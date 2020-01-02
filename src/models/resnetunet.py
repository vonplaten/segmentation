import os
import torch
import torchvision
from torchsummary import summary
import numpy as np
from contextlib import redirect_stdout

import src.util.config

class ResnetUnet18(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.config = src.util.config.Config()
        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.encoder_layers = list(self.encoder.children())

        self.layer0 = torch.nn.Sequential(*self.encoder_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = self.convrelu(64, 64, 1, 0)
        self.layer1 = torch.nn.Sequential(*self.encoder_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = self.convrelu(64, 64, 1, 0)
        self.layer2 = self.encoder_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = self.convrelu(128, 128, 1, 0)
        self.layer3 = self.encoder_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = self.convrelu(256, 256, 1, 0)
        self.layer4 = self.encoder_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = self.convrelu(512, 512, 1, 0)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = self.convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = self.convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = self.convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = self.convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = self.convrelu(3, 64, 3, 1)
        self.conv_original_size1 = self.convrelu(64, 64, 3, 1)
        self.conv_original_size2 = self.convrelu(64 + 128, 64, 3, 1)

        self.conv_last = torch.nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

    def convrelu(self, in_channels, out_channels, kernel, padding):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            torch.nn.ReLU(inplace=True),
        )

    def printmodel(self, model):
        printfolder = os.path.join(os.getcwd(), self.config.model_printfolder)
        if not os.path.exists(printfolder):
            os.makedirs(printfolder)
        with open(file=os.path.join(printfolder, "model_info.txt"), mode="w") as f:
            with redirect_stdout(f):
                print(f"{list(model.children())}\n")
                summary(model.cuda(), input_size=(3, 224, 224))
                total_param = 0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        num_param = np.prod(param.size())
                        if param.dim() > 1:
                            print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                        else:
                            print(name, ':', num_param)
                        total_param += num_param
                print(f"number of trainable parameters = {total_param}")
