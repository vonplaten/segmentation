import torch

import src.inferencers.infbase

class Inf1(src.inferencers.infbase.Inferencer):
    def __init__(self, dataloaderfactory, model, state_path):
        self.dataloaderfactory = dataloaderfactory
        self.config = src.util.config.Config()
        self.testloader = dataloaderfactory.gettestloader()
        self.model = model
        state = torch.load(state_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state["state_dict"])
        self.device = torch.device("cuda:0")
        self.model.to(self.device)

    def run(self):
        inputs, labels = next(iter(self.testloader))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        pred = self.model(inputs)
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        # # print(pred.shape)
        # if self.config.dataloader_print_inferencer_num > 0:
        #     self.dataloaderfactory.print_testexample(
        # inputs, pred, labels,
        # count=self.config.dataloader_print_inferencer_num,
        # printfolder=self.config.dataloader_printfolder, printfile="pred.png")
        # print("")


'''
        inputs, labels = next(iter(self.gettestloader()))
        inputs = inputs.to(device)
        labels = labels.to(device)

        pred = model(inputs)
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        print(pred.shape)

        # Change channel-order and make 3 channels for matplot
        input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

        # Map each channel (i.e. class) to each color
        target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
        pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

        helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

import torch
import src.util.config

class InferenceModel:
    def __init__(self, dataloaderfactory, state_path, model):
        state = torch.load(state_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        pass
        '''
