import importlib
import os
import copy
import time
from collections import defaultdict
import numpy as np
import torch

import src.trainers.base
import src.util.viz
import src.util.transform


class Train1(src.trainers.base.TrainerBase):
    def __init__(self, dataloaderfactory, model, state_path):
        super().__init__(dataloaderfactory, model, state_path)

        # Freeze encoder params
        for l in self.model.encoder_layers:
            for param in l.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.trainer_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.4,
            min_lr=1e-7,
            patience=2,
            verbose=True)

    def run(self):
        # print input and label
        images, masks = next(iter(self.trainloader))
        src.util.viz.print_images_masks(
            tuple(src.util.transform.reverse_transform(x) for x in images[:3]),
            tuple(np.transpose(x, (1, 2, 0)) for x in masks[:3].clone().numpy().astype(np.uint8)),
            printfolder=os.path.join(os.getcwd(), self.config.output_printfolder),
            printfile="traininput.png")

        # Train and validate
        best_loss = 0.2
        trloss = None
        valoss = None
        for epnr in range(self.config.trainer_numepochs):
            print(f"Ep {epnr} {'-' * 10}")
            trloss = self._train()
            valoss = self._validate()

            # Save model
            if valoss < best_loss:
                print("saving best model")
                best_loss = valoss
                state = {
                    "best_loss": best_loss,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                savepath = os.path.join(self.config.output_modelfolder, f"model_{best_loss:.3f}.pth")
                torch.save(state, savepath)
                print("Ended train!")

    def _train(self):
        self.model.train()
        metrics_dict = defaultdict(float)
        num_ep_samples = 0

        for inputs, labels in self.trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.calc_loss(outputs, labels, metrics_dict)
                loss.backward()
                self.optimizer.step()
            num_ep_samples += inputs.size(0)

        self.print_metrics(metrics_dict, num_ep_samples, "train")

        epoch_loss = metrics_dict['loss'] / num_ep_samples
        return epoch_loss

    def _validate(self):
        self.model.eval()
        metrics_dict = defaultdict(float)
        num_ep_samples = 0

        for inputs, labels in self.valloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.calc_loss(outputs, labels, metrics_dict)

            num_ep_samples += inputs.size(0)

        self.print_metrics(metrics_dict, num_ep_samples, "val")

        epoch_loss = metrics_dict['loss'] / num_ep_samples
        self.scheduler.step(epoch_loss)

        return epoch_loss
