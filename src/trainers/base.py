import src
import torch

class TrainerBase:
    def __init__(self, dataloaderfactory, model, state_path):
        self.optimizer = None

        self.config = src.util.config.Config()
        self.dataloaderfactory = dataloaderfactory
        self.trainloader = dataloaderfactory.gettrainloader()
        self.valloader = dataloaderfactory.getvalloader()
        self.model = model
        self.device = torch.device("cuda:0")
        self.model.to(self.device)

        if state_path is not None:
            state = torch.load(state_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(state["state_dict"])
            print(f"Using state: {state_path}")

    def calc_loss(self, pred, target, metrics, bce_weight=0.5):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        return loss

    def dice_loss(self, pred, target, smooth=1.):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()

    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        print("  {}: {}".format(phase, ", ".join(outputs)))
        if phase == "val":
            print("  LR", self.optimizer.param_groups[0]['lr'])
