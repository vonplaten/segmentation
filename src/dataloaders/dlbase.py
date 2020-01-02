from abc import ABC, abstractmethod
import torchvision

import src.util.config

class DataloaderFactory(ABC):
    def __init__(self):
        super().__init__()
        self.config = src.util.config.Config()
        self.trainloader = None
        self.valloader = None
        self.testloader = None

    @abstractmethod
    def gettrainloader(self):
        pass

    @abstractmethod
    def getvalloader(self):
        pass

    @abstractmethod
    def gettestloader(self):
        pass

    def _gettransform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform
