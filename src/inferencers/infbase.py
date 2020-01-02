from abc import ABC, abstractmethod

class Inferencer(ABC):

    @abstractmethod
    def run(self):
        pass