from abc import *

class BaseAttack(ABC):
    def init(self, model, criterion=None):
        self.model = model
        self.criterion = criterion

    @abstractmethod
    def attack(self, x, **kwargs):
        raise NotImplementedError
