import torch
from torch import nn

from abc import *

class BaseAttack(ABC):
    def init(self, model, criterion=nn.CrossEntropyLoss()):
        self.model = model
        self.criterion = criterion

    @abstractmethod
    def attack(self, x, **kwargs):
        raise NotImplementedError
