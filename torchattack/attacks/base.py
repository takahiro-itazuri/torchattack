import torch
from torch import nn

from abc import *

class BaseAttack(ABC):
    """Base class for adversarial attacks.

    Args:
        model (torch.nn.Module): Model to be attacked
        criterion (torch.nn.Module): Criterion to calculate loss.
    """
    def init(self, model, criterion=nn.CrossEntropyLoss()):
        self.model = model
        self.criterion = criterion

    @abstractmethod
    def attack(self, x, **kwargs):
        raise NotImplementedError
