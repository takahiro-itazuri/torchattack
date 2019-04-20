import torch
from torch import nn

from .base import BaseAttack

class FGSMAttack(BaseAttack):
    def init(self, model, criterion=nn.CrossEntropyLoss()):
        self.model = model
        self.criterion = criterion

    def attack(self, x, t=None):
        return NotImplementedError

