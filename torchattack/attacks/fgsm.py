import torch
from torch import nn

from .base import BaseAttack

class FGSMAttack(BaseAttack):
    def attack(self, x, t=None):
        return NotImplementedError

