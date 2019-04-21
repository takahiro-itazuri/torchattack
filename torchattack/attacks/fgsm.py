import torch
from torch import nn

from .base import BaseAttack

class FGSMAttack(BaseAttack):
    """FGSM attack class.

    Fast Gradient Sign Method (called FGSM) adds the sign of the gradient to the input.
    For more details, please refer to `"Explaining and Harnessing Adversarial Examples" <https://arxiv.org/abs/1412.6572>`

    Args:
        model (torch.nn.Module): Model to be attacked
        criterion (torch.nn.Module): Criterion to calculate loss.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        eps (float): Perturbation size (before normalization).
        p (int): Order of norm (-1 means infty norm).
    """
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), mean=None, std=None, eps=0.3, p=-1):
        BaseAttack.__init__(self, model, criterion, mean, std)
        self.eps = eps
        self.p = p

    def attack(self, x, t=None):
        return NotImplementedError

