import torch
from torch import nn
from torch.autograd import grad

from .base import BaseAttack

class PGDAttack(BaseAttack):
    """PGD attack class.

    Projected Gradient Descent (called PGD) is the iterative variant of FGSM.
    For more details, please refer to `"Adversarial Machine Learning at Scale" <https://arxiv.org/abs/1611.01236>`

    Args:
        model (torch.nn.Module): Model to be attacked
        criterion (torch.nn.Module): Criterion to calculate loss.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        p (int): Order of norm (-1 means infty norm).
        eps (float): Perturbation size (before normalization).
        alpha (float): Perturbation size for single step.
        num_steps (int): Number of steps.
        num_restarts (int): Times of restarts.
        randomize (bool): If True, apply random initialization
    """
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), mean=None, std=None, p=-1, eps=5.0/255.0, alpha=2.0/255.0, num_steps=5, num_restarts=1, randomize=True):
        BaseAttack.__init__(self, model, criterion, mean, std)
        self.p = p
        self.eps = self._scale(eps)[None, :, None, None]
        self.alpha = self._scale(alpha)[None, :, None, None]
        self.num_steps = num_steps
        self.num_restarts = num_restarts
        self.randomize = randomize

    def attack(self, x, t):
        """Given pais of (``x``, ``t``), returns adversarial examples.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.LongTensor): Label tensor.
        """
        raise NotImplementedError