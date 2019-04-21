import torch
from torch import nn
from torch.autograd import grad

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
        self.eps = self._scale(eps)[None, :, None, None]
        self.p = p

    def attack(self, x, t):
        """Given pais of (``x``, ``t``), returns adversarial examples.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.LongTensor): Label tensor.
        """
        assert self._is_tensor_image(x), "The input tensor should be tensor image (NCHW)."
        device = x.device
        eps = self.eps.to(device)
        delta = torch.zeros_like(x, requires_grad=True).to(device)

        perturbed_x = x + delta
        y = self.model(perturbed_x if perturbed_x.shape[1] == 3 else perturbed_x.repeat(1, 3, 1, 1))
        loss = self.criterion(y, t)
        grad_delta = grad(loss, delta)[0].detach()

        if self.p == -1:
            delta.data = eps * grad_delta.sign()
        else:
            raise NotImplementedError
        
        return self._clamp(x + delta).detach()

