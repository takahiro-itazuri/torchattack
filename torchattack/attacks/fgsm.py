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
        p (int): Order of norm (-1 means infty norm).
        eps (float): Perturbation size (before normalization).
    """
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), mean=None, std=None, p=-1, eps=5.0/255.0):
        BaseAttack.__init__(self, model, criterion, mean, std)
        self.p = p
        self.eps = self._scale(eps)[None, :, None, None]

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

        x_adv = x + delta
        y = self.model(x_adv if x_adv.shape[1] == 3 else x_adv.repeat(1, 3, 1, 1))
        loss = self.criterion(y, t)
        grad_delta = grad(loss, delta)[0].detach()

        if self.p == -1:
            delta = eps * grad_delta.sign()
        elif self.p >= 1:
            grad_delta_norm = self._calc_norm(grad_delta)
            delta = eps * grad_delta / grad_delta_norm
        else:
            raise NotImplementedError
        
        return self._clamp(x + delta).detach()

