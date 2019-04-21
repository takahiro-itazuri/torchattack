import torch
from torch import nn

from abc import *

class BaseAttack(ABC):
    """Base class for adversarial attacks.

    Args:
        model (torch.nn.Module): Model to be attacked
        criterion (torch.nn.Module): Criterion to calculate loss.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), mean=None, std=None):
        self.model = model
        self.criterion = criterion
        self.mean = mean if torch.is_tensor(mean) else torch.tensor(mean)
        self.std = std if torch.is_tensor(std) else torch.tensor(std)

    @abstractmethod
    def attack(self, x, **kwargs):
        raise NotImplementedError

    def _clamp(self, x):
        """Clamp the input ``x``.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Clamped input tensor.
        """
        return self.normalize(self.unnormalize(x).clamp(min=0.0, max=1.0))

    def _normalize(self, x):
        """Normalize batch of samples.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized input tensor.
        """
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        return x.sub(mean[None, :, None, None]).div(std[None, :, None, None])

    def _unnormalize(self, x):
        """Unnormalize batch of sampels.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Unnormalized input tensor.
        """
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        return x.mul(std[None, :, None, None]).add(mean[None, :, None, None])

    def _is_tensor_image(self, x):
        """Judge whether the input ``x`` is a tensor image or not.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if the input is a tensor image, False otherwise.
        """
        return torch.is_tensor(x) and x.ndimension == 4

    def _scale(self, x):
        """ Scale the input value ``x`` based on the ``self.std``.

        Args:
            x (float): Input value

        Returns:
            torch.Tensor: Scaled input.
        """
        return torch.tensor([x]).expand_as(std) / std
