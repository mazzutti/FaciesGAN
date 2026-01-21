"""Discriminator module used by FaciesGAN.

Provides a multi-layer convolutional discriminator with a small API that
returns per-pixel critic scores (suitable for WGAN-style losses).
"""

from typing import Self, cast

import torch
import torch.nn as nn

from models.discriminator import Discriminator
from models.torch.custom_layer import TorchSPADEDiscriminator


class TorchDiscriminator(Discriminator[torch.Tensor, nn.Module], nn.Module):
    """Convolutional critic for facies images (WGAN-GP compatible).

    The discriminator produces a single-channel feature map of scores; for a
    given input tensor of shape ``(B, C, H, W)`` the output shape is
    ``(B, 1, H_out, W_out)`` where ``H_out``/``W_out`` depend on padding and
    kernel sizes. Higher values indicate more-realistic patches.

    Parameters
    ----------
    num_features : int
        Number of features in the first convolutional layer.
    min_num_features : int
        Minimum number of features (floor for channel reduction).
    num_layer : int
        Number of convolutional layers in the discriminator.
    kernel_size : int
        Size of convolutional kernels.
    padding_size : int
        Padding size for convolutions.
    input_channels : int
        Number of input facies channels.

    Attributes
    ----------
    head : ConvBlock
        Initial convolutional block.
    body : nn.Sequential
        Intermediate convolutional blocks.
    tail : nn.Conv2d
        Final 1-channel convolution producing scores.
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
    ) -> None:
        """Initialize the convolutional discriminator.

        Parameters
        ----------
        num_features : int
            Number of features in the first convolutional layer.
        min_num_features : int
            Minimum number of features used when reducing channels.
        num_layer : int
            Number of convolutional layers.
        kernel_size : int
            Convolution kernel size.
        padding_size : int
            Padding applied to convolutions.
        input_channels : int
            Number of input image channels.
        """
        # Initialize both the framework-agnostic base and the PyTorch module
        Discriminator.__init__(  # type: ignore
            self, num_layer, kernel_size, padding_size, input_channels
        )
        nn.Module.__init__(self)  # type: ignore

        self.discs: list[nn.Module] = list()

    def __call__(self, scale: int, input_tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(scale, input_tensor)

    def eval(self) -> Self:
        """Set the module in evaluation mode.

        Returns
        -------
        Self
            The discriminator instance in evaluation mode.
        """
        return cast(Self, nn.Module.eval(self))

    def forward(self, scale: int, input_tensor: torch.Tensor) -> torch.Tensor:
        """Discriminate input tensor and return score map tensor."""
        return self.discs[scale](input_tensor)

    def create_scale(self, num_features: int, min_num_features: int) -> None:
        spade_gen = TorchSPADEDiscriminator(
            num_layer=self.num_layer,
            kernel_size=self.kernel_size,
            padding_size=self.padding_size,
            num_features=num_features,
            min_num_features=min_num_features,
            input_channels=self.input_channels,
        )
        self.discs.append(spade_gen)
