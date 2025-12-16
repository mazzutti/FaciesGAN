"""Discriminator module used by FaciesGAN.

Provides a multi-layer convolutional discriminator with a small API that
returns per-pixel critic scores (suitable for WGAN-style losses).
"""

import torch
import torch.nn as nn

from models.custom_layer import ConvBlock


class Discriminator(nn.Module):
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
        num_features: int,
        min_num_features: int,
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
        super(Discriminator, self).__init__()  # type: ignore

        self.head = ConvBlock(
            input_channels, num_features, kernel_size, padding_size, 1
        )

        self.body = nn.Sequential(
            *[
                ConvBlock(
                    max(num_features // (2**i), min_num_features),
                    max(num_features // (2 ** (i + 1)), min_num_features),
                    kernel_size,
                    padding_size,
                    1,
                )
                for i in range(num_layer - 2)
            ]
        )

        output_channels = max(num_features // (2 ** (num_layer - 2)), min_num_features)
        self.tail = nn.Conv2d(
            output_channels, 1, kernel_size=kernel_size, stride=1, padding=padding_size
        )

    def forward(self, generated_facie: torch.Tensor) -> torch.Tensor:
        """Discriminate input facies images.

        Parameters
        ----------
        generated_facie : torch.Tensor
            Input tensor containing facies images.

        Returns
        -------
        torch.Tensor
            Discrimination scores with shape ``(B, 1, H_out, W_out)``. The
            returned tensor is not reduced to a scalar so callers can compute
            patch-wise or global losses as required.
        """
        scores = self.head(generated_facie)
        scores = self.body(scores)
        scores = self.tail(scores)

        return scores
