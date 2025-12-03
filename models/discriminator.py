import torch
import torch.nn as nn

from models.custom_layer import ConvBlock


class Discriminator(nn.Module):
    """Multi-layer discriminator for FaciesGAN using WGAN-GP.

    Distinguishes between real and generated facies images using a series
    of convolutional blocks with progressively decreasing channel counts.

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
    img_num_channel : int
        Number of input facies channels.
    """

    def __init__(
        self,
        num_features: int,
        min_num_features: int,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        img_num_channel: int,
    ) -> None:

        super(Discriminator, self).__init__()  # type: ignore

        self.head = ConvBlock(img_num_channel, num_features, kernel_size, padding_size, 1)

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

        out_channels = max(num_features // (2 ** (num_layer - 2)), min_num_features)
        self.tail = nn.Conv2d(
            out_channels, 1, kernel_size=kernel_size, stride=1, padding=padding_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate input facies images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing facies images.

        Returns
        -------
        torch.Tensor
            Discrimination scores (higher for more realistic images).
        """
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x
