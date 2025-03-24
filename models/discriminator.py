import torch
import torch.nn as nn

from models.custom_layer import ConvBlock


class Discriminator(nn.Module):
    """
    A class representing the Discriminator models.

    Args:
        num_features (int): The number of features for the convolutional layers.
        min_num_features (int): The minimum number of features for the convolutional layers.
        num_layer (int): The number of layers in the discriminator.
        kernel_size (int): The size of the convolutional kernel.
        padding_size (int): The size of the padding.
        img_num_channel (int): The number of facie channels.
    """
    def __init__(self,
                 num_features: int,
                 min_num_features: int,
                 num_layer: int,
                 kernel_size: int,
                 padding_size: int,
                 img_num_channel: int):

        super(Discriminator, self).__init__()

        self.head = ConvBlock(img_num_channel, num_features, kernel_size, padding_size, 1)

        self.body = nn.Sequential(*[
            ConvBlock(
                max(num_features // (2 ** i), min_num_features),
                max(num_features // (2 ** (i + 1)), min_num_features),
                kernel_size,
                padding_size,
                1
            ) for i in range(num_layer - 2)
        ])

        out_channels = max(num_features // (2 ** (num_layer - 2)), min_num_features)
        self.tail = nn.Conv2d(out_channels, 1, kernel_size=kernel_size, stride=1, padding=padding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x
