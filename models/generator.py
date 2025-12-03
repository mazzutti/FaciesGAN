import math



import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom_layer import ConvBlock
from ops import interpolate


class Generator(nn.Module):
    """
    A class representing the Generator models.

    Args:
        num_layer (int): Number of layers in the generator.
        kernel_size (int): Size of the convolutional kernel.
        padding_size (int): Size of the padding.
        img_num_channel (int): Number of facie channels.
    """

    def __init__(self, num_layer: int, kernel_size: int, padding_size: int, img_num_channel: int):
        super(Generator, self).__init__()  # type: ignore

        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.img_num_channel = img_num_channel
        self.zero_padding = self.num_layer * math.floor(self.kernel_size / 2)
        self.full_zero_padding = 2 * self.zero_padding
        self.gens = nn.ModuleList()

    def forward(
        self,
        z: list[torch.Tensor],
        amp: list[float],
        in_noise: torch.Tensor | None = None,
        start_scale: int = 0,
        stop_scale: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the generator.

        Args:
            z (list[torch.Tensor]): List of noise tensors for each scale.
            amp (list[float]): List of amplitude values for each scale.
            in_facie (torch.Tensor, optional): Input facie tensor. Defaults to None.
            start_scale (int, optional): Starting scale index. Defaults to 0.
            stop_scale (int, optional): Stopping scale index. Defaults to None.

        Returns:
            torch.Tensor: Output facie tensor.
        """
        if in_noise is None:
            channels = z[start_scale].shape[1] - 1
            height, width = tuple(dim - self.full_zero_padding for dim in z[start_scale].shape[2:])
            facie = torch.zeros(
                (z[start_scale].shape[0], channels, height, width),
                device=z[start_scale].device,
            )
        else:
            facie: torch.Tensor  = in_noise

        stop_scale = stop_scale if stop_scale is not None else len(self.gens) - 1


        for index in range(start_scale, stop_scale + 1):
            
            facie = interpolate(
                facie,
                (
                    z[index].shape[2] - self.full_zero_padding,
                    z[index].shape[3] - self.full_zero_padding,
                ),
            )

            z_in = torch.zeros_like(z[index])
            z_in[:, 1, :, :] = z[index][:, 1, :, :]
            z_in[:, 0, :, :] = amp[index] * z[index][:, 0, :, :]
            z_in = z_in + F.pad(facie, [self.zero_padding] * 4, value=0)

            facie = self.gens[index](z_in) + facie
        return facie

    def create_scale(self, num_feature: int, min_num_feature: int) -> None:
        """
        Create a new scale for the generator.

        Args:
            num_feature (int): The number of features for the convolutional layers.
            min_num_feature (int): The minimum number of features for the convolutional layers.
        """
        head = ConvBlock(self.img_num_channel, num_feature, self.kernel_size, self.padding_size, 1)
        body = nn.Sequential()

        channels = min_num_feature
        for i in range(self.num_layer - 2):
            channels = int(num_feature / pow(2, (i + 1)))
            block = ConvBlock(
                max(2 * channels, min_num_feature),
                max(channels, min_num_feature),
                self.kernel_size,
                self.padding_size,
                1,
            )
            body.add_module(f"block{i + 1}", block)

        tail = nn.Sequential(
            nn.Conv2d(
                max(channels, min_num_feature),
                self.img_num_channel - 1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding_size,
            ),
            nn.Tanh(),
        )

        self.gens.append(nn.Sequential(head, body, tail))
