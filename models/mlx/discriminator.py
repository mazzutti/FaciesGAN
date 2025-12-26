import mlx.nn as nn  # type: ignore
import mlx.core as mx

from models.discriminator import Discriminator
from typing import cast

from models.mlx.custom_layer import MLXSPADEDiscriminator


class MLXDiscriminator(Discriminator[mx.array, nn.Module], nn.Module):
    """Convolutional critic for facies images (WGAN-GP compatible) in MLX.


    The discriminator produces a single-channel feature map of scores; for a
    given input tensor of shape ``(B, H, W, C)`` the output shape is
    ``(B, H_out, W_out, 1)`` where ``H_out``/``W_out`` depend on padding and
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
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        Discriminator.__init__(  # type: ignore
            self, num_layer, kernel_size, padding_size, input_channels
        )
        nn.Module.__init__(self)

        self.dtype = dtype
        self.discs: list[nn.Module] = list()
        self.set_dtype(self.dtype)

    def __call__(self, scale: int, input_tensor: mx.array) -> mx.array:
        return super().__call__(scale, input_tensor)

    def forward(self, scale: int, input_tensor: mx.array) -> mx.array:
        """Discriminate input tensor and return score map tensor."""
        output_tensor = cast(MLXSPADEDiscriminator, self.discs[scale])(input_tensor)  # type: ignore
        return output_tensor

    def create_scale(self, num_features: int, min_num_features: int) -> None:
        spade_gen = MLXSPADEDiscriminator(
            num_layer=self.num_layer,
            kernel_size=self.kernel_size,
            padding_size=self.padding_size,
            num_features=num_features,
            min_num_features=min_num_features,
            input_channels=self.input_channels,
            dtype=self.dtype,
        )
        self.discs.append(spade_gen)
