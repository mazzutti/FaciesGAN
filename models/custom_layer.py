import torch.nn as nn


class ConvBlock(nn.Sequential):
    """Convolutional block with Conv2D, BatchNorm, and LeakyReLU.

    A standard building block for both generator and discriminator networks,
    combining convolution, batch normalization, and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    padding : int
        Amount of padding to add.
    stride : int
        Stride of the convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
    ) -> None:
        super(ConvBlock, self).__init__()

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
