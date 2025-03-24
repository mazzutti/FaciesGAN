import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int):
        """
        A convolutional block that includes a Conv2D layer, BatchNorm2D, and LeakyReLU activation.
        """
        super(ConvBlock, self).__init__()

        self.add_module("conv", nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))

