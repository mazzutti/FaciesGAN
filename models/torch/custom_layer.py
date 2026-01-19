"""Custom neural network layers used by the generator and discriminator.

This module implements building blocks such as :class:`ConvBlock`, SPADE
normalization and SPADE-based generator blocks. These helpers are
convenience modules that keep layer definitions and small forward
utilities close to the model implementations.

All public classes in this module are documented with NumPy-style
docstrings describing parameters and returns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchConvBlock(nn.Sequential):
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
        """Initialize a ConvBlock sequential module.

        Parameters are documented on the class level.
        """
        super(TorchConvBlock, self).__init__()

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


class TorchSPADE(nn.Module):
    """Spatially-Adaptive Denormalization (SPADE) layer.

    Modulates normalized feature maps using spatially-varying scale (gamma)
    and bias (beta) learned from a conditioning input (noise + wells).

    This allows the noise to have a stronger, more meaningful impact on the
    generation by learning how to transform the features based on the noise
    at each spatial location.

    Reference: Park et al., "Semantic Image Synthesis with Spatially-Adaptive
    Normalization", CVPR 2019.

    Parameters
    ----------
    norm_nc : int
        Number of channels in the feature map to be normalized.
    cond_nc : int
        Number of channels in the conditioning input (noise + wells).
    hidden_nc : int, optional
        Number of hidden channels in the SPADE mlp. Defaults to 64.
    kernel_size : int, optional
        Kernel size for convolutions. Defaults to 3.
    """

    def __init__(
        self,
        norm_nc: int,
        cond_nc: int,
        hidden_nc: int = 64,
        kernel_size: int = 3,
    ) -> None:
        """Initialize the SPADE normalization module.

        Parameters are documented on the class level.
        """
        super().__init__()  # type: ignore

        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        padding = kernel_size // 2

        # Shared convolution for processing conditioning input
        self.mlp_shared = nn.Sequential(
            *(
                nn.Conv2d(cond_nc, hidden_nc, kernel_size=kernel_size, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # Separate convolutions for gamma (scale) and beta (bias)
        self.mlp_gamma = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )
        self.mlp_beta = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply SPADE normalization.

        Parameters
        ----------
        x : torch.Tensor
            Feature map to normalize, shape (B, norm_nc, H, W).
        cond : torch.Tensor
            Conditioning input (noise + wells), shape (B, cond_nc, H', W').
            Will be resized to match x if needed.

        Returns
        -------
        torch.Tensor
            Modulated feature map with same shape as x.
        """
        # Normalize the input features
        normalized = self.norm(x)

        # Resize conditioning to match feature map size if needed
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(
                cond, size=x.shape[2:], mode="bilinear", align_corners=True
            )

        # Generate spatially-varying gamma and beta from conditioning
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Apply modulation: out = gamma * normalized + beta
        return normalized * (1 + gamma) + beta


class TorchSPADEConvBlock(nn.Module):
    """Convolutional block with SPADE normalization for noise-conditioned generation.

    Replaces BatchNorm with SPADE to allow noise to modulate features at each
    spatial location, enabling more diverse outputs from different noise inputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    cond_channels : int
        Number of conditioning channels (noise + wells).
    kernel_size : int
        Size of the convolutional kernel.
    padding : int
        Amount of padding to add.
    stride : int
        Stride of the convolution.
    spade_hidden : int, optional
        Hidden channels in SPADE mlp. Defaults to 64.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        spade_hidden: int = 64,
    ) -> None:
        """Initialize the SPADEConvBlock used inside SPADEGenerator.

        Parameters are documented on the class level.
        """
        super().__init__()  # type: ignore

        self.spade = TorchSPADE(in_channels, cond_channels, spade_hidden, kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with SPADE conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.
        cond : torch.Tensor
            Conditioning input (noise + wells).

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        x = self.spade(x, cond)
        x = self.activation(x)
        x = self.conv(x)
        return x


class TorchSPADEGenerator(nn.Module):
    """SPADE-based generator block for the coarsest scale.

    Uses SPADE normalization to inject noise into the generation process,
    allowing the network to learn how noise should modulate features at
    each spatial location. This produces more diverse outputs compared
    to simple concatenation.

    Parameters
    ----------
    num_layer : int
        Number of convolutional layers.
    kernel_size : int
        Size of convolutional kernels.
    padding_size : int
        Padding size for convolutions.
    num_feature : int
        Number of features in the first layer.
    min_num_feature : int
        Minimum number of features.
    img_channels : int
        Number of output image channels (e.g., 3 for RGB facies).
    cond_channels : int
        Number of conditioning channels (noise + wells).
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        num_features: int,
        min_num_features: int,
        output_channels: int,
        input_channels: int,
    ) -> None:
        """Initialize SPADEGenerator parameters and layers.

        Parameters are documented on the class level.
        """
        super().__init__()  # type: ignore

        self.init_conv = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=kernel_size,
            padding=padding_size,
        )

        # SPADE blocks for the body
        self.spade_blocks = nn.ModuleList()

        block_features = min_num_features
        for i in range(num_layer - 2):
            block_features = int(num_features / pow(2, (i + 1)))
            out_ch = max(block_features, min_num_features)
            in_ch = (
                num_features if i == 0 else max(2 * block_features, min_num_features)
            )

            self.spade_blocks.append(
                TorchSPADEConvBlock(
                    in_ch, out_ch, input_channels, kernel_size, padding_size, 1
                )
            )
            num_features = out_ch

        # Final output layer
        self.tail = nn.Sequential(
            nn.Conv2d(
                max(block_features, min_num_features),
                output_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding_size,
            ),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Generate output from conditioning input using SPADE modulation.

        Parameters
        ----------
        cond : torch.Tensor
            Conditioning input containing noise and well data,
            shape (B, cond_channels, H, W).

        Returns
        -------
        torch.Tensor
            Generated facies image, shape (B, img_channels, H, W).
        """
        # Initial feature extraction from conditioning
        x = self.init_conv(cond)
        x = F.leaky_relu(x, 0.2)

        # Apply SPADE blocks - each block uses conditioning to modulate features
        for spade_block in self.spade_blocks:
            x = spade_block(x, cond)

        # Generate final output
        out = self.tail(x)
        return out


class TorchSPADEDiscriminator(nn.Module):

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
        # Initialize both the framework-agnostic base and the PyTorch module

        nn.Module.__init__(self)  # type: ignore

        self.head = TorchConvBlock(
            input_channels, num_features, kernel_size, padding_size, 1
        )

        self.body = nn.Sequential(
            *[
                TorchConvBlock(
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


class TorchColorQuantization(nn.Module):
    """Quantize output to a small set of pure colors.

    During training this module performs a soft (differentiable) assignment
    of each pixel to a small palette of pure colors using a temperature-
    scaled softmax over negative squared distances. During evaluation it
    performs a hard nearest-color lookup to produce discrete colors.

    The palette is registered as a buffer and expects generator outputs in
    the ``[-1, 1]`` range (tanh output).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        """Create a ColorQuantization module.

        Parameters
        ----------
        temperature : float, optional
            Softmax temperature used during training for soft assignments.
        """
        super().__init__()  # type: ignore
        self.temperature = temperature

        # Define pure colors in [-1, 1] range (tanh output range)
        # Black, Red, Green, Blue
        self.register_buffer(
            "pure_colors",
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )
        self.pure_colors: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize RGB output to pure colors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 3, H, W) with values in [-1, 1].

        Returns
        -------
        torch.Tensor
            Quantized tensor with same shape.
        """
        if not self.training:
            # Hard quantization during inference
            b, c, h, w = x.shape
            # Use contiguous to avoid view issues
            x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, 3)

            # Calculate distances to pure colors using explicit operations
            # Avoid cdist which can create complex views
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x*c
            x_norm = (x_flat**2).sum(dim=1, keepdim=True)
            c_norm = (self.pure_colors**2).sum(dim=1, keepdim=True)
            distances = x_norm + c_norm.t() - 2 * torch.mm(x_flat, self.pure_colors.t())

            # Get nearest color
            indices = torch.argmin(distances, dim=1)
            quantized = self.pure_colors[indices]

            return quantized.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            # Soft quantization during training (differentiable)
            b, c, h, w = x.shape
            # Use contiguous to avoid view issues
            x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, 3)

            # Calculate distances using explicit operations (avoid cdist)
            x_norm = (x_flat**2).sum(dim=1, keepdim=True)
            c_norm = (self.pure_colors**2).sum(dim=1, keepdim=True)
            distances = x_norm + c_norm.t() - 2 * torch.mm(x_flat, self.pure_colors.t())

            # Soft assignment using softmax
            weights = F.softmax(-distances / self.temperature, dim=1)

            # Weighted sum of pure colors
            quantized = torch.mm(weights, self.pure_colors)

            return quantized.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
