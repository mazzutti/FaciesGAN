from typing import cast
import mlx.core as mx
import mlx.nn as nn  # type: ignore


class MLXConvBlock(nn.Module):
    """Convolutional block composed of Conv2d, BatchNorm and LeakyReLU.

    Parameters
    ----------
    in_channels : int
        Number of input channels (C in NHWC).
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size (assumed square).
    padding : int
        Padding applied to the convolution.
    stride : int
        Convolution stride.

    Notes
    -----
    - Uses MLX modules and therefore expects NHWC-formatted arrays.
    - Activation is a LeakyReLU with negative slope 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm(num_features=out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.norm(x)
        x = cast(mx.array, nn.leaky_relu(x, negative_slope=0.2))  # type: ignore
        return x


class MLXLeakyReLU(nn.Module):
    """LeakyReLU wrapper exposed as an `nn.Module`-compatible callable.

    This small module allows using a leaky ReLU as an element of lists
    of modules (e.g. ``mlp_shared``) where a callable object is
    expected to implement the module interface.
    """

    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, x: mx.array) -> mx.array:
        return cast(mx.array, nn.leaky_relu(x, negative_slope=self.negative_slope))  # type: ignore


class MLXTanh(nn.Module):
    """Tanh activation wrapper as an `nn.Module`.

    This small module allows using `mx.tanh` inside lists of modules
    (for example, the `tail` list in `MLXScaleModule`).
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x)


class MLXSPADE(nn.Module):
    """Spatially-Adaptive Denormalization (SPADE) implementation.

    SPADE modulates normalized activations with spatially-varying
    ``gamma`` and ``beta`` parameters produced by an MLP applied to a
    conditioning tensor (e.g. segmentation map or color hints).

    Parameters
    ----------
    norm_nc : int
        Number of channels in the normalized feature map.
    cond_nc : int
        Number of channels in the conditioning tensor.
    hidden_nc : int, optional
        Hidden channels used inside the shared MLP, by default 64.
    kernel_size : int, optional
        Kernel size for conv layers inside the MLP, by default 3.

    Notes
    -----
    - Input tensors are expected in NHWC layout (B, H, W, C).
    - The internal InstanceNorm does not use affine parameters.
    """

    def __init__(
        self,
        norm_nc: int,
        cond_nc: int,
        hidden_nc: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        # MLX InstanceNorm works on (B, H, W, C)
        self.norm = nn.InstanceNorm(dims=norm_nc, affine=False)

        padding = kernel_size // 2

        self.mlp_shared: list[nn.Module] = [
            nn.Conv2d(
                cond_nc,
                hidden_nc,
                kernel_size=kernel_size,
                padding=padding,
            ),
            MLXLeakyReLU(0.2),
        ]

        self.mlp_gamma = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )
        self.mlp_beta = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )

    def __call__(self, x: mx.array, conditioning_input: mx.array) -> mx.array:
        # Normalize the input features (B, H, W, C)
        normalized = self.norm(x)

        # Resize conditioning to match feature map size if needed
        # x.shape is (B, H, W, C)
        if conditioning_input.shape[1:3] != x.shape[1:3]:
            conditioning_input = cast(
                mx.array,
                mx.image.resize(  # type: ignore
                    conditioning_input,
                    x.shape[1],
                    x.shape[2],
                ),
            )

        # Generate spatially-varying gamma and beta
        activated_input: mx.array = conditioning_input
        for layer in self.mlp_shared:
            activated_input = cast(mx.array, layer(activated_input))  # type: ignore

        gamma = self.mlp_gamma(activated_input)
        beta = self.mlp_beta(activated_input)

        return normalized * (1 + gamma) + beta


class MLXSPADEConvBlock(nn.Module):
    """Convolutional block that applies SPADE followed by a conv.

    Parameters
    ----------
    in_channels : int
        Input feature channels.
    out_channels : int
        Output feature channels.
    cond_channels : int
        Conditioning tensor channels fed to the SPADE module.
    kernel_size : int
        Convolution kernel size.
    padding : int
        Convolution padding.
    stride : int
        Convolution stride.
    spade_hidden : int, optional
        Hidden size for SPADE MLP, by default 64.
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
        super().__init__()

        self.spade = MLXSPADE(in_channels, cond_channels, spade_hidden, kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        x = self.spade(x, cond)
        x = cast(mx.array, nn.leaky_relu(x, negative_slope=0.2))  # type: ignore
        x = self.conv(x)
        return x


class MLXSPADEGenerator(nn.Module):
    """SPADE-based generator composed of a head, a sequence of SPADE
    convolutional blocks and a tail conv layer.

    Parameters
    ----------
    num_layer : int
        Total number of convolutional layers (including head and tail).
    kernel_size : int
        Kernel size for convolutions.
    padding_size : int
        Padding size for convolutions.
    num_features : int
        Number of features at the first layer.
    min_num_features : int
        Minimum features allowed when halving channels in deeper blocks.
    output_channels : int
        Number of channels in the generated output (e.g. RGB channels).
    input_channels : int
        Channels in the conditioning input passed to the generator.

    Returns
    -------
    mx.array
        Generated image tensor in NHWC layout.
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
        super().__init__()

        self.init_conv = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=kernel_size,
            padding=padding_size,
        )

        self.spade_blocks: list[MLXSPADEConvBlock] = []

        current_features = num_features
        for i in range(num_layer - 2):
            block_features = int(num_features / pow(2, (i + 1)))
            out_ch = max(block_features, min_num_features)
            in_ch = current_features

            self.spade_blocks.append(
                MLXSPADEConvBlock(
                    in_ch, out_ch, input_channels, kernel_size, padding_size, 1
                )
            )
            current_features = out_ch

        self.tail_conv = nn.Conv2d(
            current_features,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding_size,
        )

    def __call__(self, cond: mx.array) -> mx.array:
        # Initial feature extraction
        x = self.init_conv(cond)
        x = cast(mx.array, nn.leaky_relu(x, negative_slope=0.2))  # type: ignore

        # Apply SPADE blocks
        for block in self.spade_blocks:
            x = block(x, cond)

        # Final output layer
        out = mx.tanh(self.tail_conv(x))
        return out


class MLXScaleModule(nn.Module):
    """Sequential-like scale module composed of head, body and tail.

    This wrapper exposes a single-callable module that applies the
    provided callables/modules in sequence: head -> *body -> *tail.
    """

    def __init__(
        self,
        head: MLXSPADEGenerator | MLXConvBlock,
        body: nn.Sequential,
        tail: nn.Sequential,
    ) -> None:
        super().__init__()
        self.head = head
        self.body = body
        self.tail = tail

    def __call__(self, x: mx.array) -> mx.array:
        x = self.head(x)
        x = cast(mx.array, self.body(x))
        x = cast(mx.array, self.tail(x))
        return x


class MLXSPADEDiscriminator(nn.Module):
    """Patch-style discriminator built from conv blocks.

    Parameters
    ----------
    num_features : int
        Base number of features for the discriminator head.
    min_num_features : int
        Minimum number of features in deeper layers.
    num_layer : int
        Number of convolutional layers.
    kernel_size : int
        Kernel size for convolutions.
    padding_size : int
        Convolution padding to preserve spatial dimensions where desired.
    input_channels : int
        Channels in the input tensor.
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
        super().__init__()

        self.head = MLXConvBlock(
            input_channels, num_features, kernel_size, padding_size, 1
        )

        self.body: list[MLXConvBlock] = []
        for i in range(num_layer - 2):
            in_ch = max(num_features // (2**i), min_num_features)
            out_ch = max(num_features // (2 ** (i + 1)), min_num_features)
            self.body.append(MLXConvBlock(in_ch, out_ch, kernel_size, padding_size, 1))

        output_channels = max(num_features // (2 ** (num_layer - 2)), min_num_features)
        self.tail = nn.Conv2d(
            output_channels, 1, kernel_size=kernel_size, stride=1, padding=padding_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.head(x)
        for block in self.body:
            x = block(x)
        return self.tail(x)


class MLXColorQuantization(nn.Module):
    """Simple color quantization module offering hard and soft modes.

    The module projects input RGB-like values to a small set of
    predefined "pure" colors. When ``training`` is False a hard
    nearest-neighbor assignment is performed; when ``training`` is True
    a soft, differentiable weighted average is returned.

    Parameters
    ----------
    temperature : float, optional
        Temperature controlling the softness of the soft-assignment
        distribution; lower values make the distribution peakier.

    Returns
    -------
    mx.array
        Quantized colors with the same spatial layout as the input
        (NHWC).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

        # Pure colors: (4, 3)
        self.pure_colors = mx.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        b, h, w, c = x.shape
        x_flat = x.reshape(-1, c)  # (N, 3) # type: ignore

        # Distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * (x @ c.T)
        x_norm = mx.sum(x_flat**2, axis=1, keepdims=True)
        c_norm = mx.sum(self.pure_colors**2, axis=1, keepdims=True)
        distances = x_norm + c_norm.T - 2 * (x_flat @ self.pure_colors.T)

        if not training:
            # Hard quantization
            indices = mx.argmin(distances, axis=1)
            quantized = self.pure_colors[indices]
            return quantized.reshape(b, h, w, c)  # type: ignore
        else:
            # Soft quantization (differentiable)
            weights = mx.softmax(-distances / self.temperature, axis=1)
            quantized = weights @ self.pure_colors
            return quantized.reshape(b, h, w, c)  # type: ignore
