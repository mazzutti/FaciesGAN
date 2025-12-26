from typing import Literal, cast
import mlx.core as mx
import mlx.nn as nn  # type: ignore
import mlx.nn.layers.upsample as upsample  # type: ignore
from numpy import dtype  # type: ignore


class MLXLeakyReLU(nn.LeakyReLU):
    """LeakyReLU wrapper exposed as an `nn.Module`-compatible callable.

    This small module allows using a leaky ReLU as an element of lists
    of modules (e.g. ``mlp_shared``) where a callable object is
    expected to implement the module interface.
    """

    def __init__(
        self, negative_slope: float = 0.2, dtype: mx.Dtype = mx.float32
    ) -> None:
        super().__init__(negative_slope=negative_slope)
        self.set_dtype(dtype)


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
        use_norm: bool = True,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        self.set_dtype(dtype)
        self.use_norm = use_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv.set_dtype(dtype)

        # Only initialize norm if requested
        if self.use_norm:
            self.norm = nn.InstanceNorm(dims=out_channels, affine=True)
            self.norm.set_dtype(dtype)
        # self.norm = nn.BatchNorm(num_features=out_channels)
        # self.norm.set_dtype(dtype)
        self.activation = MLXLeakyReLU(negative_slope=0.2, dtype=dtype)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = cast(mx.array, self.activation(x))
        return x


class MLXUpsample(nn.Module):
    """Upsample subclass that stores expected output size and scale.

    Subclassing `nn.Upsample` keeps the operator directly available while
    allowing a fast no-op when the input already matches the expected
    output `size` to avoid unnecessary backend work/allocations.
    """

    def __init__(
        self,
        size: tuple[int, ...],
        mode: Literal["nearest", "linear", "cubic"] = "linear",
        align_corners: bool = True,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        self.height, self.width = (int(size[0]), int(size[1]))
        self.mode = mode
        self.align_corners = align_corners
        self.set_dtype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        in_height, in_width = x.shape[1:3]
        if (in_height, in_width) == (self.height, self.width):
            return x

        scale_factor = (self.height / in_height, self.width / in_width)
        if self.mode == "nearest":
            return upsample.upsample_nearest(x, scale_factor).astype(self.dtype)  # type: ignore
        elif self.mode == "linear":
            return cast(
                mx.array,
                upsample.upsample_linear(  # type: ignore
                    x,
                    scale_factor,
                    self.align_corners,
                ),
            )
        elif self.mode == "cubic":
            return cast(
                mx.array,
                upsample.upsample_cubic(  # type: ignore
                    x,
                    scale_factor,
                    self.align_corners,
                ),
            )
        else:
            raise Exception(f"Unknown interpolation mode: {self.mode}")


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
    - The internal InstanceNorm uses affine parameters.
    """

    def __init__(
        self,
        norm_nc: int,
        cond_nc: int,
        hidden_nc: int = 64,
        kernel_size: int = 3,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()

        # MLX InstanceNorm works on (B, H, W, C)
        self.dtype = dtype
        self.set_dtype(dtype)
        self.norm = nn.InstanceNorm(dims=norm_nc, affine=True)
        self.norm.set_dtype(dtype)

        padding = kernel_size // 2

        self.mlp_shared: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                cond_nc,
                hidden_nc,
                kernel_size=kernel_size,
                padding=padding,
            ),
            MLXLeakyReLU(0.2, dtype),
        )

        _ = [layer.set_dtype(dtype) for layer in self.mlp_shared.layers]  # type: ignore

        self.mlp_gamma = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )
        self.mlp_gamma.set_dtype(dtype)
        self.mlp_beta = nn.Conv2d(
            hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding
        )
        self.mlp_beta.set_dtype(dtype)
        # Cache of upsample layers keyed by target (height, width).
        # We create them lazily on first use to avoid re-allocating
        # Upsample modules on every forward call. Optionally the caller
        # can provide a list of targets to preallocate common sizes.
        self._upsample_cache: dict[tuple[int, ...], nn.Module] = {}

    def ensure_upsample(
        self,
        target: tuple[int, ...],
        mode: Literal["linear"] = "linear",
        align_corners: bool = True,
    ) -> nn.Module:
        """Return a cached Upsample module for `target` size and optional `scale`.

        `target` is the desired output size (height, width). If `scale` is
        provided the underlying `MLXUpsample` will be created with
        `scale_factor` to satisfy backends that prefer it.
        """
        up = self._upsample_cache.get(target)
        if up is None:
            up = MLXUpsample(
                size=target,
                mode=mode,
                align_corners=align_corners,
                dtype=self.dtype,
            )
            self._upsample_cache[target] = up
        return up

    def __call__(
        self,
        x: mx.array,
        conditioning_input: mx.array,
        mode: Literal["linear"] = "linear",
        align_corners: bool = True,
    ) -> mx.array:
        activated_input = cast(
            mx.array,
            self.ensure_upsample(
                x.shape[1:3],
                mode=mode,
                align_corners=align_corners,
            )(conditioning_input),
        )
        activated_input = cast(mx.array, self.mlp_shared(activated_input))
        gamma = self.mlp_gamma(activated_input)
        beta = self.mlp_beta(activated_input)
        normalized = self.norm(x)
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
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()

        self.spade = MLXSPADE(
            in_channels, cond_channels, spade_hidden, kernel_size, dtype=dtype
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv.set_dtype(dtype)
        self.activation = MLXLeakyReLU(negative_slope=0.2, dtype=dtype)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        x = self.spade(x, cond)
        x = cast(mx.array, self.activation(x))
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
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()

        self.set_dtype(dtype)

        self.init_conv = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=kernel_size,
            padding=padding_size,
        )
        self.init_conv.set_dtype(dtype)

        current_features = num_features
        spade_blocks: list[MLXSPADEConvBlock] = []
        for i in range(num_layer - 2):
            block_features = int(num_features / pow(2, (i + 1)))
            out_ch = max(block_features, min_num_features)
            in_ch = current_features

            spade_blocks.append(
                MLXSPADEConvBlock(
                    in_ch,
                    out_ch,
                    input_channels,
                    kernel_size,
                    padding_size,
                    1,
                    dtype=dtype,
                )
            )
            current_features = out_ch
        self.spade_blocks: nn.Sequential = nn.Sequential(*spade_blocks)

        self.tail_conv = nn.Conv2d(
            current_features,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding_size,
        )
        self.tail_conv.set_dtype(dtype)
        self.leaky_relu = MLXLeakyReLU(negative_slope=0.2, dtype=dtype)
        self.activation = nn.Tanh()
        self.activation.set_dtype(dtype)

    def __call__(self, cond: mx.array) -> mx.array:
        # Initial feature extraction
        x = self.init_conv(cond)
        x = cast(mx.array, self.leaky_relu(x))

        for block in self.spade_blocks.layers:  # type: ignore
            x = cast(mx.array, block(x, cond))

        # Final output layer
        out = cast(mx.array, self.activation(self.tail_conv(x)))
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
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        self.head = head
        self.body = body
        self.tail = tail
        self.set_dtype(dtype)

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
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()

        self.head = MLXConvBlock(
            input_channels,
            num_features,
            kernel_size,
            padding_size,
            1,
            use_norm=True,
            dtype=dtype,
        )

        body: list[MLXConvBlock] = []
        for i in range(num_layer - 2):
            in_ch = max(num_features // (2**i), min_num_features)
            out_ch = max(num_features // (2 ** (i + 1)), min_num_features)
            body.append(
                MLXConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    padding_size,
                    1,
                    use_norm=True,
                    dtype=dtype,
                )
            )

        self.body = nn.Sequential(*body)

        output_channels = max(num_features // (2 ** (num_layer - 2)), min_num_features)
        self.tail = nn.Conv2d(
            output_channels, 1, kernel_size=kernel_size, stride=1, padding=padding_size
        )
        self.tail.set_dtype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.head(x)
        x = cast(mx.array, self.body(x))
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

    def __init__(
        self,
        temperature: float = 0.1,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.dtype = dtype

        # Pure colors: (4, 3)
        self.pure_colors = mx.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=self.dtype,
        )

    @staticmethod
    @mx.compile  # type: ignore
    def _quantize_impl(
        x: mx.array, pure_colors: mx.array, temperature: float, training: bool
    ) -> mx.array:
        b, h, w, c = x.shape
        x_flat = x.reshape(-1, c)  # (N, 3) # type: ignore
        x_norm = mx.sum(x_flat**2, axis=1, keepdims=True)
        c_norm = mx.sum(pure_colors**2, axis=1, keepdims=True)
        distances = x_norm + c_norm.T - 2 * (x_flat @ pure_colors.T)
        if not training:
            indices = mx.argmin(distances, axis=-1)
            quantized = pure_colors[indices]
            return quantized.reshape(b, h, w, c)  # type: ignore
        else:
            weights = mx.softmax(-distances / temperature, axis=-1)
            quantized = weights @ pure_colors
            return quantized.reshape(b, h, w, c)  # type: ignore

    def __call__(self, x: mx.array) -> mx.array:
        return cast(
            mx.array,
            self._quantize_impl(  # type: ignore
                x,
                self.pure_colors,
                self.temperature,
                self.training,
            ),
        )
