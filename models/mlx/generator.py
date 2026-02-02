from typing import Any, Generator, cast

import mlx.nn.layers.upsample as upsample  # type: ignore

from models.generator import Generator
import mlx.core as mx
import mlx.nn as nn  # type: ignore

from models.mlx.custom_layer import (
    create_conv2d,
    init_conv_weights,
    MLXColorQuantization,
    MLXConvBlock,
    MLXSPADEGenerator,
    MLXScaleModule,
)


class MLXGenerator(Generator[mx.array, nn.Module], nn.Module):
    """Multi-scale progressive generator for FaciesGAN in MLX.


    Inherits from the base `Generator` class and `nn.Module` to implement
    a multi-scale generator architecture using MLX framework.

    Parameters
    ----------
    num_layer : int
        Number of convolutional layers per scale module.
    kernel_size : int
        Size of the convolutional kernels.
    padding_size : int
        Size of the padding applied to the inputs.
    input_channels : int
        Number of input channels (noise + conditioning).
    output_channels : int, optional
        Number of output channels (e.g., 3 for RGB images). Defaults to 3.

    Attributes
    ----------
    gens : list[nn.Module]
        List of per-scale generator modules.
    color_quantizer : MLXColorQuantization
        Color quantization layer to refine output colors.

    Methods
    -------
    __call__(z, amp, in_noise=None, start_scale=0, stop_scale=None) -> mx.array
        Calls the generator's forward method.
    eval() -> Self
        Sets the module in evaluation mode.
    forward(z, amp, in_noise=None, start_scale=0, stop_scale=None) -> mx.array
        Forward pass through the multi-scale generator.
    create_scale(scale, num_features, min_num_features) -> None
        Creates and appends a scale module to the generator.
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
        output_channels: int = 3,
    ) -> None:
        super(MLXGenerator, self).__init__(
            num_layer, kernel_size, padding_size, input_channels, output_channels
        )
        nn.Module.__init__(self)
        self.gens: list[nn.Module] = list()
        self.color_quantizer = MLXColorQuantization(
            temperature=0.1,
        )

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> mx.array:
        """Call the generator's forward method.

        Parameters
        ----------
        *args : Any
            Positional arguments for the forward method.
        **kwargs : Any
            Keyword arguments for the forward method.

        Returns
        -------
        mx.array
            Output of the `forward` method.
        """
        # Call `forward` directly to avoid passing unexpected keyword
        # arguments through MLX's `nn.Module.__call__` which may not
        # accept arbitrary kwargs like `stop_scale`.
        return self.forward(*args, **kwargs)

    def forward(
        self,
        z: list[mx.array],
        amp: list[float],
        in_noise: mx.array | None = None,
        start_scale: int = 0,
        stop_scale: int | None = None,
    ) -> mx.array:
        """Generate facies through progressive pyramid synthesis.

        Parameters
        ----------
        z : list[mx.array]
            List of per-scale noise tensors.
        amp : list[float]
            List of per-scale amplitude scalars.
        in_noise : mx.array | None, optional
            Optional input noise tensor for the coarsest scale. Defaults to None.
        start_scale : int, optional
            Scale index to start synthesis from. Defaults to 0.
        stop_scale : int | None, optional
            Scale index to stop synthesis at (inclusive). Defaults to None,
            which means synthesis continues to the finest scale.

        Returns
        -------
        mx.array
            Generated facies tensor at the finest requested scale.
        """
        if in_noise is None:
            channels = self.output_channels
            batch_size = z[start_scale].shape[0]
            height, width = tuple(
                dim - self.full_zero_padding for dim in z[start_scale].shape[1:3]
            )
            out_facie = mx.zeros((batch_size, height, width, channels))  # type: ignore
        else:
            out_facie = in_noise

        stop_scale = stop_scale if stop_scale is not None else len(self.gens) - 1
        for index in range(start_scale, stop_scale + 1):
            height, width = out_facie.shape[1:3]

            scale_h = (z[index].shape[1] - self.full_zero_padding) / height
            scale_w = (z[index].shape[2] - self.full_zero_padding) / width
            out_facie = upsample.upsample_linear(  # type: ignore
                out_facie,
                (scale_h, scale_w),
                align_corners=True,
            )

            z_in = cast(mx.array, z[index])  # type: ignore
            if self.has_cond_channels:
                noise = z_in[..., : self.output_channels]
                cond = z_in[..., self.output_channels :]
                noise = amp[index] * noise
                z_in = mx.concat([noise, cond], axis=-1)
            else:
                z_in = amp[index] * z_in

            p = self.zero_padding
            padded_facie = mx.pad(out_facie, [(0, 0), (p, p), (p, p), (0, 0)])  # type: ignore

            if self.has_cond_channels:
                num_repeats = self.cond_channels // self.output_channels
                padded_facie = mx.tile(padded_facie, (1, 1, 1, num_repeats))
                noise = z_in[..., : self.output_channels]
                cond = z_in[..., self.output_channels :] + padded_facie
                z_in = mx.concat([noise, cond], axis=-1)
            else:
                # If channel counts differ unexpectedly, add only to the
                # image channels and preserve any extra noise/conditioning
                # channels to avoid broadcasting errors.
                try:
                    z_ch = z_in.shape[-1]
                    img_ch = padded_facie.shape[-1]
                except Exception:
                    z_ch = None
                    img_ch = None

                if z_ch is not None and img_ch is not None and z_ch != img_ch:
                    # Add padded_facie to the first `img_ch` channels and
                    # concatenate the remaining channels unchanged.
                    first = z_in[..., :img_ch] + padded_facie
                    if z_ch > img_ch:
                        rest = z_in[..., img_ch:]
                        z_in = mx.concat([first, rest], axis=-1)
                    else:
                        # z has fewer channels than image (unlikely) — pad
                        pad = mx.zeros(
                            (z_in.shape[0], z_in.shape[1], z_in.shape[2], img_ch - z_ch)
                        )
                        z_in = mx.concat([first, pad], axis=-1)
                else:
                    z_in = z_in + padded_facie

            # Ensure module receives expected input channel count; if z_in
            # contains extra conditioning/noise channels (unexpected), trim
            # to `self.input_channels` before passing to the submodule.
            try:
                mod_in_ch = self.input_channels
                if z_in.shape[-1] != mod_in_ch:
                    z_mod = z_in[..., :mod_in_ch]
                else:
                    z_mod = z_in
            except Exception:
                z_mod = z_in

            out_mod = self.gens[index]
            out_tmp = cast(mx.array, out_mod(z_mod))
            out_facie = out_tmp + out_facie
        # Apply color quantization to enforce pure colors.
        out_facie = self.color_quantizer(out_facie)
        return out_facie

    def create_scale(
        self, scale: int, num_features: int, min_num_features: int
    ) -> None:
        """Create and append a new scale block to the generator pyramid.

        Constructs a ConvBlock sequence with progressively decreasing channel
        counts from num_features down to min_num_features.

        At scale 0, uses MLXSPADEGenerator for noise-modulated generation.
        At higher scales, uses standard MLXConvBlock architecture.

        Parameters
        ----------
        scale : int
            Scale index for which to create the module.
        num_features : int
            Number of features for the first convolutional layer.
        min_num_features : int
            Minimum number of features for the convolutional layers.
        """
        # Use SPADE-based generator at the coarsest scale
        # This allows noise to modulate features via learned gamma/beta
        if scale == 0:
            spade_gen = MLXSPADEGenerator(
                self.num_layer,
                self.kernel_size,
                self.padding_size,
                num_features,
                min_num_features,
                self.output_channels,
                self.input_channels,
            )
            self.gens.append(spade_gen)
            self.spade_scales.add(scale)
        else:
            # Standard ConvBlock-based generator for finer scales
            head = MLXConvBlock(
                self.input_channels,
                num_features,
                self.kernel_size,
                self.padding_size,
                1,
            )

            def build_body() -> tuple[list[MLXConvBlock], int]:
                block_features = min_num_features
                blocks: list[MLXConvBlock] = []
                for i in range(self.num_layer - 2):
                    block_features = int(num_features / pow(2, (i + 1)))
                    in_channels = max(2 * block_features, min_num_features)
                    out_channels = max(block_features, min_num_features)
                    blocks.append(
                        MLXConvBlock(
                            in_channels,
                            out_channels,
                            self.kernel_size,
                            self.padding_size,
                            1,
                        )
                    )
                return blocks, block_features

            body_blocks, block_features = build_body()
            body = nn.Sequential(*body_blocks)

            tail_conv = create_conv2d(
                max(block_features, min_num_features),
                self.output_channels,
                self.kernel_size,
                padding=self.padding_size,
            )
            init_conv_weights(tail_conv)
            tail = nn.Sequential(
                tail_conv,
                nn.Tanh(),
            )

            self.gens.append(MLXScaleModule(head, body, tail))

    def reset_parameters(self) -> None:
        for gen in self.gens:
            reset_fn = getattr(cast(Any, gen), "reset_parameters", None)
            if reset_fn:
                reset_fn()
