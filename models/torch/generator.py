"""Generator network and supporting modules for FaciesGAN.

This module provides the multi-scale ``Generator`` used to synthesize
facies images from per-scale noise tensors, along with a simple
``ColorQuantization`` module used to snap outputs to a small palette.
"""

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generator import Generator
from models.torch import utils
from models.torch.custom_layer import (
    TorchColorQuantization, 
    TorchConvBlock,
    TorchSPADEGenerator
)



class TorchGenerator(Generator[torch.Tensor, nn.ModuleList], nn.Module):
    """Multi-scale progressive generator for FaciesGAN.

    The generator is composed of a sequence of per-scale modules appended
    with ``create_scale``. It synthesizes images from a list of per-scale
    noise tensors ``z`` and amplitude scalars ``amp``. The network supports
    optional conditioning channels (wells/seismic) concatenated to the
    noise tensor.

    Parameters
    ----------
    num_layer : int
        Number of convolutional layers in each scale block.
    kernel_size : int
        Size of convolutional kernels.
    padding_size : int
        Padding size for convolutions.
    input_channels : int
        Number of input channels (noise + conditioning channels).

    Attributes
    ----------
    gens : nn.ModuleList
        List of per-scale generator modules (SPADE or ConvBlock stacks).
    zero_padding : int
        Padding applied per side to keep spatial alignment across scales.
    full_zero_padding : int
        Total padding applied (2 * zero_padding) used to compute output sizes.
    spade_scales : set[int]
        Set of scales that use SPADE-based generation (usually coarse scales).
    color_quantizer : ColorQuantization
        Module used to quantize outputs to a small palette of colors.
    """

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
        output_channels: int = 3,
    ):
        """Initialize the multi-scale Generator.

        Parameters
        ----------
        num_layer : int
            Number of convolutional layers per scale block.
        kernel_size : int
            Convolution kernel size.
        padding_size : int
            Padding size used for convolutions.
        input_channels : int
            Number of input channels (noise plus optional conditioning).
        output_channels : int, optional
            Number of output color channels (default 3).
        """
        # Initialize framework-agnostic configuration in the Generator base
        super(TorchGenerator, self).__init__(
            num_layer, kernel_size, padding_size, input_channels, output_channels
        )  # type: ignore
        super(nn.Module, self).__init__()

        # Replace the generic gens container with PyTorch ModuleList
        self.gens = cast(list[nn.Module], nn.ModuleList())

        # Color quantization layer (framework-specific)
        self.color_quantizer = TorchColorQuantization(temperature=0.1)

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds) # type: ignore

    def forward(
        self,
        z: list[torch.Tensor],
        amp: list[float],
        in_noise: torch.Tensor | None = None,
        start_scale: int = 0,
        stop_scale: int | None = None,
    ) -> torch.Tensor:
        """Generate facies through progressive pyramid synthesis.

        Parameters
        ----------
        z : list[torch.Tensor]
            Noise tensors for each pyramid scale.
        amp : list[float]
            Noise amplitudes for each scale.
        in_noise : torch.Tensor | None, optional
            Initial facies tensor to start from. If None, starts with zeros.
            Defaults to None.
        start_scale : int, optional
            Pyramid scale to start generation from. Defaults to 0.
        stop_scale : int | None, optional
            Final pyramid scale (inclusive). If None, uses all available scales.
            Defaults to None.

        Returns
        -------
        torch.Tensor
            Generated facies tensor at the finest requested scale.
        """
        if in_noise is None:
            channels = self.output_channels
            height, width = tuple(
                dim - self.full_zero_padding for dim in z[start_scale].shape[2:]
            )
            out_facie = torch.zeros(
                (z[start_scale].shape[0], channels, height, width),
                device=z[start_scale].device,
            )
        else:
            out_facie = in_noise

        stop_scale = stop_scale if stop_scale is not None else len(self.gens) - 1
        for index in range(start_scale, stop_scale + 1):

            out_facie = utils.interpolate(
                out_facie,
                (
                    z[index].shape[2] - self.full_zero_padding,
                    z[index].shape[3] - self.full_zero_padding,
                ),
            )

            if index in self.spade_scales:
                # SPADE-based generation at coarse scale
                # Apply amplitude scaling to noise channels
                z_in = z[index].clone()
                if self.has_cond_channels:
                    z_in[:, : self.output_channels, :, :] = (
                        amp[index] * z[index][:, : self.output_channels, :, :]
                    )
                else:
                    z_in = amp[index] * z_in

                padded_facie = F.pad(out_facie, [self.zero_padding] * 4, value=0)
                if self.has_cond_channels:
                    num_repeats = self.cond_channels // self.output_channels
                    padded_facie = padded_facie.repeat(1, num_repeats, 1, 1)
                    # Add padded output facie to well conditioning channels
                    z_in[:, self.output_channels :, :, :] = (
                        z_in[:, self.output_channels :, :, :] + padded_facie
                    )
                else:
                    z_in = z_in + padded_facie

                # SPADE generator takes full conditioning and outputs directly
                out_facie = self.gens[index](z_in) + out_facie
            else:
                # Standard concatenation-based generation at finer scales
                # Apply amplitude scaling to noise channels (first N channels)
                z_in = z[index].clone()
                if self.has_cond_channels:
                    z_in[:, : self.output_channels, :, :] = (
                        amp[index] * z[index][:, : self.output_channels, :, :]
                    )
                else:
                    z_in = amp[index] * z_in

                # Add padded output facie ONLY to conditioning channels (not noise)
                # This ensures random noise always has direct impact on generation
                padded_facie = F.pad(out_facie, [self.zero_padding] * 4, value=0)
                if self.has_cond_channels:
                    num_repeats = self.cond_channels // self.output_channels
                    padded_facie = padded_facie.repeat(1, num_repeats, 1, 1)
                    z_in[:, self.output_channels :, :, :] = (
                        z_in[:, self.output_channels :, :, :] + padded_facie
                    )
                else:
                    z_in = z_in + padded_facie

                out_facie = self.gens[index](z_in) + out_facie

        # Apply color quantization to enforce pure colors
        out_facie = self.color_quantizer(out_facie)

        return out_facie

    def create_scale(
        self, scale: int, num_features: int, min_num_features: int
    ) -> None:
        """Create and append a new scale block to the generator pyramid.

        Constructs a ConvBlock sequence with progressively decreasing channel
        counts from num_features down to min_num_features.

        At scale 0, uses SPADEGenerator for noise-modulated generation.
        At higher scales, uses standard ConvBlock architecture.

        Parameters
        ----------
        scale : int
            Pyramid scale index.
        num_features : int
            Number of features in the first convolutional layer.
        min_num_features : int
            Minimum number of features (floor for channel reduction).
        """

        if scale == 0:
            # Use SPADE-based generator at the coarsest scale
            # This allows noise to modulate features via learned gamma/beta
            spade_gen = TorchSPADEGenerator(
                num_layer=self.num_layer,
                kernel_size=self.kernel_size,
                padding_size=self.padding_size,
                num_features=num_features,
                min_num_features=min_num_features,
                output_channels=self.output_channels,
                input_channels=self.input_channels,
            )
            self.gens.append(spade_gen)
            self.spade_scales.add(scale)
        else:
            # Standard ConvBlock-based generator for finer scales
            head = TorchConvBlock(
                self.input_channels,
                num_features,
                self.kernel_size,
                self.padding_size,
                1,
            )
            body = nn.Sequential()

            block_features = min_num_features
            for i in range(self.num_layer - 2):
                block_features = int(num_features / pow(2, (i + 1)))
                block = TorchConvBlock(
                    max(2 * block_features, min_num_features),
                    max(block_features, min_num_features),
                    self.kernel_size,
                    self.padding_size,
                    1,
                )
                body.add_module(f"block{i + 1}", block)

            tail = nn.Sequential(
                nn.Conv2d(
                    max(block_features, min_num_features),
                    self.output_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.padding_size,
                ),
                nn.Tanh(),
            )

            self.gens.append(nn.Sequential(head, body, tail))
