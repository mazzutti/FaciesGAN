import math



import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom_layer import ConvBlock, SPADEGenerator
from ops import interpolate


class Generator(nn.Module):
    """Multi-scale progressive generator for FaciesGAN.

    Generates facies images through a progressive pyramid architecture,
    where each scale has its own convolutional block. Images are generated
    from coarse to fine resolution by upsampling and adding noise at each scale.
    
    At the coarsest scale (scale 0), uses SPADE-based modulation to inject
    noise into the generation process, allowing the network to learn how
    noise should influence features at each spatial location.

    Parameters
    ----------
    num_layer : int
        Number of convolutional layers in each scale block.
    kernel_size : int
        Size of convolutional kernels.
    padding_size : int
        Padding size for convolutions.
    img_num_channel : int
        Number of input channels (facies channels + noise channel).

    Attributes
    ----------
    gens : nn.ModuleList
        List of generator modules, one per pyramid scale.
    zero_padding : int
        Total padding added per layer.
    full_zero_padding : int
        Total padding for both sides (2 * zero_padding).
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
        # Track which scales use SPADE (currently only scale 0)
        self.spade_scales: set[int] = set()

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
            # Output 3 channels (input is 6: 3 noise + 3 wells)
            channels = z[start_scale].shape[1] // 2
            height, width = tuple(dim - self.full_zero_padding for dim in z[start_scale].shape[2:])
            out_facie = torch.zeros(
                (z[start_scale].shape[0], channels, height, width),
                device=z[start_scale].device,
            )
        else:
            out_facie: torch.Tensor = in_noise

        stop_scale = stop_scale if stop_scale is not None else len(self.gens) - 1

        for index in range(start_scale, stop_scale + 1):
            
            out_facie = interpolate(
                out_facie,
                (
                    z[index].shape[2] - self.full_zero_padding,
                    z[index].shape[3] - self.full_zero_padding,
                ),
            )

            # z[index] has shape (batch, 6, H, W): 3 noise channels + 3 well channels
            # out_facie has shape (batch, 3, H, W): generated facies
            half_ch = z[index].shape[1] // 2  # = 3
            
            if index in self.spade_scales:
                # SPADE-based generation at coarse scale
                # Apply amplitude scaling to noise channels
                z_in = z[index].clone()
                z_in[:, :half_ch, :, :] = amp[index] * z[index][:, :half_ch, :, :]
                
                # Add padded output facie to well conditioning channels
                padded_facie = F.pad(out_facie, [self.zero_padding] * 4, value=0)
                z_in[:, half_ch:, :, :] = z_in[:, half_ch:, :, :] + padded_facie
                
                # SPADE generator takes full conditioning and outputs directly
                out_facie = self.gens[index](z_in) + out_facie
            else:
                # Standard concatenation-based generation at finer scales
                # Apply amplitude scaling to noise channels (first 3 channels)
                z_in = z[index].clone()
                z_in[:, :half_ch, :, :] = amp[index] * z[index][:, :half_ch, :, :]
                
                # Add padded output facie ONLY to conditioning channels (not noise)
                # This ensures random noise always has direct impact on generation
                padded_facie = F.pad(out_facie, [self.zero_padding] * 4, value=0)
                z_in[:, half_ch:, :, :] = z_in[:, half_ch:, :, :] + padded_facie

                out_facie = self.gens[index](z_in) + out_facie
                
        return out_facie

    def create_scale(self, num_feature: int, min_num_feature: int) -> None:
        """Create and append a new scale block to the generator pyramid.

        Constructs a ConvBlock sequence with progressively decreasing channel
        counts from num_feature down to min_num_feature.
        
        At scale 0, uses SPADEGenerator for noise-modulated generation.
        At higher scales, uses standard ConvBlock architecture.

        Parameters
        ----------
        num_feature : int
            Number of features in the first convolutional layer.
        min_num_feature : int
            Minimum number of features (floor for channel reduction).
        """
        current_scale = len(self.gens)
        
        if current_scale == 0:
            # Use SPADE-based generator at the coarsest scale
            # This allows noise to modulate features via learned gamma/beta
            spade_gen = SPADEGenerator(
                num_layer=self.num_layer,
                kernel_size=self.kernel_size,
                padding_size=self.padding_size,
                num_feature=num_feature,
                min_num_feature=min_num_feature,
                img_channels=self.img_num_channel // 2,  # Output 3 facies channels
                cond_channels=self.img_num_channel,  # Input 6 channels (noise + wells)
            )
            self.gens.append(spade_gen)
            self.spade_scales.add(current_scale)
        else:
            # Standard ConvBlock-based generator for finer scales
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
                    self.img_num_channel // 2,  # Output 3 facies channels (input is 6)
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.padding_size,
                ),
                nn.Tanh(),
            )

            self.gens.append(nn.Sequential(head, body, tail))
