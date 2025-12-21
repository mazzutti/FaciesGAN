"""Framework-agnostic generator interface.

This module defines `GeneratorBase`, an abstract base class that exposes
the minimal API expected by training and checkpointing code. Concrete
implementations (for example the PyTorch `Generator` in
`models.torch.generator`) should implement this interface.
"""

from __future__ import annotations

import math

import mlx.nn as nn  # type: ignore
from abc import ABC, abstractmethod
from typing import Generic


from log import Any
from typedefs import TModule, TTensor


class Generator(ABC, Generic[TTensor, TModule]):
    """Abstract, framework-agnostic interface for a multi-scale generator.

    The base defines two methods:
    - `forward`: synthesize an output from per-scale noise and amplitudes.
    - `create_scale`: append a new per-scale block to the generator.

    Keeping this interface simple allows higher-level code to interact with
    generators without importing framework-specific modules.
    """

    gens: list[TModule]

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
        output_channels: int = 3,
    ) -> None:
        """Initialize the generator base class with common configuration.

        This constructor stores numeric configuration that is framework
        agnostic (layer counts, kernel/padding sizes and channel counts)
        and computes derived values used by concrete implementations.
        Concrete subclasses may overwrite framework-specific fields such
        as `gens` with framework module containers (e.g., `nn.ModuleList`).

        Parameters
        ----------
        num_layer : int
            Number of convolutional layers per scale.
        kernel_size : int
            Convolution kernel size.
        padding_size : int
            Padding applied to convolutions.
        input_channels : int
            Number of input image channels.
        output_channels : int, optional
            Number of output image channels. Defaults to 3 (RGB).
        """

        super().__init__()
        self.spade_scales: set[int] = set()

        # stored configuration fields used in the generator
        self.num_layer = num_layer

        # convolution parameters used in the generator
        self.kernel_size = kernel_size

        # padding size used in convolutions
        self.padding_size = padding_size

        # channel counts used in the generator
        self.input_channels = input_channels

        # output channel count (e.g., RGB)
        self.output_channels = output_channels

        # conditional channel count (e.g., segmentation map)
        self.cond_channels = self.input_channels - self.output_channels

        # flag indicating whether conditional channels are used
        self.has_cond_channels = self.cond_channels > 0

        # zero padding values used to align spatial sizes across scales
        self.zero_padding = self.num_layer * (math.floor(self.kernel_size / 2))

        # full padding applied to input tensors at each scale
        self.full_zero_padding = 2 * self.zero_padding

    @abstractmethod
    def forward(
        self,
        z: list[TTensor],
        amp: list[float],
        in_noise: TTensor | None = None,
        start_scale: int = 0,
        stop_scale: int | None = None,
    ) -> TTensor:
        """Synthesize an output tensor from per-scale noise and amplitudes.

        Parameters
        ----------
        z : list[TTensor]
            List of per-scale noise tensors.
        amp : list[float]
            List of per-scale amplitude scalars.
        in_noise : TTensor | None, optional
            Optional input noise tensor for the coarsest scale.
            Defaults to None.
        start_scale : int, optional
            Scale index to start synthesis from. Defaults to 0.
        stop_scale : int | None, optional
            Scale index to stop synthesis at (exclusive). Defaults to None,
            which means synthesis continues to the finest scale.

        Returns
        -------
        TTensor
            Synthesized output tensor.
        """

    @abstractmethod
    def create_scale(
        self, scale: int, num_features: int, min_num_features: int
    ) -> None:
        """Append a new per-scale block to the generator implementation.

        Parameters
        ----------
        scale : int
            Scale index for which to create the block.
        num_features : int
            Number of features in the first convolutional layer of the block.
        min_num_features : int
            Minimum number of features used when reducing channels.
        """

    @abstractmethod
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> TTensor:
        """Call the generator's forward method.

        Parameters
        ----------
        *args : Any
            Positional arguments for the `forward` method.
        **kwargs : Any
            Keyword arguments for the `forward` method.

        Returns
        -------
        TTensor
            Output of the `forward` method.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")

    # @abstractmethod
    # def eval(self) -> Self:
    #     """Set the module in evaluation mode.

    #     Returns
    #     -------
    #     Generator[torch.Tensor, nn.ModuleList]
    #         The generator instance in evaluation mode.

    #     Raises
    #     ------
    #     NotImplementedError
    #         If the subclass does not implement this method.
    #     """
    #     raise NotImplementedError("Subclasses must implement eval method.")
