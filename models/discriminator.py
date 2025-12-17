"""Framework-agnostic discriminator interface.

Provides a minimal abstract base class so higher-level training code can
program against a `Discriminator` API without importing framework
implementations. Concrete backends should call the base constructor to
store numeric configuration and then implement the abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Self

from types import TModule, TTensor


class Discriminator(ABC, Generic[TTensor, TModule]):
    """Abstract discriminator interface.

    Concrete implementations should store per-instance numeric configuration
    (num_features, num_layer, kernel_size, padding_size, input_channels)
    by calling the base constructor and implement `forward` and `__call__`.
    """

    discs: list[TModule]

    def __init__(
        self,
        num_layer: int,
        kernel_size: int,
        padding_size: int,
        input_channels: int,
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.input_channels = input_channels

    @abstractmethod
    def forward(self, scale: int, input_tensor: TTensor) -> TTensor:
        """Discriminate input tensor and return score map tensor.

        Parameters
        ----------
        scale : int
            Scale index to select the appropriate per-scale block.
        input_tensor : TTensor
            Input tensor to be discriminated.

        Returns
        -------
        TTensor
            Score map tensor produced by the discriminator.
        """

    @abstractmethod
    def create_scale(self, num_features: int, min_num_features: int) -> None:
        """Append a new per-scale block to the generator implementation.

        Parameters
        ----------
        num_features : int
            Number of features in the first convolutional layer of the block.
        min_num_features : int
            Minimum number of features used when reducing channels.
        """

    def __call__(self, *args: Any, **kwds: Any) -> TTensor:
        """ "Call the discriminator's forward method.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to `forward`.
        **kwds : Any
            Keyword arguments to pass to `forward`.

        Returns
        -------
        TTensor
            Output of the `forward` method.
        """
        return super().__call__(*args, **kwds)  # type: ignore

    def eval(self) -> Self:
        """Set the discriminator to evaluation mode (framework-specific).

        Returns
        -------
        Self
            The discriminator instance in evaluation mode.
        """
        return super().eval()  # type: ignore
