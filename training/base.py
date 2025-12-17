"""Base trainer abstraction for different training backends.

This module provides an abstract :class:`Trainer` base that defines the
minimal interface and a couple of shared utilities used by concrete
trainers such as :class:`training.torch.train.TorchTrainer`.

Keep this class lightweight: it only initialises common configuration
fields and exposes abstract methods concrete trainers must implement.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from tensorboardX import SummaryWriter  # type: ignore
from tqdm import tqdm
from typing_extensions import Generic

from models.base import FaciesGAN
from options import TrainningOptions
from typedefs import TModule, TOptimizer, TScheduler, TTensor


class Trainer(ABC, Generic[TTensor, TModule, TOptimizer, TScheduler]):
    """Abstract base class for training runners.

    Subclasses must implement :meth:`train` and :meth:`train_scales`.
    The constructor initialises a small set of commonly-used attributes
    from the provided :class:`TrainningOptions` instance.
    """

    model: FaciesGAN[TTensor, TModule]

    def __init__(
        self,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
    ) -> None:
        self.options: TrainningOptions = options
        self.fine_tuning: bool = fine_tuning
        self.checkpoint_path: str = checkpoint_path

        # Common training parameters (conservative subset)
        self.start_scale: int = options.start_scale
        self.stop_scale: int = options.stop_scale
        self.output_path: str = options.output_path
        self.num_iter: int = options.num_iter
        self.save_interval: int = options.save_interval
        self.num_parallel_scales: int = options.num_parallel_scales
        # Batch size logic: clamp to available training pyramids and ensure
        # it is not smaller than the number of well mask columns when wells
        # selection is active (keeps behaviour consistent with torch trainer).
        self.batch_size: int = (
            options.batch_size
            if (options.batch_size < options.num_train_pyramids)
            else options.num_train_pyramids
        )
        self.batch_size = (
            self.batch_size
            if not (
                len(options.wells_mask_columns) > 0
                and options.batch_size < len(options.wells_mask_columns)
            )
            else len(options.wells_mask_columns)
        )

        # Feature flags
        self.enable_tensorboard: bool = options.enable_tensorboard
        self.enable_plot_facies: bool = options.enable_plot_facies

        # Placeholder containers commonly used by concrete trainers
        self.data_loader: Any = None
        self.visualizer: Any = None
        # Framework-agnostic model/training properties (moved from concrete
        # trainer to keep common configuration in the base class).
        self.num_img_channels: int = options.num_img_channels
        self.noise_channels: int = (
            options.noise_channels
            + (self.num_img_channels if options.use_wells else 0)
            + (self.num_img_channels if options.use_seismic else 0)
        )

        self.num_real_facies: int = options.num_real_facies
        self.num_generated_per_real: int = options.num_generated_per_real
        self.wells_mask_columns: tuple[int, ...] = options.wells_mask_columns

        # Optimizer configuration (default values from options)
        self.lr_g: float = options.lr_g
        self.lr_d: float = options.lr_d
        self.beta1: float = options.beta1
        self.lr_decay: int = options.lr_decay
        self.gamma: float = options.gamma

        # Model parameters
        self.zero_padding: int = options.num_layer * math.floor(options.kernel_size / 2)
        self.noise_amp: float = options.noise_amp
        self.min_noise_amp: float = options.min_noise_amp
        self.scale0_noise_amp: float = options.scale0_noise_amp
        # Containers populated by concrete trainers at runtime
        self.facies: list[Any] = []
        self.wells: list[Any] = []
        self.seismic: list[Any] = []

    @abstractmethod
    def train(self) -> None:  # pragma: no cover - implemented by subclasses
        """Run the full training process.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement train")

    @abstractmethod
    def train_scales(
        self,
        scales: list[int],
        writers: dict[int, SummaryWriter],
        scale_paths: dict[int, str],
        results_paths: dict[int, str],
        batch_id: int,
        progress: "tqdm[Any]",
    ) -> None:
        """Train one or more scales. Signature is backend-specific.

        Parameters
        ----------
        scales : list[int]
            List of scale indices to train.
        writers : dict[int, SummaryWriter]
            TensorBoard writers per scale index.
        scale_paths : dict[int, str]
            Checkpoint paths per scale index.
        results_paths : dict[int, str]
            Output results paths per scale index.
        batch_id : int
            Current batch identifier.
        progress : tqdm[Any]
            Progress bar instance to update during training.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement train_scales")

    def load(self, path: str, until_scale: int | None = None) -> None:
        """Default loader hook; concrete trainers may override.

        Parameters
        ----------
        path : str
            Path to load the model from.
        until_scale : int | None, optional
            Optional scale index up to which to load the model. Defaults to None.


        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement load")

    # @property
    # @abstractmethod
    # def model(self) -> FaciesGAN[TTensor, TModule]:
    #     """Framework-agnostic `FaciesGAN` instance used by the trainer.

    #     The concrete trainer (for example `training.torch.train.TorchTrainer`)
    #     should assign an instance of a `models.base.FaciesGAN` subclass to this
    #     property before training begins. The getter raises an informative
    #     ``AttributeError`` if the model hasn't been set yet.

    #     Raises
    #     ------
    #     NotImplementedError
    #         If the subclass does not implement this method.
    #     """
    #     raise NotImplementedError("Subclasses must implement model getter")

    # @model.setter
    # def model(self, value: FaciesGAN[TTensor, TModule]) -> None:
    #     """Assign the `FaciesGAN` instance used by the trainer.

    #     Performs a runtime type-check against the concrete `models.base.FaciesGAN`
    #     class when available to provide early feedback on incorrect types.

    #     Parameters
    #     ----------
    #     value : FaciesGAN
    #         The `FaciesGAN` instance to assign to the trainer.

    #     Raises
    #     ------
    #     NotImplementedError
    #         If the subclass does not implement this method.
    #     """
    #     raise NotImplementedError("Subclasses must implement model setter")
