"""Base trainer abstraction for different training backends.

This module provides an abstract :class:`Trainer` base that defines the
minimal interface and a couple of shared utilities used by concrete
trainers such as :class:`training.torch.train.TorchTrainer`.

Keep this class lightweight: it only initialises common configuration
fields and exposes abstract methods concrete trainers must implement.
"""

from __future__ import annotations

import math
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Any, Iterator, cast

from tensorboardX import SummaryWriter  # type: ignore
from typing_extensions import Generic


class DummyProgress:
    def update(self, n: int = 1) -> None:
        pass

    def set_description(self, desc: str) -> None:
        pass

    def write(self, s: str) -> None:
        print(s)

    def close(self) -> None:
        pass


import utils
from config import RESULT_FACIES_PATH
from datasets import PyramidsDataset
import log
from metrics import (
    DiscriminatorMetrics,
    GeneratorMetrics,
    ScaleMetrics,
)
from models.base import FaciesGAN, IterableMetrics
from options import TrainningOptions
from tensorboard_visualizer import TensorBoardVisualizer
from typedefs import (
    IDataLoader,
    PyramidsBatch,
    TModule,
    TOptimizer,
    TScheduler,
    TTensor,
)


class Trainer(ABC, Generic[TTensor, TModule, TOptimizer, TScheduler, IDataLoader]):
    """Abstract base class for training runners.

    Subclasses must implement :meth:`train` and :meth:`train_scales`.
    The constructor initialises a small set of commonly-used attributes
    from the provided :class:`TrainningOptions` instance.
    """

    model: FaciesGAN[
        TTensor,
        TModule,
        TOptimizer,
        TScheduler,
    ]

    def __init__(
        self,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
    ) -> None:
        self.options: TrainningOptions = options
        self.fine_tuning: bool = fine_tuning
        self.checkpoint_path: str = checkpoint_path

        # Distributed training flag — only rank-0 should perform I/O.
        # Subclasses (e.g. TorchTrainer) may set this before calling super().
        if not hasattr(self, "_is_main_process"):
            self._is_main_process: bool = True

        # Common training parameters (conservative subset)
        self.start_scale: int = options.start_scale
        self.start_epoch: int = getattr(options, "start_epoch", 0)
        self.stop_scale: int = options.stop_scale
        self.output_path: str = options.output_path
        self.num_iter: int = options.num_iter
        self.save_interval: int = options.save_interval
        self.num_parallel_scales: int = options.num_parallel_scales

        # How often to flush TensorBoard scalars (epochs).  Writing every
        # epoch triggers a GPU→CPU sync per scale; batching to every N epochs
        # reduces that overhead by ~N× with minimal loss of resolution.
        self._tb_log_interval: int = 10

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
        self.enable_plot_outputs: bool = options.enable_plot_outputs

        # Placeholder containers commonly used by concrete trainers
        self.visualizer: TensorBoardVisualizer | None = None

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

        # Initialize dataset and data loader
        dataset, scales = self.init_dataset()
        self.dataset: PyramidsDataset[TTensor] = dataset
        self.num_of_batchs: int = len(self.dataset) // self.batch_size
        self.scales: tuple[tuple[int, ...], ...] = scales
        self.data_loader: IDataLoader = self.create_dataloader()

        if self._is_main_process:
            print(f"DataLoader num_workers: {self.data_loader.num_workers}")

        self.model = self.create_model()
        self.model.shapes = list(self.scales)

        # generator learning rate
        self.lr_g = options.lr_g

        # discriminator learning rate
        self.lr_d = options.lr_d

        # discriminator learning rate
        self.beta1 = options.beta1

        # learning rate decay milestone
        self.lr_decay = options.lr_decay

        # learning rate gamma
        self.gamma = options.gamma

        # generator optimizers
        self.generator_optimizers: dict[int, TOptimizer] = {}

        # discriminator optimizers
        self.discriminator_optimizers: dict[int, TOptimizer] = {}

        # generator schedulers
        self.generator_schedulers: dict[int, TScheduler] = {}

        # discriminator schedulers
        self.discriminator_schedulers: dict[int, TScheduler] = {}

        if self._is_main_process:
            lines = ["Generated facie shapes:"]
            lines.append("╔══════════╦══════════╦══════════╦══════════╗")
            lines.append(
                "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                    "Batch", "Channels", "Height", "Width"
                )
            )
            lines.append("╠══════════╬══════════╬══════════╬══════════╣")
            for shape in self.scales:
                lines.append(
                    "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                        shape[0], self.noise_channels, shape[2], shape[3]
                    )
                )
            lines.append("╚══════════╩══════════╩══════════╩══════════╝")
            print("\n".join(lines), flush=True)

        # Initialize TensorBoard visualizer if enabled
        self.enable_tensorboard = options.enable_tensorboard
        self.enable_plot_outputs = options.enable_plot_outputs
        if self.enable_tensorboard and self._is_main_process:
            viz_path = os.path.join(self.output_path, "training_visualizations")
            log_dir = os.path.join(self.output_path, "tensorboard_logs")
            dataset_info = f"{len(self.dataset)} pyramids, {self.batch_size} batch size"
            if len(options.wells_mask_columns) > 0:
                dataset_info += f", wells: {options.wells_mask_columns}"

            _purge = self.start_epoch if self.start_epoch > 0 else None
            self.visualizer = TensorBoardVisualizer(
                num_scales=self.stop_scale - self.start_scale + 1,
                output_dir=viz_path,
                log_dir=log_dir,
                update_interval=1,
                dataset_info=dataset_info,
                purge_step=_purge,
            )
            print(f"📊 TensorBoard logging enabled")
            print(f"   logdir: {log_dir}")
            print(f"   URL: http://localhost:6006")
        else:
            self.visualizer = None  # type: ignore
            if self._is_main_process:
                print("📊 TensorBoard logging disabled")

    @abstractmethod
    def create_model(self) -> FaciesGAN[TTensor, TModule, TOptimizer, TScheduler]:
        """Create the model used by the trainer.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement create_model")

    @abstractmethod
    def create_dataloader(self) -> IDataLoader:
        """Create the data loader used by the trainer.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement create_dataloader")

    @abstractmethod
    def init_dataset(
        self,
    ) -> tuple[PyramidsDataset[TTensor], tuple[tuple[int, ...], ...]]:
        """Initialize the dataset used by the trainer.

        Returns
        -------
        tuple[PyramidsDataset[TTensor], tuple[tuple[int, ...], ...]]
            A tuple containing the dataset instance and the scales list used
            by the dataset.


        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement init_dataset")

    def _ddp_barrier(self) -> None:
        """Synchronize DDP ranks.  No-op for non-distributed training.

        Overridden by framework-specific trainers (e.g.
        :class:`TorchTrainer`) to call ``dist.barrier()``.
        """

    @abstractmethod
    def generate_visualization_samples(
        self,
        scales: tuple[int, ...],
        indexes: list[int],
        wells_pyramid: dict[int, TTensor] = {},
        seismic_pyramid: dict[int, TTensor] = {},
    ) -> tuple[TTensor, ...]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to generate samples for.
        indexes : list[int]
            List of batch sample indices.
        wells_pyramid : dict[int, TTensor], optional
            Dictionary of well-conditioning tensors for all scales.
        seismic_pyramid : dict[int, TTensor], optional
            Dictionary of seismic-conditioning tensors for all scales.

        Returns
        -------
        tuple[TTensor, ...]
            A tuple mapping scale indices to generated facies tensors
            for visualization.
        """
        raise NotImplementedError(
            "Subclasses must implement generate_visualization_samples"
        )

    @abstractmethod
    def compute_rec_input(
        self,
        scale: int,
        indexes: list[int],
        facies_pyramid: dict[int, TTensor],
    ) -> TTensor:
        """Compute the reconstruction input tensor for a specific scale.

        This method upsamples the reconstruction from the previous scale
        to match the spatial dimensions of the current scale's real facies.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        indexes : list[int]
            Batch sample indices.
        facies_pyramid : dict[int, TTensor]
            Dictionary of real facies data for all scales.

        Returns
        -------
        TTensor
            The upsampled reconstruction input tensor for the current scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement compute_rec_input")

    @abstractmethod
    def init_rec_noise_and_amp(
        self,
        scale: int,
        indexes: list[int],
        real: TTensor,
        wells_pyramid: dict[int, TTensor] = {},
        seismic_pyramid: dict[int, TTensor] = {},
    ) -> None:
        """Initialize reconstruction noise and noise amplitude for a specific scale.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        indexes : list[int]
            Batch sample indices.
        real : TTensor
            Real facies tensor for the current scale.
        wells_pyramid : dict[int, TTensor], optional
            Dictionary of well-conditioning tensors for all scales.
        seismic_pyramid : dict[int, TTensor], optional
            Dictionary of seismic-conditioning tensors for all scales.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement init_rec_noise_and_amp")

    @abstractmethod
    def create_batch_iterator(
        self, loader: IDataLoader, scales: tuple[int, ...]
    ) -> Iterator[PyramidsBatch[TTensor] | None]:
        """Create an iterator that yields batches for training.

        Subclasses must implement this to define how data is fetched and
        prepared (e.g., using a prefetcher or standard iteration).

        Parameters
        ----------
        loader : IDataLoader
            The data loader to iterate over.
        scales : tuple[int, ...]
            Tuple of scale indices being trained.

        Yields
        ------
        DictBatch[TTensor] | None
            The prepared batch for training, or None if no more batches are available.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement create_batch_iterator")

    @abstractmethod
    def setup_optimizers(self, scales: tuple[int, ...]) -> None:
        """Setup optimizers and schedulers for all scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to setup optimizers for.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement setup_optimizers")

    @abstractmethod
    def reset_schedulers(self, scales: tuple[int, ...]) -> None:
        """Reset LR schedulers to their initial state for a new batch.

        Called at the start of each DataLoader batch so that every batch
        trains with the same LR schedule (e.g. decay at epoch 500, 1000).
        """
        raise NotImplementedError("Subclasses must implement reset_schedulers")

    def _warmup_compile_traces(
        self,
        scales: tuple[int, ...],
        indexes: list[int],
        facies_pyramid: dict[int, TTensor],
        rec_in_pyramid: dict[int, TTensor],
        wells_pyramid: dict[int, TTensor],
        seismic_pyramid: dict[int, TTensor],
    ) -> None:
        """Hook for subclasses to warm up compiled code paths.

        The default implementation is a no-op.  ``TorchTrainer`` overrides
        this to run dummy forwards that force ``torch.compile`` to cache
        specializations for recovery-loss and diversity-N>1 branches
        before the training loop needs them.
        """

    def train_scales(
        self,
        scales: tuple[int, ...],
        writers: dict[int, SummaryWriter], # type: ignore
        scale_paths: dict[int, str],
        results_paths: dict[int, str],
        batch_id: int,
        progress: "tqdm[Any]",  # type: ignore
        facies_pyramid: dict[int, TTensor],
        wells_pyramid: dict[int, TTensor] = {},
        masks_pyramid: dict[int, TTensor] = {},
        seismic_pyramid: dict[int, TTensor] = {},
        start_epoch: int = 0,
    ) -> None:
        """Train multiple pyramid scales simultaneously.

        Accepts prepared data dictionaries directly.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to train.
        writers : dict[int, SummaryWriter]
            Dictionary of per-scale TensorBoard writers.
        scale_paths : tuple[str, ...]
            Tuple of per-scale output directory paths.
        results_paths : tuple[str, ...]
            Tuple of per-scale results directory paths.
        batch_id : int
            Current batch index.
        progress : tqdm[Any]
            Progress bar instance to update description text.
        facies_pyramid : tuple[TTensor, ...]
            Tuple of real facies data for all scales.
        wells_pyramid : dict[int, TTensor], optional
            Dictionary of well-conditioning tensors for all scales.
        masks_pyramid : dict[int, TTensor], optional
            Dictionary of masks tensors for all scales.
        seismic_pyramid : dict[int, TTensor], optional
            Dictionary of seismic-conditioning tensors for all scales.
        """

        # Derive indexes from the actual batch size (the last DataLoader
        # batch may be smaller than self.batch_size, so a hard-coded range
        # would produce out-of-bounds indices on wells/seismic tensors).
        first_scale = min(facies_pyramid)
        actual_batch = facies_pyramid[first_scale].shape[0]  # type: ignore[union-attr]
        indexes = list(range(actual_batch))

        # if self.fine_tuning:
        #     for scale in scales:
        #         self.load_optimizers(
        #             scale,
        #             scale_paths[scale],
        #             generator_optimizers[scale],
        #             discriminator_optimizers[scale],
        #             generator_schedulers[scale],
        #             discriminator_schedulers[scale],
        #         )

        stop_scale = min(max(scales), self.stop_scale) + 1
        rec_in_pyramid: dict[int, TTensor] = {}
        active = set(self.model.active_scales)
        for scale in range(stop_scale):
            # Only compute rec_in for active scales (needed by generator)
            if scale in active:
                rec_in_pyramid[scale] = self.compute_rec_input(
                    scale, indexes, facies_pyramid
                )
            self.init_rec_noise_and_amp(
                scale,
                indexes,
                facies_pyramid[scale],
                wells_pyramid,
                seismic_pyramid,
            )

        # Pre-populate torch.compile trace caches for code paths that
        # activate later (recovery loss, diversity N>1) so the first
        # recompilation doesn't happen mid-training where it can crash.
        self._warmup_compile_traces(
            scales, indexes, facies_pyramid,
            rec_in_pyramid, wells_pyramid, seismic_pyramid,
        )

        # Training loop - iterate epochs (0-based)
        if start_epoch > 0 and self._is_main_process:
            print(
                f"  Resuming from epoch {start_epoch} (skipping epochs 0-{start_epoch - 1})"
            )

        profile_timing = os.environ.get("FG_PROFILE_TIMING", "0") == "1"
        prof_opt_time = 0.0
        prof_epoch_end_time = 0.0
        prof_barrier_time = 0.0
        prof_epochs = 0

        for epoch in range(start_epoch, self.num_iter):
            progress.set_description(  # type: ignore
                f"Batch [{self._current_batch_id + 1}/{self._total_batches}] Epoch [{epoch+1:4d}/{self.num_iter}]"
            )

            # Let the model know the current epoch so it can skip
            # recovery loss during the early-training warmup phase.
            self.model._current_epoch = epoch  # type: ignore[attr-defined]

            generated_samples: tuple[TTensor, ...] = ()

            opt_t0 = time.perf_counter() if profile_timing else 0.0
            scale_metrics = self.optimization_step(
                indexes,
                facies_pyramid,
                rec_in_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            )
            if profile_timing:
                prof_opt_time += time.perf_counter() - opt_t0

            # Visualization epochs: 0, 199, 299, … (every 200 and last).
            # Only rank-0 runs generate_visualization_samples; non-main
            # ranks used to run it too to "warm up" compiled traces, but
            # the training iterations already warm them up and the
            # inference_mode specialisations are distinct, causing rank-1
            # to lag minutes behind rank-0 at the barrier.
            _is_viz_epoch = (
                (epoch + 1) % 200 == 0
                or epoch == 0
                or epoch == (self.num_iter - 1)
            )
            if self._is_main_process and _is_viz_epoch:
                generated_samples = self.generate_visualization_samples(
                    scales,
                    indexes,
                    wells_pyramid,
                    seismic_pyramid,
                )

            ep_t0 = time.perf_counter() if profile_timing else 0.0
            self.handle_epoch_end(  # type: ignore
                scales=scales,
                epoch=epoch,
                scale_metrics=cast(ScaleMetrics[TTensor], scale_metrics),
                generated_samples=generated_samples,
                writers=writers,
                results_paths=results_paths,
                progress=progress,  # type: ignore
                facies_pyramid=facies_pyramid,
                wells_pyramid=wells_pyramid,
                masks_pyramid=masks_pyramid,
                seismic_pyramid=seismic_pyramid,
            )
            if profile_timing:
                prof_epoch_end_time += time.perf_counter() - ep_t0
            # Synchronize DDP ranks on visualization epochs and on
            # save_generated_outputs epochs. Use _is_viz_epoch (symmetric
            # pure-Python arithmetic) instead of len(generated_samples)
            # so that rank-1 (which skips generate_visualization_samples)
            # still reaches the barrier on the same epochs as rank-0.
            _needs_barrier = _is_viz_epoch or (
                self.enable_plot_outputs
                and (epoch % self.save_interval == 0 or epoch == self.num_iter - 1)
                and (epoch != 0 or self.num_iter == 1)
                and self._current_batch_id == self._total_batches - 1
            )
            br_t0 = time.perf_counter() if profile_timing else 0.0
            if _needs_barrier:
                self._ddp_barrier()
            if profile_timing:
                prof_barrier_time += time.perf_counter() - br_t0
                prof_epochs += 1

            # Save an epoch checkpoint at save_interval boundaries so
            # training can be resumed from that point.  Skip the very
            # last epoch — the full scale checkpoint saved in train()
            # after the batch loop is sufficient for that case.
            if (
                (epoch + 1) % self.save_interval == 0
                and epoch < self.num_iter - 1
                and self._is_main_process
            ):
                self.save_epoch_checkpoint(
                    scales,
                    scale_paths,
                    epoch + 1,
                    batch_id,
                )

            # Release GPU tensors that are no longer needed
            del scale_metrics
            generated_samples = ()

        if profile_timing and prof_epochs > 0 and self._is_main_process:
            avg_opt = prof_opt_time / prof_epochs
            avg_end = prof_epoch_end_time / prof_epochs
            avg_bar = prof_barrier_time / prof_epochs
            avg_total = avg_opt + avg_end + avg_bar
            print("\n[TIMING] Per-epoch averages for current batch")
            print(
                "[TIMING] "
                f"opt={avg_opt:.3f}s "
                f"epoch_end={avg_end:.3f}s "
                f"barrier={avg_bar:.3f}s "
                f"tracked_total={avg_total:.3f}s"
            )

        # for scale in scales:
        #     self.save_optimizers(
        #         scale_paths[scale],
        #         generator_optimizers[scale],
        #         discriminator_optimizers[scale],
        #         generator_schedulers[scale],
        #         discriminator_schedulers[scale],
        #     )

    def optimization_step(
        self,
        indexes: list[int],
        facies_pyramid: dict[int, TTensor],
        rec_in_pyramid: dict[int, TTensor],
        wells_pyramid: dict[int, TTensor] = {},
        masks_pyramid: dict[int, TTensor] = {},
        seismic_pyramid: dict[int, TTensor] = {},
    ) -> (
        ScaleMetrics[TTensor]
        | tuple[
            IterableMetrics[TTensor],
            ...,
        ]
    ):
        """Perform a single optimization step for the model.

        Parameters
        ----------
        indexes : list[int]
            Batch sample indices.
        facies_pyramid : dict[int, TTensor]
            Dictionary of real facies data for all scales.
        rec_in_pyramid : dict[int, TTensor]
            Dictionary mapping scale -> reconstruction input from previous scale.
        wells_pyramid : dict[int, TTensor]
            Dictionary of well-conditioning tensors for all scales.
        masks_pyramid : dict[int, TTensor]
            Dictionary of masks tensors for all scales.
        seismic_pyramid : dict[int, TTensor]
            Dictionary of seismic-conditioning tensors for all scales.

        Returns
        -------
        ScaleMetrics[TTensor]
            Collected metrics for all scales after the optimization step.
        """
        return cast(
            ScaleMetrics[TTensor],
            self.model(
                self.generator_optimizers,
                self.discriminator_optimizers,
                indexes,
                facies_pyramid,
                rec_in_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            ),
        )

    @abstractmethod
    def save_optimizers(
        self,
        scale_path: str,
        generator_optimizer: TOptimizer,
        discriminator_optimizer: TOptimizer,
        generator_scheduler: TScheduler,
        discriminator_scheduler: TScheduler,
    ) -> None:
        raise NotImplementedError("Subclasses must implement save_optimizers")

    @abstractmethod
    def save_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
        epoch: int,
        batch_id: int,
    ) -> None:
        """Save a mid-training checkpoint so training can resume from *epoch*.

        The checkpoint must capture model weights, optimizer/scheduler
        states and the noise amplitudes / reconstruction noise for the
        scales currently being trained.

        Parameters
        ----------
        scales : tuple[int, ...]
            Scale indices being trained in the current group.
        scale_paths : dict[int, str]
            Per-scale output directory paths.
        epoch : int
            The *next* epoch to run (i.e. training completed up to
            ``epoch - 1``).
        batch_id : int
            Current batch index within the DataLoader iteration.
        """
        raise NotImplementedError("Subclasses must implement save_epoch_checkpoint")

    @abstractmethod
    def load_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
    ) -> tuple[int, int]:
        """Restore a previously saved epoch checkpoint.

        Parameters
        ----------
        scales : tuple[int, ...]
            Scale indices being trained in the current group.
        scale_paths : dict[int, str]
            Per-scale output directory paths (used to locate the checkpoint
            file).

        Returns
        -------
        tuple[int, int]
            ``(start_epoch, resume_batch_id)`` — the epoch and batch id
            to resume from.  Returns ``(0, 0)`` when no checkpoint is
            found.
        """
        raise NotImplementedError("Subclasses must implement load_epoch_checkpoint")

    def load(self, path: str, until_scale: int | None = None) -> None:
        """Load saved models and set the starting scale for training.

        Parameters
        ----------
        path : str
            Path to the directory containing model checkpoint files.
        until_scale : int | None, optional
            Load models up to and including this scale. If None, loads all
            available scales. Defaults to None.
        """
        self.start_scale = self.model.load(
            path, load_shapes=False, until_scale=until_scale
        )

    @abstractmethod
    def load_model(self, scale: int) -> None:
        """Load generator and discriminator state dicts for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index to load the model for.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement load_model")

    def load_optimizers(
        self,
        scale: int,
        scale_path: str,
        generator_optimizer: TOptimizer,
        discriminator_optimizer: TOptimizer,
        generator_scheduler: TScheduler,
        discriminator_scheduler: TScheduler,
    ) -> None:
        """Load optimizer and scheduler state dicts from disk using project filename
        constants from :mod:`config`.

        Parameters
        ----------
        scale : int
            Scale index to load the optimizers for.
        scale_path : str
            Path to the scale directory where optimizers are saved.
        generator_optimizer : TOptimizer
            Generator optimizer instance to load state into.
        discriminator_optimizer : TOptimizer
            Discriminator optimizer instance to load state into.
        generator_scheduler : TScheduler
            Generator learning rate scheduler instance to load state into.
        discriminator_scheduler : TScheduler
            Discriminator learning rate scheduler instance to load state into.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement load_optimizers")

    def save_generated_outputs(
        self,
        scale: int,
        epoch: int,
        batch_id: int,
        results_path: str,
        real_facies: TTensor,
        wells_pyramid: dict[int, TTensor] = {},
        masks_pyramid: dict[int, TTensor] = {},
        seismic_pyramid: dict[int, TTensor] = {},
    ) -> None:
        """Persist generated facies for a given scale.

        Concrete trainers that can generate and save facies (e.g. torch)
        should implement this method. The base implementation is a
        no-op / hook and may be overridden.

        Parameters
        ----------
        scale : int
            Scale index to save generated facies for.
        epoch : int
            Current epoch index.
        batch_id : int
            Current batch index.
        results_path : str
            Path to the results directory for the current scale.
        real_facies : TTensor
            Real facies tensor for the current scale.
        wells_pyramid : dict[int, TTensor], optional
            Dictionary of well-conditioning tensors for all scales.
        masks_pyramid : dict[int, TTensor], optional
            Dictionary of masks tensors for all scales.
        seismic_pyramid : dict[int, TTensor], optional
            Dictionary of seismic-conditioning tensors for all scales.
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement save_generated_outputs")

    def handle_epoch_end(
        self,
        scales: tuple[int, ...],
        epoch: int,
        scale_metrics: ScaleMetrics[TTensor],
        generated_samples: tuple[TTensor, ...],
        writers: dict[int, SummaryWriter], # type: ignore
        results_paths: dict[int, str],
        progress: "tqdm[Any]",  # type: ignore
        facies_pyramid: dict[int, TTensor],
        wells_pyramid: dict[int, TTensor],
        masks_pyramid: dict[int, TTensor],
        seismic_pyramid: dict[int, TTensor],
    ) -> None:
        """Shared end-of-epoch bookkeeping for trainers.

        This consolidates visualization updates, metric printing,
        TensorBoard logging, optional facies saving and scheduler steps.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices being trained.
        epoch : int
            Current epoch index.
        scale_metrics : ScaleMetrics[TTensor]
            Collected metrics for the current epoch.
        generated_samples : tuple[TTensor, ...]
            Tuple of generated facies samples for visualization.
        writers : dict[int, SummaryWriter]
            Dictionary of per-scale TensorBoard writers.
        results_paths : dict[int, str]
            Dictionary of per-scale results directory paths.
        progress : tqdm[Any]
            Progress bar instance to update.
        facies_pyramid : dict[int, TTensor]
            Dictionary of real facies data for all scales.
        wells_pyramid : dict[int, TTensor]
            Dictionary of well-conditioning tensors for all scales.
        masks_pyramid : dict[int, TTensor]
            Dictionary of masks tensors for all scales.
        seismic_pyramid : dict[int, TTensor]
            Dictionary of seismic-conditioning tensors for all scales.
        """
        # Use a globally unique step so that different DataLoader
        # batches do not overwrite each other's TensorBoard entries.
        global_step = self._current_batch_id * self.num_iter + epoch
        samples_processed = self.batch_size * global_step

        # ── Rank-0U-only I/O ─────────────────────────────────────────
        if self._is_main_process:
            # Visualizer update
            if self.enable_tensorboard and self.visualizer:
                self.visualizer.update(  # type: ignore
                    global_step, scale_metrics, generated_samples, samples_processed
                )

            # Print formatted metrics table occasionally
            if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (self.num_iter - 1):
                lines: list[str] = []
                lines.append(
                    f"\n  Batch [{self._current_batch_id + 1}/{self._total_batches}] Epoch [{epoch + 1:4d}/{self.num_iter}]"
                )
                lines.append("  ┌" + "─" * 110 + "┐")
                lines.append(
                    (
                        f"  │ {'Scale':^5} │ {'G_total':>8} │ {'G_adv':>7} │ {'G_rec':>7} │ "
                        f"{'G_well':>7} │ {'G_div':>7} │ {'G_imp':>7} │ {'D_total':>8} │ {'D_real':>7} │ "
                        f"{'D_fake':>7} │ {'D_gp':>7} │"
                    )
                )
                lines.append("  ├" + "─" * 110 + "┤")

                import torch as _t

                for scale in scales:
                    g = scale_metrics.generator[scale]
                    d = scale_metrics.discriminator[scale]
                    v: list[float] = _t.stack(  # type: ignore[arg-type]
                        [  # type: ignore[arg-type]
                            g.total,
                            g.fake,
                            g.rec,
                            g.well,
                            g.div,
                            g.imp,
                            d.total,
                            d.real,
                            d.fake,
                            d.gp,
                        ]
                    ).tolist()
                    lines.append(
                        (
                            f"  │ {scale:^5} │ {v[0]:8.3f} │ {v[1]:7.3f} │ {v[2]:7.3f} │ "
                            f"{v[3]:7.3f} │ {v[4]:7.3f} │ {v[5]:7.3f} │ {v[6]:8.3f} │ {v[7]:7.3f} │ "
                            f"{v[8]:7.3f} │ {v[9]:7.3f} │"
                        )
                    )

                lines.append("  └" + "─" * 110 + "┘")
                progress.write("\n".join(lines))  # type: ignore

            # Save to TensorBoard and log per-scale (only when TB is
            # enabled — each call does a GPU→CPU sync via torch.stack().tolist();
            # throttled to every _tb_log_interval epochs to reduce host syncs).
            _log_tb = (
                (epoch + 1) % self._tb_log_interval == 0
                or epoch == 0
                or epoch == (self.num_iter - 1)
            )
            if self.enable_tensorboard and _log_tb:
                for scale in scales:
                    g = scale_metrics.generator[scale]
                    d = scale_metrics.discriminator[scale]
                    self.log_epoch(progress, writers[scale], epoch, g, d, global_step)  # type: ignore
                    # Log learning rates per scale
                    lr_g = self.generator_schedulers[scale].get_last_lr()[0]  # type: ignore[union-attr]
                    lr_d = self.discriminator_schedulers[scale].get_last_lr()[0]  # type: ignore[union-attr]
                    writers[scale].add_scalar("LearningRate/generator", lr_g, global_step)  # type: ignore
                    writers[scale].add_scalar("LearningRate/discriminator", lr_d, global_step)  # type: ignore

            # Log when learning rate decays (at every lr_decay interval,
            # before schedulers_step advances the count).
            if (
                self.lr_decay > 0
                and epoch > 0
                and epoch % self.lr_decay == 0
            ):
                lr_g_before = self.generator_schedulers[scales[0]].get_last_lr()[0]  # type: ignore[union-attr]
                lr_g_after = lr_g_before * self.gamma  # type: ignore[operator]
                progress.write(  # type: ignore
                    f"\n  ⚡ LR decay at epoch {epoch}: "
                    f"lr_g {lr_g_before:.2e} → {lr_g_after:.2e}, "
                    f"lr_d {lr_g_before:.2e} → {lr_g_after:.2e} "
                    f"(gamma={self.gamma})"
                )

            # Save generated facies at intervals
            if (
                (epoch % self.save_interval == 0 or epoch == self.num_iter - 1)
                and (epoch != 0 or self.num_iter == 1)
                and (self._current_batch_id == self._total_batches - 1)
            ):
                for scale in scales:
                    self.save_generated_outputs(
                        scale,
                        epoch,
                        self._current_batch_id,
                        results_paths[scale],
                        facies_pyramid[scale],
                        wells_pyramid,
                        masks_pyramid,
                        seismic_pyramid,
                    )

        # ── All ranks ─────────────────────────────────────────────────
        # Step schedulers (LR decay happens here when epoch reaches
        # the milestone — the console message above fires just before).
        self.schedulers_step(scales)
        progress.update(1)  # type: ignore

    def schedulers_step(
        self,
        scales: tuple[int, ...],
    ) -> None:
        """Step the learning-rate schedulers for the provided scales.

        Parameters
        ----------
        generator_schedulers : dict[int, LRScheduler]
            Generator learning-rate schedulers per scale.
        discriminator_schedulers : dict[int, LRScheduler]
            Discriminator learning-rate schedulers per scale.
        scales : tuple[int, ...]
            Tuple of scale indices to step the schedulers for.
        """
        for scale in scales:
            self.generator_schedulers[scale].step()
            self.discriminator_schedulers[scale].step()

    def _release_accelerator_memory(self) -> None:
        """Release unused accelerator (GPU) memory back to the OS.

        Calls the caching allocator to release unused blocks.
        ``empty_cache()`` already triggers an implicit device sync, so
        an explicit ``synchronize()`` is unnecessary.  GC is skipped
        because this is only called between scale groups and the
        allocator handles freed tensors without a Python GC pass.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have empty_cache but we can still trigger GC
                pass
        except ImportError:
            pass

    def train(self) -> None:
        """Train the FaciesGAN model with parallel scale training.

        Trains multiple pyramid scales simultaneously in groups. Processes
        scales in batches of num_parallel_scales at a time.

        When running under DDP only the main process (``_is_main_process``)
        performs I/O (directory creation, model saving, TensorBoard logging,
        progress bar updates).  All ranks participate in model init,
        optimizer setup and training iterations.
        """
        start_train_time = time.time()

        # Train scales in parallel groups
        scale = self.start_scale
        while scale <= self.stop_scale:
            # Determine how many scales to train in this parallel group
            num_scales_in_group = min(
                self.num_parallel_scales, self.stop_scale - scale + 1
            )

            scales_to_train = tuple(range(scale, scale + num_scales_in_group))
            if self._is_main_process:
                print(f"\n{'='*60}")
                print(f"Training scales {scales_to_train} in parallel")
                print(f"{'='*60}\n")

            group_start_time = time.time()

            # Initialize all scales in the group (all ranks)
            self.model.init_scales(scale, num_scales_in_group)

            # Freeze generator blocks from previous groups so backward()
            # does not allocate gradient tensors on them.  The forward
            # pass still runs through these scales (progressive
            # synthesis), but without requires_grad the autograd engine
            # skips gradient storage, saving significant CUDA memory.
            self.model.freeze_generator_scales(scales_to_train)

            # Discard rec_noise for scales outside the current group to
            # free CUDA memory.  Only noise for the active scales (and
            # the scales needed by the progressive forward pass) is
            # kept — the rest is regenerated if needed later.
            self.model.trim_rec_noise(min(scales_to_train))

            # Prune optimizers/schedulers from previous groups to free
            # Adam state buffers and avoid training frozen scales.
            for s in list(self.generator_optimizers):
                if s not in scales_to_train:
                    del self.generator_optimizers[s]
                    del self.generator_schedulers[s]
            for s in list(self.discriminator_optimizers):
                if s not in scales_to_train:
                    del self.discriminator_optimizers[s]
                    del self.discriminator_schedulers[s]

            # Setup optimizers for active scales (all ranks)
            self.setup_optimizers(scales_to_train)

            # Create directories for all scales (use dict to map scale -> path)
            scale_paths: dict[int, str] = {
                s: os.path.join(self.output_path, str(s)) for s in scales_to_train
            }
            results_paths: dict[int, str] = {
                s: os.path.join(scale_paths[s], RESULT_FACIES_PATH)
                for s in scales_to_train
            }

            # Only main process creates directories, writers
            writers: dict[int, SummaryWriter] = {} # type: ignore
            if self._is_main_process:
                for s in scales_to_train:
                    utils.create_dirs(scale_paths[s])
                    utils.create_dirs(results_paths[s])

            if self.fine_tuning:
                for s in scales_to_train:
                    self.load_model(s)

            # ── Epoch-level resume ──────────────────────────────────
            # When start_epoch > 0 for the *first* scale group, load the
            # epoch checkpoint (model weights + optimiser/scheduler states)
            # and figure out which batch to resume from.
            resume_epoch: int = 0
            resume_batch_id: int = 0
            # base_epoch: the epoch that unprocessed batches should start
            # from.  When resuming mid-batch, batches after the resumed
            # one already completed up to the previous full run's epoch
            # count (recorded in completed_epoch.txt).  Without this,
            # they would incorrectly restart from epoch 0.
            base_epoch: int = 0
            if self.start_epoch > 0:
                resume_epoch, resume_batch_id = self.load_epoch_checkpoint(
                    scales_to_train,
                    scale_paths,
                )
                if resume_epoch > 0 and self._is_main_process:
                    print(
                        f"Epoch checkpoint loaded: resuming from batch {resume_batch_id}, "
                        f"epoch {resume_epoch}"
                    )
                if resume_batch_id > 0:
                    # Determine base_epoch: the epoch that all batches
                    # before resume_batch_id already completed.
                    # 1) Check completed_epoch.txt (written when a full
                    #    scale group finishes).
                    # 2) Fall back to self.start_epoch (the CLI value
                    #    that triggered checkpoint loading — this is the
                    #    epoch the scale-level models were trained to in
                    #    the previous run).
                    from config import COMPLETED_EPOCH_FILE

                    meta_path = os.path.join(
                        scale_paths[min(scales_to_train)], COMPLETED_EPOCH_FILE
                    )
                    if os.path.isfile(meta_path):
                        with open(meta_path) as f:
                            base_epoch = int(f.read().strip())
                    else:
                        base_epoch = self.start_epoch

            # Create per-scale TensorBoard writers AFTER resume logic so
            # purge_step can use the correct global_step value.  The
            # first stale step is resume_batch_id * num_iter + resume_epoch.
            if self._is_main_process and self.enable_tensorboard:
                if resume_epoch > 0:
                    _purge: int | None = resume_batch_id * self.num_iter + resume_epoch
                else:
                    _purge = None
                writers = {
                    s: SummaryWriter(log_dir=scale_paths[s], purge_step=_purge)
                    for s in scales_to_train
                }

            # Iterate over DataLoader batches for this group and train on each
            # Single progress bar for all batches and epochs in this group
            total_batches = len(self.data_loader)
            if resume_epoch > 0:
                if resume_batch_id == 0:
                    # All batches completed through resume_epoch (scale-level
                    # checkpoint); every batch runs the remaining epochs.
                    progress_total = (self.num_iter - resume_epoch) * total_batches
                else:
                    # Mid-batch resume: the resumed batch runs fewer epochs;
                    # later ones start from base_epoch (the last fully-completed
                    # epoch count from the previous training run).
                    batches_after_resume = max(0, total_batches - resume_batch_id - 1)
                    progress_total = (
                        self.num_iter - resume_epoch
                    ) + batches_after_resume * (self.num_iter - base_epoch)
            else:
                progress_total = self.num_iter * total_batches
            progress = tqdm(  # type: ignore
                total=progress_total,
                position=0,
                disable=not self._is_main_process,
            )

            # Use create_batch_iterator to allow subclasses to inject prefetching logic
            # Need to load all scales from 0 to max(scales_to_train) for noise initialization
            max_scale = max(scales_to_train)
            all_scales = tuple(range(max_scale + 1))
            batch_iterator = self.create_batch_iterator(self.data_loader, all_scales)

            for batch_id, batch in enumerate(batch_iterator):

                # Skip batches already processed before the resume point
                if resume_epoch > 0 and batch_id < resume_batch_id:
                    continue

                # Determine the epoch to start from for this batch.
                # When resume_batch_id == 0, all batches completed through
                # resume_epoch (scale-level checkpoint), so every batch
                # starts from resume_epoch.  When resume_batch_id > 0
                # (mid-batch epoch checkpoint), only the interrupted batch
                # resumes from resume_epoch; later ones start from
                # base_epoch (the last fully-completed epoch count).
                batch_start_epoch = (
                    resume_epoch
                    if resume_epoch > 0
                    and (resume_batch_id == 0 or batch_id == resume_batch_id)
                    else base_epoch
                )

                # Reset LR schedulers so each batch trains with the
                # same schedule (decay at epoch lr_decay, 2*lr_decay, ...).
                self.reset_schedulers(scales_to_train)

                # Expose batch info to train_scales so it can show epoch progress
                self._total_batches = total_batches
                self._current_batch_id = batch_id

                # Enforce that prepared_batch is present
                if batch is None:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.create_batch_iterator returned None "
                        "for prepared_batch. Data preparation must be handled by the iterator."
                    )

                facies_pyramid, wells_pyramid, masks_pyramid, seismic_pyramid = batch

                # Pass prepared dictionaries directly to train_scales
                self.train_scales(  # type: ignore
                    scales_to_train,
                    writers,
                    scale_paths,
                    results_paths,
                    batch_id,
                    progress,  # type: ignore
                    facies_pyramid=facies_pyramid,
                    wells_pyramid=wells_pyramid,
                    masks_pyramid=masks_pyramid,
                    seismic_pyramid=seismic_pyramid,
                    start_epoch=batch_start_epoch,
                )

            progress.close()  # type: ignore

            # After processing all batches for this group, save models (rank 0 only)
            if self._is_main_process:
                for s in scales_to_train:
                    self.model.save_scale(s, scale_paths[s])
                    self.save_optimizers(
                        scale_paths[s],
                        self.generator_optimizers[s],
                        self.discriminator_optimizers[s],
                        self.generator_schedulers[s],
                        self.discriminator_schedulers[s],
                    )

                # Record the actual last epoch trained so that resume
                # can detect the real training progress.  We write the
                # last epoch index (0-based), i.e. self.num_iter - 1 is
                # the last completed epoch when training ran to the end.
                # NOTE: we deliberately write num_iter (the count of
                # completed epochs) rather than num_iter-1 so that the
                # value can be compared directly with start_epoch.
                from config import COMPLETED_EPOCH_FILE

                for s in scales_to_train:
                    meta_path = os.path.join(scale_paths[s], COMPLETED_EPOCH_FILE)
                    with open(meta_path, "w") as f:
                        f.write(str(self.num_iter))

                # Save a final epoch checkpoint so that the next resume
                # can use the fast-path load instead of falling back to
                # piece-meal scale-level file loading.
                self.save_epoch_checkpoint(
                    scales_to_train,
                    scale_paths,
                    self.num_iter,
                    0,
                )

            # Synchronise so non-zero ranks wait for rank 0 to finish
            # saving before proceeding to the next scale group (or to
            # DDP teardown at the end of training).
            self._ddp_barrier()

            # Close writers (rank 0 only)
            for writer in writers.values():
                writer.close()

            # Release GPU memory cached by the allocator between groups
            self._release_accelerator_memory()

            group_end_time = time.time()
            elapsed = log.format_time(int(group_end_time - group_start_time))
            if self._is_main_process:
                print(f"\nScales {scales_to_train} training time: {elapsed}")

            scale += num_scales_in_group

        end_train_time = time.time()
        if self._is_main_process:
            print(
                "\nTotal training time:",
                log.format_time(int(end_train_time - start_train_time)),
            )

        # Close TensorBoard writer
        if self.enable_tensorboard and self.visualizer:
            self.visualizer.close()
        if self._is_main_process:
            print("\n✅ Training complete!")
        if self.enable_tensorboard:
            print("📊 View results in TensorBoard (if still running)")

        # Final DDP barrier: ensure all ranks have finished training
        # before returning so the caller can safely tear down the
        # process group without one rank still doing I/O.
        self._ddp_barrier()

    def log_epoch(
        self,
        epochs: "tqdm[int]",  # type: ignore
        writer: SummaryWriter, # type: ignore
        epoch: int,
        generator_metrics: GeneratorMetrics[TTensor],
        discriminator_metrics: DiscriminatorMetrics[TTensor],
        global_step: int | None = None,
    ) -> None:
        """Log training metrics for the current epoch to TensorBoard and console.

        Parameters
        ----------
        epochs : tqdm[int]
            Progress bar instance to update description text.
        writer : SummaryWriter
            Per-scale TensorBoard writer to record scalars.
        epoch : int
            Current epoch index (0-based).
        generator_metrics : GeneratorMetrics
            Dataclass carrying tensor-valued generator losses for the scale.
        discriminator_metrics : DiscriminatorMetrics
            Dataclass carrying tensor-valued discriminator losses for the scale.

        Notes
        -----
        Metric dataclass fields are tensor scalars; this function batch-converts
        them to Python floats via ``torch.stack().tolist()`` (single GPU sync)
        before writing to TensorBoard or formatting for display.
        """
        g = generator_metrics
        d = discriminator_metrics

        # Batch all .item() calls into one GPU→CPU sync (single
        # cudaMemcpy instead of 9 separate ones per scale per epoch).
        import torch as _t

        vals: list[float] = _t.stack(  # type: ignore[arg-type]
            [  # type: ignore[arg-type]
                g.total,
                g.fake,
                g.rec,
                g.well,
                g.div,
                g.imp,
                d.total,
                d.real,
                d.fake,
                d.gp,
            ]
        ).tolist()
        g_total, g_fake, g_rec, g_well, g_div, g_imp = vals[:6]
        d_total, d_real, d_fake, d_gp = vals[6:]

        step = global_step if global_step is not None else epoch

        # Update progress bar description with more detailed info
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (self.num_iter - 1):
            epochs.set_description(  # type: ignore
                "Epoch [{:4d}/{}] Scales {} | G: {:.3f} | D: {:.3f}".format(
                    epoch + 1,
                    self.num_iter,
                    list(self.model.active_scales),
                    g_total,
                    d_total,
                )
            )

        # Log to TensorBoard - discriminator losses
        writer.add_scalar("Loss/train/discriminator/real", -d_real, step)  # type: ignore
        writer.add_scalar("Loss/train/discriminator/fake", d_fake, step)  # type: ignore
        writer.add_scalar(  # type: ignore
            "Loss/train/discriminator/gradient_penalty", d_gp, step
        )
        writer.add_scalar("Loss/train/discriminator", d_total, step)  # type: ignore

        # Log to TensorBoard - generator losses
        writer.add_scalar("Loss/train/generator/adversarial", g_fake, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/reconstruction", g_rec, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/well_constraint", g_well, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/diversity", g_div, step)  # type: ignore
        writer.add_scalar("Loss/train/generator/impedance", g_imp, step)  # type: ignore
        writer.add_scalar("Loss/train/generator", g_total, step)  # type: ignore
