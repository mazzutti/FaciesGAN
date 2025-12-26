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
from abc import ABC, abstractmethod
from typing import Any, Iterator

from tensorboardX import SummaryWriter  # type: ignore
from tqdm import tqdm
from typing_extensions import Generic

import utils
from config import RESULT_FACIES_PATH
from datasets import PyramidsDataset
import log
from training.metrics import DiscriminatorMetrics, GeneratorMetrics, ScaleMetrics
from models.base import FaciesGAN
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

        # Common training parameters (conservative subset)
        self.start_scale: int = options.start_scale
        self.stop_scale: int = options.stop_scale
        self.output_path: str = options.output_path
        self.num_iter: int = options.num_iter
        self.save_interval: int = options.save_interval
        self.num_parallel_scales: int = options.num_parallel_scales

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

        print(f"DataLoader num_workers: {self.data_loader.num_workers}")

        self.model = self.create_model()
        self.model.shapes = list(self.scales)

        print("Generated facie shapes:")
        print("╔══════════╦══════════╦══════════╦══════════╗")
        print(
            "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                "Batch", "Channels", "Height", "Width"
            )
        )
        print("╠══════════╬══════════╬══════════╬══════════╣")
        for shape in self.scales:
            print(
                "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                    shape[0], shape[1], shape[2], shape[3]
                )
            )
        print("╚══════════╩══════════╩══════════╩══════════╝")

        # Initialize TensorBoard visualizer if enabled
        self.enable_tensorboard = options.enable_tensorboard
        self.enable_plot_facies = options.enable_plot_facies
        if self.enable_tensorboard:
            viz_path = os.path.join(self.output_path, "training_visualizations")
            log_dir = os.path.join(self.output_path, "tensorboard_logs")
            dataset_info = f"{len(self.dataset)} pyramids, {self.batch_size} batch size"
            if len(options.wells_mask_columns) > 0:
                dataset_info += f", wells: {options.wells_mask_columns}"

            self.visualizer = TensorBoardVisualizer(
                num_scales=self.stop_scale - self.start_scale + 1,
                output_dir=viz_path,
                log_dir=log_dir,
                update_interval=1,
                dataset_info=dataset_info,
            )
            print("📊 TensorBoard logging enabled!")
            print(f"   View training progress: tensorboard --logdir={log_dir}")
            print("   Then open: http://localhost:6006")
        else:
            self.visualizer = None  # type: ignore
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

    @abstractmethod
    def initialize_noise(
        self,
        scale: int,
        indexes: list[int],
        facies_pyramid: tuple[TTensor, ...],
        wells_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
    ) -> TTensor:
        """Initialize reconstruction noise for a specific scale.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        indexes : list[int]
            Batch sample indices.
        facies_pyramid : tuple[TTensor, ...]
            Tuple of real facies data for all scales.
        wells_pyramid : tuple[TTensor, ...], optional
            Tuple of well-conditioning tensors for all scales.
        seismic_pyramid : tuple[TTensor, ...], optional
            Tuple of seismic-conditioning tensors for all scales.

        Returns
        -------
        TTensor
            The upsampled previous reconstruction (``prev_rec``) matching
            ``facies_pyramid`` spatial shape. Note: reconstruction noise ``z_rec``
            is stored in ``self.model.rec_noise`` and is not returned.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement initialize_noise")

    @abstractmethod
    def generate_visualization_samples(
        self,
        scales: tuple[int, ...],
        indexes: list[int],
        wells_pyramid: tuple[TTensor, ...],
        seismic_pyramid: tuple[TTensor, ...],
    ) -> tuple[TTensor, ...]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to generate samples for.
        indexes : list[int]
            List of batch sample indices.
        wells_pyramid : tuple[TTensor, ...], optional
            Tuple of well-conditioning tensors for all scales.
        seismic_pyramid : tuple[TTensor, ...], optional
            Tuple of seismic-conditioning tensors for all scales.

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

    def train_scales(
        self,
        scales: tuple[int, ...],
        writers: tuple[SummaryWriter, ...],
        scale_paths: tuple[str, ...],
        results_paths: tuple[str, ...],
        batch_id: int,
        progress: "tqdm[Any]",
        facies_pyramid: tuple[TTensor, ...],
        wells_pyramid: tuple[TTensor, ...],
        masks_pyramid: tuple[TTensor, ...],
        seismic_pyramid: tuple[TTensor, ...],
    ) -> None:
        """Train multiple pyramid scales simultaneously.

        Accepts prepared data dictionaries directly.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to train.
        writers : tuple[SummaryWriter, ...]
            Tuple of per-scale TensorBoard writers.
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
        wells_pyramid : tuple[TTensor, ...]
            Tuple of well-conditioning tensors for all scales.
        masks_pyramid : tuple[TTensor, ...]
            Tuple of masks tensors for all scales.
        seismic_pyramid : tuple[TTensor, ...]
            Tuple of seismic-conditioning tensors for all scales.
        """

        indexes = list(range(self.batch_size))

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

        # Initialize noise for all scales
        rec_in_pyramid: tuple[TTensor, ...] = tuple(
            self.initialize_noise(
                scale,
                indexes,
                facies_pyramid,
                wells_pyramid,
                seismic_pyramid,
            )
            for scale in scales
        )

        # Training loop - iterate epochs (0-based)
        for epoch in range(self.num_iter):
            progress.set_description(
                f"Batch [{self._current_batch_id + 1}/{self._total_batches}] Epoch [{epoch+1:4d}/{self.num_iter}]"
            )

            generated_samples: tuple[TTensor, ...] = ()

            scale_metrics = self.model(
                indexes,
                facies_pyramid,
                rec_in_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            )

            if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (self.num_iter - 1):
                generated_samples = self.generate_visualization_samples(
                    scales,
                    indexes,
                    wells_pyramid,
                    seismic_pyramid,
                )

            self.handle_epoch_end(
                scales=scales,
                epoch=epoch,
                scale_metrics=scale_metrics,
                generated_samples=generated_samples,
                writers=writers,
                results_paths=results_paths,
                progress=progress,
                real_facies_pyramid=facies_pyramid,
                wells_pyramid=wells_pyramid,
                masks_pyramid=masks_pyramid,
                seismic_pyramid=seismic_pyramid,
            )

        # for scale in scales:
        #     self.save_optimizers(
        #         scale_paths[scale],
        #         generator_optimizers[scale],
        #         discriminator_optimizers[scale],
        #         generator_schedulers[scale],
        #         discriminator_schedulers[scale],
        #     )

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

    def save_generated_facies(
        self,
        scale: int,
        epoch: int,
        results_path: str,
        real_facies: TTensor,
        wells_pyramid: tuple[TTensor, ...] = (),
        masks_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
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
        results_path : str
            Path to the results directory for the current scale.
        real_facies : TTensor
            Real facies tensor for the current scale.
        wells_pyramid : tuple[TTensor, ...]
            Tuple of well-conditioning tensors for all scales.
        masks_pyramid : tuple[TTensor, ...]
            Tuple of masks tensors for all scales.
        seismic_pyramid : tuple[TTensor, ...]
            Tuple of seismic-conditioning tensors for all scales.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement save_generated_facies")

    def handle_epoch_end(
        self,
        scales: tuple[int, ...],
        epoch: int,
        scale_metrics: ScaleMetrics[TTensor],
        generated_samples: tuple[TTensor, ...],
        writers: tuple[SummaryWriter, ...],
        results_paths: tuple[str, ...],
        progress: "tqdm[Any]",
        real_facies_pyramid: tuple[TTensor, ...],
        wells_pyramid: tuple[TTensor, ...],
        masks_pyramid: tuple[TTensor, ...],
        seismic_pyramid: tuple[TTensor, ...],
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
        writers : tuple[SummaryWriter, ...]
            Tuple of per-scale TensorBoard writers.
        results_paths : tuple[str, ...]
            Tuple of per-scale results directory paths.
        progress : tqdm[Any]
            Progress bar instance to update.
        real_facies_pyramid : tuple[TTensor, ...]
            Tuple of real facies data for all scales.
        wells_pyramid : tuple[TTensor, ...]
            Tuple of well-conditioning tensors for all scales.
        masks_pyramid : tuple[TTensor, ...]
            Tuple of masks tensors for all scales.
        seismic_pyramid : tuple[TTensor, ...]
            Tuple of seismic-conditioning tensors for all scales.
        """
        samples_processed = self.batch_size * epoch

        # Visualizer update
        if self.enable_tensorboard and self.visualizer:
            self.visualizer.update(
                epoch, scale_metrics, generated_samples, samples_processed
            )

        # Print formatted metrics table occasionally
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (self.num_iter - 1):
            # Build the entire metrics box as a single string and write it
            # with one `progress.write` call to avoid interleaving with other
            # prints from other threads/processes.
            lines: list[str] = []
            lines.append(
                f"\n  Batch [{self._current_batch_id + 1}/{self._total_batches}] Epoch [{epoch + 1:4d}/{self.num_iter}]"
            )
            lines.append("  ┌" + "─" * 99 + "┐")
            lines.append(
                (
                    f"  │ {'Scale':^5} │ {'G_total':>8} │ {'G_adv':>7} │ {'G_rec':>7} │ "
                    f"{'G_well':>7} │ {'G_div':>7} │ {'D_total':>8} │ {'D_real':>7} │ "
                    f"{'D_fake':>7} │ {'D_gp':>7} │"
                )
            )
            lines.append("  ├" + "─" * 99 + "┤")

            for scale in scales:
                g = scale_metrics.generator[scale]
                d = scale_metrics.discriminator[scale]
                lines.append(
                    (
                        f"  │ {scale:^5} │ {g.total.item():8.3f} │ {g.fake.item():7.3f} │ {g.rec.item():7.3f} │ "
                        f"{g.well.item():7.3f} │ {g.div.item():7.3f} │ {d.total.item():8.3f} │ {d.real.item():7.3f} │ "
                        f"{d.fake.item():7.3f} │ {d.gp.item():7.3f} │"
                    )
                )

            lines.append("  └" + "─" * 99 + "┘")
            progress.write("\n".join(lines))

        # Save to TensorBoard and log per-scale
        for scale in scales:
            g = scale_metrics.generator[scale]
            d = scale_metrics.discriminator[scale]
            self.log_epoch(progress, writers[scale], epoch, g, d)

        # Save generated facies at intervals
        if (
            epoch % self.save_interval == 0 or epoch == self.num_iter - 1
        ) and epoch != 0:
            for scale in scales:
                self.save_generated_facies(
                    scale,
                    epoch,
                    results_paths[scale],
                    real_facies_pyramid[scale],
                    wells_pyramid,
                    masks_pyramid,
                    seismic_pyramid,
                )

        # Step schedulers
        self.model.schedulers_step(scales)
        progress.update(1)

    def train(self) -> None:
        """Train the FaciesGAN model with parallel scale training.

        Trains multiple pyramid scales simultaneously in groups. Processes
        scales in batches of num_parallel_scales at a time.
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
            print(f"\n{'='*60}")
            print(f"Training scales {scales_to_train} in parallel")
            print(f"{'='*60}\n")

            group_start_time = time.time()

            # Initialize all scales in the group
            self.model.init_scales(scale, num_scales_in_group)

            # Create directories for all scales
            scale_paths: tuple[str, ...] = tuple(
                os.path.join(self.output_path, str(s)) for s in scales_to_train
            )
            results_paths: tuple[str, ...] = tuple(
                os.path.join(scale_paths[s], RESULT_FACIES_PATH)
                for s in scales_to_train
            )
            writers: tuple[SummaryWriter, ...] = tuple(
                SummaryWriter(log_dir=scale_paths[s]) for s in scales_to_train
            )
            for s in scales_to_train:
                utils.create_dirs(scale_paths[s])
                utils.create_dirs(results_paths[s])

            if self.fine_tuning:
                for s in scales_to_train:
                    self.load_model(s)

            # Iterate over DataLoader batches for this group and train on each
            # Single progress bar for all batches and epochs in this group
            total_batches = len(self.data_loader)
            progress = tqdm(total=self.num_iter * total_batches, position=0)

            # Use create_batch_iterator to allow subclasses to inject prefetching logic
            batch_iterator = self.create_batch_iterator(
                self.data_loader, scales_to_train
            )

            for batch_id, batch in enumerate(batch_iterator):

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
                self.train_scales(
                    scales_to_train,
                    writers,
                    scale_paths,
                    results_paths,
                    batch_id,
                    progress,
                    facies_pyramid=facies_pyramid,
                    wells_pyramid=wells_pyramid,
                    masks_pyramid=masks_pyramid,
                    seismic_pyramid=seismic_pyramid,
                )

            progress.close()

            # After processing all batches for this group, save models
            for s in scales_to_train:
                self.model.save_scale(s, scale_paths[s])

            # Close writers
            for writer in writers.values():
                writer.close()

            group_end_time = time.time()
            elapsed = log.format_time(int(group_end_time - group_start_time))
            print(f"\nScales {scales_to_train} training time: {elapsed}")

            scale += num_scales_in_group

        end_train_time = time.time()
        print(
            "\nTotal training time:",
            log.format_time(int(end_train_time - start_train_time)),
        )

        # Close TensorBoard writer
        if self.enable_tensorboard and self.visualizer:
            self.visualizer.close()
        print("\n✅ Training complete!")
        if self.enable_tensorboard:
            print("📊 View results in TensorBoard (if still running)")

    def log_epoch(
        self,
        epochs: "tqdm[int]",
        writer: SummaryWriter,
        epoch: int,
        generator_metrics: GeneratorMetrics[TTensor],
        discriminator_metrics: DiscriminatorMetrics[TTensor],
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
        Metric dataclass fields are tensor scalars; this function converts
        them to Python floats via `.item()` before writing to TensorBoard or
        formatting for display.
        """
        g = generator_metrics
        d = discriminator_metrics

        # Update progress bar description with more detailed info
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (self.num_iter - 1):
            epochs.set_description(
                "Epoch [{:4d}/{}] Scales {} | G: {:.3f} | D: {:.3f}".format(
                    epoch + 1,
                    self.num_iter,
                    list(self.model.active_scales),
                    g.total.item(),
                    d.total.item(),
                )
            )

        # Log to TensorBoard - discriminator losses
        writer.add_scalar("Loss/train/discriminator/real", -d.real.item(), epoch)  # type: ignore
        writer.add_scalar("Loss/train/discriminator/fake", d.fake.item(), epoch)  # type: ignore
        writer.add_scalar(  # type: ignore
            "Loss/train/discriminator/gradient_penalty", d.gp.item(), epoch
        )
        writer.add_scalar("Loss/train/discriminator", d.total.item(), epoch)  # type: ignore

        # Log to TensorBoard - generator losses
        writer.add_scalar("Loss/train/generator/adversarial", g.fake.item(), epoch)  # type: ignore
        writer.add_scalar("Loss/train/generator/reconstruction", g.rec.item(), epoch)  # type: ignore
        writer.add_scalar("Loss/train/generator/well_constraint", g.well.item(), epoch)  # type: ignore
        writer.add_scalar("Loss/train/generator/diversity", g.div.item(), epoch)  # type: ignore
        writer.add_scalar("Loss/train/generator", g.total.item(), epoch)  # type: ignore
