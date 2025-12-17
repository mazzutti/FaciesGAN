"""Parallel trainer for multi-scale FaciesGAN training.

This module implements a training pipeline that trains multiple pyramid
scales simultaneously in parallel groups. The parallel trainer processes
multiple scales at once (controlled by ``num_parallel_scales``) instead of
the sequential scale-by-scale training used in the original progressive
implementation. Each scale keeps its own discriminator and optimizer, while
the generator is managed by the central ``FaciesGAN`` model instance.

Notes
-----
- For efficiency this trainer typically uses a single data batch per group
    of scales (the DataLoader yields batches of pyramids and a group consumes
    one batch to train all its scales in parallel).
- The trainer stores per-scale reconstruction noise and noise amplitudes in
    the model's ``rec_noise`` and ``noise_amp`` lists respectively.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

import ops
import utils
from background_workers import submit_plot_generated_facies
from config import (D_FILE, G_FILE, OPT_D_FILE, OPT_G_FILE, RESULT_FACIES_PATH,
                    SCH_D_FILE, SCH_G_FILE)
from dataset import PyramidsDataset
from log import format_time
from metrics import DiscriminatorMetrics, GeneratorMetrics, ScaleMetrics
from models.torch import utils as torch_utils
from models.torch.facies_gan import TorchFaciesGAN
from options import TrainningOptions
from tensorboard_visualizer import TensorBoardVisualizer


class Trainer:
    """Parallel trainer for multi-scale progressive FaciesGAN training.

    This class manages simultaneous training of multiple pyramid scales by
    grouping scales and training each group in parallel. Each scale retains
    its own discriminator and optimizer while the shared generator is
    exposed through the central :class:`models.facies_gan.FaciesGAN` model.

    Parameters
    ----------
    device : torch.device
        Device for training (CPU, CUDA, or MPS).
    options : TrainningOptions
        Training configuration containing hyperparameters and paths.
    fine_tuning : bool, optional
        Whether to load and fine-tune from existing checkpoints. Defaults to False.
    checkpoint_path : str, optional
        Path to load checkpoints from when fine-tuning. Defaults to ".checkpoints/".

    Attributes
    ----------
    device : torch.device
        Training device.
    start_scale : int
        Starting pyramid scale index.
    stop_scale : int
        Final pyramid scale index.
    batch_size : int
        Effective batch size for training. Note: the dataset yields per-pyramid
        batches; groups of scales commonly consume a single batch when trained
        in parallel.
    model : FaciesGAN
        The FaciesGAN model instance. Per-scale state such as ``rec_noise`` and
        ``noise_amp`` is stored in the model and updated during initialization
        and training.
    scales_list : tuple[tuple[int, ...], ...]
        Pyramid resolutions for each scale.
    data_loader : DataLoader
        PyTorch DataLoader for training data.
    visualizer : TensorBoardVisualizer
        Handles per-epoch visualization and writes global logs under the
        configured `log_dir` (``<output_path>/tensorboard_logs``). Each scale
        also writes a per-scale SummaryWriter to its scale directory.
    """

    def __init__(
        self,
        device: torch.device,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints/",
    ) -> None:
        """Create a Trainer instance and prepare datasets, model and logging.

        Parameters
        ----------
        device : torch.device
            Device used for training (cpu/cuda/mps).
        options : TrainningOptions
            Training options with hyperparameters and paths.
        fine_tuning : bool, optional
            Whether to attempt to load existing checkpoints, by default False.
        checkpoint_path : str, optional
            Base path for checkpoint files, by default ".checkpoints/".
        """
        self.device: torch.device = device

        # Training parameters
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
        self.batch_size: int = (
            self.batch_size
            if not (
                len(options.wells_mask_columns) > 0
                and options.batch_size < len(options.wells_mask_columns)
            )
            else len(options.wells_mask_columns)
        )
        self.fine_tuning: bool = fine_tuning
        self.checkpoint_path: str = checkpoint_path

        self.num_img_channels: int = options.num_img_channels
        self.noise_channels: int = (
            options.noise_channels
            + (self.num_img_channels if options.use_wells else 0)
            + (self.num_img_channels if options.use_seismic else 0)
        )

        self.num_real_facies: int = options.num_real_facies
        self.num_generated_per_real: int = options.num_generated_per_real
        self.wells_mask_columns: tuple[int, ...] = options.wells_mask_columns

        # Optimizer configuration
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
        self.facies: list[torch.Tensor] = []
        self.wells: list[torch.Tensor] = []
        self.seismic: list[torch.Tensor] = []

        dataset: PyramidsDataset = PyramidsDataset(options)
        self.scales_list: tuple[tuple[int, ...], ...] = dataset.scales_list  # type: ignore

        if len(options.wells_mask_columns) > 0:
            for i in range(len(self.scales_list)):
                dataset.facies_pyramids[i] = dataset.facies_pyramids[i][
                    options.wells_mask_columns
                ]
                if options.use_wells:
                    dataset.wells_pyramids[i] = dataset.wells_pyramids[i][
                        options.wells_mask_columns
                    ]
                if options.use_seismic:
                    dataset.seismic_pyramids[i] = dataset.seismic_pyramids[i][
                        options.wells_mask_columns
                    ]
        elif options.num_train_pyramids < len(dataset):
            idxs = torch.randperm(len(dataset))[: options.num_train_pyramids]
            for i in range(len(self.scales_list)):
                dataset.facies_pyramids[i] = dataset.facies_pyramids[i][idxs]
                if options.use_wells:
                    dataset.wells_pyramids[i] = dataset.wells_pyramids[i][idxs]
                if options.use_seismic:
                    dataset.seismic_pyramids[i] = dataset.seismic_pyramids[i][idxs]

        self.data_loader = DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=(options.num_workers > 0),
        )
        print(f"DataLoader num_workers: {self.data_loader.num_workers}")

        self.num_of_batchs: int = len(dataset) // self.batch_size

        self.model: TorchFaciesGAN = TorchFaciesGAN(
            options,
            self.wells,
            self.seismic,
            device,
            noise_channels=self.noise_channels,
        )
        self.model.shapes = list(self.scales_list)

        print("Generated facie shapes:")
        print("╔══════════╦══════════╦══════════╦══════════╗")
        print(
            "║ {:^8} ║ {:^8} ║ {:^8} ║ {:^8} ║".format(
                "Batch", "Channels", "Height", "Width"
            )
        )
        print("╠══════════╬══════════╬══════════╬══════════╣")
        for shape in self.scales_list:
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
            dataset_info = f"{len(dataset)} pyramids, {self.batch_size} batch size"
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

            scales_to_train = list(range(scale, scale + num_scales_in_group))
            print(f"\n{'='*60}")
            print(f"Training scales {scales_to_train} in parallel")
            print(f"{'='*60}\n")

            group_start_time = time.time()

            # Initialize all scales in the group
            self.model.init_scales(scale, num_scales_in_group)

            # Create directories for all scales
            scale_paths: dict[int, str] = {}
            results_paths: dict[int, str] = {}
            writers: dict[int, SummaryWriter] = {}
            for s in scales_to_train:
                scale_path = os.path.join(self.output_path, str(s))
                results_path = os.path.join(scale_path, RESULT_FACIES_PATH)
                ops.create_dirs(scale_path)
                ops.create_dirs(results_path)
                scale_paths[s] = scale_path
                results_paths[s] = results_path
                writers[s] = SummaryWriter(log_dir=scale_path)

            if self.fine_tuning:
                for s in scales_to_train:
                    self.__load_model(s)

            # Iterate over DataLoader batches for this group and train on each
            # Single progress bar for all batches and epochs in this group
            total_batches = len(self.data_loader)
            progress = tqdm(total=self.num_iter * total_batches, position=0)
            for batch_id, batch in enumerate(self.data_loader):
                # Each iteration yields a batch of pyramids (facies, wells, seismic)
                self.facies, self.wells, self.seismic = batch

                self.model.wells = self.wells
                self.model.seismic = self.seismic

                # Expose batch info to train_scales so it can show epoch progress
                self._total_batches = total_batches
                self._current_batch_id = batch_id

                # Train all scales in this group using the current batch
                self.train_scales(
                    scales_to_train,
                    writers,
                    scale_paths,
                    results_paths,
                    batch_id,
                    progress,
                )

            progress.close()

            # After processing all batches for this group, save models
            for s in scales_to_train:
                self.model.save_scale(s, scale_paths[s])

            # Close writers
            for writer in writers.values():
                writer.close()

            group_end_time = time.time()
            elapsed = format_time(int(group_end_time - group_start_time))
            print(f"\nScales {scales_to_train} training time: {elapsed}")

            scale += num_scales_in_group

        end_train_time = time.time()
        print(
            "\nTotal training time:",
            format_time(int(end_train_time - start_train_time)),
        )

        # Close TensorBoard writer
        if self.enable_tensorboard and self.visualizer:
            self.visualizer.close()
        print("\n✅ Training complete!")
        if self.enable_tensorboard:
            print("📊 View results in TensorBoard (if still running)")

    def train_scales(
        self,
        scales: list[int],
        writers: dict[int, SummaryWriter],
        scale_paths: dict[int, str],
        results_paths: dict[int, str],
        batch_id: int,
        progress: "tqdm[Any]",
    ) -> None:
        """Train multiple pyramid scales simultaneously.

        Parameters
        ----------
        scales : list[int]
            List of scale indices to train in parallel.
        writers : dict[int, SummaryWriter]
            Dictionary mapping scale indices to TensorBoard writers.
        scale_paths : dict[int, str]
            Dictionary mapping scale indices to checkpoint directories.
        results_paths : dict[int, str]
            Dictionary mapping scale indices to results directories.
        batch_id : int
            Current batch index within the epoch.
        """
        # Batch is provided by caller (`train()` sets `self.facies`, `self.wells`, `self.seismic`)

        indexes = list(range(self.batch_size))

        # Prepare data for all scales
        real_facies_dict: dict[int, torch.Tensor] = {}
        masks_dict: dict[int, torch.Tensor] = {}
        wells_dict: dict[int, torch.Tensor] = {}
        seismic_dict: dict[int, torch.Tensor] = {}

        for scale in scales:
            facies_batch = self.facies[scale][indexes]

            # Wells (optional)
            if len(self.wells) > 0:
                wells_batch = self.wells[scale][indexes]
                wells_dev = utils.to_device(
                    wells_batch, self.device, channels_last=True
                )
                masks_dev = (wells_dev.abs().sum(dim=1, keepdim=True) > 0).int()
                masks_dev = utils.to_device(masks_dev, self.device, channels_last=True)
                wells_dict[scale] = wells_dev
                masks_dict[scale] = masks_dev

            # Seismic (optional)
            if len(self.seismic) > 0:
                seismic_batch = self.seismic[scale][indexes]
                seismic_dev = utils.to_device(
                    seismic_batch, self.device, channels_last=True
                )
                seismic_dict[scale] = seismic_dev

            # Real facies
            real_facies_dict[scale] = utils.to_device(
                facies_batch, self.device, channels_last=True
            )

        # Create optimizers for all scales
        generator_optimizers: dict[int, optim.Optimizer] = {}
        discriminator_optimizers: dict[int, optim.Optimizer] = {}
        generator_schedulers: dict[int, optim.lr_scheduler.LRScheduler] = {}
        discriminator_schedulers: dict[int, optim.lr_scheduler.LRScheduler] = {}

        for scale in scales:
            generator_optimizers[scale] = optim.Adam(
                self.model.generator.gens[scale].parameters(),
                lr=self.lr_g,
                betas=(self.beta1, 0.999),
            )
            discriminator_optimizers[scale] = optim.Adam(
                self.model.discriminator.discs[scale].parameters(),
                lr=self.lr_d,
                betas=(self.beta1, 0.999),
            )
            generator_schedulers[scale] = optim.lr_scheduler.MultiStepLR(
                generator_optimizers[scale],
                milestones=[self.lr_decay],
                gamma=self.gamma,
            )
            discriminator_schedulers[scale] = optim.lr_scheduler.MultiStepLR(
                discriminator_optimizers[scale],
                milestones=[self.lr_decay],
                gamma=self.gamma,
            )

        if self.fine_tuning:
            for scale in scales:
                self.__load_optimizers(
                    scale,
                    scale_paths[scale],
                    generator_optimizers[scale],
                    discriminator_optimizers[scale],
                    generator_schedulers[scale],
                    discriminator_schedulers[scale],
                )

        # Initialize noise for all scales
        rec_in_dict: dict[int, torch.Tensor] = {}
        for scale in scales:
            wells = wells_dict[scale] if len(self.wells) > 0 else None
            seismic = seismic_dict[scale] if len(self.seismic) > 0 else None
            prev_rec = self.__initialize_noise(
                scale, real_facies_dict[scale], indexes, wells, seismic
            )
            rec_in_dict[scale] = prev_rec

        # Training loop - iterate epochs (0-based) and update the single progress bar
        for epoch in range(self.num_iter):
            # Update progress description with current batch and epoch
            progress.set_description(
                f"Batch [{self._current_batch_id + 1}/{self._total_batches}] Epoch [{epoch+1:4d}/{self.num_iter}]"
            )

            generated_samples: dict[int, torch.Tensor] = {}

            discriminator_metrics = self.model.optimize_discriminator(
                indexes, real_facies_dict, discriminator_optimizers
            )
            generator_metrics = self.model.optimize_generator(
                indexes, real_facies_dict, masks_dict, rec_in_dict, generator_optimizers
            )

            scale_metrics = ScaleMetrics(
                generator=generator_metrics, discriminator=discriminator_metrics
            )

            # Update visualizer
            if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (self.num_iter - 1):
                # Generate samples for visualization every 50 epochs
                with torch.no_grad():
                    for scale in scales:
                        fake = self.model.generator(
                            self.model.get_noise(indexes, scale),
                            [1.0] * (scale + 1),
                            stop_scale=scale,
                        )
                        generated_samples[scale] = fake[0]

                samples_processed = self.batch_size * epoch
                if self.enable_tensorboard and self.visualizer:
                    self.visualizer.update(
                        epoch, scale_metrics, generated_samples, samples_processed
                    )

                # Always print formatted metrics table for all scales (multi-line table)
                progress.write(
                    f"\n  Batch [{self._current_batch_id + 1}/{self._total_batches}] Epoch [{epoch + 1:4d}/{self.num_iter}]"
                )
                progress.write("  ┌" + "─" * 99 + "┐")
                progress.write(
                    f"  │ {'Scale':^5} │ {'G_total':>8} │ {'G_adv':>7} │ {'G_rec':>7} │ "
                    f"{'G_well':>7} │ {'G_div':>7} │ {'D_total':>8} │ {'D_real':>7} │ "
                    f"{'D_fake':>7} │ {'D_gp':>7} │"
                )
                progress.write("  ├" + "─" * 99 + "┤")

                for scale in scales:
                    g = scale_metrics.generator[scale]
                    d = scale_metrics.discriminator[scale]

                    progress.write(
                        f"  │ {scale:^5} │ {g.total.item():8.3f} │ {g.fake.item():7.3f} │ {g.rec.item():7.3f} │ "
                        f"{g.well.item():7.3f} │ {g.div.item():7.3f} │ {d.total.item():8.3f} │ {d.real.item():7.3f} │ "
                        f"{d.fake.item():7.3f} │ {d.gp.item():7.3f} │"
                    )

                progress.write("  └" + "─" * 99 + "┘")
            else:
                # Update visualizer without generating samples (just metrics)
                samples_processed = self.batch_size * epoch
                if self.enable_tensorboard and self.visualizer:
                    self.visualizer.update(
                        epoch, scale_metrics, None, samples_processed
                    )

            # Save to TensorBoard for all scales (pass `progress` as the bar)
            for scale in scales:
                g = scale_metrics.generator[scale]
                d = scale_metrics.discriminator[scale]

                self.__log_epoch(progress, writers[scale], epoch, g, d)

            # Save generated facies at intervals
            if (
                epoch % self.save_interval == 0 or epoch == self.num_iter - 1
            ) and epoch != 0:
                for scale in scales:
                    self.__save_generated_facies(
                        scale,
                        epoch,
                        results_paths[scale],
                        masks_dict[scale] if len(self.wells) > 0 else None,
                    )

            # Step schedulers
            for scale in scales:
                generator_schedulers[scale].step()
                discriminator_schedulers[scale].step()

            # Advance the single progress bar (one step per epoch)
            try:
                progress.update(1)
            except Exception:
                pass

        # Save optimizers for all scales
        for scale in scales:
            self.__save_optimizers(
                scale_paths[scale],
                generator_optimizers[scale],
                discriminator_optimizers[scale],
                generator_schedulers[scale],
                discriminator_schedulers[scale],
            )

    def __initialize_noise(
        self,
        scale: int,
        real_facies: torch.Tensor,
        indexes: list[int],
        wells: torch.Tensor | None = None,
        seismic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Initialize reconstruction noise for a specific scale.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        wells : torch.Tensor
            Well conditioning data.
        real_facies : torch.Tensor
            Real facies data at the current scale.
        indexes : list[int]
            Batch sample indices.

                Returns
                -------
                torch.Tensor
                        The upsampled previous reconstruction (``prev_rec``) matching
                        ``real_facies`` spatial shape. Note: reconstruction noise ``z_rec``
                        is stored in ``self.model.rec_noise`` and is not returned.
        """
        # Prepare previous reconstruction
        if scale == 0:
            prev_rec = torch.zeros_like(real_facies)
            z_rec = torch_utils.generate_noise(
                (self.noise_channels, *real_facies.shape[2:]),
                device=self.device,
                num_samp=self.batch_size,
            )
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            self.model.rec_noise.append(z_rec)

            # Calculate noise amplitude for scale 0
            with torch.no_grad():
                fake = self.model.generator(
                    self.model.get_noise(indexes, scale),
                    [1.0] * (scale + 1),
                    stop_scale=scale,
                )

            rmse = torch.sqrt(F.mse_loss(fake, real_facies))
            amp = self.scale0_noise_amp * rmse.item()
            self.model.noise_amp.append(amp)

        else:
            # For higher scales, upsample previous facies to current resolution
            prev_rec = torch_utils.interpolate(
                self.facies[scale - 1][indexes], real_facies.shape[2:]
            ).to(self.device)

            # noise channel sizing for higher scales (empirical split)
            if wells is None:
                if seismic is None:
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels,
                            *real_facies.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                else:
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels - self.num_img_channels,
                            *real_facies.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                    z_rec = torch.cat([z_rec, seismic], dim=1)
            else:
                if seismic is None:
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels - self.num_img_channels,
                            *real_facies.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                    z_rec = torch.cat([z_rec, wells], dim=1)
                else:
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels - 2 * self.num_img_channels,
                            *real_facies.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                    z_rec = torch.cat([z_rec, wells, seismic], dim=1)
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            self.model.rec_noise.append(z_rec)

            # Calculate noise amplitude based on reconstruction error
            with torch.no_grad():
                fake = self.model.generator(
                    self.model.get_noise(indexes, scale),
                    self.model.noise_amp + [1.0],
                    stop_scale=scale,
                )

            rmse = torch.sqrt(F.mse_loss(fake, real_facies))
            amp = max(self.noise_amp * rmse.item(), self.min_noise_amp)

            if scale < len(self.model.noise_amp):
                self.model.noise_amp[scale] = (amp + self.model.noise_amp[scale]) / 2
            else:
                self.model.noise_amp.append(amp)

        return prev_rec

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

    def __load_model(self, scale: int) -> None:
        """Load generator and discriminator state dicts for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index to load models for.
        """
        try:
            generator_path = os.path.join(str(self.checkpoint_path), str(scale), G_FILE)
            discriminator_path = os.path.join(
                str(self.checkpoint_path), str(scale), D_FILE
            )

            self.model.generator.gens[scale].load_state_dict(
                torch_utils.load(generator_path, self.device, as_type=Mapping[str, Any])
            )
            self.model.discriminator.discs[scale].load_state_dict(
                torch_utils.load(
                    discriminator_path, self.device, as_type=Mapping[str, Any]
                )
            )
        except Exception as e:
            print(f"Error loading models from {self.checkpoint_path}/{scale}: {e}")
            raise

    def __load_optimizers(
        self,
        scale: int,
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: optim.lr_scheduler.LRScheduler,
        discriminator_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Load optimizer and scheduler state dictionaries from checkpoint."""
        try:
            generator_optimizer.load_state_dict(
                torch_utils.load(
                    os.path.join(scale_path, OPT_G_FILE),
                    self.device,
                    as_type=dict[str, Any],
                )
            )
            discriminator_optimizer.load_state_dict(
                torch_utils.load(
                    os.path.join(scale_path, OPT_D_FILE),
                    self.device,
                    as_type=dict[str, Any],
                )
            )
            generator_scheduler.load_state_dict(
                torch_utils.load(
                    os.path.join(scale_path, SCH_G_FILE),
                    self.device,
                    as_type=dict[str, Any],
                )
            )
            discriminator_scheduler.load_state_dict(
                torch_utils.load(
                    os.path.join(scale_path, SCH_D_FILE),
                    self.device,
                    as_type=dict[str, Any],
                )
            )
        except Exception as e:
            print(f"Warning: Could not load optimizers for scale {scale}: {e}")

    def __log_epoch(
        self,
        epochs: "tqdm[int]",
        writer: SummaryWriter,
        epoch: int,
        generator_metrics: GeneratorMetrics[torch.Tensor],
        discriminator_metrics: DiscriminatorMetrics[torch.Tensor],
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

    def __save_generated_facies(
        self,
        scale: int,
        epoch: int,
        results_path: str,
        masks: torch.Tensor | None = None,
    ) -> None:
        """Save generated facies visualizations to disk asynchronously.

        This method samples noises, generates multiple facies images per real
        sample, clips them to [-1, 1], moves them to CPU and submits a
        background worker job to save the visualization images. Masks are
        passed through for overlay if provided.
        """
        indexes = torch.randint(self.batch_size, (self.num_real_facies,))
        real_facies = self.facies[scale][indexes]

        noises = [
            self.model.get_noise(
                [int(index.item())] * self.num_generated_per_real, scale
            )
            for index in indexes
        ]

        with torch.no_grad():
            generated_facies = [
                self.model.generator(
                    noise, self.model.noise_amp[: scale + 1], stop_scale=scale
                )
                for noise in noises
            ]
            generated_facies = [gen.clip(-1, 1) for gen in generated_facies]

        if self.enable_plot_facies:

            generated_facies_cpu = [g.detach().cpu() for g in generated_facies]
            real_facies_cpu = real_facies.detach().cpu()
            masks_cpu = masks[indexes].detach().cpu() if masks is not None else None

            # Submit background job to save the plot (non-blocking)
            submit_plot_generated_facies(
                generated_facies_cpu,
                real_facies_cpu,
                scale,
                epoch,
                results_path,
                masks_cpu,
            )

    @staticmethod
    def __save_optimizers(
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: optim.lr_scheduler.LRScheduler,
        discriminator_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Save optimizer and scheduler state dictionaries to disk."""
        torch.save(
            generator_optimizer.state_dict(), os.path.join(scale_path, OPT_G_FILE)
        )
        torch.save(
            discriminator_optimizer.state_dict(), os.path.join(scale_path, OPT_D_FILE)
        )
        torch.save(
            generator_scheduler.state_dict(), os.path.join(scale_path, SCH_G_FILE)
        )
        torch.save(
            discriminator_scheduler.state_dict(), os.path.join(scale_path, SCH_D_FILE)
        )
