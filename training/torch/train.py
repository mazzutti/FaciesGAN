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

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import background_workers as bw
import utils
from config import D_FILE, G_FILE, OPT_D_FILE, OPT_G_FILE, SCH_D_FILE, SCH_G_FILE
from datasets import PyramidsDataset
from datasets.torch.dataset import TorchPyramidsDataset
from models import FaciesGAN, TorchFaciesGAN
from models.torch import utils as torch_utils
from options import TrainningOptions
from training.base import Trainer
from typedefs import Batch


class TorchTrainer(
    Trainer[
        torch.Tensor,
        torch.nn.Module,
        optim.Optimizer,
        optim.lr_scheduler.LRScheduler,
        DataLoader[Batch[torch.Tensor]],
    ]
):
    """Parallel trainer for multi-scale progressive FaciesGAN training.

    Manages simultaneous training of multiple pyramid scales by grouping
    scales and training each group in parallel. Each scale keeps its own
    discriminator and optimizer while the shared generator is exposed via
    the :class:`models.facies_gan.FaciesGAN` instance attached to this
    trainer.

    Parameters
    ----------
    options : TrainningOptions
        Training configuration containing hyperparameters and paths.
    fine_tuning : bool, optional
        Whether to load and fine-tune from existing checkpoints.
    checkpoint_path : str, optional
        Base path used to load/save per-scale checkpoints.
    device : torch.device
        Device used for training (cpu/cuda/mps).

    Attributes
    ----------
    device : torch.device
        Training device.
    model : FaciesGAN
        The multi-scale model instance managed by the trainer.

    Notes
    -----
    - The trainer updates ``self.model.rec_noise`` and ``self.model.noise_amp``
      as part of noise initialization (see :meth:`initialize_noise`).
    - Conditioning tensors (wells/seismic) are expected channels-last when
      prepared and returned by :meth:`prepare_scale_batch`.
    """

    def __init__(
        self,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
        device: torch.device = torch.device("cpu"),
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
        super().__init__(options, fine_tuning, checkpoint_path)

    def create_dataloader(self) -> DataLoader[Batch[torch.Tensor]]:
        """Create and return a :class:`torch.utils.data.DataLoader` for the
        trainer's dataset using configured batch size and worker settings.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.options.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=(self.options.num_workers > 0),
        )

    def create_model(self) -> FaciesGAN[torch.Tensor, torch.nn.Module]:
        """Instantiate and return the :class:`TorchFaciesGAN` configured
        with the trainer options and device.
        """
        return TorchFaciesGAN(
            self.options,
            self.wells,
            self.seismic,
            self.device,
            noise_channels=self.noise_channels,
        )

    def create_optimizers_and_schedulers(self, scales: list[int]) -> tuple[
        dict[int, optim.Optimizer],
        dict[int, optim.Optimizer],
        dict[int, optim.lr_scheduler.LRScheduler],
        dict[int, optim.lr_scheduler.LRScheduler],
    ]:
        """Create per-scale optimizers and learning-rate schedulers.

        Parameters
        ----------
        scales : list[int]
            List of scale indices to create optimizers/schedulers for.

        Returns
        -------
        tuple
            Four dictionaries mapping scale index -> optimizer/scheduler in the
            order: ``(generator_optimizers, discriminator_optimizers,
            generator_schedulers, discriminator_schedulers)``.
        """
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

        return (
            generator_optimizers,
            discriminator_optimizers,
            generator_schedulers,
            discriminator_schedulers,
        )

    def generate_visualization_samples(
        self, scales: list[int], indexes: list[int]
    ) -> dict[int, torch.Tensor]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : list[int]
            List of scale indices to generate samples for.
        indexes : list[int]
            List of batch sample indices.

        Returns
        -------
        dict[int, torch.Tensor]
            A dictionary mapping scale indices to generated facies tensors
            for visualization.
        """
        generated_samples: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for scale in scales:
                generated_samples[scale] = self.model.generate_fake(
                    self.model.get_noise(indexes, scale), scale
                )
        return generated_samples

    def initialize_noise(
        self,
        scale: int,
        real_facies: torch.Tensor,
        indexes: list[int],
        wells: torch.Tensor | None = None,
        seismic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Initialize and append reconstruction noise for a specific scale.

        For ``scale == 0`` this generates an initial ``z_rec`` and computes the
        initial noise amplitude appended to ``self.model.noise_amp``. For
        higher scales it upsamples the previous reconstruction, composes the
        correct noise channels (optionally concatenating wells/seismic), and
        stores the resulting ``z_rec`` in ``self.model.rec_noise``.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        real_facies : torch.Tensor
            Real facies data at the current scale (channels-last expected).
        indexes : list[int]
            Batch sample indices used to generate noise with deterministic
            indexing.
        wells : torch.Tensor | None
            Optional wells conditioning tensor for this batch (channels-last).
        seismic : torch.Tensor | None
            Optional seismic conditioning tensor for this batch (channels-last).

        Returns
        -------
        torch.Tensor
            The upsampled previous reconstruction (``prev_rec``) matching
            ``real_facies`` spatial shape.

        Side effects
        ------------
        - Appends the generated reconstruction noise ``z_rec`` to
          ``self.model.rec_noise``.
        - Updates or appends the noise amplitude entry in ``self.model.noise_amp``.
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

    def init_dataset(
        self,
    ) -> tuple[PyramidsDataset[torch.Tensor], tuple[tuple[int, ...], ...]]:
        """Initialize and possibly subsample the pyramids dataset.

        Applies optional selection via ``options.wells_mask_columns`` or
        subsamples the dataset randomly to ``options.num_train_pyramids``
        if that value is smaller than the dataset size.

        Returns
        -------
        tuple
            A pair ``(dataset, scales)`` where ``dataset`` is a
            :class:`datasets.torch.dataset.TorchPyramidsDataset` instance and
            ``scales`` is the tuple of pyramid scales present in the dataset.
        """
        dataset = TorchPyramidsDataset(self.options)
        if len(self.options.wells_mask_columns) > 0:
            sel = [int(i) for i in self.options.wells_mask_columns]
            dataset.batches = [dataset.batches[i] for i in sel]
        elif self.options.num_train_pyramids < len(dataset):
            idxs = torch.randperm(len(dataset))[: self.options.num_train_pyramids]
            dataset.batches = [dataset.batches[i] for i in idxs]
        return dataset, dataset.scales

    def load_model(self, scale: int) -> None:
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

    def load_optimizers(
        self,
        scale: int,
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: optim.lr_scheduler.LRScheduler,
        discriminator_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Load optimizer and scheduler state dictionaries from checkpoint.

        If any checkpoint files are missing or incompatible a warning is
        printed and the trainer continues without restoring those states.
        """
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

    def prepare_scale_batch(self, scales: list[int], indexes: list[int]) -> tuple[
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
    ]:
        """Prepare and move to device the batch tensors for given scales.

        Parameters
        ----------
        scales : list[int]
            List of scale indices to prepare batches for.
        indexes : list[int]
            List of batch sample indices.

        Returns
        -------
        tuple containing:
        - real_facies_dict: dict[int, torch.Tensor]
            Real facies tensors per scale moved to ``self.device`` (channels-last).
        - masks_dict: dict[int, torch.Tensor]
            Well masks per scale moved to device (present when wells are used).
        - wells_dict: dict[int, torch.Tensor]
            Well conditioning tensors per scale moved to device (channels-last).
        - seismic_dict: dict[int, torch.Tensor]
            Seismic conditioning tensors per scale moved to device (channels-last).
        """
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

        return real_facies_dict, masks_dict, wells_dict, seismic_dict

    def save_generated_facies(
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
            bw.submit_plot_generated_facies(
                generated_facies_cpu,
                real_facies_cpu,
                scale,
                epoch,
                results_path,
                masks_cpu,
            )

    def save_optimizers(
        self,
        scale_path: str,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        generator_scheduler: LRScheduler,
        discriminator_scheduler: LRScheduler,
    ) -> None:
        """Save optimizer and scheduler state dicts to disk using project
        filename constants from :mod:`config`.
        """
        os.makedirs(scale_path, exist_ok=True)
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

    def schedulers_step(
        self,
        generator_schedulers: dict[int, LRScheduler],
        discriminator_schedulers: dict[int, LRScheduler],
        scales: list[int],
    ) -> None:
        """Step the learning-rate schedulers for the provided scales.

        Parameters
        ----------
        generator_schedulers : dict[int, LRScheduler]
            Generator learning-rate schedulers per scale.
        discriminator_schedulers : dict[int, LRScheduler]
            Discriminator learning-rate schedulers per scale.
        scales : list[int]
            Scale indices to step the schedulers for.
        """
        for scale in scales:
            generator_schedulers[scale].step()
            discriminator_schedulers[scale].step()
