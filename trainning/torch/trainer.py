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
from collections.abc import Iterator, Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import background_workers as bw
from datasets.data_prefetcher import PyramidsBatch
from datasets.torch.data_prefetcher import TorchDataPrefetcher
from config import D_FILE, G_FILE, OPT_D_FILE, OPT_G_FILE, SCH_D_FILE, SCH_G_FILE
from datasets import PyramidsDataset
from datasets.torch.dataset import TorchPyramidsDataset
from models import FaciesGAN, TorchFaciesGAN
from models.torch import utils as torch_utils
from options import TrainningOptions
from trainning.base import Trainer
from typedefs import Batch
from utils import torch2np


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
      prepared and returned by :meth:`TorchDataPrefetcher`.
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

    def create_model(
        self,
    ) -> FaciesGAN[
        torch.Tensor, torch.nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler
    ]:
        """Instantiate and return the :class:`TorchFaciesGAN` configured
        with the trainer options and device.
        """
        return TorchFaciesGAN(
            self.options,
            self.device,
            noise_channels=self.noise_channels,
        )

    def generate_visualization_samples(
        self,
        scales: tuple[int, ...],
        indexes: list[int],
        wells_pyramid: tuple[torch.Tensor, ...],
        seismic_pyramid: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to generate samples for.
        indexes : tuple[int, ...]
            Tuple of batch sample indices.
        wells_pyramid : tuple[torch.Tensor, ...]
            Tuple of well-conditioning tensors per scale.
        seismic_pyramid : tuple[torch.Tensor, ...]
            Tuple of seismic-conditioning tensors per scale.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of generated facies tensors for visualization, one per scale.
        """
        with torch.no_grad():
            return tuple(
                self.model.generate_fake(
                    self.model.get_pyramid_noise(
                        scale,
                        indexes,
                        wells_pyramid,
                        seismic_pyramid,
                    ),
                    scale,
                )
                for scale in scales
            )

    def initialize_noise(
        self,
        scale: int,
        indexes: list[int],
        facies_pyramid: tuple[torch.Tensor, ...],
        wells_pyramid: tuple[torch.Tensor, ...] = (),
        seismic_pyramid: tuple[torch.Tensor, ...] = (),
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
        indexes : list[int]
            List of batch sample indices.
        facies_pyramid : tuple[torch.Tensor, ...]
            Tuple of real facies data for all scales.
        wells_pyramid : tuple[torch.Tensor, ...] | None, optional
            Tuple of well-conditioning tensors per scale.
        seismic_pyramid : tuple[torch.Tensor, ...] | None, optional
            Tuple of seismic-conditioning tensors per scale.

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
        real = facies_pyramid[scale]
        if scale == 0:
            prev_rec = torch.zeros_like(real)
            z_rec = torch_utils.generate_noise(
                (self.noise_channels, *real.shape[2:]),
                device=self.device,
                num_samp=self.batch_size,
            )
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            self.model.rec_noise.append(z_rec)

            # Calculate noise amplitude for scale 0
            with torch.no_grad():
                fake = self.model.generator(
                    self.model.get_pyramid_noise(scale, indexes),
                    [1.0] * (scale + 1),
                    stop_scale=scale,
                )

            rmse = torch.sqrt(F.mse_loss(fake, real))
            amp = self.scale0_noise_amp * rmse.item()
            self.model.noise_amps.append(amp)

        else:
            # For higher scales, upsample previous facies to current resolution
            prev_rec = torch_utils.interpolate(
                facies_pyramid[scale - 1][indexes], real.shape[2:]
            ).to(self.device)

            # noise channel sizing for higher scales (empirical split)
            if len(wells_pyramid) == 0 and len(seismic_pyramid) == 0:
                if len(seismic_pyramid) == 0:
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels,
                            *real.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                else:
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels - self.num_img_channels,
                            *real.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                    z_rec = torch.cat([z_rec, seismic_pyramid[scale]], dim=1)
            else:
                if wells_pyramid == ():
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels - self.num_img_channels,
                            *real.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                    z_rec = torch.cat([z_rec, wells_pyramid[scale]], dim=1)
                else:
                    z_rec = torch_utils.generate_noise(
                        (
                            self.noise_channels - 2 * self.num_img_channels,
                            *real.shape[2:],
                        ),
                        device=self.device,
                        num_samp=self.batch_size,
                    )
                    z_rec = torch.cat(
                        [z_rec, wells_pyramid[scale], seismic_pyramid[scale]], dim=1
                    )
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            self.model.rec_noise.append(z_rec)

            # Calculate noise amplitude based on reconstruction error
            with torch.no_grad():
                fake = self.model.generator(
                    self.model.get_pyramid_noise(
                        scale,
                        indexes,
                        wells_pyramid,
                        seismic_pyramid,
                    ),
                    self.model.noise_amps + [1.0],
                    stop_scale=scale,
                )

            rmse = torch.sqrt(F.mse_loss(fake, real))
            amp = max(self.noise_amp * rmse.item(), self.min_noise_amp)

            if scale < len(self.model.noise_amps):
                self.model.noise_amps[scale] = (amp + self.model.noise_amps[scale]) / 2
            else:
                self.model.noise_amps.append(amp)

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

    def create_batch_iterator(
        self,
        loader: DataLoader[Batch[torch.Tensor]],
        scales: tuple[int, ...],
    ) -> Iterator[PyramidsBatch[torch.Tensor] | None]:
        """Create a prefetching iterator for the DataLoader.

        Overrides the base implementation to use :class:`TorchDataPrefetcher`,
        which moves tensors to the GPU asynchronously.
        """
        prefetcher = TorchDataPrefetcher(loader, scales, self.device)
        batch = prefetcher.next()
        while batch is not None:
            yield batch
            batch = prefetcher.next()

    def save_generated_facies(
        self,
        scale: int,
        epoch: int,
        results_path: str,
        real_facies: torch.Tensor,
        wells_pyramid: tuple[torch.Tensor, ...] = (),
        masks_pyramid: tuple[torch.Tensor, ...] = (),
        seismic_pyramid: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Save generated facies visualizations to disk asynchronously.

        This method samples noises, generates multiple facies images per real
        sample, clips them to [-1, 1], moves them to CPU and submits a
        background worker job to save the visualization images. Masks are
        passed through for overlay if provided.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        epoch : int
            Current epoch number (used for logging).
        results_path : str
            Base path where results are saved.
        real_facies : torch.Tensor
            Tensor of real facies samples at the current scale.
        wells_pyramid : tuple[torch.Tensor, ...]
            Tuple of well-conditioning tensors per scale.
        masks_pyramid : tuple[torch.Tensor, ...]
            Tuple of mask tensors per scale.
        seismic_pyramid : tuple[torch.Tensor, ...]
            Tuple of seismic-conditioning tensors per scale.
        """
        if self.enable_plot_facies:
            indexes = torch.randint(self.batch_size, (self.num_real_facies,))

            # Repeat each index num_generated_per_real times
            tiled_indexes: list[int] = cast(
                list[int], indexes.repeat(self.num_generated_per_real).tolist()  # type: ignore
            )
            noises = self.model.get_pyramid_noise(
                scale,
                tiled_indexes,
                wells_pyramid,
                seismic_pyramid,
            )

            with torch.no_grad():
                generated_facies = self.model.generator(
                    noises,
                    self.model.noise_amps[: scale + 1],
                    stop_scale=scale,
                ).clamp(-1, 1)

            facies_tensor = generated_facies.reshape(  # type: ignore
                self.num_real_facies,
                self.num_generated_per_real,
                *generated_facies.shape[1:],
            )

            real_facies_tensor = real_facies[indexes]

            bw.submit_plot_generated_facies(
                torch2np(facies_tensor.detach().cpu(), denormalize=True),
                torch2np(real_facies_tensor.detach().cpu(), denormalize=True),
                scale,
                epoch,
                results_path,
                (
                    torch2np(masks_pyramid[scale][indexes].detach().cpu())
                    if len(masks_pyramid) > 0
                    else None
                ),
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
