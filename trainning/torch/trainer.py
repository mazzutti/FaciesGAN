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
import threading
from collections.abc import Iterator, Mapping
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import background_workers as bw
from apex_utils import FusedAdam
from datasets.data_prefetcher import PyramidsBatch
from datasets.torch.data_prefetcher import TorchDataPrefetcher
from config import (
    D_FILE,
    EPOCH_CKPT_FILE,
    G_FILE,
    OPT_D_FILE,
    OPT_G_FILE,
    RESULT_FACIES_PATH,
    RESULT_IMPEDANCE_PATH,
    SCH_D_FILE,
    SCH_G_FILE,
)
from datasets import PyramidsDataset
from datasets.torch.dataset import TorchPyramidsDataset
from models import FaciesGAN, TorchFaciesGAN
from models.torch import utils as torch_utils
from models.torch.facies_gan import unwrap_ddp
from options import TrainningOptions
from trainning.base import Trainer
from typedefs import Batch
from utils import denorm, torch2np


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
    model : TorchFaciesGAN
        The multi-scale model instance managed by the trainer.

    Notes
    -----
    - The trainer updates ``self.model.rec_noise`` and ``self.model.noise_amp``
      as part of noise initialization (see :meth:`initialize_noise`).
    - Conditioning tensors (wells/seismic) are expected channels-last when
      prepared and returned by :meth:`TorchDataPrefetcher`.
    """

    model: TorchFaciesGAN  # type: ignore[assignment]

    def __init__(
        self,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
        device: torch.device = torch.device("cpu"),
        distributed: bool = False,
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
        distributed : bool, optional
            Whether running under ``DistributedDataParallel`` with
            ``torchrun``.  When ``True`` the batch size is divided by
            ``world_size`` and a ``DistributedSampler`` is used.
        """
        self.device: torch.device = device
        self.distributed: bool = distributed
        # Must be set *before* super().__init__() so that prints inside
        # the base __init__ are correctly guarded on non-main ranks.
        if distributed:
            self._is_main_process = dist.get_rank() == 0
        super().__init__(options, fine_tuning, checkpoint_path)
        self._ckpt_thread: threading.Thread | None = None

    def _ddp_barrier(self) -> None:
        """Synchronize DDP ranks via NCCL barrier."""
        if self.distributed and dist.is_initialized():
            dist.barrier()  # type: ignore

    def create_dataloader(self) -> DataLoader[Batch[torch.Tensor]]:
        """Create and return a :class:`torch.utils.data.DataLoader` for the
        trainer's dataset using configured batch size and worker settings.

        When running under DDP a :class:`DistributedSampler` is used so
        each rank gets a non-overlapping subset of the data.  The per-rank
        batch size is also halved here (rather than later) so the DataLoader
        is constructed with the correct value.
        """
        sampler: DistributedSampler[Batch[torch.Tensor]] | None = None
        shuffle = False
        if self.distributed:
            # Halve per-GPU batch size so the effective batch is unchanged.
            world_size = dist.get_world_size()
            self.batch_size = max(1, self.batch_size // world_size)
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
            )
        has_workers = self.options.num_workers > 0
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.options.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=has_workers,
            prefetch_factor=8 if has_workers else None,
            # Drop the last incomplete batch to avoid shape mismatches
            # when the per-rank sample count is not divisible by
            # batch_size (the training loop assumes full batches).
            drop_last=True,
            # Prevent a hung DataLoader worker from silently blocking the
            # training loop.  Under DDP a stalled rank causes every other
            # rank to block in the next NCCL collective.
            timeout=120 if has_workers else 0,
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
            use_ddp=self.distributed,
        )

    def generate_visualization_samples(
        self,
        scales: tuple[int, ...],
        indexes: list[int],
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[torch.Tensor, ...]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to generate samples for.
        indexes : tuple[int, ...]
            Tuple of batch sample indices.
        wells_pyramid : dict[int, torch.Tensor], optional
            Dictionary of well-conditioning tensors per scale.
        seismic_pyramid : dict[int, torch.Tensor], optional
            Dictionary of seismic-conditioning tensors per scale.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of generated facies tensors for visualization, one per scale.
        """
        with torch.inference_mode():
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

    def _warmup_compile_traces(
        self,
        scales: tuple[int, ...],
        indexes: list[int],
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor],
        seismic_pyramid: dict[int, torch.Tensor],
    ) -> None:
        """Run dummy forwards to pre-populate torch.compile trace caches.

        Recovery loss and N>1 diversity activate after a warmup window
        (``_rec_skip_epochs``).  Their generator calls use different
        arguments (``in_noise``, ``start_scale``, larger batch) that
        Dynamo has never traced.  If the first real recompilation
        happens mid-training it can crash under debugpy or cause a long
        stall that triggers a DDP timeout.

        This method forces those traces at startup (within
        ``inference_mode`` so no gradients are computed) so every
        compiled specialization is cached before the training loop.
        """
        if not getattr(self.model, "_use_compile", False):
            return

        N = self.model.num_diversity_samples
        if N <= 1 and self.model.alpha == 0:
            return  # nothing extra to warm up

        if self._is_main_process:
            print("  Warming up torch.compile traces …")

        with torch.inference_mode():
            for scale in scales:
                amps = self.model.get_noise_aplitude(scale)

                # 1) Recovery-loss path: generator(rec_noise, amps,
                #    in_noise=rec_in, start_scale=s, stop_scale=s)
                if self.model.alpha > 0:
                    rec_noise = self.model.get_pyramid_noise(
                        scale, indexes, wells_pyramid, seismic_pyramid,
                        rec=True,
                    )
                    rec_in = rec_in_pyramid[scale]
                    self.model.generator(
                        rec_noise, amps,
                        in_noise=rec_in,
                        start_scale=scale,
                        stop_scale=scale,
                    )

                # 2) Diversity N>1 path: generator(batched_N*B, amps,
                #    stop_scale=s)
                if N > 1:
                    noise_sets = [
                        self.model.get_pyramid_noise(
                            scale, indexes, wells_pyramid, seismic_pyramid,
                        )
                        for _ in range(N)
                    ]
                    batched_noises = [
                        torch.cat(
                            [noise_sets[k][lvl] for k in range(N)], dim=0,
                        )
                        for lvl in range(scale + 1)
                    ]
                    self.model.generator(
                        batched_noises, amps, stop_scale=scale,
                    )

        if self._is_main_process:
            print("  Compile warmup done.")

    def compute_rec_input(
        self,
        scale: int,
        indexes: list[int],
        facies_pyramid: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        real = facies_pyramid[scale]
        if scale == 0:
            return torch.zeros_like(real).to(self.device)

        return torch_utils.interpolate(
            facies_pyramid[scale - 1][indexes],
            real.shape[2:],
        ).to(self.device)

    def init_rec_noise_and_amp(
        self,
        scale: int,
        indexes: list[int],
        real: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> None:
        if len(self.model.rec_noise) >= scale + 1:
            return

        actual_batch = real.shape[0]
        if scale == 0:
            z_rec = torch_utils.generate_noise(
                (self.noise_channels, *real.shape[2:]),
                device=self.device,
                num_samp=actual_batch,
            )
            z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
            self.model.rec_noise.append(z_rec)

            with torch.no_grad():
                fake = self.model.generator(
                    self.model.get_pyramid_noise(scale, indexes),
                    [1.0] * (scale + 1),
                    stop_scale=scale,
                )

            rmse = torch.sqrt(F.mse_loss(fake, real.to(fake.device)))
            amp = self.scale0_noise_amp * rmse.item()
            if len(self.model.noise_amps) <= scale:
                self.model.noise_amps.append(amp)
            else:
                self.model.noise_amps[scale] = amp
            return

        # Determine how many noise channels we need after conditioning
        num_cond_channels = 0
        if len(wells_pyramid) > 0:
            num_cond_channels += self.num_img_channels
        if len(seismic_pyramid) > 0:
            num_cond_channels += self.num_img_channels

        noise_ch = self.noise_channels - num_cond_channels

        z_rec = torch_utils.generate_noise(
            (
                noise_ch,
                *real.shape[2:],
            ),
            device=self.device,
            num_samp=actual_batch,
        )

        to_concat = [z_rec]
        if len(wells_pyramid) > 0:
            to_concat.append(wells_pyramid[scale].to(self.device))
        if len(seismic_pyramid) > 0:
            to_concat.append(seismic_pyramid[scale].to(self.device))

        if len(to_concat) > 1:
            z_rec = torch.cat(to_concat, dim=1)

        z_rec = F.pad(z_rec, [self.zero_padding] * 4, value=0)
        self.model.rec_noise.append(z_rec)

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

        rmse = torch.sqrt(F.mse_loss(fake, real.to(fake.device)))
        amp = max(self.noise_amp * rmse.item(), self.min_noise_amp)

        if scale < len(self.model.noise_amps):
            self.model.noise_amps[scale] = (amp + self.model.noise_amps[scale]) / 2
        else:
            self.model.noise_amps.append(amp)

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

            gen = unwrap_ddp(self.model.generator.gens[scale])
            gen.load_state_dict(
                torch_utils.load(generator_path, self.device, as_type=Mapping[str, Any])
            )
            disc = unwrap_ddp(self.model.discriminator.discs[scale])
            disc.load_state_dict(
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

        NVLink optimization: barriers are now sparse to maximize compute/comm
        overlap. Critical barriers (epoch/checkpoint) remain; periodic sync
        barriers have been removed to allow async all-reduce to proceed.
        """
        prefetcher = TorchDataPrefetcher(loader, scales, self.device)
        batch = prefetcher.next()
        while batch is not None:
            yield batch
            batch = prefetcher.next()

    def save_generated_outputs(
        self,
        scale: int,
        epoch: int,
        batch_id: int,
        results_path: str,
        real_facies: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
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
        wells_pyramid : dict[int, torch.Tensor]
            Dictionary of well-conditioning tensors per scale.
        masks_pyramid : tuple[torch.Tensor, ...]
            Tuple of mask tensors per scale.
        seismic_pyramid : dict[int, torch.Tensor]
            Dictionary of seismic-conditioning tensors per scale.
        """
        if self.enable_plot_outputs:
            actual_batch = real_facies.shape[0]
            # Generate on CPU so .tolist() avoids a GPU→CPU sync.
            indexes = torch.randint(actual_batch, (self.num_real_facies,), device="cpu")

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

            with torch.inference_mode():
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

            # Batch GPU→CPU transfers with non_blocking=True so the
            # three DMA copies overlap, then sync once before numpy
            # conversion.  Denormalization runs on GPU to avoid a
            # redundant per-element copy on CPU.
            facies_cpu: torch.Tensor = cast(torch.Tensor, denorm(facies_tensor.detach()).to("cpu", non_blocking=True))  # type: ignore[arg-type]
            real_cpu: torch.Tensor = cast(torch.Tensor, denorm(real_facies_tensor.detach()).to("cpu", non_blocking=True))  # type: ignore[arg-type]
            masks_cpu: torch.Tensor | None = None
            if len(masks_pyramid) > 0:
                masks_cpu = masks_pyramid[scale][indexes].detach().to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()

            use_impedance: bool = getattr(self.model.options, "use_impedance", False)
            # When impedance is active the tensor has 6 channels: first 3 are
            # facies, last 3 are impedance.  Split before plotting.
            num_facies_ch: int = self.model.options.noise_channels
            if use_impedance:
                facies_only_cpu = facies_cpu[:, :, :num_facies_ch]
                real_facies_only_cpu = real_cpu[:, :num_facies_ch]
                imp_cpu = facies_cpu[:, :, num_facies_ch:]
                real_imp_cpu = real_cpu[:, num_facies_ch:]
            else:
                facies_only_cpu = facies_cpu
                real_facies_only_cpu = real_cpu
                imp_cpu = None
                real_imp_cpu = None

            masks_np = torch2np(masks_cpu) if masks_cpu is not None else None
            bw.submit_plot_generated_outputs(
                torch2np(facies_only_cpu),
                torch2np(real_facies_only_cpu),
                scale,
                epoch,
                results_path,
                masks_np,
                batch_id=batch_id,
            )

            if use_impedance and imp_cpu is not None and real_imp_cpu is not None:
                impedance_results_path = results_path.replace(
                    RESULT_FACIES_PATH, RESULT_IMPEDANCE_PATH
                )
                os.makedirs(impedance_results_path, exist_ok=True)
                bw.submit_plot_generated_outputs(
                    torch2np(imp_cpu),
                    torch2np(real_imp_cpu),
                    scale,
                    epoch,
                    impedance_results_path,
                    None,  # impedance is not well-conditioned
                    batch_id=batch_id,
                    plot_title="Impedance",
                    quantize=False,
                )

    def setup_optimizers(self, scales: tuple[int, ...]) -> None:
        for scale in scales:
            self.discriminator_optimizers[scale] = FusedAdam(
                self.model.discriminator.discs[scale].parameters(),
                lr=self.lr_d,
                betas=(self.beta1, 0.999),
                set_grad_none=True,
            )
            self.discriminator_schedulers[scale] = torch.optim.lr_scheduler.StepLR(
                self.discriminator_optimizers[scale],
                step_size=self.lr_decay,
                gamma=self.gamma,
            )

            self.generator_optimizers[scale] = FusedAdam(
                self.model.generator.gens[scale].parameters(),
                lr=self.lr_g,
                betas=(self.beta1, 0.999),
                set_grad_none=True,
            )

            self.generator_schedulers[scale] = torch.optim.lr_scheduler.StepLR(
                self.generator_optimizers[scale],
                step_size=self.lr_decay,
                gamma=self.gamma,
            )

    def reset_schedulers(self, scales: tuple[int, ...]) -> None:
        for scale in scales:
            # Reset optimizer param group LRs to initial values before
            # creating new schedulers, otherwise StepLR picks up the
            # already-decayed LR as its base.
            for pg in self.generator_optimizers[scale].param_groups:
                pg["lr"] = self.lr_g
            for pg in self.discriminator_optimizers[scale].param_groups:
                pg["lr"] = self.lr_d

            self.generator_schedulers[scale] = torch.optim.lr_scheduler.StepLR(
                self.generator_optimizers[scale],
                step_size=self.lr_decay,
                gamma=self.gamma,
            )
            self.discriminator_schedulers[scale] = torch.optim.lr_scheduler.StepLR(
                self.discriminator_optimizers[scale],
                step_size=self.lr_decay,
                gamma=self.gamma,
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

    def save_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
        epoch: int,
        batch_id: int,
    ) -> None:
        checkpoint: dict[str, object] = {
            "epoch": epoch,
            "batch_id": batch_id,
            "noise_amps": list(self.model.noise_amps),
        }
        per_scale: dict[int, dict[str, object]] = {}
        for s in scales:
            gen = unwrap_ddp(self.model.generator.gens[s])
            disc = unwrap_ddp(self.model.discriminator.discs[s])
            per_scale[s] = {
                "generator": gen.state_dict(),
                "discriminator": disc.state_dict(),
                "opt_g": self.generator_optimizers[s].state_dict(),
                "opt_d": self.discriminator_optimizers[s].state_dict(),
                "sch_g": self.generator_schedulers[s].state_dict(),
                "sch_d": self.discriminator_schedulers[s].state_dict(),
            }
        checkpoint["scales"] = per_scale

        # Save into the first scale's directory (arbitrary but deterministic)
        ckpt_path = os.path.join(scale_paths[min(scales)], EPOCH_CKPT_FILE)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        # Join any in-flight save before spawning a new one so we never
        # have two concurrent writes to the same path.
        if self._ckpt_thread is not None and self._ckpt_thread.is_alive():
            self._ckpt_thread.join()
        self._ckpt_thread = threading.Thread(
            target=torch.save,
            args=(checkpoint, ckpt_path),
            daemon=True,
        )
        self._ckpt_thread.start()
        print(f"  Epoch checkpoint save started at epoch {epoch} (batch {batch_id})")

    def load_epoch_checkpoint(
        self,
        scales: tuple[int, ...],
        scale_paths: dict[int, str],
    ) -> tuple[int, int]:
        ckpt_path = os.path.join(scale_paths[min(scales)], EPOCH_CKPT_FILE)

        # ── Fast path: a full epoch checkpoint exists ──────────────
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
            epoch: int = checkpoint["epoch"]
            batch_id: int = checkpoint["batch_id"]

            self.model.noise_amps = checkpoint["noise_amps"]

            per_scale: dict[int, dict[str, object]] = checkpoint["scales"]
            for s, state in per_scale.items():
                gen = unwrap_ddp(self.model.generator.gens[s])
                gen.load_state_dict(state["generator"])  # type: ignore[arg-type]
                disc = unwrap_ddp(self.model.discriminator.discs[s])
                disc.load_state_dict(state["discriminator"])  # type: ignore[arg-type]
                self.generator_optimizers[s].load_state_dict(state["opt_g"])  # type: ignore[arg-type]
                self.discriminator_optimizers[s].load_state_dict(state["opt_d"])  # type: ignore[arg-type]
                self.generator_schedulers[s].load_state_dict(state["sch_g"])  # type: ignore[arg-type]
                self.discriminator_schedulers[s].load_state_dict(state["sch_d"])  # type: ignore[arg-type]

            return epoch, batch_id

        # ── Fallback: load from scale-level checkpoints ────────────
        # When no epoch checkpoint exists but the user passed
        # --start-epoch, reconstruct the training state from the
        # per-scale generator/discriminator files that the normal
        # training loop writes at the end of each scale group.
        #
        # If a completed_epoch.txt metadata file exists, use it to
        # determine the actual last-completed epoch instead of relying
        # on the CLI --start-epoch value (which may be stale).
        from config import COMPLETED_EPOCH_FILE

        effective_start = self.start_epoch
        meta_path = os.path.join(scale_paths[min(scales)], COMPLETED_EPOCH_FILE)
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                completed = int(f.read().strip())
            if completed >= self.num_iter:
                # Training already finished for this scale group;
                # nothing to resume.
                print(
                    f"  Scale group already completed {completed} epochs "
                    f"(num_iter={self.num_iter}); nothing to resume."
                )
                return self.num_iter, 0
            effective_start = max(effective_start, completed)

        if effective_start <= 0:
            return 0, 0

        loaded_any = False
        for s in scales:
            gen_path = os.path.join(scale_paths[s], G_FILE)
            disc_path = os.path.join(scale_paths[s], D_FILE)
            if os.path.isfile(gen_path):
                gen = unwrap_ddp(self.model.generator.gens[s])
                gen.load_state_dict(
                    torch_utils.load(gen_path, self.device, as_type=Mapping[str, Any])
                )
                loaded_any = True
            if os.path.isfile(disc_path):
                disc = unwrap_ddp(self.model.discriminator.discs[s])
                disc.load_state_dict(
                    torch_utils.load(disc_path, self.device, as_type=Mapping[str, Any])
                )

        if not loaded_any:
            return 0, 0

        # Load noise amplitudes from scale-level files
        from config import AMP_FILE

        for s in sorted(scales):
            amp_path = os.path.join(scale_paths[s], AMP_FILE)
            if os.path.isfile(amp_path):
                with open(amp_path) as f:
                    amp_val = float(f.read().strip())
                while len(self.model.noise_amps) <= s:
                    self.model.noise_amps.append(0.0)
                self.model.noise_amps[s] = amp_val

        # Try to restore optimizer/scheduler states from scale-level files.
        # If they exist the LR schedulers are fully restored; otherwise
        # we fast-forward them to approximate the right learning rate.
        opts_loaded = False
        for s in scales:
            try:
                self.load_optimizers(
                    s,
                    scale_paths[s],
                    self.generator_optimizers[s],
                    self.discriminator_optimizers[s],
                    self.generator_schedulers[s],
                    self.discriminator_schedulers[s],
                )
                opts_loaded = True
            except Exception:
                pass

        if not opts_loaded:
            # Fast-forward the LR schedulers to match the requested epoch
            for _ in range(effective_start):
                for s in scales:
                    self.generator_schedulers[s].step()
                    self.discriminator_schedulers[s].step()

        if self._is_main_process:
            print(
                f"  Loaded scale-level checkpoints"
                f" (optimizers {'restored' if opts_loaded else 'reset, schedulers fast-forwarded'})"
                f" at epoch {effective_start}"
            )
        return effective_start, 0
