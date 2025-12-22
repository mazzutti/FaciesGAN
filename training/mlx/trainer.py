import os
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn  # type: ignore
import mlx.optimizers as optim  # type: ignore
from torch.utils.data import DataLoader

import background_workers as bw
from config import D_FILE, G_FILE, OPT_D_FILE, OPT_G_FILE, SCH_D_FILE, SCH_G_FILE
from datasets import PyramidsDataset
from datasets.mlx.data_prefetcher import MLXDataPrefetcher
from datasets.mlx.dataset import MLXPyramidsDataset
from models import FaciesGAN
from options import TrainningOptions

from models.mlx.facies_gan import MLXFaciesGAN
from training.base import Trainer
from training.mlx.collate import mlx_collate
from training.mlx.schedulers import MultiStepLR
from typedefs import Batch
from models.mlx import utils as mlx_utils
import utils


class MLXTrainer(
    Trainer[
        mx.array,
        nn.Module,
        optim.Optimizer,
        MultiStepLR,
        DataLoader[Batch[mx.array]],
    ]
):
    """Parallel trainer for multi-scale progressive FaciesGAN training in MLX."""

    def __init__(
        self,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
    ) -> None:
        super().__init__(options, fine_tuning, checkpoint_path)
        self.gpu_stream: mx.Stream = mx.default_stream(mx.gpu)  # type: ignore

    def create_dataloader(self) -> DataLoader[Batch[mx.array]]:
        """Create and return a :class:`torch.utils.data.DataLoader` for the
        trainer's dataset using configured batch size and worker settings.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.options.num_workers,
            persistent_workers=(self.options.num_workers > 0),
            collate_fn=mlx_collate,
        )

    def create_model(self) -> FaciesGAN[mx.array, nn.Module]:
        """Instantiate and return the :class:`TorchFaciesGAN` configured
        with the trainer options and device.
        """
        return MLXFaciesGAN(
            self.options,
            self.wells,
            self.seismic,
            noise_channels=self.noise_channels,
        )

    def create_optimizers_and_schedulers(self, scales: list[int]) -> tuple[
        dict[int, optim.Optimizer],
        dict[int, optim.Optimizer],
        dict[int, MultiStepLR],
        dict[int, MultiStepLR],
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
        generator_schedulers: dict[int, MultiStepLR] = {}
        discriminator_schedulers: dict[int, MultiStepLR] = {}

        for scale in scales:
            generator_optimizers[scale] = optim.Adam(
                learning_rate=self.lr_g, betas=[self.beta1, 0.999]
            )
            generator_optimizers[scale].init(  # type: ignore
                self.model.generator.gens[scale].trainable_parameters()
            )
            discriminator_optimizers[scale] = optim.Adam(
                learning_rate=self.lr_d, betas=[self.beta1, 0.999]
            )
            discriminator_optimizers[scale].init(  # type: ignore
                self.model.discriminator.discs[scale].trainable_parameters()
            )
            # Create and attach MLX MultiStepLR schedulers to optimizers
            generator_schedulers[scale] = MultiStepLR(
                init_lr=self.lr_g,
                milestones=[self.lr_decay],
                gamma=self.gamma,
                optimizer=generator_optimizers[scale],
            )
            discriminator_schedulers[scale] = MultiStepLR(
                init_lr=self.lr_d,
                milestones=[self.lr_decay],
                gamma=self.gamma,
                optimizer=discriminator_optimizers[scale],
            )
        return (
            generator_optimizers,
            discriminator_optimizers,
            generator_schedulers,
            discriminator_schedulers,
        )

    def generate_visualization_samples(
        self, scales: list[int], indexes: list[int]
    ) -> dict[int, mx.array]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : list[int]
            List of scale indices to generate samples for.
        indexes : list[int]
            List of batch sample indices.

        Returns
        -------
        dict[int, mx.array]
            A dictionary mapping scale indices to generated facies tensors
            for visualization.
        """
        generated_samples: dict[int, mx.array] = {}

        # Ensure generation happens on GPU
        with mx.stream(self.gpu_stream):
            for scale in scales:
                generated_samples[scale] = self.model.generate_fake(
                    self.model.get_noise(indexes, scale), scale
                )
        return generated_samples

    def initialize_noise(
        self,
        scale: int,
        real_facies: mx.array,
        indexes: list[int],
        wells: mx.array | None = None,
        seismic: mx.array | None = None,
    ) -> mx.array:
        """Initialize reconstruction noise for a scale.

        Parameters
        ----------
        scale : int
            The current scale index.
        real_facies : mx.array
            The real facies tensor for the current scale.
        indexes : list[int]
            List of batch sample indices.
        wells : mx.array | None, optional
            Well-conditioning tensor for the current scale, by default None.
        seismic : mx.array | None, optional
            Seismic-conditioning tensor for the current scale, by default None.

        Returns
        -------
        mx.array
            The upsampled previous reconstruction (``prev_rec``) matching
            ``real_facies`` spatial shape.

        Side effects
        ------------
        - Appends the generated reconstruction noise ``z_rec`` to
          ``self.model.rec_noise``.
        - Updates or appends the noise amplitude entry in ``self.model.noise_amp``.
        """
        # Ensure heavy ops use default stream
        with mx.stream(self.gpu_stream):
            if scale == 0:
                prev_rec = mx.zeros_like(real_facies, stream=mx.cpu)  # type: ignore
                z_rec = mlx_utils.generate_noise(
                    (*real_facies.shape[1:3], self.noise_channels),
                    num_samp=self.batch_size,
                )
                p = self.model.zero_padding
                z_rec = mx.pad(z_rec, [(0, 0), (p, p), (p, p), (0, 0)])  # type: ignore
                self.model.rec_noise.append(z_rec)

                # Calculate noise amplitude for scale 0
                fake = self.model.generator(
                    self.model.get_noise(indexes, scale),
                    [1.0] * (scale + 1),
                    stop_scale=scale,
                )

                rmse = mx.sqrt(nn.losses.mse_loss(fake, real_facies))
                amp = self.options.scale0_noise_amp * rmse.item()
                self.model.noise_amp.append(amp)
            else:
                # Interpolate previous scale facies
                prev_rec = mlx_utils.interpolate(
                    self.facies[scale - 1][indexes],
                    real_facies.shape[1:3],
                )

                # Logic for noise generation based on conditioning
                shape = (*real_facies.shape[1:3], self.noise_channels)
                if wells is not None:
                    shape = (shape[0], shape[1], shape[2] - self.num_img_channels)
                if seismic is not None:
                    shape = (shape[0], shape[1], shape[2] - self.num_img_channels)

                z_rec = mlx_utils.generate_noise(shape, num_samp=self.batch_size)

                to_concat = [z_rec]
                if wells is not None:
                    to_concat.append(wells)
                if seismic is not None:
                    to_concat.append(seismic)

                if len(to_concat) > 1:
                    z_rec = mx.concat(to_concat, axis=-1)  # type: ignore

                p = self.model.zero_padding
                z_rec = mx.pad(z_rec, [(0, 0), (p, p), (p, p), (0, 0)])  # type: ignore
                self.model.rec_noise.append(z_rec)

                # Calculate noise amplitude based on reconstruction error
                fake = self.model.generator(
                    self.model.get_noise(indexes, scale),
                    self.model.noise_amp + [1.0],
                    stop_scale=scale,
                )

                rmse = mx.sqrt(nn.losses.mse_loss(fake, real_facies))
                amp = (
                    max(self.model.noise_amp[-1] * rmse.item(), self.min_noise_amp)
                    if self.model.noise_amp
                    else self.min_noise_amp
                )
                self.model.noise_amp.append(amp)

                if scale < len(self.model.noise_amp):
                    self.model.noise_amp[scale] = (
                        amp + self.model.noise_amp[scale]
                    ) / 2
                else:
                    self.model.noise_amp.append(amp)

            # Critical: eval here forces synchronization before returning
            mx.eval(  # type: ignore
                self.model.rec_noise[-1],
                prev_rec,
                mx.array(self.model.noise_amp[-1]),
            )
        return prev_rec

    def init_dataset(
        self,
    ) -> tuple[PyramidsDataset[mx.array], tuple[tuple[int, ...], ...]]:
        """Initialize and possibly subsample the pyramids dataset.

        Applies optional selection via ``options.wells_mask_columns`` or
        subsamples the dataset randomly to ``options.num_train_pyramids``
        if that value is smaller than the dataset size.

        Returns
        -------
        tuple
            A pair ``(dataset, scales)`` where ``dataset`` is a
            :class:`datasets.mlx.dataset.MLXPyramidsDataset` instance and
            ``scales`` is the tuple of pyramid scales present in the dataset.
        """
        dataset = MLXPyramidsDataset(self.options, channels_last=True)
        if len(self.options.wells_mask_columns) > 0:
            sel = [int(i) for i in self.options.wells_mask_columns]
            dataset.batches = [dataset.batches[i] for i in sel]
        elif self.options.num_train_pyramids < len(dataset):
            idxs = mx.random.permutation(mx.arange(len(dataset)))[
                : self.options.num_train_pyramids
            ]
            dataset.batches = [dataset.batches[int(i)] for i in idxs]
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

            self.model.generator.gens[scale].load_weights(
                mlx_utils.load(generator_path)
            )
            self.model.discriminator.discs[scale].load_weights(
                mlx_utils.load(discriminator_path)
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
        generator_scheduler: MultiStepLR,
        discriminator_scheduler: MultiStepLR,
    ) -> None:
        """Load optimizer and scheduler state dictionaries from checkpoint.

        If any checkpoint files are missing or incompatible a warning is
        printed and the trainer continues without restoring those states.
        """
        try:
            generator_optimizer.state = mlx_utils.load(
                os.path.join(scale_path, OPT_G_FILE),
                as_type=dict[str, Any],
            )
            discriminator_optimizer.state = mlx_utils.load(
                os.path.join(scale_path, OPT_D_FILE),
                as_type=dict[str, Any],
            )
            generator_scheduler.load_state_dict(
                mlx_utils.load(
                    os.path.join(scale_path, SCH_G_FILE),
                    as_type=dict[str, Any],
                )
            )
            discriminator_scheduler.load_state_dict(
                mlx_utils.load(
                    os.path.join(scale_path, SCH_D_FILE),
                    as_type=dict[str, Any],
                )
            )
        except Exception as e:
            print(f"Warning: Could not load optimizers for scale {scale}: {e}")

    def create_batch_iterator(
        self, loader: DataLoader[mx.array], scales: list[int]
    ) -> Iterator[tuple[Batch[mx.array], Any]]:
        """Override to use MLXDataPrefetcher."""
        prefetcher = MLXDataPrefetcher(loader, scales)
        batch, prepared = prefetcher.next()
        while batch is not None:
            yield batch, prepared
            batch, prepared = prefetcher.next()

    def save_generated_facies(
        self,
        scale: int,
        epoch: int,
        results_path: str,
        masks: mx.array | None = None,
    ) -> None:
        """Save generated facies visualizations to disk asynchronously.

        This method samples noises, generates multiple facies images per real
        sample, clips them to [-1, 1], moves them to CPU and submits a
        background worker job to save the visualization images. Masks are
        passed through for overlay if provided.
        """
        indexes = mx.random.randint(
            0,
            self.batch_size,
            shape=(self.num_real_facies,),
        )
        real_facies = self.facies[scale][indexes]

        # Generate on GPU (lazy)
        noises = [
            self.model.get_noise(
                [int(index.item())] * self.num_generated_per_real, scale
            )
            for index in indexes
        ]
        generated_facies = [
            self.model.generator(
                noise, self.model.noise_amp[: scale + 1], stop_scale=scale
            )
            for noise in noises
        ]
        generated_facies = [utils.clamp(gen, -1, 1) for gen in generated_facies]

        if self.enable_plot_facies:
            # We want to minimize blocking here.
            # Convert to numpy (this blocks until generation is done)
            # We can't easily avoid this block without changing background_workers to accept MLX.
            # However, since this runs only once per epoch, blocking here is acceptable.
            generated_facies_cpu = [
                utils.tensor2np(g, denormalize=True) for g in generated_facies
            ]
            real_facies_cpu = utils.tensor2np(real_facies, denormalize=True)
            masks_cpu = utils.tensor2np(masks[indexes]) if masks is not None else None

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
        generator_scheduler: MultiStepLR,
        discriminator_scheduler: MultiStepLR,
    ) -> None:
        """Save optimizer and scheduler state dicts to disk using project
        filename constants from :mod:`config`.
        """
        os.makedirs(scale_path, exist_ok=True)
        # Create a dump of optimizer states for debugging (non-fatal)
        try:
            mx.savez(  # type: ignore
                os.path.join(scale_path, OPT_G_FILE),
                **generator_optimizer.state,  # type: ignore
            )
        except Exception as e:
            print(f"Failed to save generator optimizer state: {e}")

        try:
            mx.savez(  # type: ignore
                os.path.join(scale_path, OPT_D_FILE),
                **discriminator_optimizer.state,  # type: ignore
            )
        except Exception as e:
            print(f"Failed to save discriminator optimizer state: {e}")

        try:
            mx.savez(  # type: ignore
                os.path.join(scale_path, SCH_G_FILE),
                **generator_scheduler.state_dict(),
            )
        except Exception as e:
            print(f"Failed to save generator scheduler state: {e}")

        try:
            mx.savez(  # type: ignore
                os.path.join(scale_path, SCH_D_FILE),
                **discriminator_scheduler.state_dict(),
            )
        except Exception as e:
            print(f"Failed to save discriminator scheduler state: {e}")

    def schedulers_step(
        self,
        generator_schedulers: dict[int, MultiStepLR],
        discriminator_schedulers: dict[int, MultiStepLR],
        scales: list[int],
    ) -> None:
        """Step the learning-rate schedulers for the provided scales.

        Parameters
        ----------
        generator_schedulers : dict[int, MultiStepLR]
            Generator learning-rate schedulers per scale.
        discriminator_schedulers : dict[int, MultiStepLR]
            Discriminator learning-rate schedulers per scale.
        scales : list[int]
            Scale indices to step the schedulers for.
        """
        for scale in scales:
            generator_schedulers[scale].step()
            discriminator_schedulers[scale].step()
