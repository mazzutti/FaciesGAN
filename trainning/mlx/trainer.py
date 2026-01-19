from collections.abc import Callable
import os
from typing import Any, Iterator, cast

import mlx.core as mx
import mlx.nn as nn  # type: ignore
import mlx.optimizers as optim  # type: ignore
from torch.utils.data import DataLoader
import utils


import background_workers as bw
from config import D_FILE, G_FILE, OPT_D_FILE, OPT_G_FILE, SCH_D_FILE, SCH_G_FILE
from datasets import PyramidsDataset
from datasets.data_prefetcher import PyramidsBatch
from datasets.mlx.data_prefetcher import MLXDataPrefetcher
from datasets.mlx.dataset import MLXPyramidsDataset
from models import FaciesGAN
from models.mlx.facies_gan import MLXFaciesGAN
from options import TrainningOptions

from trainning.base import Trainer
from trainning.metrics import (
    DiscriminatorMetrics,
    GeneratorMetrics,
    IterableMetrics,
    ScaleMetrics,
)
from trainning.mlx.collate import collate
from trainning.mlx.schedulers import MultiStepLR
from typedefs import Batch
from models.mlx import utils as mlx_utils

OptimizationStep = Callable[
    [
        mx.array | list[int],
        dict[int, mx.array],
        dict[int, mx.array],
        dict[int, mx.array],
        dict[int, mx.array],
        dict[int, mx.array],
    ],
    tuple[IterableMetrics[mx.array], ...],
]


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

    _compiled_optimization_step: OptimizationStep | None
    _optimization_results: list[tuple[IterableMetrics[mx.array], ...]]
    _state_to_eval: list[Any]

    def __init__(
        self,
        options: TrainningOptions,
        fine_tuning: bool = False,
        checkpoint_path: str = ".checkpoints",
    ) -> None:
        self.compile_backend = options.compile_backend
        self._compiled_optimization_step = None
        super().__init__(options, fine_tuning, checkpoint_path)

        # Container to hold results - must be part of outputs for compile
        self._optimization_results: list[tuple[IterableMetrics[mx.array], ...]] = []

        # State list to eval after compiled step
        self._state_to_eval: list[Any] = [
            cast(MLXFaciesGAN, self.model).state,
            mx.random.state,  # type: ignore
            self._optimization_results,
        ]

        try:
            # Set memory limit for better memory management
            mx.set_memory_limit(48 * 1024**3)  # 48GB limit
            mx.set_default_device(mx.gpu)  # type: ignore

            print(f"MLX Metal Configuration:")
            print(f"  Device: {mx.default_device()}")
            print(f"  Active memory: {mx.get_active_memory() / 1024**3:.3f} GB")
            print(f"  Peak memory: {mx.get_peak_memory() / 1024**3:.3f} GB")
            print(f"  Compilation: {'Enabled' if self.compile_backend else 'Disabled'}")
        except Exception as e:
            print(f"Warning: Could not configure Metal backend: {e}")

    def create_dataloader(self) -> DataLoader[Batch[mx.array]]:
        """Create and return a :class:`torch.utils.data.DataLoader` for the
        trainer's dataset using configured batch size and worker settings.

        Returns
        -------
        DataLoader[Batch[mx.array]]
            The data loader for the trainer's dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.options.num_workers,
            persistent_workers=(self.options.num_workers > 0),
            collate_fn=collate,
        )

    def create_model(
        self,
    ) -> FaciesGAN[mx.array, nn.Module, optim.Optimizer, MultiStepLR]:
        """Instantiate and return the :class:`TorchFaciesGAN` configured
        with the trainer options and device.

        Returns
        -------
        TorchFaciesGAN
            The instantiated FaciesGAN model for MLX.
        """
        return MLXFaciesGAN(
            self.options,
            noise_channels=self.noise_channels,
            compile_backend=self.compile_backend,
        )

    def generate_visualization_samples(
        self,
        scales: tuple[int, ...],
        indexes: list[int],
        wells_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> tuple[mx.array, ...]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to generate samples for.
        indexes : list[int]
            List of batch sample indices.
        wells_pyramid : dict[int, mx.array]
            Dictionary of well-conditioning tensors for all scales.
        seismic_pyramid : dict[int, mx.array]
            Dictionary of seismic-conditioning tensors for all scales.

        Returns
        -------
        tuple[mx.array, ...]
            A tuple of generated facies tensors for visualization at each specified scale.
        """

        # Ensure generation happens on GPU
        # with mx.stream(self.gpu_stream):
        return tuple(
            self.model.generate_fake(
                self.model.get_pyramid_noise(
                    scale, indexes, wells_pyramid, seismic_pyramid
                ),
                scale,
            )
            for scale in scales
        )

    def compute_rec_input(
        self,
        scale: int,
        indexes: list[int],
        facies_pyramid: dict[int, mx.array],
    ) -> mx.array:
        real_facies = facies_pyramid[scale]
        if scale == 0:
            return mx.zeros_like(real_facies)

        tensor = facies_pyramid[scale - 1][indexes]
        return mlx_utils.interpolate(
            tensor,
            real_facies.shape[1:3],
        )

    def init_rec_noise_and_amp(
        self,
        scale: int,
        indexes: list[int],
        real: mx.array,
        wells_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> None:
        """Initialize reconstruction noise and noise amplitude for a specific scale.

        Parameters
        ----------
        scale : int
            Current pyramid scale index.
        indexes : list[int]
            Batch sample indices.
        real : mx.array
            Real facies tensor for the current scale.
        wells_pyramid : dict[int, mx.array], optional
            Dictionary of well-conditioning tensors for all scales.
        seismic_pyramid : dict[int, mx.array], optional
            Dictionary of seismic-conditioning tensors for all scales.
        """
        if len(self.model.rec_noise) > scale:
            return

        if scale == 0:
            z_rec = mlx_utils.generate_noise(
                (*real.shape[1:3], self.noise_channels),
                num_samp=self.batch_size,
            )

            p = self.model.zero_padding
            z_rec = mx.pad(z_rec, [(0, 0), (p, p), (p, p), (0, 0)])  # type: ignore
            self.model.rec_noise.append(z_rec)

            fake = self.model.generator(
                self.model.get_pyramid_noise(scale, indexes),
                [1.0] * (scale + 1),
                stop_scale=scale,
            )
            rmse = mx.sqrt(nn.losses.mse_loss(fake, real))
            amp = self.options.scale0_noise_amp * rmse.item()
            if len(self.model.noise_amps) <= scale:
                self.model.noise_amps.append(amp)
            else:
                self.model.noise_amps[scale] = amp
            return

        # Logic for noise generation based on conditioning
        shape = (*real.shape[1:3], self.noise_channels)
        # Adjust shape only if conditioning is present for this scale
        if wells_pyramid.get(scale, None) is not None:
            shape = (shape[0], shape[1], shape[2] - self.num_img_channels)
        if seismic_pyramid.get(scale, None) is not None:
            shape = (shape[0], shape[1], shape[2] - self.num_img_channels)

        z_rec = mlx_utils.generate_noise(shape, num_samp=self.batch_size)

        to_concat = [z_rec]
        if wells_pyramid.get(scale, None) is not None:
            to_concat.append(wells_pyramid[scale])
        if seismic_pyramid.get(scale, None) is not None:
            to_concat.append(seismic_pyramid[scale])
        if len(to_concat) > 1:
            z_rec = mx.concat(to_concat, axis=-1)  # type: ignore

        p = self.model.zero_padding
        z_rec = mx.pad(z_rec, [(0, 0), (p, p), (p, p), (0, 0)])  # type: ignore
        self.model.rec_noise.append(z_rec)

        fake = self.model.generator(
            self.model.get_pyramid_noise(
                scale, indexes, wells_pyramid, seismic_pyramid
            ),
            self.model.noise_amps + [1.0],
            stop_scale=scale,
        )

        rmse = mx.sqrt(nn.losses.mse_loss(fake, real))
        amp = max(self.noise_amp * rmse.item(), self.min_noise_amp)
        if scale < len(self.model.noise_amps):
            self.model.noise_amps[scale] = (amp + self.model.noise_amps[scale]) / 2
        else:
            self.model.noise_amps.append(amp)

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
            idxs = mx.random.permutation(mx.arange(len(dataset)))[  # type: ignore
                : self.options.num_train_pyramids
            ]
            dataset.batches = [dataset.batches[int(i)] for i in idxs]  # type: ignore
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

        Parameters
        ----------
        scale : int
            Scale index to load optimizers for.
        scale_path : str
            Filesystem path to the scale checkpoint directory.
        generator_optimizer : optim.Optimizer
            The generator optimizer instance to load state into.
        discriminator_optimizer : optim.Optimizer
            The discriminator optimizer instance to load state into.
        generator_scheduler : MultiStepLR
            The generator learning rate scheduler to load state into.
        discriminator_scheduler : MultiStepLR
            The discriminator learning rate scheduler to load state into.
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
        self, loader: DataLoader[Batch[mx.array]], scales: tuple[int, ...]
    ) -> Iterator[PyramidsBatch[mx.array] | None]:
        """Override to use MLXDataPrefetcher."""
        # Need ALL scales from 0 to max(scales) for proper pyramid structure
        # (not just the scales being trained, since we need previous scales for interpolation)
        all_scales = tuple(range(0, max(scales) + 1))
        prefetcher = MLXDataPrefetcher(loader, all_scales)
        batch = prefetcher.next()
        while batch is not None:
            yield batch
            batch = prefetcher.next()

    def optimization_step(
        self,
        indexes: list[int],
        facies_pyramid: dict[int, mx.array],
        rec_in_pyramid: dict[int, mx.array],
        wells_pyramid: dict[int, mx.array] = {},
        masks_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> ScaleMetrics[mx.array] | tuple[IterableMetrics[mx.array], ...]:
        """Run a single optimization step using a compiled forward if available.

        This method prefers a compiled forward cached on the model (via
        `MLXFaciesGAN.compile_forward`). If compilation is disabled or fails
        it falls back to the Python-callable `self.model(...)`.

        Parameters
        ----------
        indexes : list[int]
            List of batch sample indices.
        facies_pyramid : dict[int, mx.array]
            Dictionary of real facies tensors for all scales.
        rec_in_pyramid : dict[int, mx.array]
            Dictionary mapping scale -> reconstruction input from previous scale.
        wells_pyramid : dict[int, mx.array], optional
            Dictionary of well-conditioning tensors for all scales.
        masks_pyramid : dict[int, mx.array], optional
            Dictionary of mask tensors for all scales.
        seismic_pyramid : dict[int, mx.array], optional
            Dictionary of seismic-conditioning tensors for all scales.
        Returns
        -------
        ScaleMetrics[mx.array]
            The scale metrics resulting from the optimization step.
        """
        if self.compile_backend:
            if getattr(self, "_compiled_optimization_step", None) is None:
                self._compile_optimization_step()

            # Call compiled step (stores results in self._compiled_results)
            result = cast(OptimizationStep, self._compiled_optimization_step)(
                mx.array(indexes),
                facies_pyramid,
                rec_in_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            )
        else:
            result = cast(
                tuple[IterableMetrics[mx.array], ...],
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

        self._optimization_results.clear()
        self._optimization_results.append(result)

        # Evaluate all lazy computations
        mx.eval(self._state_to_eval)  # type: ignore

        disc_results, gen_results = self._optimization_results[0]

        # Vectorized metric aggregation using vmap-like stacking
        def aggregate_metrics_vectorized(
            results_dict: dict[int, list[tuple[mx.array, ...]]],
        ) -> dict[int, tuple[mx.array, ...]]:
            aggregated: dict[int, tuple[mx.array, ...]] = {}
            for scale in sorted(results_dict.keys()):
                # Stack all metric tuples for this scale
                stacked = mx.stack([mx.stack(list(m)) for m in results_dict[scale]])
                # Compute mean across batch dimension
                means = mx.mean(stacked, axis=0)
                aggregated[scale] = tuple(means)
            return aggregated

        discriminator_metrics = {
            scale: DiscriminatorMetrics(*metrics)
            for scale, metrics in aggregate_metrics_vectorized(disc_results[0]).items()
        }
        generator_metrics = {
            scale: GeneratorMetrics(*metrics)
            for scale, metrics in aggregate_metrics_vectorized(gen_results[0]).items()
        }

        return ScaleMetrics[mx.array](
            generator=generator_metrics,
            discriminator=discriminator_metrics,
        )

    def _compile_optimization_step(self) -> None:
        """Compile the forward call with MLX `mx.compile` for faster execution.

        Uses partial to specify `inputs` and `outputs` with model.state,
        optimizer states, and random state to ensure all updates are tracked.
        """

        from functools import partial

        @partial(mx.compile, inputs=self._state_to_eval, outputs=self._state_to_eval)  # type: ignore
        def _optimization_step(
            indexes: list[int],
            facies_pyramid: dict[int, mx.array],
            rec_in_pyramid: dict[int, mx.array],
            wells_pyramid: dict[int, mx.array] = {},
            masks_pyramid: dict[int, mx.array] = {},
            seismic_pyramid: dict[int, mx.array] = {},
        ) -> tuple[IterableMetrics[mx.array], ...]:
            return cast(
                tuple[IterableMetrics[mx.array], ...],
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

        self._compiled_optimization_step = _optimization_step  # type: ignore

    def setup_optimizers(self, scales: tuple[int, ...]) -> None:
        """Setup optimizers and schedulers on the model.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to setup optimizers for.
        """

        generators = self.model.generator.gens
        discriminators = self.model.discriminator.discs

        for scale in scales:

            self.generator_optimizers[scale] = optim.Adam(
                learning_rate=self.lr_g,
                betas=[self.beta1, 0.999],
            )
            self.generator_optimizers[scale].init(  # type: ignore
                generators[scale].parameters()
            )

            self.generator_schedulers[scale] = MultiStepLR(
                init_lr=self.lr_g,
                milestones=[self.lr_decay],
                gamma=self.gamma,
                optimizer=self.generator_optimizers[scale],
            )

            self.discriminator_optimizers[scale] = optim.Adam(
                learning_rate=self.lr_d,
                betas=[self.beta1, 0.999],
            )
            self.discriminator_optimizers[scale].init(  # type: ignore
                discriminators[scale].parameters()
            )
            self.discriminator_schedulers[scale] = MultiStepLR(
                init_lr=self.lr_d,
                milestones=[self.lr_decay],
                gamma=self.gamma,
                optimizer=self.discriminator_optimizers[scale],
            )

            # Build optimizer state lists
            gen_opt_states = [
                cast(dict[str, Any], opt.state)  # type: ignore
                for opt in self.generator_optimizers.values()
            ]
            disc_opt_states = [
                cast(dict[str, Any], opt.state)  # type: ignore
                for opt in self.discriminator_optimizers.values()
            ]

            self._state_to_eval.extend([gen_opt_states, disc_opt_states])

            # Invalidate compiled step on optimizer change
            self._compiled_optimization_step = None

    def save_generated_facies(
        self,
        scale: int,
        epoch: int,
        results_path: str,
        real_facies: mx.array,
        wells_pyramid: dict[int, mx.array] = {},
        masks_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> None:
        """Save generated facies visualizations to disk asynchronously.

        This method samples noises, generates multiple facies images per real
        sample, clips them to [-1, 1], and submits a background worker job to save
        the visualization images. Masks are passed through for overlay if provided.
        All data is kept as tensors/arrays until the worker process.

        Parameters
        ----------
        scale : int
            The current scale index.
        epoch : int
            The current epoch number.
        results_path : str
            Path to save results to.
        facies_pyramid : dict[int, mx.array]
            Dictionary of real facies tensors for all scales.
        wells_pyramid : dict[int, mx.array], optional
            Dictionary of well-conditioning tensors for all scales.
        masks_pyramid : dict[int, mx.array], optional
            Dictionary of mask tensors for all scales.
        seismic_pyramid : dict[int, mx.array], optional
            Dictionary of seismic-conditioning tensors for all scales.
        """
        indexes = mx.random.randint(  # type: ignore
            0,
            self.batch_size,
            shape=(self.num_real_facies,),
            dtype=mx.int32,
        )

        idx_list = cast(list[int], indexes.tolist())  # type: ignore
        repeated_mx = mx.repeat(indexes, repeats=self.num_generated_per_real)  # type: ignore
        repeated_indexes = cast(list[int], repeated_mx.tolist())
        noises = self.model.get_pyramid_noise(
            scale,
            repeated_indexes,
            wells_pyramid,
            seismic_pyramid,
        )

        generated_facies = utils.clamp(
            self.model.generator(
                noises,
                self.model.noise_amps[: scale + 1],
                stop_scale=scale,
            ),
            min_val=-1,
            max_val=1,
        )
        facies_tensor = generated_facies.reshape(  # type: ignore
            self.num_real_facies,
            self.num_generated_per_real,
            *generated_facies.shape[1:],
        )

        masks_tensor = (
            masks_pyramid[scale][idx_list] if len(masks_pyramid) > 0 else None
        )
        real_facies_tensor = real_facies[idx_list]

        if self.enable_plot_facies:
            bw.submit_plot_generated_facies(
                utils.mlx2np(facies_tensor, denormalize=True),
                utils.mlx2np(real_facies_tensor, denormalize=True),
                scale,
                epoch,
                results_path,
                utils.mlx2np(masks_tensor) if masks_tensor is not None else None,
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
