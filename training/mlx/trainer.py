import os
from typing import Any, Iterator, cast
from typing_extensions import Callable

import mlx.core as mx
import mlx.nn as nn  # type: ignore
import mlx.optimizers as optim  # type: ignore
from torch.utils.data import DataLoader

import background_workers as bw
from config import D_FILE, G_FILE, OPT_D_FILE, OPT_G_FILE, SCH_D_FILE, SCH_G_FILE
from datasets import PyramidsDataset
from datasets.data_prefetcher import PyramidsBatch
from datasets.mlx.data_prefetcher import MLXDataPrefetcher
from datasets.mlx.dataset import MLXPyramidsDataset
from models import FaciesGAN
from options import TrainningOptions

from models.mlx.facies_gan import MLXFaciesGAN
from training.base import Trainer
from training.mlx.collate import collate
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
        compile_backend: bool = False,
    ) -> None:
        self.compile_backend = compile_backend
        super().__init__(options, fine_tuning, checkpoint_path)
        self.gpu_stream: mx.Stream = mx.default_stream(mx.gpu)  # type: ignore

        # Initialize cache for the compiled step function
        self._compiled_step: (
            Callable[
                [
                    mx.array,
                    dict[int, mx.array],
                    dict[int, mx.array],
                    dict[int, mx.array],
                ],
                tuple[
                    dict[int, list[mx.array]],
                    dict[int, list[mx.array]],
                    list[dict[str, Any]],
                ],
            ]
            | None
        ) = None
        self._compiled_step_key: tuple[int, ...] | None = None

    # def optimization_step(
    #     self,
    #     scales: tuple[int, ...],
    #     indexes: mx.array,
    #     real_facies_dict: dict[int, mx.array],
    #     masks_dict: dict[int, mx.array],
    #     rec_in_dict: dict[int, mx.array],
    # ) -> ScaleMetrics[mx.array]:
    #     """
    #     Execute a compiled optimization step using mx.compile.
    #     This overrides the default loop-based optimization in the BaseTrainer.
    #     """

    #     def step_fn(
    #         model: MLXFaciesGAN,
    #         idxs: mx.array,
    #         rf_dict: dict[int, mx.array],
    #         m_dict: dict[int, mx.array],
    #         ri_dict: dict[int, mx.array],
    #     ) -> tuple[
    #         dict[int, list[mx.array]],
    #         dict[int, list[mx.array]],
    #         dict[str, Any],
    #     ]:
    #         metrics: ScaleMetrics[mx.array] = self.model(idxs, rf_dict, m_dict, ri_dict)
    #         return (
    #             {k: v.as_list() for k, v in metrics.discriminator.items()},
    #             {k: v.as_list() for k, v in metrics.generator.items()},
    #             cast(dict[str, Any], model.trainable_parameters()),
    #         )

    #     if self.compile_backend:
    #         # Create a stable key for caching the compiled function based on active scales
    #         key = tuple(sorted(scales))

    #         # Check if we need to (re)compile. This happens when the set of active scales changes.
    #         if self._compiled_step_key != key:
    #             self._compiled_step = mx.compile(  # type: ignore
    #                 partial(
    #                     step_fn,
    #                     cast(MLXFaciesGAN, self.model),
    #                 )
    #             )  # type: ignore
    #             self._compiled_step_key = key

    #         # Execute the compiled step
    #         if self._compiled_step is not None:
    #             d_metrics, g_metrics, gradients = self._compiled_step(
    #                 indexes,
    #                 real_facies_dict,
    #                 wells_dict,
    #                 masks_dict,
    #                 rec_in_dict,
    #             )
    #             mx.eval(*d_metrics, *g_metrics, *gradients)  # type: ignore
    #             metrics = cast(
    #                 ScaleMetrics[mx.array],
    #                 ScaleMetrics.from_dicts(g_metrics, d_metrics),  # type: ignore
    #             )
    #         else:
    #             raise RuntimeError("Compiled step is None. Compilation failed.")

    #     else:
    #         metrics = self.model(indexes, real_facies_dict, masks_dict, rec_in_dict)
    #     return metrics

    @staticmethod
    def evaluate_step(gradients: dict[str, Any], metrics: list[mx.array]) -> None:
        """Evaluate and return the scalar values of the metrics.

        Parameters
        ----------
        metrics : ScaleMetrics[mx.array]
            The scale metrics containing MX arrays to evaluate.
        """
        mx.eval(*metrics, *gradients)  # type: ignore

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
            collate_fn=collate,
        )

    def create_model(
        self,
    ) -> FaciesGAN[mx.array, nn.Module, optim.Optimizer, MultiStepLR]:
        """Instantiate and return the :class:`TorchFaciesGAN` configured
        with the trainer options and device.
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
        wells_pyramid: tuple[mx.array, ...] = (),
        seismic_pyramid: tuple[mx.array, ...] = (),
    ) -> tuple[mx.array, ...]:
        """Generate fixed samples for visualization at specified scales.

        Parameters
        ----------
        scales : tuple[int, ...]
            Tuple of scale indices to generate samples for.
        indexes : list[int]
            List of batch sample indices.
        wells_pyramid : tuple[mx.array, ...]
            Tuple of well-conditioning tensors for all scales.
        seismic_pyramid : tuple[mx.array, ...]
            Tuple of seismic-conditioning tensors for all scales.

        Returns
        -------
        tuple[mx.array, ...]
            A tuple of generated facies tensors for visualization at each specified scale.
        """

        # Ensure generation happens on GPU
        with mx.stream(self.gpu_stream):
            return tuple(
                self.model.generate_fake(
                    self.model.get_pyramid_noise(
                        scale, indexes, wells_pyramid, seismic_pyramid
                    ),
                    scale,
                )
                for scale in scales
            )
        return generated_samples

    def initialize_noise(
        self,
        scale: int,
        indexes: list[int],
        facies_pyramid: tuple[mx.array, ...],
        wells_pyramid: tuple[mx.array, ...] = (),
        seismic_pyramid: tuple[mx.array, ...] = (),
    ) -> mx.array:
        """Initialize reconstruction noise for a scale.

        Parameters
        ----------
        scale : int
            The current scale index.
        indexes : list[int]
            List of batch sample indices.
        facies_pyramid : tuple[mx.array, ...]
            Tuple of real facies tensors for all scales.
        wells_pyramid : tuple[mx.array, ...], optional
            Tuple of well-conditioning tensors for all scales.
        seismic_pyramid : tuple[mx.array, ...], optional
            Tuple of seismic-conditioning tensors for all scales.

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
            real_facies = facies_pyramid[scale]
            if scale == 0:
                prev_rec = mx.zeros_like(real_facies)
                z_rec = mlx_utils.generate_noise(
                    (*real_facies.shape[1:3], self.noise_channels),
                    num_samp=self.batch_size,
                )
                p = self.model.zero_padding
                z_rec = mx.pad(z_rec, [(0, 0), (p, p), (p, p), (0, 0)])  # type: ignore
                self.model.rec_noise.append(z_rec)

                # Calculate noise amplitude for scale 0
                fake = self.model.generator(
                    self.model.get_pyramid_noise(scale, indexes),
                    [1.0] * (scale + 1),
                    stop_scale=scale,
                )

                rmse = mx.sqrt(nn.losses.mse_loss(fake, real_facies))
                amp = self.options.scale0_noise_amp * rmse.item()
                self.model.noise_amps.append(amp)
            else:
                # Interpolate previous scale facies
                tensor = facies_pyramid[scale - 1][indexes]
                prev_rec = mlx_utils.interpolate(
                    tensor,
                    real_facies.shape[1:3],
                )

                # Logic for noise generation based on conditioning
                shape = (*real_facies.shape[1:3], self.noise_channels)
                if len(wells_pyramid) > 0:
                    shape = (shape[0], shape[1], shape[2] - self.num_img_channels)
                if len(seismic_pyramid) > 0:
                    shape = (shape[0], shape[1], shape[2] - self.num_img_channels)

                z_rec = mlx_utils.generate_noise(shape, num_samp=self.batch_size)

                to_concat = [z_rec]
                if len(wells_pyramid) > 0:
                    to_concat.append(wells_pyramid[scale])
                if len(seismic_pyramid) > 0:
                    to_concat.append(seismic_pyramid[scale])

                if len(to_concat) > 1:
                    z_rec = mx.concat(to_concat, axis=-1)  # type: ignore

                p = self.model.zero_padding
                z_rec = mx.pad(z_rec, [(0, 0), (p, p), (p, p), (0, 0)])  # type: ignore
                self.model.rec_noise.append(z_rec)

                # Calculate noise amplitude based on reconstruction error
                fake = self.model.generator(
                    self.model.get_pyramid_noise(
                        scale, indexes, wells_pyramid, seismic_pyramid
                    ),
                    self.model.noise_amps + [1.0],
                    stop_scale=scale,
                )

                rmse = mx.sqrt(nn.losses.mse_loss(fake, real_facies))
                amp = max(self.noise_amp * rmse.item(), self.min_noise_amp)

                if scale < len(self.model.noise_amps):
                    self.model.noise_amps[scale] = (
                        amp + self.model.noise_amps[scale]
                    ) / 2
                else:
                    self.model.noise_amps.append(amp)

            # Critical: eval here forces synchronization before returning
            mx.eval(  # type: ignore
                self.model.rec_noise[-1],
                prev_rec,
                mx.array(self.model.noise_amps[-1]),
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
        self, loader: DataLoader[mx.array], scales: tuple[int, ...]
    ) -> Iterator[PyramidsBatch[mx.array] | None]:
        """Override to use MLXDataPrefetcher."""
        prefetcher = MLXDataPrefetcher(loader, scales)
        batch = prefetcher.next()
        while batch is not None:
            yield batch
            batch = prefetcher.next()

    def save_generated_facies(
        self,
        scale: int,
        epoch: int,
        results_path: str,
        real_facies: mx.array,
        wells_pyramid: tuple[mx.array, ...] = (),
        masks_pyramid: tuple[mx.array, ...] = (),
        seismic_pyramid: tuple[mx.array, ...] = (),
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
        facies_pyramid : tuple[mx.array, ...]
            Tuple of real facies tensors for all scales.
        wells_pyramid : tuple[mx.array, ...], optional
            Tuple of well-conditioning tensors for all scales.
        masks_pyramid : tuple[mx.array, ...], optional
            Tuple of mask tensors for all scales.
        seismic_pyramid : tuple[mx.array, ...], optional
            Tuple of seismic-conditioning tensors for all scales.
        """
        indexes = mx.random.randint(
            0,
            self.batch_size,
            shape=(self.num_real_facies,),
            dtype=mx.int32,
        )

        idx_list = cast(list[int], indexes.tolist())
        repeated_mx = mx.repeat(indexes, repeats=self.num_generated_per_real)
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
