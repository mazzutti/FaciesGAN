import os
import numpy as np
from typing import Any, cast

import math

import mlx.core as mx
import mlx.nn as nn  # type: ignore
import mlx.utils as mlx_utils
from mlx.optimizers import Optimizer  # type: ignore


from config import D_FILE, G_FILE, M_FILE, SHAPE_FILE
from models.base import FaciesGAN
from models.mlx.discriminator import MLXDiscriminator
from models.mlx.generator import MLXGenerator
from options import TrainningOptions
import models.mlx.utils as utils
from trainning.metrics import (
    DiscriminatorMetrics,
    GeneratorMetrics,
    IterableMetrics,
)
from trainning.mlx.schedulers import MultiStepLR


class MLXFaciesGAN(FaciesGAN[mx.array, nn.Module, Optimizer, MultiStepLR], nn.Module):
    """MLX adapter for the FaciesGAN architecture.

    This class implements the abstract FaciesGAN base for the Apple MLX framework.
    It handles:
    - Construction of MLX-based Generator and Discriminator.
    - Lazy evaluation of computation graphs via `mx.eval`.
    - Native MLX optimization steps.
    """

    def __init__(
        self,
        options: TrainningOptions,
        noise_channels: int = 3,
        compile_backend: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLX FaciesGAN model.

        Parameters
        ----------
        options : TrainningOptions
            Configuration options.
        wells : list[mx.array], optional
            List of well log tensors for conditioning, by default [].
        seismic : list[mx.array], optional
            List of seismic volume tensors for conditioning, by default [].
        noise_channels : int, optional
            Number of input noise channels, by default 3.
        """
        super().__init__(options, noise_channels, *args, **kwargs)
        nn.Module.__init__(self)

        self.dtype = mx.float32
        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))
        self.compile_backend = compile_backend
        self.setup_framework()
        self._gp_counter = 0

    def reset_parameters(self) -> None:
        gens = getattr(self.generator, "gens", [])
        discs = getattr(self.discriminator, "discs", [])
        n = min(len(gens), len(discs))
        for idx in range(n):
            gen_reset = getattr(gens[idx], "reset_parameters", None)
            if gen_reset:
                gen_reset()
            disc_reset = getattr(discs[idx], "reset_parameters", None)
            if disc_reset:
                disc_reset()

    def __call__(
        self, *args: Any, **kwds: Any
    ) -> tuple[IterableMetrics[mx.array], ...]:
        return self.forward(*args, **kwds)

    def forward(
        self,
        generator_optimizers: dict[int, Optimizer],
        discriminator_optimizers: dict[int, Optimizer],
        indexes: list[int],
        facies_pyramid: dict[int, mx.array],
        rec_in_pyramid: dict[int, mx.array],
        wells_pyramid: dict[int, mx.array] = {},
        masks_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> tuple[IterableMetrics[mx.array], ...]:
        """Execute a forward pass for both discriminator and generator.

        Parameters
        ----------
        generator_optimizers : dict[int, Optimizer]
            Dictionary mapping scale indices to optimizers for generator.
        discriminator_optimizers : dict[int, Optimizer]
            Dictionary mapping scale indices to optimizers for discriminator.
        indexes : list[int]
            Batch/sample indices used for consistent noise generation.
        facies_pyramid : dict[int, mx.array]
            Dictionary of real facies tensors at each scale.
        rec_in_pyramid : dict[int, mx.array]
            Dictionary of reconstruction input tensors at each scale.
        wells_pyramid : dict[int, mx.array], optional
            Dictionary of well log tensors for conditioning, by default ().
        masks_pyramid : dict[int, mx.array], optional
            Dictionary of well/mask tensors for conditioning, by default ().
        seismic_pyramid : dict[int, mx.array], optional
            Dictionary of seismic volume tensors for conditioning, by default ().
        Returns
        -------
        ScaleMetrics[mx.array]
            The computed metrics for discriminator and generator.
        """

        discriminator_metrics: IterableMetrics[mx.array] = cast(
            IterableMetrics[mx.array],
            self.optimize_discriminator(
                indexes,
                discriminator_optimizers,
                facies_pyramid,
                wells_pyramid,
                seismic_pyramid,
            ),
        )

        generator_metrics: IterableMetrics[mx.array] = cast(
            IterableMetrics[mx.array],
            self.optimize_generator(
                indexes,
                generator_optimizers,
                facies_pyramid,
                rec_in_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            ),
        )

        return discriminator_metrics, generator_metrics

    def build_discriminator(self) -> MLXDiscriminator:
        """Build the MLX Discriminator model.

        Returns
        -------
        MLXDiscriminator
            The instantiated discriminator.
        """
        return MLXDiscriminator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.disc_input_channels,
        )

    def build_generator(self) -> MLXGenerator:
        """Build the MLX Generator model.

        Returns
        -------
        MLXGenerator
            The instantiated generator.
        """
        return MLXGenerator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.gen_input_channels,
            self.gen_output_channels,
        )

    def compute_discriminator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: mx.array,
        wells_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> tuple[
        DiscriminatorMetrics[mx.array] | IterableMetrics[mx.array],
        dict[str, Any],
    ]:
        """Compute discriminator losses and gradients for a specific scale.

        Parameters
        ----------
        indexes : list[int]
            Batch indices used for consistent noise generation.
        scale : int
            Current pyramid scale.
        real_facies : mx.array
            Real facies images for the current scale.
        wells_pyramid : dict[int, mx.array]
            Dictionary of well log tensors for conditioning.
        seismic_pyramid : dict[int, mx.array]
            Dictionary of seismic volume tensors for conditioning.
        Returns
        -------
        tuple[DiscriminatorMetrics[mx.array], dict[str, Any] | None]
            A tuple containing the calculated metrics and the gradients
            (to be passed to the update method).
        """
        metrics: DiscriminatorMetrics[mx.array] = DiscriminatorMetrics(
            mx.array(0.0),
            mx.array(0.0),
            mx.array(0.0),
            mx.array(0.0),
        )

        noises = self.get_pyramid_noise(
            scale,
            indexes,
            wells_pyramid,
            seismic_pyramid,
        )
        fake = self.generate_fake(noises, scale)

        outputs: dict[str, mx.array] = {}

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            cast(MLXDiscriminator, self.discriminator.discs[scale]).update(params)  # type: ignore
            outputs["d_real"] = self.discriminator(scale, real)
            outputs["d_fake"] = self.discriminator(scale, fake)
            metrics.real = -outputs["d_real"].mean()
            metrics.fake = outputs["d_fake"].mean()
            metrics.gp = self.compute_gradient_penalty(scale, real, fake)
            return metrics.real + metrics.fake + metrics.gp

        params = cast(dict[str, Any], self.discriminator.discs[scale].parameters())
        metrics.total, gradients = mx.value_and_grad(compute_metrics)(params)  # type: ignore

        # Return gradients so they can be passed to update_discriminator_weights
        return metrics, cast(dict[str, Any], gradients)

    def compute_generator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: mx.array,
        rec_in_pyramid: dict[int, mx.array],
        wells_pyramid: dict[int, mx.array] = {},
        masks_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> tuple[GeneratorMetrics[mx.array], dict[str, Any]]:
        """Compute generator losses and gradients for a specific scale.

        Parameters
        ----------
        indexes : list[int]
            Batch indices used for consistent noise generation.
        scale : int
            Current pyramid scale.
        real_facies : mx.array
            Real facies images for the current scale.
        rec_in_pyramid : dict[int, mx.array]
            Reconstruction input tensors for the current scale (from previous scale).
        wells_pyramid : dict[int, mx.array]
            Dictionary of well log tensors for conditioning.
        masks_pyramid : dict[int, mx.array]
            Dictionary of well/mask tensors for conditioning.
        seismic_pyramid : dict[int, mx.array]
            Dictionary of seismic volume tensors for conditioning.

        Returns
        -------
        tuple[GeneratorMetrics[mx.array], dict[str, Any] | None]
            A tuple containing the calculated metrics and the gradients
            (to be passed to the update method).
        """
        metrics: GeneratorMetrics[mx.array] = GeneratorMetrics(
            mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)
        )

        rec_in = rec_in_pyramid[scale]
        well = wells_pyramid.get(scale, None)
        mask = masks_pyramid.get(scale, None)

        outputs: dict[str, mx.array] = {}

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            self.generator.gens[scale].update(params)  # type: ignore
            fake_samples = self.generate_diverse_samples(
                indexes, scale, wells_pyramid, seismic_pyramid
            )
            fake = fake_samples[0]
            outputs["fake"] = fake

            metrics.fake = self.compute_adversarial_loss(scale, fake)

            metrics.well = self.compute_masked_loss(
                fake,
                real,
                well,
                mask,
            )

            metrics.div = self.compute_diversity_loss(fake_samples)
            metrics.rec = self.compute_recovery_loss(
                indexes,
                scale,
                real,
                rec_in,
                wells_pyramid,
                seismic_pyramid,
            )
            total = metrics.fake + metrics.well + metrics.rec + metrics.div
            return total

        params = cast(dict[str, Any], self.generator.gens[scale].parameters())
        metrics.total, gradients = mx.value_and_grad(compute_metrics)(params)  # type: ignore

        return metrics, cast(dict[str, Any], gradients)

    def generate_diverse_samples(
        self,
        indexes: list[int],
        scale: int,
        wells_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> list[mx.array]:
        samples: list[mx.array] = []
        for i in range(self.num_diversity_samples):
            noises = self.get_pyramid_noise(
                scale, indexes, wells_pyramid, seismic_pyramid
            )
            amps = self.get_noise_aplitude(scale)
            samples.append(self.generator(noises, amps, stop_scale=scale))
        return samples

    def optimize_discriminator(
        self,
        indexes: list[int],
        optimizers: dict[int, Optimizer],
        facies_pyramid: dict[int, mx.array],
        wells_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> tuple[DiscriminatorMetrics[mx.array], ...] | IterableMetrics[mx.array]:
        """Framework-agnostic discriminator optimization orchestration.

        This method zeroes gradients, delegates framework-specific forward
        computations to small abstract hooks implemented by subclasses,
        aggregates tensor losses for a single backward call, and steps the
        provided optimizers. It intentionally avoids importing heavy
        frameworks.

        Parameters
        ----------
        indexes : list[int]
            List of batch/sample indices used to generate noise.
        facies_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to real tensor samples.
        wells_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to well-conditioning tensors.
        seismic_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to seismic-conditioning tensors.
        Returns
        -------
        tuple[
            dict[int, list[tuple[mx.array, ...]]],
            dict[int, list[dict[str, Any]]],
            dict[int, list[dict[str, Any]]],
        ]
            Tuple containing:
            - metrics: dict[int, list[tuple[mx.array, ...]]]
                Dictionary mapping scale indices to lists of discriminator metrics tuples.
            - gradients: dict[int, list[dict[str, Any]]]
                Dictionary mapping scale indices to lists of gradients dicts.
        """
        metrics: dict[int, list[tuple[mx.array, ...]]] = {
            scale: [] for scale in self.active_scales
        }
        gradients: dict[int, list[dict[str, Any]]] = {
            scale: [] for scale in self.active_scales
        }

        for _ in range(self.discriminator_steps):

            for scale in self.active_scales:

                met, grad = self.compute_discriminator_metrics(
                    indexes,
                    scale,
                    facies_pyramid[scale],
                    wells_pyramid,
                    seismic_pyramid,
                )

                met = cast(DiscriminatorMetrics[mx.array], met)
                metrics[scale].append(met.as_tuple())
                gradients[scale].append(grad)

                self.update_discriminator_weights(
                    scale,
                    optimizers[scale],
                    met.total,
                    grad,
                )

        # return metrics, gradients, parameters, states
        return metrics, gradients

    def optimize_generator(
        self,
        indexes: list[int],
        optimizers: dict[int, Optimizer],
        facies_pyramid: dict[int, mx.array],
        rec_in_pyramid: dict[int, mx.array],
        wells_pyramid: dict[int, mx.array] = {},
        masks_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> tuple[GeneratorMetrics[mx.array], ...] | IterableMetrics[mx.array]:
        """Framework-agnostic generator optimization orchestration.

        This method handles zeroing grads, calling the per-scale
        computation hook, aggregating totals for backward, and stepping
        optimizers. Subclasses must implement
        `compute_generator_metrics` to return scale-level
        metrics and produced tensors.

        Parameters
        ----------
        indexes : list[int]
            List of batch/sample indices used to generate noise.
        optimizers : dict[int, Optimizer]
            Dictionary mapping scale indices to generator optimizers.
        facies_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to real tensor samples.
        rec_in_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to reconstruction inputs.
        wells_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to well-conditioning tensors.
        masks_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to well/mask tensors.
        seismic_pyramid : dict[int, mx.array]
            Dictionary mapping scale indices to seismic-conditioning tensors.

        Returns
        -------
        tuple[
            dict[int, list[tuple[mx.array, ...]]],
            dict[int, list[dict[str, Any]]],
            dict[int, list[dict[str, Any]]],
            dict[int, list[dict[str, Any]]],
        ]
            Tuple containing:
            - metrics: dict[int, list[tuple[mx.array, ...]]]
                Dictionary mapping scale indices to lists of generator metrics tuples.
            - gradients: dict[int, list[dict[str, Any]]]
                Dictionary mapping scale indices to lists of gradients dicts.
        """
        metrics: dict[int, list[tuple[mx.array, ...]]] = {
            scale: [] for scale in self.active_scales
        }
        gradients: dict[int, list[dict[str, Any]]] = {
            scale: [] for scale in self.active_scales
        }

        for _ in range(self.generator_steps):

            for scale in self.active_scales:
                if scale not in facies_pyramid:
                    continue

                real = facies_pyramid[scale]

                if len(self.noise_amps) < scale + 1:
                    raise RuntimeError(
                        f"noise_amp not initialized for scale {scale}. "
                        "Call the project's noise initialization before training."
                    )

                met, grad = self.compute_generator_metrics(
                    indexes,
                    scale,
                    real,
                    rec_in_pyramid,
                    wells_pyramid,
                    masks_pyramid,
                    seismic_pyramid,
                )

                metrics[scale].append(met.as_tuple())
                gradients[scale].append(grad)

                # Delegate the optimization step to subclass
                self.update_generator_weights(
                    scale,
                    optimizers[scale],
                    met.total,
                    grad,
                )

        # return metrics, gradients
        return metrics, gradients

    def update_discriminator_weights(
        self, scale: int, optimizer: Optimizer, loss: mx.array, gradients: Any | None
    ) -> None:
        """Update discriminator weights using MLX optimizer.

        Parameters
        ----------
        scale : int
            Current scale index.
        optimizer :
            The MLX optimizer instance.
        loss : mx.array
            The total loss (evaluated lazily).
        gradients : Any | None
            The gradients computed by `value_and_grad`.
        """
        if gradients:
            discriminator = self.discriminator.discs[scale]
            optimizer.update(discriminator, gradients)  # type: ignore

    def update_generator_weights(
        self, scale: int, optimizer: Optimizer, loss: mx.array, gradients: Any | None
    ) -> None:
        """Update generator weights using MLX optimizer.

        Parameters
        ----------
        scale : int
            Current scale index.
        optimizer : mx.Optimizer
            The MLX optimizer instance.
        loss : mx.array
            The total loss (evaluated lazily).
        gradients :  Any| None
            The gradients computed by `value_and_grad`.
        """
        if gradients:
            generator = self.generator.gens[scale]
            optimizer.update(generator, gradients)  # type: ignore

    def concatenate_tensors(self, tensors: list[mx.array]) -> mx.array:
        """Concatenate a list of MLX arrays along the channel axis (last dimension).

        Parameters
        ----------
        tensors : list[mx.array]
            List of tensors to concatenate.

        Returns
        -------
        mx.array
            The concatenated tensor.
        """
        concat_tensor = mx.concat(tensors, axis=-1)  # type: ignore
        return concat_tensor

    def compute_diversity_loss(self, fake_samples: list[mx.array]) -> mx.array:
        """Compute diversity loss across multiple generated `fake_samples`.

        Encourages different noise inputs to produce diverse outputs by
        penalizing small pairwise distances between flattened samples.
        Uses vmap for efficient vectorization.

        Parameters
        ----------
            fake_samples : list[mx.array]
                List of generated samples to compare for diversity.

        Returns
        -------
        mx.array
            Scalar diversity loss.
        """
        if self.lambda_diversity <= 0 or len(fake_samples) < 2:
            return mx.array(0.0)

        stacked = mx.stack([f.flatten() for f in fake_samples])
        n = stacked.shape[0]
        diffs = stacked[:, None] - stacked[None, :]
        sq_diffs = (diffs**2).mean(axis=-1)
        mask = mx.triu(mx.ones((n, n)), k=1)
        diversity_matrix = mx.exp(-sq_diffs * 10)
        diversity_loss = (diversity_matrix * mask).sum()
        num_pairs = mask.sum()
        if num_pairs == 0:
            return mx.array(0.0)

        return self.lambda_diversity * (diversity_loss / num_pairs)

    def compute_gradient_penalty(
        self, scale: int, real: mx.array, fake: mx.array
    ) -> mx.array:
        """Compute WGAN-GP gradient penalty.

        Parameters
        ----------
        scale : int
            Current scale index.
        real : mx.array
            Real images.
        fake : mx.array
            Fake images.

        Returns
        -------
        mx.array
            Scalar gradient penalty term.
        """
        # Random interpolation factor
        batch_size = real.shape[0]
        alpha = mx.random.uniform(shape=(batch_size, 1, 1, 1))  # type: ignore
        interpolates = alpha * real + (1 - alpha) * fake  # type: ignore

        def grad_fn(x: mx.array) -> mx.array:
            # Discriminator output for interpolates
            out: mx.array = self.discriminator(scale, x)
            return mx.sum(out)  # type: ignore

        # Calculate gradients of the output w.r.t. the interpolates
        gradients = cast(mx.array, mx.grad(grad_fn)(interpolates))  # type: ignore
        grad_norm = mx.sqrt(mx.sum(mx.square(gradients), axis=-1) + 1e-12)
        gradient_penalty = mx.mean(mx.square(grad_norm - 1.0)) * self.lambda_grad
        return gradient_penalty

    def compute_masked_loss(
        self,
        fake: mx.array,
        real: mx.array,
        well: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Compute MSE loss constrained to well locations.

        Parameters
        ----------
        scale : int
            Current scale index.
        fake : mx.array
            Generated images.
        real : mx.array
            Real images.
        wells_dict : dict[int, mx.array]
            Dictionary of well log tensors for conditioning.
        masks_dict : dict[int, mx.array]
            Dictionary of masks indicating well locations.

        Returns
        -------
        mx.array
            Masked MSE loss.
        """
        if well is None or mask is None:
            return mx.array(0.0)
        mse = nn.losses.mse_loss(fake * mask, real * mask)
        return self.well_loss_penalty * mse

    def compute_recovery_loss(
        self,
        indexes: list[int],
        scale: int,
        real: mx.array,
        rec_in: mx.array,
        wells_pyramid: dict[int, mx.array] = {},
        seismic_pyramid: dict[int, mx.array] = {},
    ) -> mx.array:
        """Compute reconstruction loss for the image pyramid.

        Parameters
        ----------
        indexes : list[int]
            Batch indices.
        scale : int
            Current scale index.
        real : mx.array
            Real images.
        rec_in : mx.array
            Reconstruction input tensor for the current scale (upsampled from previous scale).
        wells_pyramid : dict[int, mx.array], optional
            Dictionary of well log tensors for conditioning, by default {}.
        seismic_pyramid : dict[int, mx.array], optional
            Dictionary of seismic volume tensors for conditioning, by default {}.

        Returns
        -------
        mx.array
            Reconstruction loss.
        """
        if self.alpha == 0:
            return mx.array(0.0)
        rec_noise = self.get_pyramid_noise(
            scale,
            indexes,
            wells_pyramid,
            seismic_pyramid,
            rec=True,
        )
        rec = self.generator(
            rec_noise,
            self.noise_amps[: scale + 1],
            in_noise=rec_in,
            start_scale=scale,
            stop_scale=scale,
        )
        rec_loss = self.alpha * nn.losses.mse_loss(rec, real)
        return rec_loss

    def finalize_discriminator_scale(self, scale: int) -> None:
        """Initialize weights for the new discriminator scale.

        Parameters
        ----------
        scale : int
            The new scale index.
        """
        utils.init_weights(self.discriminator.discs[scale])

        # discriminator = self.discriminator.discs[scale]
        # self.compute_gradient_penalty_.append(
        #     mx.compile(
        #         partial(utils.calc_gradient_penalty, discriminator),
        #         inputs=[discriminator],
        #         outputs=[discriminator],  # type: ignore
        #     )
        # )

    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Initialize weights for the new generator scale.

        Parameters
        ----------
        scale : int
            The new scale index.
        reinit : bool
            If True, reinitialize weights. If False, copy from previous scale.
        """
        if scale == 0 or reinit:
            utils.init_weights(self.generator.gens[scale])
        else:
            self.generator.gens[scale].update(  # type: ignore
                self.generator.gens[scale - 1].parameters(),
            )

    def generate_fake(self, noises: list[mx.array], scale: int) -> mx.array:
        """Generate fake images using the provided noise inputs.

        Parameters
        ----------
        noises : list[mx.array]
            List of noise tensors for each scale.
        scale : int
            The target scale to generate up to.

        Returns
        -------
        mx.array
            Generated image tensor.
        """
        amps = (
            self.noise_amps[: scale + 1]
            if hasattr(self, "noise_amps")
            else [1.0] * (scale + 1)
        )
        fake = self.generator(noises, amps, stop_scale=scale)
        return fake

    def generate_noise(
        self,
        scale: int,
        indexes: list[int],
        well: mx.array | None = None,
        seismic: mx.array | None = None,
    ) -> mx.array:
        """Create a noise tensor for a single pyramid level, optionally
        concatenating conditioning channels and applying padding.

        Parameters
        ----------
        scale : int
            Pyramid level index used to select shapes and conditioning tensors.
        indexes : list[int]
            Batch/sample indices to select conditioning slices from stored per-scale tensors.
        wells_dict : dict[int, mx.array], optional
            Dictionary of well log tensors for conditioning, by default {}.
        seismic_dict : dict[int, mx.array], optional
            Dictionary of seismic volume tensors for conditioning, by default {}.

        Returns
        -------
        mx.array
            Padded noise tensor for the requested level, possibly concatenated with well
            and/or seismic conditioning.
        """

        batch = len(indexes)

        if well is not None and seismic is not None:
            shape = self.get_noise_shape(scale)
            z = utils.generate_noise(shape, num_samp=batch)
            well = well[indexes]
            seismic = seismic[indexes]
            z = self.concatenate_tensors([z, well, seismic])
        elif well is not None:
            shape = self.get_noise_shape(scale)
            z = utils.generate_noise(shape, num_samp=batch)
            well = well[indexes]
            z = self.concatenate_tensors([z, well])
        elif seismic is not None:
            shape = self.get_noise_shape(scale)
            z = utils.generate_noise(shape, num_samp=batch)
            seismic = seismic[indexes]
            z = self.concatenate_tensors([z, seismic])
        else:
            shape = self.get_noise_shape(scale, use_base_channel=False)
            # Quiet path when no conditioning is present.

            z = utils.generate_noise((*shape, self.gen_input_channels), num_samp=batch)

        z = self.generate_padding(z, value=0)
        return z

    def generate_padding(self, z: mx.array, value: int = 0) -> mx.array:
        """Pad the input tensor with zeros or a specified value.

        Parameters
        ----------
        z : mx.array
            Input tensor.
        value : int, optional
            Padding value, by default 0.

        Returns
        -------
        mx.array
            Padded tensor.
        """
        p = self.zero_padding
        return mx.pad(  # type: ignore
            z, [(0, 0), (p, p), (p, p), (0, 0)], constant_values=value
        )

    def get_noise_shape(
        self, scale: int, use_base_channel: bool = True
    ) -> tuple[int, ...]:
        """Determine the noise shape for a given scale.

        Parameters
        ----------
        scale : int
            Scale index.
        use_base_channel : bool, optional
            Whether to use the base channel count (typically includes conditioning),
            by default True.

        Returns
        -------
        tuple[int, ...]
            Shape tuple (Height, Width, Channels) for MLX (NHWC).
        """
        return (
            (*self.shapes[scale][1:3], self.base_channel)
            if use_base_channel
            else self.shapes[scale][1:3]
        )

    def get_rec_noise(self, scale: int) -> list[mx.array]:
        """Retrieve reconstruction noise for the specified scale.

        Parameters
        ----------
        scale : int
            Scale index.

        Returns
        -------
        list[mx.array]
            List of reconstruction noise tensors.
        """
        return [mx.array(tensor) for tensor in self.rec_noise[: scale + 1]]

    def load_amp(self, scale_path: str) -> None:
        """Load the noise amplitude value from disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory.
        """
        amp_path = os.path.join(scale_path, "amp.txt")
        if os.path.exists(amp_path):
            with open(amp_path, "r") as f:
                self.noise_amps.append(float(f.read().strip()))

    def load_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Load discriminator weights from a checkpoint.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory.
        scale : int
            Scale index.
        """
        # MLX uses .npz extension instead of .pth
        disc_path = os.path.join(scale_path, D_FILE.replace(".pth", ".npz"))
        if os.path.exists(disc_path):
            self.discriminator.discs[scale] = utils.load(
                disc_path, as_type=type(self.discriminator.discs[scale])
            )

    def load_generator_state(self, scale_path: str, scale: int) -> None:
        """Load generator weights from a checkpoint.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory.
        scale : int
            Scale index.
        """
        # MLX uses .npz extension instead of .pth
        gen_path = os.path.join(scale_path, G_FILE.replace(".pth", ".npz"))
        if os.path.exists(gen_path):
            self.generator.gens[scale] = utils.load(
                gen_path, as_type=type(self.generator.gens[scale])
            )

    def load_shape(self, scale_path: str) -> None:
        """Load image shape metadata from disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory.
        """
        shape_path = os.path.join(scale_path, SHAPE_FILE)
        if os.path.exists(shape_path):
            # Load array and convert to tuple
            loaded = mx.load(shape_path)  # type: ignore
            # mx.load returns a dict, get the array from it
            if isinstance(loaded, dict):
                shape_array = list(loaded.values())[0]
            else:
                shape_array = loaded
            self.shapes.append(tuple(int(x) for x in shape_array.tolist()))  # type: ignore

    def load_wells(self, scale_path: str) -> None:
        """Load well data from disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory.
        """
        well_path = os.path.join(scale_path, M_FILE)
        wells: list[mx.array] = []
        if os.path.exists(well_path):
            # append to private `_wells` list used for conditioning
            wells.append(utils.load(well_path, as_type=mx.array))
        self._wells = tuple(wells)

    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Save discriminator weights to disk.

        Parameters
        ----------
        scale_path : str
            Output directory.
        scale : int
            Scale index.
        """
        if scale < len(self.discriminator.discs):
            # MLX requires .npz or .safetensors extension
            path = os.path.join(scale_path, D_FILE.replace(".pth", ".npz"))
            self.discriminator.discs[scale].save_weights(path)

    def save_discriminator_state_c_compat(self, scale_path: str, scale: int) -> None:
        """Save discriminator weights in C-compatible safetensors format.

        Parameters
        ----------
        scale_path : str
            Output directory.
        scale : int
            Scale index.
        """
        if scale >= len(self.discriminator.discs):
            return
        try:
            ref_path = os.path.join(scale_path, D_FILE.replace(".pth", ".npz"))
            c_map: dict[str, mx.array] = {}
            if os.path.exists(ref_path):
                loaded = mx.load(ref_path)
                if isinstance(loaded, dict) and loaded:
                    c_map = {str(k): v for k, v in loaded.items()}
            if not c_map:
                params = self.discriminator.discs[scale].parameters()
                flat = mlx_utils.tree_flatten(params)
                if not flat:
                    return
                if isinstance(flat[0], tuple) and len(flat[0]) == 2:
                    c_map = {str(k): v for k, v in flat}
                else:
                    c_map = {f"param_{i:06d}": v for i, v in enumerate(flat)}
            path = os.path.join(scale_path, "discriminator.safetensors")
            mx.save_safetensors(path, c_map)
        except Exception:
            return

    def save_generator_state(self, scale_path: str, scale: int) -> None:
        """Save generator weights to disk.

        Parameters
        ----------
        scale_path : str
            Output directory.
        scale : int
            Scale index.
        """
        if scale < len(self.generator.gens):
            # MLX requires .npz or .safetensors extension
            path = os.path.join(scale_path, G_FILE.replace(".pth", ".npz"))
            self.generator.gens[scale].save_weights(path)

    def save_generator_state_c_compat(self, scale_path: str, scale: int) -> None:
        """Save generator weights in C-compatible safetensors format.

        Parameters
        ----------
        scale_path : str
            Output directory.
        scale : int
            Scale index.
        """
        if scale >= len(self.generator.gens):
            return
        try:
            ref_path = os.path.join(scale_path, G_FILE.replace(".pth", ".npz"))
            c_map: dict[str, mx.array] = {}
            if os.path.exists(ref_path):
                loaded = mx.load(ref_path)
                if isinstance(loaded, dict) and loaded:
                    c_map = {str(k): v for k, v in loaded.items()}
            if not c_map:
                params = self.generator.gens[scale].parameters()
                flat = mlx_utils.tree_flatten(params)
                if not flat:
                    return
                if isinstance(flat[0], tuple) and len(flat[0]) == 2:
                    c_map = {str(k): v for k, v in flat}
                else:
                    c_map = {f"param_{i:06d}": v for i, v in enumerate(flat)}
            path = os.path.join(scale_path, "generator.safetensors")
            mx.save_safetensors(path, c_map)
        except Exception:
            return

    def save_shape(self, scale_path: str, scale: int) -> None:
        """Save shape metadata to disk.

        Parameters
        ----------
        scale_path : str
            Output directory.
        scale : int
            Scale index.
        """
        if scale < len(self.shapes):
            path = os.path.join(scale_path, SHAPE_FILE)
            # Convert tuple to mx.array for saving
            shape_array = mx.array(self.shapes[scale])
            mx.savez(path, shape_array)  # type: ignore
