import os
from typing import Any, cast

import math

import mlx.core as mx
import mlx.nn as nn  # type: ignore
import gc


from config import D_FILE, G_FILE, M_FILE, SHAPE_FILE
from models.base import FaciesGAN
from models.mlx.discriminator import MLXDiscriminator
from models.mlx.generator import MLXGenerator
from options import TrainningOptions
import models.mlx.utils as utils
from training.metrics import DiscriminatorMetrics, GeneratorMetrics


class MLXFaciesGAN(FaciesGAN[mx.array, nn.Module]):
    """MLX adapter for the FaciesGAN architecture.

    This class manages the lifecycle of Generators and Discriminators,
    initializes them, and provides helpers for the training loop.
    Unlike PyTorch, we don't inherit from a base class with a strict
    call graph here, but rather provide the necessary functional hooks.
    """

    def __init__(
        self,
        options: TrainningOptions,
        wells: list[mx.array] = [],
        seismic: list[mx.array] = [],
        noise_channels: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the parallel FaciesGAN model.

        Parameters
        ----------
        options : TrainningOptions
            Training configuration containing hyperparameters.
        wells : list[torch.Tensor], optional
            Optional per-scale well-conditioning tensors, by default [].
        seismic : list[torch.Tensor], optional
            Optional per-scale seismic-conditioning tensors, by default [].
        device : torch.device, optional
            Device for computation, by default CPU.
        noise_channels : int, optional
            Number of input noise channels, by default 3.
        """
        super().__init__(options, noise_channels, *args, **kwargs)

        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))
        # Create framework objects via the base class helper (calls build_* hooks)
        self.setup_framework()
        self.wells, self.seismic = wells, seismic

    def backward_grads(
        self,
        losses: list[mx.array],
        gradients: list[dict[str, Any]] | None = None,
    ) -> None:
        """Call MLX backward on aggregated gradients.

        Parameters:
        ----------
        losses (list[mx.array]): List of gradient tensors to evaluate
            through the tree via `mx.eval`.
        gradients (list[dict[str, Any]] | None, optional):
            Optional list of dictionaries mapping parameter names to gradients
            to populate during the backward call (default is None).
        """

        mx.eval(*losses, *(gradients or []))  # type: ignore
        gc.collect()

    def build_discriminator(self) -> MLXDiscriminator:
        return MLXDiscriminator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.disc_input_channels,
        )

    def build_generator(self) -> MLXGenerator:
        """Build and return the MLX `Generator` instance (not moved).

        Returns:
            MLXGenerator: Newly constructed generator instance.
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
        real_facies: mx.array,
    ) -> tuple[DiscriminatorMetrics[mx.array], list[dict[str, Any]] | None]:
        """Compute discriminator losses and gradient penalty for a scale.

        Parameters
        ----------
        indexes (list[int]):
            Batch/sample indices used to generate fake inputs.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real_facies (mx.array):
            Ground-truth tensor for the current scale.

        Returns
        -------
        tuple[DiscriminatorMetrics[mx.array], list[dict[Any, Any]] | None]:
            Container with total, real, fake and gp losses, and optional gradients list.
        """
        metrics: DiscriminatorMetrics[mx.array] = DiscriminatorMetrics(
            mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)
        )

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            cast(MLXDiscriminator, self.discriminator.discs[scale]).update(params)  # type: ignore
            noises = self.get_noise(indexes, scale)
            fake = self.generate_fake(noises, scale)
            metrics.real = -self.discriminator(scale, real_facies).mean()
            metrics.fake = self.discriminator(scale, fake).mean()
            metrics.gp = self.compute_gradient_penalty(scale, real_facies, fake)
            return metrics.real + metrics.fake + metrics.gp

        params = cast(
            dict[str, Any], self.discriminator.discs[scale].trainable_parameters()
        )
        metrics.total, gradients = mx.value_and_grad(compute_metrics)(params)  # type: ignore # type: ignore

        return metrics, [
            gradients,
            params,
            self.discriminator.discs[scale].parameters(),
        ]

    def compute_generator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real_facies: mx.array,
        masks_dict: dict[int, mx.array],
        rec_in_dict: dict[int, mx.array],
    ) -> tuple[GeneratorMetrics[mx.array], list[dict[str, Any]] | None]:
        """Common generator-metrics flow shared by frameworks.

        Parameters
        ----------
        indexes (list[int]):
            Batch/sample indices used to generate noise.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real_facies (TTensor):
            Ground-truth tensor for the current scale.
        masks_dict (dict[int, TTensor]):
            Dictionary mapping scale indices to well/mask tensors.
        rec_in_dict (dict[int, TTensor]):
            Dictionary mapping scale indices to reconstruction inputs.

        Returns
        -------
        tuple[
            GeneratorMetrics[TTensor], list[dict[Any, Any]] | None
        ]:
            Container with total, fake, rec, well and div losses, and optional gradients list.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        metrics: GeneratorMetrics[mx.array] = GeneratorMetrics(
            mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)
        )
        rec_in = rec_in_dict[scale]

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            # Ensure the generator uses the provided parameters for the
            # functional/gradient computation.
            self.generator.gens[scale].update(params)  # type: ignore
            # Generate diversity candidates inside the gradient function so
            # intermediate activations are scoped to the autodiff call and
            # can be freed promptly. Generating them outside kept the
            # forward graph alive across iterations and caused memory growth.
            fake_samples = self.generate_diverse_samples(indexes, scale)
            fake = fake_samples[0]

            metrics.fake = self.compute_adversarial_loss(scale, fake)
            metrics.well = self.compute_masked_loss(
                scale, fake, real_facies, masks_dict
            )
            metrics.div = self.compute_diversity_loss(fake_samples)
            metrics.rec = self.compute_recovery_loss(
                indexes, scale, rec_in, real_facies
            )
            total = metrics.fake + metrics.well + metrics.rec + metrics.div
            return total

        params = cast(dict[str, Any], self.generator.gens[scale].trainable_parameters())
        metrics.total, gradients = mx.value_and_grad(compute_metrics)(params)  # type: ignore
        mx.eval(metrics.total, *gradients)  # type: ignore
        return metrics, cast(
            list[dict[Any, Any]],
            [gradients, params, self.generator.gens[scale].parameters],
        )

    def concatenate_tensors(self, tensors: list[mx.array]) -> mx.array:
        """Concatenate a list of tensors along dimension `dim`.

        Uses MLX `mx.concat` and preserves device placement.
        """
        concat_tensor = mx.concat(tensors, axis=-1)  # type: ignore
        mx.eval(concat_tensor)  # type: ignore
        return concat_tensor

    def compute_diversity_loss(self, fake_samples: list[mx.array]) -> mx.array:
        """Compute diversity loss across multiple generated `fake_samples`.

        Encourages different noise inputs to produce diverse outputs by
        penalizing small pairwise distances between flattened samples.

        Args:
            fake_samples (list[mx.array]): List of generated samples to
                compare for diversity.

        Returns:
            mx.array: Scalar diversity loss; zero when disabled or when
                fewer than two samples are provided.
        """
        if self.lambda_diversity <= 0 or len(fake_samples) < 2:
            return mx.array(0.0)

        n = len(fake_samples)
        # Flatten each sample to (D,) and stack -> (N, D)
        stacked = mx.stack([x.flatten() for x in fake_samples])

        # Compute pairwise squared distances via dot-product identity to
        # avoid allocating the (N, N, D) diffs tensor which can blow up
        # memory for large images. distances[i,j] = ||x_i - x_j||^2
        norms = mx.sum(stacked * stacked, axis=1)  # (N,)
        # (N,1) + (1,N) - 2 * (N,N) -> pairwise squared distances
        distances = norms[:, None] + norms[None, :] - 2 * (stacked @ stacked.T)
        diversity_loss = mx.exp(-distances * 10)
        mask = 1 - mx.eye(n)
        masked_loss = diversity_loss * mask
        avg_loss = mx.sum(masked_loss) / (n * (n - 1))
        return self.lambda_diversity * (avg_loss if n > 1 else mx.array(0.0))

    def compute_gradient_penalty(
        self, scale: int, real: mx.array, fake: mx.array
    ) -> mx.array:
        return utils.calc_gradient_penalty(
            self.discriminator.discs[scale],
            real,
            fake,
            self.lambda_grad,
        )

    def compute_masked_loss(
        self,
        scale: int,
        fake: mx.array,
        real: mx.array,
        masks_dict: dict[int, mx.array],
    ) -> mx.array:
        if len(self.wells) == 0:
            return mx.array(0.0)

        masks = masks_dict[scale]
        mse = nn.losses.mse_loss(fake * masks, real * masks)
        return self.well_loss_penalty * mse

    def compute_recovery_loss(
        self,
        indexes: list[int],
        scale: int,
        rec_in: mx.array | None,
        real: mx.array,
    ) -> mx.array:
        if self.alpha == 0 or rec_in is None:
            return mx.array(0.0)

        rec_noise = self.get_noise(indexes, scale, rec=True)

        rec = self.generator(
            rec_noise,
            self.noise_amp[: scale + 1],
            in_noise=rec_in,
            start_scale=scale,
            stop_scale=scale,
        )
        rec_loss = self.alpha * nn.losses.mse_loss(rec, real)
        return rec_loss

    def finalize_discriminator_scale(self, scale: int) -> None:
        utils.init_weights(self.discriminator.discs[scale])

    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        if reinit:
            utils.init_weights(self.generator.gens[scale])
        else:
            self.generator.gens[scale].update(  # type: ignore
                self.generator.gens[scale - 1].parameters(),
            )

    def generate_fake(self, noises: list[mx.array], scale: int) -> mx.array:
        amps = (
            self.noise_amp[: scale + 1]
            if hasattr(self, "noise_amp")
            else [1.0] * (scale + 1)
        )
        fake = self.generator(noises, amps, stop_scale=scale)
        return fake

    def generate_noise(self, shape: tuple[int, ...], num_samp: int) -> mx.array:
        # shape is (H, W, C), returns (N, H, W, C)
        return utils.generate_noise(shape, num_samp=num_samp)

    def generate_padding(self, z: mx.array, value: int = 0) -> mx.array:
        # z is (N, H, W, C)
        # Pad H and W
        p = self.zero_padding
        return mx.pad(  # type: ignore
            z, [(0, 0), (p, p), (p, p), (0, 0)], constant_values=value
        )

    def get_noise_shape(
        self, scale: int, use_base_channel: bool = True
    ) -> tuple[int, ...]:
        """Return the noise shape tuple for a given `scale`.

        Args:
            scale (int): Scale index for which to get the noise shape.

        Returns:
            tuple[int, ...]: Noise shape tuple as (channels, height, width).
        """
        return (
            (*self.shapes[scale][1:3], self.base_channel)
            if use_base_channel
            else self.shapes[scale][1:3]
        )

    def get_rec_noise(self, scale: int) -> list[mx.array]:
        return [mx.array(tensor) for tensor in self.rec_noise[: scale + 1]]

    def load_amp(self, scale_path: str) -> None:
        amp_path = os.path.join(scale_path, "amp.txt")  # Assuming filename
        if os.path.exists(amp_path):
            with open(amp_path, "r") as f:
                self.noise_amp.append(float(f.read().strip()))

    def load_discriminator_state(self, scale_path: str, scale: int) -> None:
        disc_path = os.path.join(
            scale_path,
        )  # Changed to safetensors
        if os.path.exists(disc_path):
            self.discriminator.discs[scale] = utils.load(
                disc_path, as_type=type(self.discriminator.discs[scale])
            )

    def load_generator_state(self, scale_path: str, scale: int) -> None:
        gen_path = os.path.join(
            scale_path,
        )
        if os.path.exists(gen_path):
            self.generator.gens[scale] = utils.load(
                gen_path, as_type=type(self.generator.gens[scale])
            )

    def load_shape(self, scale_path: str) -> None:
        shape_path = os.path.join(scale_path, SHAPE_FILE)  # Changed to .npy
        if os.path.exists(shape_path):
            self.shapes.append(cast(tuple[int, ...], mx.load(shape_path)))  # type: ignore

    def load_wells(self, scale_path: str) -> None:
        well_path = os.path.join(scale_path, M_FILE)
        if os.path.exists(well_path):
            self.wells.append(utils.load(well_path, as_type=mx.array))

    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        if scale < len(self.discriminator.discs):
            path = os.path.join(scale_path, D_FILE)
            self.discriminator.discs[scale].save_weights(path)

    def save_generator_state(self, scale_path: str, scale: int) -> None:
        if scale < len(self.generator.gens):
            path = os.path.join(scale_path, G_FILE)
            self.generator.gens[scale].save_weights(path)

    def save_shape(self, scale_path: str, scale: int) -> None:
        if scale < len(self.shapes):
            path = os.path.join(scale_path, SHAPE_FILE)
            mx.savez(path, self.shapes[scale])  # type: ignore
