import os
from typing import Any, cast

import math

import mlx.core as mx
import mlx.nn as nn  # type: ignore


from config import D_FILE, G_FILE, M_FILE, SHAPE_FILE
from models.base import FaciesGAN
from models.mlx.discriminator import MLXDiscriminator
from models.mlx.generator import MLXGenerator
from options import TrainningOptions
import models.mlx.utils as utils
from training.metrics import DiscriminatorMetrics, GeneratorMetrics


class MLXFaciesGAN(FaciesGAN[mx.array, nn.Module]):
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
        wells: tuple[mx.array, ...] = (),
        seismic: tuple[mx.array, ...] = (),
        noise_channels: int = 3,
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
        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))
        self.setup_framework()
        self.wells, self.seismic = wells, seismic

    def backward_grads(
        self,
        losses: list[mx.array],
        gradients: list[Any] | None = None,
    ) -> None:
        """Execute the backward pass and evaluate the computation graph.

        In MLX, operations are lazy. This method forces the evaluation of
        losses and gradient updates, ensuring the computation graph is executed
        and memory is freed.

        Parameters
        ----------
        losses : list[mx.array]
            List of loss tensors to evaluate.
        gradients : list[Any] | None, optional
            List of gradients or optimizer states to evaluate/update, by default None.
        """
        mx.eval(*losses, *(gradients or []))  # type: ignore

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
        indexes: mx.array,
        scale: int,
        real_facies: mx.array,
    ) -> tuple[DiscriminatorMetrics[mx.array], dict[str, Any] | None]:
        """Compute discriminator losses and gradients for a specific scale.

        Parameters
        ----------
        indexes : mx.array
            Batch indices used for consistent noise generation.
        scale : int
            Current pyramid scale.
        real_facies : mx.array
            Real facies images for the current scale.

        Returns
        -------
        tuple[DiscriminatorMetrics[mx.array], dict[str, Any] | None]
            A tuple containing the calculated metrics and the gradients
            (to be passed to the update method).
        """
        metrics: DiscriminatorMetrics[mx.array] = DiscriminatorMetrics(
            mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)
        )

        noises = self.get_noise(indexes, scale)
        fake = self.generate_fake(noises, scale)

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            cast(MLXDiscriminator, self.discriminator.discs[scale]).update(params)  # type: ignore
            metrics.real = -self.discriminator(scale, real_facies).mean()
            metrics.fake = self.discriminator(scale, fake).mean()
            metrics.gp = self.compute_gradient_penalty(scale, real_facies, fake)
            return metrics.real + metrics.fake + metrics.gp

        params = cast(
            dict[str, Any], self.discriminator.discs[scale].trainable_parameters()
        )
        metrics.total, gradients = mx.value_and_grad(compute_metrics)(params)  # type: ignore

        # Return gradients so they can be passed to update_discriminator_weights
        return metrics, cast(dict[str, Any] | None, gradients)

    def compute_generator_metrics(
        self,
        indexes: mx.array,
        scale: int,
        real_facies: mx.array,
        masks_dict: dict[int, mx.array],
        rec_in_dict: dict[int, mx.array],
    ) -> tuple[GeneratorMetrics[mx.array], dict[str, Any] | None]:
        """Compute generator losses and gradients for a specific scale.

        Parameters
        ----------
        indexes : mx.array
            Batch indices used for consistent noise generation.
        scale : int
            Current pyramid scale.
        real_facies : mx.array
            Real facies images for the current scale.
        masks_dict : dict[int, mx.array]
            Dictionary of well masks for conditioning.
        rec_in_dict : dict[int, mx.array]
            Dictionary of reconstruction inputs.

        Returns
        -------
        tuple[GeneratorMetrics[mx.array], dict[str, Any] | None]
            A tuple containing the calculated metrics and the gradients
            (to be passed to the update method).
        """
        metrics: GeneratorMetrics[mx.array] = GeneratorMetrics(
            mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)
        )
        rec_in = rec_in_dict[scale]

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            self.generator.gens[scale].update(params)  # type: ignore
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

        return metrics, cast(dict[str, Any] | None, gradients)

    def update_discriminator_weights(
        self, scale: int, optimizer: Any, loss: mx.array, gradients: Any | None
    ) -> dict[str, Any] | None:
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

        Returns
        -------
        list[Any]
            A list containing the optimizer state and updated parameters.
            These must be passed to `mx.eval` to trigger the update.
        """
        if gradients:
            optimizer.update(self.discriminator.discs[scale], gradients)
            # Return updated state elements for lazy evaluation
            return cast(dict[str, Any], self.discriminator.discs[scale].parameters())

    def update_generator_weights(
        self, scale: int, optimizer: Any, loss: mx.array, gradients: Any | None
    ) -> dict[str, Any] | None:
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

        Returns
        -------
        dict[str, Any]
            A list containing the optimizer state and updated parameters.
            These must be passed to `mx.eval` to trigger the update.
        """
        if gradients:
            optimizer.update(self.generator.gens[scale], gradients)
            # Return updated state elements for lazy evaluation
            return cast(dict[str, Any], self.generator.gens[scale].parameters())

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
        """Compute diversity loss to encourage variation in generated samples.

        Parameters
        ----------
        fake_samples : list[mx.array]
            List of generated samples.

        Returns
        -------
        mx.array
            Calculated diversity loss.
        """
        if self.lambda_diversity <= 0 or len(fake_samples) < 2:
            return mx.array(0.0)
        n = len(fake_samples)
        stacked = mx.stack([x.flatten() for x in fake_samples])
        norms = mx.sum(stacked * stacked, axis=1)  # (N,)
        distances = norms[:, None] + norms[None, :] - 2 * (stacked @ stacked.T)
        diversity_loss = mx.exp(-distances * 10)
        mask = 1 - mx.eye(n)
        masked_loss = diversity_loss * mask
        avg_loss = mx.sum(masked_loss) / (n * (n - 1))
        return self.lambda_diversity * (avg_loss if n > 1 else mx.array(0.0))

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
        """Compute MSE loss constrained to well locations.

        Parameters
        ----------
        scale : int
            Current scale index.
        fake : mx.array
            Generated images.
        real : mx.array
            Real images.
        masks_dict : dict[int, mx.array]
            Dictionary of masks indicating well locations.

        Returns
        -------
        mx.array
            Masked MSE loss.
        """
        if len(self.wells) == 0:
            return mx.array(0.0)
        masks = masks_dict[scale]
        mse = nn.losses.mse_loss(fake * masks, real * masks)
        return self.well_loss_penalty * mse

    def compute_recovery_loss(
        self,
        indexes: mx.array,
        scale: int,
        rec_in: mx.array | None,
        real: mx.array,
    ) -> mx.array:
        """Compute reconstruction loss for the image pyramid.

        Parameters
        ----------
        indexes :  mx.array
            Batch indices.
        scale : int
            Current scale index.
        rec_in : mx.array | None
            Input for reconstruction from previous scale.
        real : mx.array
            Real images.

        Returns
        -------
        mx.array
            Reconstruction loss.
        """
        if self.alpha == 0 or rec_in is None:
            return mx.array(0.0)
        rec_noise = self.get_noise(indexes, scale, rec=True)
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

    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Initialize weights for the new generator scale.

        Parameters
        ----------
        scale : int
            The new scale index.
        reinit : bool
            If True, reinitialize weights. If False, copy from previous scale.
        """
        if reinit:
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
            if hasattr(self, "noise_amp")
            else [1.0] * (scale + 1)
        )
        fake = self.generator(noises, amps, stop_scale=scale)
        return fake

    def generate_noise(self, index: int, indexes: mx.array) -> mx.array:
        """Create a noise tensor for a single pyramid level, optionally
        concatenating conditioning channels and applying padding.

        Parameters
        ----------
        index : int
            Pyramid level index used to select shapes and conditioning tensors.
        indexes : mx.array
            Batch/sample indices to select conditioning slices from stored per-scale tensors.

        Returns
        -------
        mx.array
            Padded noise tensor for the requested level, possibly concatenated with well
            and/or seismic conditioning.
        """

        batch = len(indexes)

        if self.use_wells(index) and self.use_seismic(index):
            shape = self.get_noise_shape(index)
            z = utils.generate_noise(shape, num_samp=batch)
            well = self._wells[index][indexes]
            seismic = self._seismic[index][indexes]
            z = self.concatenate_tensors([z, well, seismic])
        elif self.use_wells(index):
            shape = self.get_noise_shape(index)
            z = utils.generate_noise(shape, num_samp=batch)
            well = self._wells[index][indexes]
            z = self.concatenate_tensors([z, well])
        elif self.use_seismic(index):
            shape = self.get_noise_shape(index)
            z = utils.generate_noise(shape, num_samp=batch)
            seismic = self._seismic[index][indexes]
            z = self.concatenate_tensors([z, seismic])
        else:
            shape = self.get_noise_shape(index, use_base_channel=False)
            z = utils.generate_noise((*shape, self.gen_input_channels), num_samp=batch)

        return self.generate_padding(z, value=0)

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
        disc_path = os.path.join(scale_path, D_FILE)
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
        gen_path = os.path.join(scale_path, G_FILE)
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
            self.shapes.append(cast(tuple[int, ...], mx.load(shape_path)))  # type: ignore

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
            wells.append(utils.load(well_path, as_type=mx.array))
        self.wells = tuple(wells)

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
            path = os.path.join(scale_path, D_FILE)
            self.discriminator.discs[scale].save_weights(path)

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
            path = os.path.join(scale_path, G_FILE)
            self.generator.gens[scale].save_weights(path)

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
            mx.savez(path, self.shapes[scale])  # type: ignore
