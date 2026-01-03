import os
from typing import Any, cast

import math

import mlx.core as mx
import mlx.nn as nn  # type: ignore
from mlx.optimizers import Adam, Optimizer  # type: ignore


from config import D_FILE, G_FILE, M_FILE, SHAPE_FILE
from models.base import FaciesGAN
from models.mlx.discriminator import MLXDiscriminator
from models.mlx.generator import MLXGenerator
from options import TrainningOptions
import models.mlx.utils as utils
from trainning.metrics import DiscriminatorMetrics, GeneratorMetrics, ScaleMetrics
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

        # Keep reference to options so setup can access flags like mixed precision
        self.dtype = mx.bfloat16 if options.use_mixed_precision else mx.float32
        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))
        self.compile_backend = compile_backend
        self.setup_framework()

    def __call__(self, *args: Any, **kwds: Any) -> ScaleMetrics[mx.array]:
        return self.forward(*args, **kwds)

    def forward(
        self,
        indexes: list[int],
        facies_pyramid: tuple[mx.array, ...],
        rec_in_pyramid: tuple[mx.array, ...],
        wells_pyramid: tuple[mx.array, ...] = (),
        masks_pyramid: tuple[mx.array, ...] = (),
        seismic_pyramid: tuple[mx.array, ...] = (),
    ) -> ScaleMetrics[mx.array]:
        """Execute a forward pass for both discriminator and generator.

        Parameters
        ----------
        indexes : list[int]
            Batch/sample indices used for consistent noise generation.
        facies_pyramid : tuple[mx.array, ...]
            Tuple of real facies tensors at each scale.
        rec_in_pyramid : tuple[mx.array, ...]
            Tuple of reconstruction input tensors at each scale.
        wells_pyramid : tuple[mx.array, ...], optional
            Tuple of well log tensors for conditioning, by default ().
        masks_pyramid : tuple[mx.array, ...], optional
            Tuple of well/mask tensors for conditioning, by default ().
        seismic_pyramid : tuple[mx.array, ...], optional
            Tuple of seismic volume tensors for conditioning, by default ().

        Returns
        -------
        ScaleMetrics[mx.array]
            The computed metrics for discriminator and generator.
        """

        return ScaleMetrics(
            discriminator=self.optimize_discriminator(
                indexes, facies_pyramid, wells_pyramid, seismic_pyramid
            ),
            generator=self.optimize_generator(
                indexes,
                facies_pyramid,
                rec_in_pyramid,
                wells_pyramid,
                masks_pyramid,
                seismic_pyramid,
            ),
        )

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
        if not self.compile_backend:
            # FIX: flattening gradients dict to ensure values are evaluated
            eval_items = list(losses)
            if gradients:
                eval_items.extend(utils.flatten_to_list(gradients))
            mx.eval(*eval_items)  # type: ignore

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
            dtype=self.dtype,
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
            dtype=self.dtype,
        )

    def compute_discriminator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: mx.array,
        wells_pyramid: tuple[mx.array, ...] = (),
        seismic_pyramid: tuple[mx.array, ...] = (),
    ) -> tuple[DiscriminatorMetrics[mx.array], dict[str, Any] | None]:
        """Compute discriminator losses and gradients for a specific scale.

        Parameters
        ----------
        indexes : list[int]
            Batch indices used for consistent noise generation.
        scale : int
            Current pyramid scale.
        real_facies : mx.array
            Real facies images for the current scale.
        wells_pyramid : tuple[mx.array, ...]
            Tuple of well log tensors for conditioning.
        seismic_pyramid : tuple[mx.array, ...]
            Tuple of seismic volume tensors for conditioning.

        Returns
        -------
        tuple[DiscriminatorMetrics[mx.array], dict[str, Any] | None]
            A tuple containing the calculated metrics and the gradients
            (to be passed to the update method).
        """
        metrics: DiscriminatorMetrics[mx.array] = DiscriminatorMetrics(
            mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)
        )

        noises = self.get_pyramid_noise(
            scale,
            indexes,
            wells_pyramid,
            seismic_pyramid,
        )
        fake = self.generate_fake(noises, scale)

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            cast(MLXDiscriminator, self.discriminator.discs[scale]).update(params)  # type: ignore
            metrics.real = -self.discriminator(scale, real).mean()
            metrics.fake = self.discriminator(scale, fake).mean()
            metrics.gp = self.compute_gradient_penalty(scale, real, fake)
            return metrics.real + metrics.fake + metrics.gp

        params = cast(dict[str, Any], self.discriminator.discs[scale].parameters())
        metrics.total, gradients = mx.value_and_grad(compute_metrics)(params)  # type: ignore

        # Return gradients so they can be passed to update_discriminator_weights
        return metrics, cast(dict[str, Any] | None, gradients)

    def compute_generator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: mx.array,
        rec_in: mx.array,
        wells_pyramid: tuple[mx.array, ...] = (),
        seismic_pyramid: tuple[mx.array, ...] = (),
        mask: mx.array | None = None,
    ) -> tuple[GeneratorMetrics[mx.array], dict[str, Any] | None]:
        """Compute generator losses and gradients for a specific scale.

        Parameters
        ----------
        indexes : list[int]
            Batch indices used for consistent noise generation.
        scale : int
            Current pyramid scale.
        real_facies : mx.array
            Real facies images for the current scale.
        rec_in : mx.array
             Array of reconstruction input.
        wells_pyramid : tuple[mx.array, ...]
            Tuple of well log tensors for conditioning.
        seismic_pyramid : tuple[mx.array, ...]
            Tuple of seismic volume tensors for conditioning.
        mask : mx.array | None
            Well/mask tensor for conditioning.

        Returns
        -------
        tuple[GeneratorMetrics[mx.array], dict[str, Any] | None]
            A tuple containing the calculated metrics and the gradients
            (to be passed to the update method).
        """
        metrics: GeneratorMetrics[mx.array] = GeneratorMetrics(
            mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)
        )

        def compute_metrics(params: dict[str, Any]) -> mx.array:
            self.generator.gens[scale].update(params)  # type: ignore
            fake_samples = self.generate_diverse_samples(
                indexes, scale, wells_pyramid, seismic_pyramid
            )
            fake = fake_samples[0]

            metrics.fake = self.compute_adversarial_loss(scale, fake)
            metrics.well = self.compute_masked_loss(
                fake,
                real,
                wells_pyramid[scale] if len(wells_pyramid) > 0 else None,
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

        return metrics, cast(dict[str, Any] | None, gradients)

    def update_discriminator_weights(
        self, scale: int, loss: mx.array, gradients: Any | None
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
            optimizer = self.discriminator_optimizers[scale]
            discriminator = self.discriminator.discs[scale]
            optimizer.update(discriminator, gradients)  # type: ignore
            # Return updated state elements for lazy evaluation
            return {
                "discriminator": discriminator.parameters(),
                "discriminator_optimizer": optimizer.state,  # type: ignore
            }

    def update_generator_weights(
        self, scale: int, loss: mx.array, gradients: Any | None
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
            optimizer = self.generator_optimizers[scale]
            generator = self.generator.gens[scale]
            optimizer.update(generator, gradients)  # type: ignore
            # Return updated state elements for lazy evaluation
            return {
                "generator": generator.parameters(),
                "generator_optimizer": optimizer.state,  # type: ignore
            }

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
        return utils.calc_gradient_penalty(
            self.discriminator.discs[scale],
            real,
            fake,
            self.lambda_grad,
        )

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
        wells_pyramid: tuple[mx.array, ...] = (),
        seismic_pyramid: tuple[mx.array, ...] = (),
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
            Input for reconstruction from previous scale.
        wells_pyramid : tuple[mx.array, ...], optional
            Tuple of well log tensors for conditioning, by default ().
        seismic_pyramid : tuple[mx.array, ...], optional
            Tuple of seismic volume tensors for conditioning, by default ().

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

        self.discriminator_optimizers[scale] = Adam(
            learning_rate=self.lr_d,
            betas=[self.beta1, 0.999],
        )
        self.discriminator_optimizers[scale].init(  # type: ignore
            self.discriminator.discs[scale].parameters()
        )
        self.discriminator_schedulers[scale] = MultiStepLR(
            init_lr=self.lr_d,
            milestones=[self.lr_decay],
            gamma=self.gamma,
            optimizer=self.discriminator_optimizers[scale],
        )

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

        self.generator_optimizers[scale] = Adam(
            learning_rate=self.lr_g,
            betas=[self.beta1, 0.999],
        )
        self.generator_optimizers[scale].init(  # type: ignore
            self.generator.gens[scale].parameters()
        )

        self.generator_schedulers[scale] = MultiStepLR(
            init_lr=self.lr_g,
            milestones=[self.lr_decay],
            gamma=self.gamma,
            optimizer=self.generator_optimizers[scale],
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
