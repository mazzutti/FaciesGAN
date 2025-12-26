"""Parallel LAPGAN implementation for training multiple scales simultaneously.

This module extends the standard FaciesGAN to support parallel training of
multiple pyramid scales. Instead of training scales sequentially, this
implementation can train multiple scales at once using separate optimizers
and discriminators for each scale.
"""

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AMP_FILE, D_FILE, G_FILE, M_FILE, SHAPE_FILE
from models.base import FaciesGAN
from models.torch import utils
from models.torch.discriminator import TorchDiscriminator
from models.torch.generator import TorchGenerator
from options import TrainningOptions
from training.metrics import DiscriminatorMetrics, GeneratorMetrics, ScaleMetrics


class TorchFaciesGAN(
    FaciesGAN[
        torch.Tensor,
        nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
    ],
    nn.Module,
):
    """PyTorch adapter for the framework-agnostic FaciesGAN base.

    This class manages the lifecycle of Generators and Discriminators,
    initializes them, and provides helpers for the training loop.
    Unlike PyTorch, we don't inherit from a base class with a strict
    call graph here, but rather provide the necessary functional hooks.
    """

    def __init__(
        self,
        options: TrainningOptions,
        device: torch.device = torch.device("cpu"),
        noise_channels: int = 3,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the parallel FaciesGAN model.

        Parameters
        ----------
        device : torch.device
            Device for computation.
        options : TrainningOptions
            Training configuration containing hyperparameters.
        wells : list[torch.Tensor], optional
            Optional per-scale well-conditioning tensors, by default [].
        seismic : list[torch.Tensor], optional
            Optional per-scale seismic-conditioning tensors, by default [].
        noise_channels : int, optional
            Number of input noise channels, by default 3.
        """
        nn.Module.__init__(self)
        # Initialize framework-agnostic attributes in the base class
        super().__init__(options, noise_channels, *args, **kwargs)

        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))

        # Framework-specific attributes
        self.device = device

        # Create framework objects via the base class helper (calls build_* hooks)
        self.setup_framework()

    def __call__(self, *args: Any, **kwds: Any) -> ScaleMetrics[torch.Tensor]:
        return nn.Module.__call__(self, *args, **kwds)

    def backward_grads(
        self,
        losses: list[torch.Tensor],
        gradients: list[dict[str, Any]] | None = None,
    ) -> None:
        """Call PyTorch backward on aggregated losses.

        Parameters:
        ----------
        losses  (list[torch.Tensor]): List of gradient tensors to backpropagate
            through the graph via `torch.autograd.backward`.
        gradients (list[dict[str, Any]] | None, optional):
            Optional list of dictionaries mapping parameter names to gradients
            to populate during the backward call (default is None).
        """
        torch.autograd.backward(losses)

    def build_discriminator(self) -> TorchDiscriminator:
        """Build and return the PyTorch `Discriminator` instance (not moved).

        Returns:
            TorchDiscriminator: Newly constructed discriminator instance.
        """
        return TorchDiscriminator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.disc_input_channels,
        ).to(self.device)

    def build_generator(self) -> TorchGenerator:
        """Build and return the PyTorch `Generator` instance (not moved).

        Returns:
            TorchGenerator: Newly constructed generator instance.
        """
        return TorchGenerator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.gen_input_channels,
            self.gen_output_channels,
        ).to(self.device)

    def compute_discriminator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: torch.Tensor,
        wells_pyramid: tuple[torch.Tensor, ...] = (),
        seismic_pyramid: tuple[torch.Tensor, ...] = (),
    ) -> tuple[DiscriminatorMetrics[torch.Tensor], dict[str, Any] | None]:
        """Compute discriminator losses and gradient penalty for a scale.

        Parameters
        ----------
        indexes (tuple[int, ...]):
            Batch/sample indices used to generate fake inputs.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real_facies (torch.Tensor):
            Ground-truth tensor for the current scale.
        wells_pyramid (tuple[torch.Tensor, ...], optional):
            Wells tensors dict for conditioning, keyed by scale.
        seismic_pyramid (tuple[torch.Tensor, ...], optional):
            Seismic  tensors dict for conditioning, keyed by scale.

        Returns
        -------
        tuple[DiscriminatorMetrics[torch.Tensor], dict[Any, Any] | None]:
            Container with total, real, fake and gp losses, and optional gradients dict.
        """

        real_loss = -self.discriminator(scale, real).mean()
        noises = self.get_pyramid_noise(scale, indexes, wells_pyramid, seismic_pyramid)
        fake = self.generate_fake(noises, scale)
        fake_loss = self.discriminator(scale, fake.detach()).mean()  # type: ignore
        gp_loss = self.compute_gradient_penalty(scale, real, fake)
        return (
            DiscriminatorMetrics(
                total=(real_loss + fake_loss + gp_loss),
                real=real_loss,
                fake=fake_loss,
                gp=gp_loss,
            ),
            None,
        )

    def compute_generator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: torch.Tensor,
        rec_in: torch.Tensor,
        wells_pyramid: tuple[torch.Tensor, ...] = (),
        seismic_pyramid: tuple[torch.Tensor, ...] = (),
        mask: torch.Tensor | None = None,
    ) -> tuple[GeneratorMetrics[torch.Tensor], dict[str, Any] | None]:
        """Common generator-metrics flow shared by frameworks.

        Parameters
        ----------
        indexes (list[int]):
            Batch/sample indices used to generate noise.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real (torch.Tensor):
            Ground-truth tensor for the current scale.
        rec_in (torch.Tensor):
            Reconstruction input tensor for the current scale.
        wells_pyramid (tuple[torch.Tensor, ...], optional):
            Wells tensors dict for conditioning, keyed by scale.
        seismic_pyramid (tuple[torch.Tensor, ...], optional):
            Seismic tensors dict for conditioning, keyed by scale.
        mask (torch.Tensor | None, optional):
            Well mask tensor for the current scale, by default None.
        Returns
        -------
        tuple[
            GeneratorMetrics[TTensor] | dict[Any, Any] | None
        ]:
            Container with total, fake, rec, well and div losses, and optional gradients dict.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """

        # Generate diversity candidates (framework-agnostic forward)
        fake_samples = self.generate_diverse_samples(
            indexes,
            scale,
            wells_pyramid,
            seismic_pyramid,
        )
        fake = fake_samples[0]

        # Delegate component computations to subclass hooks
        adv = self.compute_adversarial_loss(scale, fake)
        well = self.compute_masked_loss(
            fake,
            real,
            wells_pyramid[scale] if wells_pyramid else None,
            mask,
        )
        div = self.compute_diversity_loss(fake_samples)
        rec_loss = self.compute_recovery_loss(
            indexes,
            scale,
            real,
            rec_in,
            wells_pyramid,
            seismic_pyramid,
        )

        total = adv + well + rec_loss + div

        metrics = GeneratorMetrics(
            total=total,
            fake=adv,
            rec=rec_loss,
            well=well,
            div=div,
        )

        return metrics, None

    def concatenate_tensors(
        self, tensors: list[torch.Tensor], dim: int = 1
    ) -> torch.Tensor:
        """Concatenate a list of tensors along dimension `dim`.

        Uses PyTorch `torch.cat` and preserves device placement.

        Parameters
        ----------
        tensors (list[torch.Tensor]):
            List of tensors to concatenate.
        dim (int, optional):

            Dimension along which to concatenate, by default 1.
        """
        return torch.cat(tensors, dim=dim)

    def compute_diversity_loss(self, fake_samples: list[torch.Tensor]) -> torch.Tensor:
        """Compute diversity loss across multiple generated `fake_samples`.

        Encourages different noise inputs to produce diverse outputs by
        penalizing small pairwise distances between flattened samples.

        Parameters:
            fake_samples (list[torch.Tensor]): List of generated samples to
                compare for diversity.

        Returns:
            torch.Tensor: Scalar diversity loss; zero when disabled or when
                fewer than two samples are provided.
        """
        if self.lambda_diversity <= 0 or len(fake_samples) < 2:
            return torch.tensor(0.0, device=self.device)
        n = len(fake_samples)
        stacked = torch.stack([f.flatten() for f in fake_samples])
        sq_diffs = ((stacked.unsqueeze(1) - stacked.unsqueeze(0)) ** 2).mean(dim=2)
        mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1).bool()
        distances = sq_diffs[mask]
        diversity_loss = torch.exp(-distances * 10).sum()
        num_pairs = distances.numel()
        return self.lambda_diversity * (
            diversity_loss / num_pairs
            if num_pairs > 0
            else torch.tensor(0.0, device=self.device)
        )

    def compute_gradient_penalty(
        self, scale: int, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute the gradient penalty for WGAN-GP style regularization.

        Args:
            scale (int): Discriminator scale index used for the penalty.
            real (torch.Tensor): Real samples tensor.
            fake (torch.Tensor): Fake samples tensor.

        Returns:
            torch.Tensor: Scalar gradient penalty term.
        """
        return utils.calc_gradient_penalty(
            self.discriminator.discs[scale],
            real,
            fake,
            self.lambda_grad,
            self.device,
        )

    def compute_masked_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        well: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mask-weighted MSE between `fake` and `real` at `scale`.

        parameters
        ----------
        fake (torch.Tensor):
            Generated tensor samples for the current scale.
        real (torch.Tensor):
            Ground-truth tensor samples for the current scale.
        wells (torch.Tensor):
            Well-conditioning tensor for the current scale.
        masks (torch.Tensor):
            Well mask tensor for the current scale.

        Returns:
            torch.Tensor: Scalar masked MSE loss scaled by
                `self.well_loss_penalty`, or zero if no wells are used.
        """
        if well is None or mask is None:
            return torch.zeros(1, device=self.device)
        return self.well_loss_penalty * nn.MSELoss(reduction="mean")(
            fake * mask, real * mask
        )

    def compute_recovery_loss(
        self,
        indexes: list[int],
        scale: int,
        real: torch.Tensor,
        rec_in: torch.Tensor,
        wells_pyramid: tuple[torch.Tensor, ...] = (),
        seismic_pyramid: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        """Compute reconstruction (recovery) loss for given inputs.

        parameters
        ----------
        indexes (tuple[int, ...]):
            Batch/sample indices used to generate reconstruction noise.
        scale (int):
            Pyramid scale index for which to compute the loss.
        real (torch.Tensor):
            Ground-truth tensor for the current scale.
        rec_in (torch.Tensor):
            Reconstruction input tensor for the current scale.
        wells_pyramid (tuple[torch.Tensor, ...]), optional):
            Wells tensors tuple for conditioning, keyed by scale.
        seismic_pyramid (tuple[torch.Tensor, ...]), optional):
            Seismic tensors tuple for conditioning, keyed by scale.

        Returns:
            torch.Tensor: Scalar reconstruction loss weighted by `self.alpha`,
                or zero when recovery is disabled.
        """
        if self.alpha == 0:
            return torch.zeros(1, device=self.device)
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
        rec_loss = self.alpha * nn.MSELoss()(rec, real)
        return rec_loss

    def finalize_discriminator_scale(self, scale: int) -> None:
        """Finalize discriminator block after creation.

        This applies weight initialization to the newly created discriminator
        block and ensures it is moved to `self.device`.

        Args:
            scale (int): Index of the discriminator scale to finalize.
        """
        self.discriminator.discs[scale].apply(utils.weights_init)
        self.discriminator.discs[scale] = self.discriminator.discs[scale].to(
            self.device
        )

        self.discriminator_optimizers[scale] = torch.optim.Adam(
            self.discriminator.discs[scale].parameters(),
            lr=self.lr_d,
            betas=(self.beta1, 0.999),
        )
        self.discriminator_schedulers[scale] = torch.optim.lr_scheduler.MultiStepLR(
            self.discriminator_optimizers[scale],
            milestones=[self.lr_decay],
            gamma=self.gamma,
        )

    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Finalize generator block after creation.

        Either initialize weights for a freshly reinitialized block or copy
        weights from the previous scale, then move the block to `self.device`.

        Args:
            scale (int): Index of the generator scale to finalize.
            reinit (bool): Whether to initialize weights instead of copying.
        """
        if reinit:
            self.generator.gens[scale].apply(utils.weights_init)
        else:
            self.generator.gens[scale].load_state_dict(
                self.generator.gens[scale - 1].state_dict()
            )
        self.generator.gens[scale] = self.generator.gens[scale].to(self.device)

        self.generator_optimizers[scale] = torch.optim.Adam(
            self.generator.gens[scale].parameters(),
            lr=self.lr_g,
            betas=(self.beta1, 0.999),
        )

        self.generator_schedulers[scale] = torch.optim.lr_scheduler.MultiStepLR(
            self.generator_optimizers[scale],
            milestones=[self.lr_decay],
            gamma=self.gamma,
        )

    def forward(
        self,
        indexes: list[int],
        facies_pyramid: tuple[torch.Tensor, ...],
        rec_in_pyramid: tuple[torch.Tensor, ...],
        wells_pyramid: tuple[torch.Tensor, ...] = (),
        masks_pyramid: tuple[torch.Tensor, ...] = (),
        seismic_pyramid: tuple[torch.Tensor, ...] = (),
    ) -> ScaleMetrics[torch.Tensor]:
        """Perform a forward pass and compute scale metrics.

        Parameters
        ----------
        indexes (list[int]):
            List of batch/sample indices used to generate noise.
        facies_pyramid (tuple[torch.Tensor, ...]):
            Tuple mapping scale indices to real tensor samples.
        rec_in_pyramid (tuple[torch.Tensor, ...]):
            Tuple mapping scale indices to reconstruction input tensors.
        wells_pyramid (tuple[torch.Tensor, ...], optional):
            Wells tensors tuple for conditioning, keyed by scale.
        masks_pyramid (tuple[torch.Tensor, ...], optional):
            Well masks tuple for conditioning, keyed by scale.
        seismic_pyramid (tuple[torch.Tensor, ...], optional):
            Seismic tensors tuple for conditioning, keyed by scale.

        Returns
        -------
        ScaleMetrics[torch.Tensor]:
            Container with discriminator and generator metrics for the scale.
        """

        return ScaleMetrics(
            discriminator=self.optimize_discriminator(
                indexes,
                facies_pyramid,
                wells_pyramid,
                seismic_pyramid,
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

    def generate_fake(self, noises: list[torch.Tensor], scale: int) -> torch.Tensor:
        """Generate a fake sample at the requested `scale` using `noises`.

        Args:
            noises (list[torch.Tensor]): Noise inputs for the generator per scale.
            scale (int): Target scale index to generate.

        Returns:
            torch.Tensor: Generated fake tensor for the requested scale.
        """
        with torch.no_grad():
            amps = self.get_noise_aplitude(scale)
            fake = self.generator(noises, amps, stop_scale=scale)
        return fake

    def generate_noise(
        self,
        scale: int,
        indexes: list[int],
        well: torch.Tensor | None = None,
        seismic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create a noise tensor for a single pyramid level, optionally
        concatenating conditioning channels and applying padding.

        Parameters
        ----------
        scale : int
            Pyramid level index used to select shapes and conditioning tensors.
        indexes : list[int]
            Batch/sample indices to select conditioning slices from stored per-scale tensors
        wells : torch.Tensor, optional
            Well-conditioning tensor for the current scale, by default torch.Tensor().
        seismic : torch.Tensor, optional
            Seismic-conditioning tensor for the current scale, by default torch.Tensor().

        Returns
        -------
        torch.Tensor
            Padded noise tensor for the requested level, possibly concatenated with well
            and/or seismic conditioning.
        """

        batch = len(indexes)

        if well is not None and seismic is not None:
            shape = self.get_noise_shape(scale)
            z = utils.generate_noise(shape, num_samp=batch, device=self.device)
            well = well[indexes].to(self.device)
            seismic = seismic[indexes].to(self.device)
            z = self.concatenate_tensors([z, well, seismic])
        elif well is not None:
            shape = self.get_noise_shape(scale)
            z = utils.generate_noise(shape, num_samp=batch, device=self.device)
            well = well[indexes].to(self.device)
            z = self.concatenate_tensors([z, well])
        elif seismic is not None:
            shape = self.get_noise_shape(scale)
            z = utils.generate_noise(shape, num_samp=batch, device=self.device)
            seismic = seismic[indexes].to(self.device)
            z = self.concatenate_tensors([z, seismic])
        else:
            shape = self.get_noise_shape(scale, use_base_channel=False)
            z = utils.generate_noise(
                (self.gen_input_channels, *shape),
                num_samp=batch,
                device=self.device,
            )

        return self.generate_padding(z, value=0)

    def get_rec_noise(self, scale: int) -> list[torch.Tensor]:
        return [tensor.clone() for tensor in self.rec_noise[: scale + 1]]

    def generate_padding(self, z: torch.Tensor, value: int = 0) -> torch.Tensor:
        """Pad tensor `z` using the model's zero-padding size.

        Args:
            z (torch.Tensor): Input tensor to pad.
            value (int): Padding fill value (default: 0).

        Returns:
            torch.Tensor: Padded tensor.
        """
        return F.pad(z, [self.zero_padding] * 4, value=value)

    def load_amp(self, scale_path: str) -> None:
        """Default loader for amplitude files created by `save_amp`.

        Reads the text file named by `AMP_FILE` and appends the parsed float to
        `self.noise_amp` if present. This keeps the base class framework-agnostic
        while providing a sensible default implementation.
        """
        amp_path = os.path.join(scale_path, AMP_FILE)
        if os.path.exists(amp_path):
            with open(amp_path, "r") as f:
                self.noise_amps.append(float(f.read().strip()))

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
            (self.base_channel, *self.shapes[scale][2:])
            if use_base_channel
            else self.shapes[scale][2:]
        )

    def load_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Load discriminator state dict for `scale` from `scale_path` if present.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the discriminator scale to load.
        """
        disc_path = os.path.join(scale_path, D_FILE)
        if os.path.exists(disc_path):
            self.discriminator.discs[scale].load_state_dict(
                torch.load(disc_path, map_location=self.device)
            )

    def load_generator_state(self, scale_path: str, scale: int) -> None:
        """Load generator state dict for the latest generator in the scale.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the generator scale to load (unused here).
        """
        gen_path = os.path.join(scale_path, G_FILE)
        if os.path.exists(gen_path):
            self.generator.gens[-1].load_state_dict(
                torch.load(gen_path, map_location=self.device)
            )

    def load_shape(self, scale_path: str) -> None:
        """Load saved shape tensor for a scale and append to `self.shapes`.

        Args:
            scale_path (str): Directory path for the given scale.
        """
        shape_path = os.path.join(scale_path, SHAPE_FILE)
        if os.path.exists(shape_path):
            self.shapes.append(torch.load(shape_path, map_location=self.device))

    def load_wells(self, scale_path: str) -> None:
        """Load well-conditioning mask for a scale and append to `self.wells`.

        Args:
            scale_path (str): Directory path for the given scale.
        """
        wells: list[torch.Tensor] = []
        wells.append(
            utils.load(
                os.path.join(scale_path, M_FILE),
                self.device,
                as_type=torch.Tensor,
            )
        )
        self.wells = tuple(wells)

    def move_to_device(self, obj: Any, device: torch.device | None = None) -> Any:
        """Move PyTorch modules or tensors to a target device.

        Args:
            obj (Any): Module or tensor to move.
            device (torch.device | None): Destination device. If None, uses
                `self.device`.

        Returns:
            Any: The object moved to the target device.
        """
        return obj.to(device or self.device)

    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Save discriminator state dict for `scale` to `scale_path` if present.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the discriminator scale to save.
        """
        if scale < len(self.discriminator.discs):
            discriminator_path = os.path.join(scale_path, f"{D_FILE}")
            torch.save(self.discriminator.discs[scale].state_dict(), discriminator_path)

    def save_generator_state(self, scale_path: str, scale: int) -> None:
        """Save generator state dict for `scale` to `scale_path` if present.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the generator scale to save.
        """
        if scale < len(self.generator.gens):
            generator_path = os.path.join(scale_path, f"{G_FILE}")
            torch.save(self.generator.gens[scale].state_dict(), generator_path)

    def save_shape(self, scale_path: str, scale: int) -> None:
        """Save shape tensor for `scale` to disk at `scale_path`.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the shape to save.
        """
        if scale < len(self.shapes):
            shape_path = os.path.join(scale_path, SHAPE_FILE)
            torch.save(self.shapes[scale], shape_path)

    def update_discriminator_weights(
        self,
        scale: int,
        loss: torch.Tensor,
        gradients: Any | None,
    ) -> dict[str, Any] | None:
        """Perform standard PyTorch optimization step."""
        self.discriminator_optimizers[scale].zero_grad()
        torch.autograd.backward(loss)
        self.discriminator_optimizers[scale].step()

    def update_generator_weights(
        self,
        scale: int,
        loss: torch.Tensor,
        gradients: Any | None,
    ) -> dict[str, Any] | None:
        """Perform standard PyTorch optimization step."""
        self.generator_optimizers[scale].zero_grad()
        torch.autograd.backward(loss)
        self.generator_optimizers[scale].step()
