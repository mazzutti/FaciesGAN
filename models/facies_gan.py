"""Parallel LAPGAN implementation for training multiple scales simultaneously.

This module extends the standard FaciesGAN to support parallel training of
multiple pyramid scales. Instead of training scales sequentially, this
implementation can train multiple scales at once using separate optimizers
and discriminators for each scale.
"""

import math
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import ops
from config import AMP_FILE, D_FILE, G_FILE, M_FILE, SHAPE_FILE
from models.discriminator import Discriminator
from models.generator import Generator
from options import TrainningOptions
from metrics import DiscriminatorMetrics, GeneratorMetrics


class FaciesGAN:
    """Parallel multi-scale FaciesGAN for simultaneous scale training.

    This class manages multiple discriminators and enables training several
    pyramid scales in parallel rather than sequentially. Each scale has its
    own discriminator and optimization state.

    Parameters
    ----------
    device : torch.device
        Device for model computation (CPU, CUDA, or MPS).
    options : argparse.Namespace | SimpleNamespace
        Configuration containing all hyperparameters.
    wells : Sequence[torch.Tensor], optional
        Well location data for conditioning. Defaults to empty tuple.
    *args : tuple[Any, ...]
        Additional positional arguments.
    **kwargs : dict[str, Any]
        Additional keyword arguments.

    Attributes
    ----------
    generator : Generator
        Multi-scale progressive generator network.
    discriminators : nn.ModuleDict
        Dictionary mapping scale indices to discriminator networks.
    rec_noise : list[torch.Tensor]
        Reconstruction noise tensors for each scale.
    noise_amp : list[float]
        Noise amplitudes for each scale.
    wells : list[torch.Tensor]
        Well location data for conditioning.
    shapes : list[tuple[int, ...]]
        Pyramid resolutions for each scale.
    active_scales : set[int]
        Set of scale indices currently being trained.
    """

    def __init__(
        self,
        device: torch.device,
        options: TrainningOptions,
        wells: list[torch.Tensor] = [],
        seismic: list[torch.Tensor] = [],
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
        super().__init__(*args, **kwargs)
        self.device = device
        self.num_parallel_scales = options.num_parallel_scales

        # Image parameters
        self.zero_padding = options.num_layer * math.floor(options.kernel_size / 2)

        self.num_img_channels = options.num_img_channels

        # Input/output channels
        self.disc_input_channels: int = self.num_img_channels
        self.disc_output_channels: int = self.num_img_channels

        # total channels fed to the generator (noise + conditioning)
        self.gen_input_channels: int = noise_channels
        self.gen_output_channels: int = self.num_img_channels

        # Training parameters
        self.discriminator_steps = options.discriminator_steps
        self.generator_steps = options.generator_steps
        self.lambda_grad = options.lambda_grad
        self.alpha = options.alpha
        self.lambda_diversity = options.lambda_diversity

        # Network parameters
        self.num_feature = options.num_feature
        self.min_num_feature = options.min_num_feature
        self.num_layer = options.num_layer
        self.kernel_size = options.kernel_size
        self.padding_size = options.padding_size

        self.shapes: list[tuple[int, ...]] = []
        self.rec_noise: list[torch.Tensor] = []
        self.noise_amp: list[float] = []

        # Track active scales being trained
        self.active_scales: set[int] = set()

        self.wells: list[torch.Tensor] = wells
        self.seismic: list[torch.Tensor] = seismic

        self.generator: Generator = Generator(
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.gen_input_channels,
            self.gen_output_channels,
        ).to(self.device)

        # Multiple discriminators for parallel training
        self.discriminators: nn.ModuleDict = nn.ModuleDict()

    def init_scales(self, start_scale: int, num_scales: int) -> None:
        """Initialize multiple scales for parallel training.

        Parameters
        ----------
        start_scale : int
            Starting pyramid scale index.
        num_scales : int
            Number of consecutive scales to initialize.
        """
        for scale in range(start_scale, start_scale + num_scales):
            self.init_scale_generator(scale)
            self.init_scale_discriminator(scale)
            self.active_scales.add(scale)

    def init_scale_generator(self, scale: int) -> None:
        """Initialize generator for a new pyramid scale.

        Creates a new scale block with appropriate feature counts, initializes
        or copies weights, and freezes previous scale generators.

        Parameters
        ----------
        scale : int
            Pyramid scale index to initialize.
        """
        num_feature, min_num_feature = self.get_num_features(scale)

        self.generator.create_scale(num_feature, min_num_feature)
        # Move newly created module to device with channels-last memory format
        self.generator.gens[-1] = self.generator.gens[-1].to(self.device)

        # Reinitialize the weights if features were doubled or using SPADE
        prev_is_spade = (
            (scale - 1) in self.generator.spade_scales if scale > 0 else False
        )
        curr_is_spade = scale in self.generator.spade_scales

        if scale % 4 == 0 or prev_is_spade or curr_is_spade:
            self.generator.gens[-1].apply(ops.weights_init)
        else:
            self.generator.gens[-1].load_state_dict(
                self.generator.gens[-2].state_dict()
            )

        # Don't freeze generators - allow joint optimization
        # This is key for parallel training

    def init_scale_discriminator(self, scale: int) -> None:
        """Initialize discriminator for a new pyramid scale.

        Creates a new discriminator with appropriate feature counts. Each
        scale gets its own discriminator for parallel training.

        Parameters
        ----------
        scale : int
            Pyramid scale index to initialize.
        """
        num_feature, min_num_feature = self.get_num_features(scale)

        # Create a new discriminator for each scale
        discriminator = Discriminator(
            num_feature,
            min_num_feature,
            self.num_layer,
            self.kernel_size,
            self.padding_size,
            self.disc_input_channels,
        ).to(self.device)
        discriminator.apply(ops.weights_init)

        self.discriminators[str(scale)] = discriminator

    def get_num_features(self, scale: int) -> tuple[int, int]:
        """Calculate feature counts for networks at a given scale.

        Features double every 4 scales up to a maximum of 128.

        Parameters
        ----------
        scale : int
            Pyramid scale index.

        Returns
        -------
        tuple[int, int]
            (num_features, min_num_features) for the scale.
        """
        num_feature = min(self.num_feature * pow(2, math.floor(scale / 4)), 128)
        min_num_feature = min(self.min_num_feature * pow(2, math.floor(scale / 4)), 128)

        return num_feature, min_num_feature

    def get_noise(
        self,
        indexes: list[int],
        scale: int,
        rec: bool = False,
    ) -> list[torch.Tensor]:
        """Generate noise tensors up to a specific pyramid scale.

        Parameters
        ----------
        indexes : list[int]
            Indices specifying which well conditioning to use.
        scale : int
            Generate noise up to and including this scale.
        rec : bool, optional
            If True, return stored reconstruction noise. Defaults to False.

        Returns
        -------
        list[torch.Tensor]
            List of noise tensors from scale 0 to the specified scale.
        """

        def generate_noise(index: int) -> torch.Tensor:
            """Generate per-scale noise, optionally concatenating well tensors.

            Rules (preserve existing behavior):
            - If `self.wells` is empty or `index >= len(self.wells)`: generate
              `self.num_channels` noise channels.
            - If `self.wells` exists and `index == 0`: generate `self.num_channels` noise.
            - If `self.wells` exists and `index > 0`: generate `self.num_channels // 2`
              noise channels and concatenate the corresponding well tensor.
            """
            shape = self.shapes[index][2:]
            batch = len(indexes)

            # Determine whether we have a well  or seimic tensor or both for this index
            has_well = len(self.wells) > 0 and index < len(self.wells)
            has_seismic = len(self.seismic) > 0 and index < len(self.seismic)

            if has_well and index > 0:
                if has_seismic:
                    base_ch = max(
                        1, self.gen_input_channels - 2 * self.num_img_channels
                    )
                    z = ops.generate_noise(
                        (base_ch, *shape), device=self.device, num_samp=batch
                    )
                    # Ensure well tensor is on the correct device and select batch indices
                    well = self.wells[index][indexes].to(self.device)
                    seismic = self.seismic[index][indexes].to(self.device)
                    z = torch.cat([z, well, seismic], dim=1)
                else:
                    base_ch = max(1, self.gen_input_channels - self.num_img_channels)
                    z = ops.generate_noise(
                        (base_ch, *shape), device=self.device, num_samp=batch
                    )
                    # Ensure well tensor is on the correct device and select batch indices
                    well = self.wells[index][indexes].to(self.device)
                    z = torch.cat([z, well], dim=1)
            elif has_seismic and index > 0:
                base_ch = max(1, self.gen_input_channels - self.num_img_channels)
                z = ops.generate_noise(
                    (base_ch, *shape), device=self.device, num_samp=batch
                )
                # Ensure seismic tensor is on the correct device and select batch indices
                seismic = self.seismic[index][indexes].to(self.device)
                z = torch.cat([z, seismic], dim=1)
            else:
                z = ops.generate_noise(
                    (self.gen_input_channels, *shape),
                    device=self.device,
                    num_samp=batch,
                )

            return F.pad(z, [self.zero_padding] * 4, value=0)

        if rec:
            return self.rec_noise[: scale + 1].copy()
        return [generate_noise(i) for i in range(scale + 1)]

    def optimize_discriminator(
        self,
        indexes: list[int],
        real_facies_dict: dict[int, torch.Tensor],
        discriminator_optimizers: dict[int, torch.optim.Optimizer],
    ) -> dict[int, DiscriminatorMetrics]:
        """Optimize multiple discriminators in TRUE parallel (simultaneous GPU computation).

        All discriminators are optimized in a single pass with gradient accumulation,
        allowing true parallel execution on the GPU.

        Parameters
        ----------
        indexes : list[int]
            Indices of the samples in the batch.
        real_facies_dict : dict[int, torch.Tensor]
            Dictionary mapping scale indices to real facies tensors.
        discriminator_optimizers : dict[int, torch.optim.Optimizer]
            Dictionary mapping scale indices to discriminator optimizers.

        Returns
        -------
        dict[int, DiscriminatorMetrics]
            Mapping from scale index to `DiscriminatorMetrics`. Each field is a
            tensor scalar (autograd-enabled) representing losses computed for
            that scale. Callers are expected to convert to Python floats when
            logging (e.g., via `.item()`).
        """
        metrics: dict[int, DiscriminatorMetrics] = {}
        # Temporarily hold tensor-valued metrics per-scale for backward()
        scale_computations: dict[int, DiscriminatorMetrics] = {}

        for _ in range(self.discriminator_steps):
            # Zero all gradients at once
            for scale in self.active_scales:
                if scale in real_facies_dict:
                    discriminator_optimizers[scale].zero_grad()

                # Compute all forward passes in parallel (triggers GPU parallelism)
                # Map scale -> DiscriminatorStep (tensor losses) so we can call
                # autograd.backward with named fields and then convert to floats.

            for scale in self.active_scales:
                if scale not in real_facies_dict:
                    continue

                real_facies = real_facies_dict[scale]
                discriminator = self.discriminators[str(scale)]

                # Forward pass for real images
                real_output = discriminator(real_facies)
                real_loss: torch.Tensor = -real_output.mean()

                # Generate fake images
                noises = self.get_noise(indexes, scale)
                with torch.no_grad():
                    fake = self.generator(
                        noises, self.noise_amp[: scale + 1], stop_scale=scale
                    )

                # Forward pass for fake images
                fake_output = discriminator(fake.detach())
                fake_loss: torch.Tensor = fake_output.mean()

                # Gradient penalty
                gp_loss: torch.Tensor = ops.calc_gradient_penalty(
                    discriminator,
                    real_facies,
                    fake,
                    self.lambda_grad,
                    self.device,
                )

                # Store tensor-valued metrics directly so we can backward()
                scale_computations[scale] = DiscriminatorMetrics(
                    total=(real_loss + fake_loss + gp_loss),
                    real=real_loss,
                    fake=fake_loss,
                    gp=gp_loss,
                )

            # Backward pass for all scales (GPU can parallelize)
            for scale, step in scale_computations.items():
                # step.real/step.fake/step.gp are autograd tensors
                torch.autograd.backward([step.real, step.fake, step.gp])

            # Update all discriminators
            for scale in self.active_scales:
                if scale in real_facies_dict:
                    discriminator_optimizers[scale].step()

            # Record losses from last step
            for scale, step in scale_computations.items():
                # Keep tensor metrics; conversion to python floats happens
                # at logging time in the trainer/visualizer.
                metrics[scale] = DiscriminatorMetrics(
                    total=step.total,
                    real=step.real,
                    fake=step.fake,
                    gp=step.gp,
                )

        return metrics

    def optimize_generator(
        self,
        indexes: list[int],
        real_facies_dict: dict[int, torch.Tensor],
        masks_dict: dict[int, torch.Tensor],
        rec_in_dict: dict[int, torch.Tensor],
        generator_optimizers: dict[int, torch.optim.Optimizer],
    ) -> Dict[int, GeneratorMetrics]:
        """Optimize generator for multiple scales.

        All generator blocks are optimized together with gradient accumulation,
        allowing true parallel execution on the GPU.

        Parameters
        ----------
        indexes : list[int]
            Indices of the samples.
        real_facies_dict : dict[int, torch.Tensor]
            Dictionary mapping scale indices to real facies tensors.
        masks_dict : dict[int, torch.Tensor]
            Dictionary mapping scale indices to binary mask tensors.
        rec_in_dict : dict[int, torch.Tensor]
            Dictionary mapping scale indices to reconstruction input tensors.
        generator_optimizers : dict[int, torch.optim.Optimizer]
            Dictionary mapping scale indices to generator optimizers.

        Returns
        -------
        Dict[int, GeneratorMetrics]
            Mapping from scale index to `GeneratorMetrics`. Each field is a
            tensor scalar or tensor (for generated samples); metrics remain
            as tensors to preserve autograd until the caller logs or detaches
            them.
        """
        scale_metrics: dict[int, GeneratorMetrics] = {}
        num_diversity_samples = 3

        for _ in range(self.generator_steps):
            # Zero all generator gradients at once
            for scale in self.active_scales:
                if scale in real_facies_dict:
                    generator_optimizers[scale].zero_grad()

            # Compute all forward passes in parallel for all scales
            # Map scale -> GeneratorMetrics (tensor-valued) for clarity and
            # easier backward() invocation.

            scale_fakes: dict[int, torch.Tensor] = {}
            scale_recs: dict[int, torch.Tensor | None] = {}

            for scale in self.active_scales:
                if scale not in real_facies_dict:
                    continue

                real_facies = real_facies_dict[scale]

                rec_in = rec_in_dict.get(scale)
                discriminator = self.discriminators[str(scale)]

                # Generate multiple samples for diversity loss
                fake_samples: list[torch.Tensor] = []

                for _ in range(num_diversity_samples):
                    noises = self.get_noise(indexes, scale)
                    fake_sample = self.generator(
                        noises, self.noise_amp[: scale + 1], stop_scale=scale
                    )
                    fake_samples.append(fake_sample)

                # Use the first sample as the main fake
                fake = fake_samples[0]

                # Adversarial loss
                generator_loss_fake: torch.Tensor = -discriminator(fake).mean()

                # Well conditioning loss (strong constraint at well locations)
                generator_masked_loss: torch.Tensor = torch.zeros(1, device=self.device)
                if len(self.wells) > 0:
                    masks = masks_dict[scale]
                    generator_masked_loss: torch.Tensor = 10 * nn.MSELoss(
                        reduction="mean"
                    )(fake * masks, real_facies * masks)
                # Diversity loss - encourage different outputs for different noise
                # This prevents mode collapse where all samples look the same
                if self.lambda_diversity > 0 and len(fake_samples) >= 2:
                    # Vectorized pairwise distance calculation
                    n = len(fake_samples)
                    stacked = torch.stack([f.flatten() for f in fake_samples])

                    # Compute pairwise L2 distances using broadcasting
                    # (n, 1, -1) - (1, n, -1) -> (n, n, -1)
                    sq_diffs = (
                        (stacked.unsqueeze(1) - stacked.unsqueeze(0)) ** 2
                    ).mean(dim=2)

                    # Extract upper triangular part (excluding diagonal) for unique pairs
                    mask = torch.triu(
                        torch.ones(n, n, device=self.device), diagonal=1
                    ).bool()
                    distances = sq_diffs[mask]

                    # Penalize similarity (want samples to be DIFFERENT)
                    diversity_loss = torch.exp(-distances * 10).sum()
                    num_pairs = distances.numel()

                    generator_loss_diversity: torch.Tensor = self.lambda_diversity * (
                        diversity_loss / num_pairs
                        if num_pairs > 0
                        else torch.tensor(0.0, device=self.device)
                    )
                else:
                    generator_loss_diversity: torch.Tensor = torch.tensor(
                        0.0, device=self.device
                    )

                generator_loss_rec: torch.Tensor = torch.zeros(1, device=self.device)
                rec = None

                if self.alpha != 0 and rec_in is not None:
                    rec_noise = self.get_noise(indexes, scale, rec=True)
                    rec = self.generator(
                        rec_noise,
                        self.noise_amp[: scale + 1],
                        in_noise=rec_in,
                        start_scale=scale,
                        stop_scale=scale,
                    )

                    generator_loss_rec = self.alpha * nn.MSELoss()(rec, real_facies)

                # Combine all losses for this scale
                total_gen_loss: torch.Tensor = (
                    generator_loss_fake
                    + generator_masked_loss
                    + generator_loss_rec
                    + generator_loss_diversity
                )

                # Store tensor-valued GeneratorMetrics for backward pass
                scale_metrics[scale] = GeneratorMetrics(
                    total=total_gen_loss,
                    fake=generator_loss_fake,
                    rec=generator_loss_rec,
                    well=generator_masked_loss,
                    div=generator_loss_diversity,
                )
                scale_fakes[scale] = fake
                scale_recs[scale] = rec

            if scale_metrics:
                totals = [gm.total for gm in scale_metrics.values()]
                torch.autograd.backward(totals)

            # Update all generators
            for scale in self.active_scales:
                if scale in real_facies_dict:
                    generator_optimizers[scale].step()

        return scale_metrics

    def save_scale(self, scale: int, path: str) -> None:
        """Save the generator and discriminator for a specific scale.

        Parameters
        ----------
        scale : int
            The scale index to save.
        path : str
            Directory path to save the models.
        """
        # Save generator for this scale
        if scale < len(self.generator.gens):
            generator_path = os.path.join(path, f"{G_FILE}")
            torch.save(self.generator.gens[scale].state_dict(), generator_path)

        # Save discriminator for this scale
        if str(scale) in self.discriminators:
            discriminator_path = os.path.join(path, f"{D_FILE}")
            torch.save(self.discriminators[str(scale)].state_dict(), discriminator_path)

        # Save noise amplitude
        if scale < len(self.noise_amp):
            amp_path = os.path.join(path, AMP_FILE)
            with open(amp_path, "w") as f:
                f.write(str(self.noise_amp[scale]))

        # Save shapes
        if scale < len(self.shapes):
            shape_path = os.path.join(path, SHAPE_FILE)
            torch.save(self.shapes[scale], shape_path)

    def load(
        self,
        path: str,
        load_shapes: bool = True,
        until_scale: int | None = None,
        load_discriminator: bool = False,
        load_wells: bool = False,
    ) -> int:
        """Load saved models and return the starting scale.

        Parameters
        ----------
        path : str
            Path to the directory containing model checkpoint files.
        load_shapes : bool, optional
            Whether to load shape information. Defaults to True.
        until_scale : int | None, optional
            Load models up to and including this scale. Defaults to None.

        Returns
        -------
        int
            The next scale to start training from.
        """
        scale = 0
        if load_wells:
            self.wells = []
        while os.path.exists(os.path.join(path, str(scale))):
            if until_scale is not None and scale > until_scale:
                break

            scale_path = os.path.join(path, str(scale))

            # Load generator
            gen_path = os.path.join(scale_path, G_FILE)
            if os.path.exists(gen_path):
                self.init_scale_generator(scale)
                self.generator.gens[-1].load_state_dict(
                    torch.load(gen_path, map_location=self.device)
                )

            if load_discriminator:

                # Load discriminator
                disc_path = os.path.join(scale_path, D_FILE)
                if os.path.exists(disc_path):
                    self.init_scale_discriminator(scale)
                    self.discriminators[str(scale)].load_state_dict(
                        torch.load(disc_path, map_location=self.device)
                    )

            # Load noise amplitude
            amp_path = os.path.join(scale_path, AMP_FILE)
            if os.path.exists(amp_path):
                with open(amp_path, "r") as f:
                    self.noise_amp.append(float(f.read().strip()))

            # Load shapes
            if load_shapes:
                shape_path = os.path.join(scale_path, SHAPE_FILE)
                if os.path.exists(shape_path):
                    self.shapes.append(torch.load(shape_path, map_location=self.device))

            if load_wells:
                self.wells.append(
                    ops.load(
                        os.path.join(path, str(scale), M_FILE),
                        self.device,
                        as_type=torch.Tensor,
                    )
                )

            scale += 1

        return scale
