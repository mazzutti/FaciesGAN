"""Parallel LAPGAN implementation for training multiple scales simultaneously.

This module extends the standard FaciesGAN to support parallel training of
multiple pyramid scales. Instead of training scales sequentially, this
implementation can train multiple scales at once using separate optimizers
and discriminators for each scale.
"""

import math
import os
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from config import AMP_FILE, D_FILE, G_FILE, M_FILE, SHAPE_FILE
from models.base import FaciesGAN, IterableMetrics
from models.torch import utils
from models.torch.discriminator import TorchDiscriminator
from models.torch.generator import TorchGenerator
from options import TrainningOptions
from trainning.metrics import DiscriminatorMetrics, GeneratorMetrics, ScaleMetrics


def unwrap_ddp(module: nn.Module) -> nn.Module:
    """Return the inner module if wrapped in ``DistributedDataParallel``."""
    return getattr(module, "module", module)  # type: ignore[no-any-return]


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
        use_ddp: bool = False,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the parallel FaciesGAN model.

        Parameters
        ----------
        device : torch.device
            Primary device for computation.
        options : TrainningOptions
            Training configuration containing hyperparameters.
        noise_channels : int, optional
            Number of input noise channels, by default 3.
        use_ddp : bool, optional
            When ``True``, each per-scale sub-module is wrapped with
            ``DistributedDataParallel`` after creation.  Requires that
            ``torch.distributed`` has been initialised before training
            starts.  Defaults to ``False``.
        """
        nn.Module.__init__(self)  # type: ignore
        # Initialize framework-agnostic attributes in the base class
        super().__init__(options, noise_channels, *args, **kwargs)

        self.zero_padding = int(options.num_layer * math.floor(options.kernel_size / 2))

        # Framework-specific attributes
        self.device = device

        # Multi-GPU setup via manual gradient all-reduce.
        self.use_ddp = use_ddp

        # AMP (Automatic Mixed Precision) for faster CUDA training.
        # Only the *generator* path uses AMP.  The discriminator stays
        # in fp32 because:
        #   (a) WGAN critic scores need full precision for a good
        #       Wasserstein distance approximation,
        #   (b) the gradient penalty uses create_graph=True; GradScaler's
        #       scale factor can cause inf/NaN in the second-order
        #       backward, making the scaler silently skip steps,
        #   (c) the disc is small — fp16 saves negligible time.
        self._use_amp = device.type == "cuda"
        self._grad_scaler_g = GradScaler(enabled=self._use_amp)

        # torch.compile gives a meaningful speedup on CUDA when gradient
        # checkpointing is OFF (the two features are incompatible because
        # compiled graphs reorder saved tensors, breaking checkpoint
        # recomputation metadata checks).  Enabled by default on CUDA;
        # pass ``--no-compile`` to disable.
        self._use_compile = device.type == "cuda" and getattr(
            options, "compile_backend", True
        )

        # Pre-allocate constant zero scalars on device so the hot path
        # avoids repeated small CUDA allocations.
        self._zero_scalar = torch.tensor(0.0, device=device)
        self._zero_one = torch.zeros(1, device=device)

        # Create framework objects via the base class helper (calls build_* hooks)
        self.setup_framework()

    def __call__(self, *args: Any, **kwds: Any) -> ScaleMetrics[torch.Tensor]:
        return nn.Module.__call__(self, *args, **kwds)

    def device_for_scale(self, scale: int) -> torch.device:
        """Return the primary device (all modules live there).

        With DataParallel the modules are replicated at forward time;
        their parameters always reside on ``self.device``.

        Parameters
        ----------
        scale : int
            Pyramid scale index (unused — kept for API compatibility).

        Returns
        -------
        torch.device
            The primary CUDA device.
        """
        return self.device

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
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
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
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dict for conditioning, keyed by scale.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic  tensors dict for conditioning, keyed by scale.

        Returns
        -------
        tuple[DiscriminatorMetrics[torch.Tensor], dict[Any, Any] | None]:
            Container with total, real, fake and gp losses, and optional gradients dict.
        """

        d_real = self.discriminator(scale, real.to(self.device))
        noises = self.get_pyramid_noise(scale, indexes, wells_pyramid, seismic_pyramid)
        fake = self.generate_fake(noises, scale)
        d_fake = self.discriminator(scale, fake.detach())  # type: ignore

        # WGAN losses (matching MLX implementation).
        real_loss = -d_real.mean()
        fake_loss = d_fake.mean()
        gp = self.compute_gradient_penalty(scale, real, fake.detach())

        total = real_loss + fake_loss + gp
        return (
            DiscriminatorMetrics(
                total=total,
                real=real_loss.detach(),
                fake=fake_loss.detach(),
                gp=gp.detach(),
            ),
            None,
        )

    def optimize_discriminator(
        self,
        indexes: list[int],
        optimizers: dict[int, torch.optim.Optimizer],
        facies_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[DiscriminatorMetrics[torch.Tensor], ...]:
        """Discriminator optimization with pre-batched fake generation.

        All D fakes per scale are generated in a single batched generator
        forward (batch = D × B) before the D-step loop begins.  This is
        safe because generator weights are frozen during D optimisation.

        ``torch.compile(dynamic=True)`` on the gen blocks prevents
        recompilation when the batch dimension changes between the D and
        G phases.

        Lazy gradient penalty (``gp_interval``) amortises the expensive
        ``create_graph=True`` double backward.

        Returns
        -------
        tuple[DiscriminatorMetrics[torch.Tensor], ...]
            Metrics from the last discriminator step for each active scale.
        """
        D = self.discriminator_steps
        if D <= 0:
            return ()

        sorted_scales = sorted(self.active_scales)
        B = len(indexes)

        # ── Pre-generate all D fakes per scale in one batched forward ──
        # Generator weights are frozen during D optimisation, so all
        # fakes can be produced upfront.  One forward with batch = D*B
        # is cheaper than D separate forwards with batch = B (fewer
        # kernel launches).
        prefaked: dict[int, list[torch.Tensor]] = {}
        with torch.no_grad():
            for scale in sorted_scales:
                noise_sets = [
                    self.get_pyramid_noise(
                        scale, indexes, wells_pyramid, seismic_pyramid
                    )
                    for _ in range(D)
                ]
                # Concatenate along batch dim: each level gets D*B samples.
                batched_noises: list[torch.Tensor] = [
                    torch.cat([noise_sets[k][lvl] for k in range(D)], dim=0)
                    for lvl in range(scale + 1)
                ]
                amps = self.get_noise_aplitude(scale)
                batched_fake = self.generator(batched_noises, amps, stop_scale=scale)
                # Split back into D chunks of size B.
                prefaked[scale] = list(batched_fake.split(B, dim=0))  # type: ignore[arg-type]

        # ── D-step loop using pre-generated fakes ──
        step_metrics: list[DiscriminatorMetrics[torch.Tensor]] = []
        for step_idx in range(D):
            step_metrics = []
            self._disc_step_counter += 1
            compute_gp = (self._disc_step_counter % self.gp_interval) == 0

            for scale in sorted_scales:
                fake = prefaked[scale][step_idx]
                real = facies_pyramid[scale].to(self.device, non_blocking=True)

                # Discriminator forward runs in fp32 (no autocast) —
                # WGAN critic scores need full precision and the GP
                # backward is incompatible with GradScaler.
                d_real = self.discriminator(scale, real)
                d_fake = self.discriminator(scale, fake)

                # WGAN losses (matching MLX implementation).
                real_loss = -d_real.mean()
                fake_loss = d_fake.mean()

                if compute_gp:
                    gp = (
                        self.compute_gradient_penalty(scale, real, fake.detach())
                        * self.gp_interval
                    )
                else:
                    gp = self._zero_scalar

                total = real_loss + fake_loss + gp

                self.update_discriminator_weights(scale, optimizers[scale], total, None)

                step_metrics.append(
                    DiscriminatorMetrics(
                        total=total.detach(),
                        real=real_loss.detach(),
                        fake=fake_loss.detach(),
                        gp=gp.detach(),
                    )
                )

        return tuple(step_metrics)

    def optimize_generator(
        self,
        indexes: list[int],
        optimizers: dict[int, torch.optim.Optimizer],
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[GeneratorMetrics[torch.Tensor], ...]:
        """Generator optimization with per-scale parameter freezing.

        Overrides the base ``optimize_generator`` to temporarily freeze
        all active-group gen blocks **except** the one being trained.
        This prevents ``backward()`` from computing (and then discarding)
        gradients for gen blocks that participate in the progressive
        forward pass but whose optimizer is not being stepped.

        For *S* parallel scales training simultaneously, the base
        implementation computes gradients for 1+2+…+S = S(S+1)/2 block
        instances per G step.  This override reduces that to just *S*
        block instances — a ~(S+1)/2× speedup in backward computation
        (e.g. ~4× for 7 parallel scales).

        Returns
        -------
        tuple[GeneratorMetrics[torch.Tensor], ...]
            Metrics from the last generator step for each active scale.
        """
        sorted_scales = sorted(self.active_scales)
        G = self.generator_steps
        if G <= 0:
            return ()

        step_metrics: list[GeneratorMetrics[torch.Tensor]] = []

        for _ in range(G):
            step_metrics = []

            # Freeze all active gen blocks up front.  Blocks from
            # previous groups are already frozen by freeze_generator_scales.
            for s in sorted_scales:
                self.generator.gens[s].requires_grad_(False)

            for scale in sorted_scales:
                if scale >= len(facies_pyramid):
                    continue

                # Ensure noise amplitudes have been initialized.
                if len(self.noise_amps) < scale + 1:
                    raise RuntimeError(
                        f"noise_amp not initialized for scale {scale}. "
                        "Call the project's noise initialization before training."
                    )

                # Unfreeze only the target scale's gen block.
                self.generator.gens[scale].requires_grad_(True)

                result, gradients = self.compute_generator_metrics(
                    indexes,
                    scale,
                    facies_pyramid[scale],
                    rec_in_pyramid,
                    wells_pyramid,
                    masks_pyramid,
                    seismic_pyramid,
                )
                metrics = cast(GeneratorMetrics[torch.Tensor], result)

                self.update_generator_weights(
                    scale, optimizers[scale], metrics.total, gradients
                )

                # Re-freeze so the next scale's backward skips this block.
                self.generator.gens[scale].requires_grad_(False)

                step_metrics.append(
                    GeneratorMetrics(
                        total=metrics.total.detach(),
                        fake=metrics.fake.detach(),
                        rec=metrics.rec.detach(),
                        well=metrics.well.detach(),
                        div=metrics.div.detach(),
                    )
                )

            # Restore requires_grad on all active blocks for next step.
            for s in sorted_scales:
                self.generator.gens[s].requires_grad_(True)

            # Reset scaler state so the next G-step can call unscale_()
            # on the same optimizers.  One update per G-step (G calls)
            # instead of per-scale (G×S calls).
            self._grad_scaler_g.update()

        return tuple(step_metrics)

    def compute_generator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: torch.Tensor,
        rec_in_pyramid: dict[int, torch.Tensor] = {},
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> tuple[
        GeneratorMetrics[torch.Tensor] | IterableMetrics[torch.Tensor],
        dict[str, Any] | None,
    ]:
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
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dict for conditioning, keyed by scale.
        masks_pyramid (dict[int, torch.Tensor], optional):
            Well mask tensors dict for conditioning, keyed by scale.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic tensors dict for conditioning, keyed by scale.

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

        with autocast("cuda", enabled=self._use_amp):
            # Generate diversity candidates (framework-agnostic forward)
            fake_samples = self.generate_diverse_samples(
                indexes,
                scale,
                wells_pyramid,
                seismic_pyramid,
            )
            fake = fake_samples[0]

            # WGAN generator adversarial loss: -E[D(fake)].
            # Temporarily disable gradient tracking on discriminator
            # parameters so backward() only updates generator params.
            disc_module = self.discriminator.discs[scale]
            disc_module.requires_grad_(False)
            adv = self.compute_adversarial_loss(scale, fake)
            disc_module.requires_grad_(True)
            mask = masks_pyramid.get(scale, None)
            well = wells_pyramid.get(scale, None)
            well = self.compute_masked_loss(
                fake,
                real,
                well,
                mask,
            )
            div = self.compute_diversity_loss(fake_samples)
            rec_in = rec_in_pyramid[scale]
            rec_loss = self.compute_recovery_loss(
                indexes,
                scale,
                real,
                rec_in,
                wells_pyramid,
                seismic_pyramid,
            )

            well_d = well.to(adv.device)
            rec_d = rec_loss.to(adv.device)
            div_d = div.to(adv.device)
            total = adv + well_d + rec_d + div_d

        del fake_samples  # free diversity candidates early

        # Detach component losses — backward() will be called on total;
        # the individual components are only needed as scalar logs.
        adv = adv.detach()
        rec_d = rec_d.detach()
        well_d = well_d.detach()
        div_d = div_d.detach()

        metrics = GeneratorMetrics(
            total=total,
            fake=adv,
            rec=rec_d,
            well=well_d,
            div=div_d,
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

    def split_tensor(self, tensor: torch.Tensor, chunks: int) -> list[torch.Tensor]:
        """Split a tensor into ``chunks`` equal parts along the batch dimension.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to split (batch dimension is dim 0).
        chunks : int
            Number of equal-sized chunks.

        Returns
        -------
        list[torch.Tensor]
            List of ``chunks`` tensors.
        """
        return list(torch.chunk(tensor, chunks, dim=0))

    def cat_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate tensors along the batch (first) dimension."""
        return torch.cat(tensors, dim=0)

    def compute_adversarial_loss(self, scale: int, fake: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss.

        Parameters
        ----------
        scale : int
            Pyramid scale index.
        fake : torch.Tensor
            Generated tensor.

        Returns
        -------
        torch.Tensor
            Negative mean discriminator score.
        """
        return -self.discriminator.discs[scale](fake.to(self.device)).mean()

    def compute_diversity_loss(self, fake_samples: list[torch.Tensor]) -> torch.Tensor:
        """Compute diversity loss across multiple generated `fake_samples`.

        Encourages different noise inputs to produce diverse outputs by
        penalizing small pairwise distances between flattened samples.

        Uses a vectorized approach: stacks all samples into an ``(N, -1)``
        matrix, computes the full pairwise squared-distance matrix with a
        single matmul, and extracts the upper-triangular pairs.

        Parameters:
            fake_samples (list[torch.Tensor]): List of generated samples to
                compare for diversity.

        Returns:
            torch.Tensor: Scalar diversity loss; zero when disabled or when
                fewer than two samples are provided.
        """
        if self.lambda_diversity <= 0 or len(fake_samples) < 2:
            return self._zero_scalar
        n = len(fake_samples)
        if n == 2:
            # Fast path for the common N=2 case: single pairwise distance,
            # avoids triu_indices / sq_norms / indexing overhead.
            diff = fake_samples[0] - fake_samples[1]
            pair_dist = (diff * diff).mean()
            return self.lambda_diversity * torch.exp(-pair_dist * 10)
        # Stack into (N, D) where D = B*C*H*W — single flatten + stack.
        flat = torch.stack([s.flatten() for s in fake_samples])  # (N, D)
        # Pairwise squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        sq_norms = (flat * flat).sum(dim=1)  # (N,)
        # Only compute upper-triangle pairs (i < j)
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=flat.device)
        pair_dists = (
            sq_norms[idx_i] + sq_norms[idx_j] - 2 * (flat[idx_i] * flat[idx_j]).sum(1)
        ) / flat.shape[
            1
        ]  # mean over D
        div_loss = torch.exp(-pair_dists * 10).mean()
        return self.lambda_diversity * div_loss

    def compute_gradient_penalty(
        self, scale: int, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute the gradient penalty for WGAN-GP style regularization.

        The gradient penalty uses ``autograd.grad(create_graph=True)``
        which requires float32 tensors, so AMP autocast is explicitly
        disabled here.

        Args:
            scale (int): Discriminator scale index used for the penalty.
            real (torch.Tensor): Real samples tensor.
            fake (torch.Tensor): Fake samples tensor.

        Returns:
            torch.Tensor: Scalar gradient penalty term.
        """
        disc = unwrap_ddp(self.discriminator.discs[scale])
        with autocast("cuda", enabled=False):
            return utils.calc_gradient_penalty(
                disc,
                real.float(),
                fake.float(),
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
            return self._zero_one
        mask_d = mask.to(fake.device, non_blocking=True)
        return self.well_loss_penalty * F.mse_loss(
            fake * mask_d, real.to(fake.device, non_blocking=True) * mask_d
        )

    def compute_recovery_loss(
        self,
        indexes: list[int],
        scale: int,
        real: torch.Tensor,
        rec_in: torch.Tensor,
        wells_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
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
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dictionary for conditioning, keyed by scale.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic tensors dictionary for conditioning, keyed by scale.

        Returns:
            torch.Tensor: Scalar reconstruction loss weighted by `self.alpha`,
                or zero when recovery is disabled.
        """
        if self.alpha == 0:
            return self._zero_one
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
            in_noise=rec_in.to(self.device, non_blocking=True),
            start_scale=scale,
            stop_scale=scale,
        )
        rec_loss = self.alpha * F.mse_loss(rec, real.to(rec.device, non_blocking=True))
        return rec_loss

    def finalize_discriminator_scale(self, scale: int) -> None:
        """Finalize discriminator block after creation.

        Applies weight initialization, moves the block to the primary
        device, and broadcasts parameters from rank 0 when DDP is
        enabled.

        Note: Discriminators are **not** wrapped with DDP because the
        WGAN-GP gradient penalty uses ``autograd.grad(create_graph=True)``
        which is incompatible with DDP's in-place backward hooks.
        Gradients are instead all-reduced manually in
        :meth:`update_discriminator_weights`.

        Args:
            scale (int): Index of the discriminator scale to finalize.
        """
        self.discriminator.discs[scale].apply(utils.weights_init)
        self.discriminator.discs[scale] = self.discriminator.discs[scale].to(
            self.device
        )
        self.discriminator.discs[scale] = self.discriminator.discs[scale].to(  # type: ignore[call-overload]
            memory_format=torch.channels_last
        )
        if self.use_ddp:
            # Broadcast initial weights from rank 0 (DDP constructor does
            # this automatically for wrapped modules; we replicate it here).
            for p in self.discriminator.discs[scale].parameters():
                dist.broadcast(p.data, src=0)
        # Discriminator is intentionally NOT compiled even when
        # ``_use_compile`` is True.  The R1 gradient penalty uses
        # ``autograd.grad(create_graph=True)`` (double backward) which
        # is incompatible with torch.compile.  The disc is small anyway
        # so the throughput impact is negligible.

    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Finalize generator block after creation.

        Either initialize weights for a freshly reinitialized block or copy
        weights from the previous scale, then move to primary device and
        broadcast parameters from rank 0 when DDP is enabled.

        Note: Generators are **not** wrapped with DDP because the
        multi-scale forward pass shares intermediate tensors across
        scales, and DDP's in-place backward hooks corrupt the
        computation graph.  Gradients are instead all-reduced manually
        in :meth:`update_generator_weights`.

        Args:
            scale (int): Index of the generator scale to finalize.
            reinit (bool): Whether to initialize weights instead of copying.
        """
        if reinit:
            self.generator.gens[scale].apply(utils.weights_init)
        else:
            # Attempt to copy parameters from previous scale but only for
            # matching parameter shapes. This handles cases where feature
            # counts change between scales (e.g., parallel initialization)
            prev = self.generator.gens[scale - 1]
            src_state = prev.state_dict()
            tgt_state = self.generator.gens[scale].state_dict()

            # Build filtered state with only keys present in both and with
            # identical tensor shapes.
            filtered: dict[str, torch.Tensor] = {}
            for k, v in src_state.items():
                if k in tgt_state and v.shape == tgt_state[k].shape:
                    filtered[k] = v

            if filtered:
                # Load only the matching parameters; allow missing keys.
                self.generator.gens[scale].load_state_dict(filtered, strict=False)
            else:
                # No compatible parameters to copy; fall back to weight init.
                self.generator.gens[scale].apply(utils.weights_init)

        self.generator.gens[scale] = self.generator.gens[scale].to(self.device)
        self.generator.gens[scale] = self.generator.gens[scale].to(  # type: ignore[call-overload]
            memory_format=torch.channels_last
        )
        if self.use_ddp:
            # Broadcast initial weights from rank 0.
            for p in self.generator.gens[scale].parameters():
                dist.broadcast(p.data, src=0)
        # torch.compile is incompatible with torch.utils.checkpoint:
        # the compiled graph reorders saved tensors, causing metadata
        # mismatches during checkpoint recomputation.  Skip compile on
        # gen blocks when gradient checkpointing is active.
        if self._use_compile and not getattr(
            self.generator, "use_gradient_checkpointing", False
        ):
            self.generator.gens[scale] = torch.compile(  # type: ignore[assignment]
                self.generator.gens[scale],
            )

    def forward(
        self,
        generator_optimizers: dict[int, torch.optim.Optimizer],
        discriminator_optimizers: dict[int, torch.optim.Optimizer],
        indexes: list[int],
        facies_pyramid: dict[int, torch.Tensor],
        rec_in_pyramid: dict[int, torch.Tensor],
        wells_pyramid: dict[int, torch.Tensor] = {},
        masks_pyramid: dict[int, torch.Tensor] = {},
        seismic_pyramid: dict[int, torch.Tensor] = {},
    ) -> ScaleMetrics[torch.Tensor]:
        """Perform a forward pass and compute scale metrics.

        Parameters
        ----------
        indexes (list[int]):
            List of batch/sample indices used to generate noise.
        facies_pyramid (dict[int, torch.Tensor]):
            Dictionary mapping scale indices to real tensor samples.
        rec_in_pyramid (dict[int, torch.Tensor]):
            Dictionary mapping scale indices to reconstruction input tensors.
        wells_pyramid (dict[int, torch.Tensor], optional):
            Wells tensors dictionary for conditioning, keyed by scale.
        masks_pyramid (dict[int, torch.Tensor], optional):
            Well masks dictionary for conditioning, keyed by scale.
        seismic_pyramid (dict[int, torch.Tensor], optional):
            Seismic tensors dictionary for conditioning, keyed by scale.
        Returns
        -------
        ScaleMetrics[torch.Tensor]:
            Container with discriminator and generator metrics for the scale.
        """
        disc_metrics_tuple = self.optimize_discriminator(
            indexes,
            discriminator_optimizers,
            facies_pyramid,
            wells_pyramid,
            seismic_pyramid,
        )
        gen_metrics_tuple = self.optimize_generator(
            indexes,
            generator_optimizers,
            facies_pyramid,
            rec_in_pyramid,
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
        )

        # Metrics from both optimizers are already detached; pass through
        # directly without redundant .detach() calls.

        # Convert tuples to dicts mapping scale index to metrics
        discriminator_metrics = {
            scale: disc_metrics_tuple[i]
            for i, scale in enumerate(sorted(self.active_scales))
        }
        generator_metrics = {
            scale: gen_metrics_tuple[i]
            for i, scale in enumerate(sorted(self.active_scales))
        }

        return ScaleMetrics(
            discriminator=discriminator_metrics,
            generator=generator_metrics,
        )

    def generate_fake(self, noises: list[torch.Tensor], scale: int) -> torch.Tensor:
        """Generate a fake sample at the requested `scale` using `noises`.

        Uses ``no_grad`` to avoid tracking generator computation during
        discriminator optimization.  ``inference_mode`` cannot be used here
        because WGAN-GP's gradient penalty passes the resulting tensor
        through the discriminator with ``create_graph=True``, and inference
        tensors cannot be saved for backward.

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
        spatial_shape = self.get_noise_shape(scale, use_base_channel=False)
        noise_channels = self.gen_input_channels
        tensors_to_concat: list[torch.Tensor] = []

        w: torch.Tensor | None = None
        s: torch.Tensor | None = None

        if well is not None:
            w = well[indexes].to(self.device)
            noise_channels -= w.shape[1]

        if seismic is not None:
            s = seismic[indexes].to(self.device)
            noise_channels -= s.shape[1]

        z = utils.generate_noise(
            (noise_channels, *spatial_shape),
            num_samp=batch,
            device=self.device,
        )
        tensors_to_concat.append(z)

        if w is not None:
            tensors_to_concat.append(w)

        if s is not None:
            tensors_to_concat.append(s)

        if len(tensors_to_concat) > 1:
            z = self.concatenate_tensors(tensors_to_concat)

        return self.generate_padding(z, value=0)

    def get_rec_noise(self, scale: int) -> list[torch.Tensor]:
        return self.rec_noise[: scale + 1]

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

        Handles DDP-wrapped modules by loading into the inner
        ``.module`` when applicable.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the discriminator scale to load.
        """
        disc_path = os.path.join(scale_path, D_FILE)
        if os.path.exists(disc_path):
            unwrap_ddp(self.discriminator.discs[scale]).load_state_dict(
                torch.load(disc_path, map_location=self.device)
            )

    def load_generator_state(self, scale_path: str, scale: int) -> None:
        """Load generator state dict for the latest generator in the scale.

        Handles DDP-wrapped modules by loading into the inner
        ``.module`` when applicable.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the generator scale to load (unused here).
        """
        gen_path = os.path.join(scale_path, G_FILE)
        if os.path.exists(gen_path):
            unwrap_ddp(self.generator.gens[scale]).load_state_dict(
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

    @staticmethod
    def _allreduce_grads(module: nn.Module) -> None:
        """Average gradients across all DDP ranks with a single all-reduce.

        Every trainable parameter is included in the flattened buffer even
        if ``backward()`` left its ``.grad`` as ``None`` (which can happen
        when a parameter was not part of the computation graph on a given
        rank).  A zero tensor is substituted for missing gradients so that
        **all ranks always call ``all_reduce`` on identically-sized
        buffers**.  Without this, one rank could skip the collective while
        another blocks — an immediate NCCL deadlock.
        """
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return
        # Materialise missing grads as zeros so the flat buffer size is
        # identical across ranks regardless of local backward() paths.
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
        grads = [p.grad for p in params]
        flat = torch._utils._flatten_dense_tensors(grads)  # type: ignore[attr-defined]
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)  # type: ignore[arg-type]
        for g, synced in zip(  # type: ignore[assignment]
            grads, torch._utils._unflatten_dense_tensors(flat, grads)  # type: ignore[attr-defined]
        ):
            g.copy_(synced)  # type: ignore[arg-type]

    def update_discriminator_weights(
        self,
        scale: int,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        gradients: Any | None,
    ) -> None:
        """Perform standard PyTorch discriminator optimization step (fp32).

        The discriminator does **not** use AMP / GradScaler because the
        gradient penalty's ``create_graph=True`` backward is
        incompatible with loss scaling (the scale factor leaks into
        second-order gradient terms, causing frequent inf/NaN and
        silently skipped optimizer steps).

        When running under DDP, discriminator gradients are manually
        all-reduced across ranks because the discriminator is not
        wrapped with DDP (see :meth:`finalize_discriminator_scale`).
        """
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.use_ddp:
            self._allreduce_grads(self.discriminator.discs[scale])
        optimizer.step()

    def update_generator_weights(
        self,
        scale: int,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        gradients: Any | None,
    ) -> None:
        """Perform standard PyTorch generator optimization step with AMP.

        When running under DDP, generator gradients are manually
        all-reduced across ranks because the generator is not wrapped
        with DDP (see :meth:`finalize_generator_scale`).
        """
        optimizer.zero_grad(set_to_none=True)
        self._grad_scaler_g.scale(loss).backward()  # type: ignore[no-untyped-call]
        if self.use_ddp:
            self._allreduce_grads(self.generator.gens[scale])
        self._grad_scaler_g.unscale_(optimizer)
        self._grad_scaler_g.step(optimizer)
        # NOTE: scaler.update() is called once per G iteration in
        # optimize_generator, not here — calling it per-scale would
        # adjust the scale factor 7× too often.

    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Save discriminator state dict for `scale` to `scale_path`.

        Unwraps DDP to save clean state dicts without ``module.``
        prefix.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the discriminator scale to save.
        """
        if scale < len(self.discriminator.discs):
            discriminator_path = os.path.join(scale_path, f"{D_FILE}")
            torch.save(
                unwrap_ddp(self.discriminator.discs[scale]).state_dict(),
                discriminator_path,
            )

    def save_generator_state(self, scale_path: str, scale: int) -> None:
        """Save generator state dict for `scale` to `scale_path`.

        Unwraps DDP to save clean state dicts without ``module.``
        prefix.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the generator scale to save.
        """
        if scale < len(self.generator.gens):
            generator_path = os.path.join(scale_path, f"{G_FILE}")
            torch.save(
                unwrap_ddp(self.generator.gens[scale]).state_dict(), generator_path
            )

    def save_shape(self, scale_path: str, scale: int) -> None:
        """Save shape tensor for `scale` to disk at `scale_path`.

        Args:
            scale_path (str): Directory path for the given scale.
            scale (int): Index of the shape to save.
        """
        if scale < len(self.shapes):
            shape_path = os.path.join(scale_path, SHAPE_FILE)
            torch.save(self.shapes[scale], shape_path)
