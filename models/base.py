import math
import os
from abc import ABC, abstractmethod
from typing import Any, Generic

from config import AMP_FILE, D_FILE, G_FILE, M_FILE, SHAPE_FILE
from trainning.metrics import (
    DiscriminatorMetrics,
    GeneratorMetrics,
    IterableMetrics,
    ScaleMetrics,
)
from models.discriminator import Discriminator
from models.generator import Generator
from options import TrainningOptions
from typedefs import TModule, TOptimizer, TScheduler, TTensor


class FaciesGAN(ABC, Generic[TTensor, TModule, TOptimizer, TScheduler]):
    """Framework-agnostic FaciesGAN base class.

    Responsibilities:
    - Store training/configuration parameters from `TrainningOptions`.
    - Provide framework-agnostic utilities (scale bookkeeping, feature
        calculation, orchestration of scale initialization).
    - Provide generic checkpoint traversal (`save_scale` / `load`) while
        delegating actual serialization to concrete subclasses via hooks.

    Subclass contract (short):
    - Implement `build_generator`, `create_discriminators_container`, and
        `move_to_device` for framework object construction/device placement.
    - Implement I/O hooks named `save_*`, `load_*`, and presence checks
        used by the base `save_scale` / `load` orchestration.

    Example
    -------
    A framework-specific subclass (PyTorch|MLX) should call:

            super().__init__(options, wells, seismic, noise_channels)
            self.setup_framework(device)

    After that the base will manage scale-level orchestration while the
    subclass constructs modules and performs framework-specific ops.

    """

    # The framework-specific generator instance
    generator: Generator[TTensor, TModule]

    # The framework-specific discriminator instance
    discriminator: Discriminator[TTensor, TModule]

    # Padding size for noise tensors
    zero_padding: int

    def __init__(
        self,
        options: TrainningOptions,
        noise_channels: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the FaciesGAN base with training options.

        Parameters
        ----------
        options : TrainningOptions
            Training options containing hyperparameters and configuration.
        noise_channels : int, optional
            Number of noise channels for the generator input (default is 3).

        args : Any
            Additional positional arguments (not used).
        kwargs : Any
            Additional keyword arguments (not used).
        """

        # Basic training / architecture parameters (framework-agnostic)
        self.num_parallel_scales = options.num_parallel_scales

        # image channels
        self.num_img_channels = options.num_img_channels

        # input/output channels
        self.disc_input_channels: int = self.num_img_channels

        # output channels
        self.disc_output_channels: int = self.num_img_channels

        # generator channels
        self.gen_input_channels: int = noise_channels

        # output channels
        self.gen_output_channels: int = self.num_img_channels

        # base channels for generator (adjusted for conditioning)
        self.base_channel = self.num_img_channels

        # training hyperparameters
        self.discriminator_steps = options.discriminator_steps

        # generator hyperparameters
        self.generator_steps = options.generator_steps

        # loss weights
        self.lambda_grad = options.lambda_grad

        # other loss/configuration params
        self.alpha = options.alpha

        # well/mask loss weight
        self.well_loss_penalty = options.well_loss_penalty

        # diversity loss params
        self.lambda_diversity = options.lambda_diversity

        # number of diversity samples
        self.num_diversity_samples = options.num_diversity_samples

        # network sizing params
        self.num_feature = options.num_feature

        # minimum number of features
        self.min_num_feature = options.min_num_feature

        # network architecture params
        self.num_layer = options.num_layer

        # kernel size
        self.kernel_size = options.kernel_size

        # padding size
        self.padding_size = options.padding_size

        # pyramid scales
        self.shapes: list[tuple[int, ...]] = []

        # noise/reconstruction data
        self.rec_noise: list[TTensor] = []

        # noise amplitudes
        self.noise_amps: list[float] = []

        # active scales set
        self.active_scales: set[int] = set()

        # generator learning rate
        self.lr_g = options.lr_g

        # discriminator learning rate
        self.lr_d = options.lr_d

        # discriminator learning rate
        self.beta1 = options.beta1

        # learning rate decay milestone
        self.lr_decay = options.lr_decay

        # learning rate gamma
        self.gamma = options.gamma

        # generator optimizers
        self.generator_optimizers: dict[int, TOptimizer] = {}

        # discriminator optimizers
        self.discriminator_optimizers: dict[int, TOptimizer] = {}

        # generator schedulers
        self.generator_schedulers: dict[int, TScheduler] = {}

        # discriminator schedulers
        self.discriminator_schedulers: dict[int, TScheduler] = {}

    def backward_grads(
        self,
        losses: list[TTensor],
        gradients: list[Any] | None = None,
    ) -> TTensor | None:
        """Framework-specific backward operation for aggregated losses.

        Subclasses should implement this to call their framework's backward
        API (e.g., `torch.autograd.backward(losses)`, `mx.eval(losses)`). Keeping this
        abstract prevents importing heavy frameworks into the base module.

        Parameters
        ----------
        losses : list[Any]
            List of tensors from multiple scales to backpropagate.
        gradients : list[Any] | None, optional
            Optional list of parameter gradients/states to populate during the backward call (default is None).

        Returns
        -------
        TTensor | None
            Optional aggregated tensor returned by the backward call.


        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement backward_grads")

    @abstractmethod
    def __call__(
        self, *args: Any, **kwds: Any
    ) -> ScaleMetrics[TTensor] | tuple[IterableMetrics[TTensor], ...]:
        """Framework-specific forward method for training step.

        Parameters
        ----------
        args : Any
            Positional arguments for the forward call.
        kwds : Any
            Keyword arguments for the forward call.
        Returns
        -------
        ScaleMetrics[TTensor]
            Computed scale metrics from the forward pass.
        """
        raise NotImplementedError("Subclasses must implement __call__")

    @abstractmethod
    def build_discriminator(self) -> Discriminator[TTensor, TModule]:
        """Construct and return a framework-specific discriminator object.

        Concrete subclasses must implement this factory to create the
        discriminator instance (but should not move it to any device here).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.

        Returns
        -------
        Discriminator[TTensor, TModule]
            Constructed discriminator instance.
        """
        raise NotImplementedError("Subclasses must implement build_discriminator")

    @abstractmethod
    def build_generator(self) -> Generator[TTensor, TModule]:
        """Construct and return a framework-specific generator object.

        Concrete subclasses must implement this factory to create the
        generator instance (but should not move it to any device here).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.

        Returns
        -------
        Generator[TTensor, TModule]
            Constructed generator instance.
        """
        raise NotImplementedError("Subclasses must implement build_generator")

    @abstractmethod
    def concatenate_tensors(self, tensors: list[TTensor]) -> TTensor:
        """Concatenate a list of tensors along a specified dimension.

        Parameters
        ----------
        tensors : list[TTensor]
            List of tensors to concatenate.
        Returns
        -------
        TTensor
            Concatenated tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement concatenate_tensors")

    @abstractmethod
    def compute_diversity_loss(self, fake_samples: list[TTensor]) -> TTensor:
        """Return diversity loss tensor computed from `fake_samples`.

        Parameters
        ----------
        fake_samples : list[TTensor]
            Generated tensor samples used to compute diversity loss.

        Returns
        -------
        TTensor
            Diversity loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_diversity_loss")

    @abstractmethod
    def compute_gradient_penalty(
        self, scale: int, real: TTensor, fake: TTensor
    ) -> TTensor:
        """Return gradient penalty tensor for `scale` computed by subclass.

        Parameters
        ----------
        scale : int
            Scale index for which to compute the gradient penalty.
        real : TTensor
            Real tensor samples for the current scale.
        fake : TTensor
            Generated fake tensor samples for the current scale.

        Returns
        -------
        TTensor
            Gradient penalty loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_gradient_penalty")

    @abstractmethod
    def compute_masked_loss(
        self,
        fake: TTensor,
        real: TTensor,
        well: TTensor,
        mask: TTensor,
    ) -> TTensor:
        """Return well/mask-based loss tensor for `fake` at `scale`.

        Parameters
        ----------
        fake : TTensor
            Generated tensor samples for the current scale.
        real : TTensor
            Real tensor samples for the current scale.
        well : TTensor
            Well-conditioning tensor for the current scale.
        mask : TTensor
            Well mask tensor for the current scale.

        Returns
        -------
        TTensor
            Masked loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_masked_loss")

    @abstractmethod
    def compute_recovery_loss(
        self,
        indexes: list[int],
        scale: int,
        real: TTensor,
        rec_in: TTensor,
        wells_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
    ) -> TTensor:
        """Return reconstruction loss tensor for the provided inputs.

        Parameters
        ----------
        indexes : list[int]
            List of batch/sample indices used to generate noise.
        scale : int
            Scale index for which to compute the recovery loss.
        real : TTensor
            Real tensor samples for the current scale.
        rec_in : TTensor
            Input tensor for reconstruction at the current scale.
        wells_pyramid : tuple[TTensor, ...], optional
            Well-conditioning tensor tuple for the current scale.
        seismic_pyramid : tuple[TTensor, ...], optional
            Seismic-conditioning tensor tuple for the current scale.
        Returns
        -------
        TTensor
            Reconstruction loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_recovery_loss")

    @abstractmethod
    def finalize_discriminator_scale(self, scale: int) -> None:
        """Finalize discriminator scale after creation and optional device move.

        Parameters
        ----------
        scale : int
            Scale index that was just created.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement finalize_discriminator_scale"
        )

    @abstractmethod
    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Finalize generator scale after creation and optional device move.

        If `reinit` is True subclasses should initialize weights for the new
        block; otherwise they should copy weights from the previous block.

        Parameters
        ----------
        scale : int
            Scale index that was just created.
        reinit : bool
            Whether to reinitialize weights or copy from previous scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement finalize_generator_scale")

    @abstractmethod
    def generate_fake(self, noises: list[TTensor], scale: int) -> TTensor:
        """Generate fake images from `noises` without tracking gradients.

        Subclasses should implement this using their framework's no-grad
        mechanism (e.g., `torch.no_grad()`), returning the produced tensor.
        """
        raise NotImplementedError("Subclasses must implement generate_fake")

    @abstractmethod
    def generate_padding(self, z: TTensor, value: int = 0) -> TTensor:
        """Apply padding to a noise tensor.

        Parameters
        ----------
        z : TTensor
            Input noise tensor to pad.
        value : int, optional
            Padding fill value (default is 0).

        Returns
        -------
        TTensor
            Padded tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement generate_padding")

    @abstractmethod
    def get_noise_shape(
        self, scale: int, use_base_channel: bool = True
    ) -> tuple[int, ...]:
        """Get the noise tensor shape for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index for which to get the noise shape.

        Returns
        -------
        tuple[int, ...]
            Shape of the noise tensor for the specified scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement get_noise_shape")

    def get_rec_noise(self, scale: int) -> list[TTensor]:
        """Get the reconstruction noise tensor for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index for which to get the reconstruction noise.

        Returns
        -------
        TTensor
            Reconstruction noise tensor for the specified scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement get_rec_noise")

    @abstractmethod
    def load_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Load discriminator state for a given scale from disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing saved discriminator state.
        scale : int
            The pyramid scale index being loaded.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_discriminator_state")

    @abstractmethod
    def load_amp(self, scale_path: str) -> None:
        """Load amplitude information for a scale and append to `self.noise_amp`.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing the amplitude file.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_amp")

    @abstractmethod
    def load_shape(self, scale_path: str) -> None:
        """Load shape metadata for a scale and append to `self.shapes`.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing the shape file.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_shape")

    @abstractmethod
    def load_wells(self, scale_path: str) -> None:
        """Load well-conditioning data for a scale and append to `self.wells`.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing wells data.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_wells")

    @abstractmethod
    def load_generator_state(self, scale_path: str, scale: int) -> None:
        """Load generator state for a given scale from disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing saved generator state.
        scale : int
            The pyramid scale index being loaded.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_generator_state")

    @abstractmethod
    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Save discriminator state for a given scale to disk.

        Parameters
        ----------
        scale_path : str
            Directory path where discriminator state should be saved.
        scale : int
            Pyramid scale index being saved.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement save_discriminator_state")

    @abstractmethod
    def save_generator_state(self, scale_path: str, scale: int) -> None:
        """Save generator state for a given scale to disk.

        Parameters
        ----------
        scale_path : str
            Directory path where generator state should be saved.
        scale : int
            Pyramid scale index being saved.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement save_generator_state")

    @abstractmethod
    def save_shape(self, scale_path: str, scale: int) -> None:
        """Save shape metadata for a given scale to disk.

        Parameters
        ----------
        scale_path : str
            Directory path where shape metadata should be written.
        scale : int
            Pyramid scale index being saved.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement save_shape")

    def compute_adversarial_loss(self, scale: int, fake: TTensor) -> TTensor:
        """Compute adversarial loss for a generated tensor at a scale.

        Parameters
        ----------
        scale : int
            Pyramid scale index for which to compute the loss.
        fake : TTensor
            Generated tensor produced by the generator for the given scale.

        Returns
        -------
        TTensor
            Scalar tensor equal to the negative mean score from the discriminator.
        """
        discriminator = self.discriminator.discs[scale]
        return -discriminator(fake).mean()  # type: ignore

    @abstractmethod
    def compute_discriminator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: TTensor,
        wells_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
    ) -> tuple[DiscriminatorMetrics[TTensor], dict[str, Any] | None]:
        """Compute discriminator losses and gradient penalty for a scale.

        Parameters
        ----------
        indexes (list[int]):
            Batch/sample indices used to generate fake inputs.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real (TTensor):
            Ground-truth tensor for the current scale.
        wells_pyramid (tuple[TTensor, ...]):
            Wells tensors tuple for conditioning, keyed by scale.
        seismic_pyramid (tuple[TTensor, ...]):
            Seismic tensors tuple for conditioning, keyed by scale.
        Returns
        -------
        tuple[DiscriminatorMetrics[TTensor], dict[str, Any] | None]:
            Container with total, real, fake and gp losses, and optional gradients dict.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_discriminator_metrics"
        )

    @abstractmethod
    def compute_generator_metrics(
        self,
        indexes: list[int],
        scale: int,
        real: TTensor,
        rec_in: TTensor,
        wells_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
        mask: TTensor | None = None,
    ) -> tuple[GeneratorMetrics[TTensor], dict[str, Any] | None]:
        """Common generator-metrics flow shared by frameworks.

        Parameters
        ----------
        indexes (list[int]):
            List of batch/sample indices used to generate noise.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real (TTensor):
            Ground-truth tensor for the current scale.
        rec_in (TTensor):
            Reconstruction input tensor for the current scale.
        wells_pyramid (tuple[TTensor, ...]):
            Well log tensors for conditioning, keyed by scale.
        seismic_pyramid (tuple[TTensor, ...]):
            Seismic volume tensors for conditioning, keyed by scale.
        mask (TTensor | None):
            Well/mask tensor for conditioning.

        Returns
        -------
        tuple[
            GeneratorMetrics[TTensor], dict[str, Any] | None
        ]:
            Container with total, fake, rec, well and div losses, and optional gradients list.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_generator_metrics")

    @abstractmethod
    def update_discriminator_weights(
        self, scale: int, loss: TTensor, gradients: Any | None
    ) -> dict[str, Any] | None:
        """Update discriminator weights using the provided optimizer and loss/gradients.

        This method encapsulates framework-specific optimization steps (e.g.,
        `loss.backward()` and `optimizer.step()` for PyTorch, or
        `optimizer.update()` for MLX).

        Parameters
        ----------
        scale : int
            Scale index.
        loss : TTensor
            Total loss tensor.
        gradients : Any | None
            Computed gradients (if applicable, e.g. MLX).

        Returns
        -------
        dict[str, Any] | None
            Dictionary mapping scale indices to updated state elements for
            lazy evaluation (for frameworks like MLX), or None (for eager
            frameworks like PyTorch).
        """
        raise NotImplementedError(
            "Subclasses must implement update_discriminator_weights"
        )

    @abstractmethod
    def update_generator_weights(
        self, scale: int, loss: TTensor, gradients: Any | None
    ) -> dict[str, Any] | None:
        """Update generator weights using the provided optimizer and loss/gradients.

        Parameters
        ----------
        scale : int
            Scale index.
        loss : TTensor
            Total loss tensor.
        gradients : Any | None
            Computed gradients (if applicable, e.g. MLX).

        Returns
        -------
        dict[str, Any] | None
            Dictionary mapping scale indices to updated state elements for
            lazy evaluation (for frameworks like MLX), or None (for eager
            frameworks like PyTorch).
        """
        raise NotImplementedError("Subclasses must implement update_generator_weights")

    @abstractmethod
    def forward(
        self,
        indexes: list[int],
        facies_pyramid: tuple[TTensor, ...],
        rec_in_pyramid: tuple[TTensor, ...],
        wells_pyramid: tuple[TTensor, ...] = (),
        masks_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
    ) -> ScaleMetrics[TTensor] | tuple[IterableMetrics[TTensor], ...]:
        """Perform a forward pass for both discriminator and generator.

        Parameters
        ----------
        indexes : tuple[int, ...]
            Tuple of batch/sample indices used to generate noise.
        facies_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to ground-truth tensors.
        rec_in_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to reconstruction inputs.
        wells_pyramid : tuple[TTensor, ...], optional
            Tuple mapping scale indices to well-conditioning tensors.
        masks_pyramid : tuple[TTensor, ...], optional
            Tuple mapping scale indices to well/mask tensors.
        seismic_pyramid : tuple[TTensor, ...], optional
            Tuple mapping scale indices to seismic-conditioning tensors.
        Returns
        -------
        ScaleMetrics[TTensor]
            Container with discriminator and generator metrics for the forward pass.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement forward")

    def generate_diverse_samples(
        self,
        indexes: list[int],
        scale: int,
        wells_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
    ) -> list[TTensor]:
        """Generate multiple candidate outputs for `scale` using current generator.

        This centralizes the pattern of sampling multiple noise realizations
        and forwarding them through `self.generator` so concrete subclasses
        can reuse the logic without duplicating code.

        Parameters
        ----------
        indexes : tuple[int, ...]
            Tuple of batch/sample indices used to generate noise.
        scale : int
            Pyramid scale index for which to generate samples.
        wells_pyramid : tuple[TTensor, ...], optional
            Tuple mapping scale indices to well-conditioning tensors for the current scale.
        seismic_pyramid : tuple[TTensor, ...], optional
            Tuple mapping scale indices to seismic-conditioning tensors for the current scale.

        Returns
        -------
        list[TTensor]
            List of generated tensors from multiple noise realizations.
        """
        samples: list[TTensor] = []
        for _ in range(self.num_diversity_samples):
            noises = self.get_pyramid_noise(
                scale, indexes, wells_pyramid, seismic_pyramid
            )
            amps = self.get_noise_aplitude(scale)
            samples.append(self.generator(noises, amps, stop_scale=scale))
        return samples

    def get_pyramid_noise(
        self,
        scale: int,
        indexes: list[int],
        wells_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
        rec: bool = False,
    ) -> list[TTensor]:
        """Generate noise tensors up to a specific pyramid scale (generic).

        Uses `NoiseSpec` and framework-provided callables supplied by the
        subclass (via `get_noise_gen_fn`, `get_pad_fn`, `get_cat_fn`) so the
        base implementation remains framework-agnostic.

        Parameters
        ----------
        indexes : list[int]
            Batch/sample indices used to generate noise.
        scale : int
            Pyramid scale index up to which to generate noise tensors.
        wells_pyramid : tuple[TTensor, ...], optional
            Tuple mapping scale indices to well-conditioning tensors for the current scale.
        seismic_pyramid : tuple[TTensor, ...], optional
            Tuple mapping scale indices to seismic-conditioning tensors for the current scale.
        rec : bool, optional
            If True, return stored reconstruction noise instead of new noise.
            (default is False).

        Returns
        -------
        list[TTensor]
            List of noise tensors from scale 0 up to `scale`.
        """
        if rec:
            return self.get_rec_noise(scale)
        return [
            self.generate_noise(
                i,
                indexes,
                wells_pyramid[i] if wells_pyramid else None,
                seismic_pyramid[i] if seismic_pyramid else None,
            )
            for i in range(scale + 1)
        ]

    def get_noise_aplitude(self, scale: int) -> list[float]:
        """Return noise amplitude for a given scale.

        Parameters
        ----------
        scale : int
            Pyramid scale index for which to get noise amplitude.

        Returns
        -------
        list[float]
            List of noise amplitudes up to the requested scale.
        """
        return (
            self.noise_amps[: scale + 1]
            if len(self.noise_amps) >= scale + 1
            else [1.0] * (scale + 1)
        )

    def get_num_features(self, scale: int) -> tuple[int, int]:
        """Calculate feature counts for networks at a given scale.

        Features double every 4 scales up to a maximum of 128. This logic is
        framework-agnostic and therefore lives in the base class.

        Parameters
        ----------
        scale : int
            Pyramid scale index for which to compute feature counts.

        Returns
        -------
        tuple[int, int]
            A 2-tuple of integers: (num_feature, min_num_feature).
        """
        num_feature = min(self.num_feature * pow(2, math.floor(scale / 4)), 128)
        min_num_feature = min(self.min_num_feature * pow(2, math.floor(scale / 4)), 128)

        return num_feature, min_num_feature

    def has_amp_file(self, scale_path: str) -> bool:
        """Return True if amplitude file exists in `scale_path`.

        Default implementation checks for the presence of `AMP_FILE` in the
        given scale directory. Subclasses can override if they store
        amplitude data differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for amplitude file.

        Returns
        -------
        bool
            True if amplitude file exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, AMP_FILE))

    def has_discriminator_checkpoint(self, scale_path: str) -> bool:
        """Return True if discriminator checkpoint exists in `scale_path`.

        Default implementation checks for the presence of `D_FILE` in the
        given scale directory. Subclasses can override if they store
        discriminator checkpoints differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for discriminator checkpoint.

        Returns
        -------
        bool
            True if discriminator checkpoint exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, D_FILE))

    def has_generator_checkpoint(self, scale_path: str) -> bool:
        """Return True if generator checkpoint exists in `scale_path`.

        Default implementation checks for the presence of `G_FILE` in the
        given scale directory. Subclasses can override if they store
        generator checkpoints differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for generator checkpoint.

        Returns
        -------
        bool
            True if generator checkpoint exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, G_FILE))

    def has_shape_file(self, scale_path: str) -> bool:
        """Return True if shape file exists in `scale_path`.

        Default implementation checks for the presence of `SHAPE_FILE` in
        the given scale directory. Subclasses can override if they store
        shapes differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for shape file.

        Returns
        -------
        bool
            True if shape file exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, SHAPE_FILE))

    def has_wells_file(self, scale_path: str) -> bool:
        """Return True if wells file exists in `scale_path`.

        Default implementation checks for the presence of `M_FILE` in the
        given scale directory. Subclasses may override if they use a
        different wells storage layout.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for wells file.

        Returns
        -------
        bool
            True if wells file exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, M_FILE))

    def init_discriminator_for_scale(self, scale: int) -> None:
        """Initialize discriminator for a new pyramid scale.

        Creates a new discriminator with appropriate feature counts. Each
        scale gets its own discriminator for parallel training.

        Parameters
        ----------
        scale : int
            Pyramid scale index to initialize.
        """
        num_feature, min_num_feature = self.get_num_features(scale)

        # Create the framework-specific discriminator scale block (subclass).
        self.discriminator.create_scale(num_feature, min_num_feature)

        # Let the subclass finalize the new discriminator block and apply weights.
        self.finalize_discriminator_scale(scale)

    def init_generator_for_scale(self, scale: int) -> None:
        """Generic generator-scale initializer.

        This base implementation computes the appropriate feature counts for
        `scale`, delegates creation and device-placement to framework hooks,
        and then asks the subclass to finalize the new block (either
        reinitializing weights or copying previous weights). Subclasses must
        implement the three hooks documented below.


        Parameters
        ----------
        scale : int
            Pyramid scale index to initialize.
        """
        num_feature, min_num_feature = self.get_num_features(scale)

        # Create the framework-specific generator scale block (subclass).
        self.generator.create_scale(scale, num_feature, min_num_feature)

        # Determine whether the new scale should be reinitialized or copied
        # from previous scale; delegate SPADE/scale logic to subclass.
        prev_is_spade = self.is_spade_scale(scale - 1) if scale > 0 else False
        curr_is_spade = self.is_spade_scale(scale)
        reinit = (scale % 4 == 0) or prev_is_spade or curr_is_spade

        # Let the subclass finalize the new generator block (apply weights or
        # copy previous weights) according to the `reinit` decision.
        self.finalize_generator_scale(scale, reinit)

    def init_scales(self, start_scale: int, num_scales: int) -> None:
        """Initialize a consecutive range of scales.

        This generic implementation delegates the framework-specific work to
        the abstract methods `init_scale_generator` and
        `init_scale_discriminator` which concrete subclasses must implement.

        Parameters
        ----------
        start_scale : int
            Pyramid scale index to start initializing from.
        num_scales : int
            Number of consecutive scales to initialize.
        """
        for scale in range(start_scale, start_scale + num_scales):
            self.init_generator_for_scale(scale)
            self.init_discriminator_for_scale(scale)
            self.active_scales.add(scale)

    def is_spade_scale(self, scale: int) -> bool:
        """Return True if `scale` uses SPADE (or other scale-specific flag).

        Default implementation looks for a `spade_scales` attribute on
        `self.generator` and checks membership; returns False when the
        attribute is missing. Subclasses may override for different
        generator implementations.

        Parameters
        ----------
        scale : int
            Pyramid scale index to check.

        Returns
        -------
        bool
            True if `scale` is a SPADE scale, False otherwise.
        """
        return scale in self.generator.spade_scales

    def load(
        self,
        path: str,
        load_shapes: bool = True,
        until_scale: int | None = None,
        load_discriminator: bool = False,
        load_wells: bool = False,
    ) -> int:
        """Load saved models and return the next starting scale.

        The base implementation walks the checkpoint directory structure and
        delegates actual model/state loading to subclass hooks.

        Parameters
        ----------
        path : str
            Root directory path where scale subdirectories are located.
        load_shapes : bool, optional
            Whether to load shape metadata for each scale (default is True).
        until_scale : int | None, optional
            If provided, load scales only up to (and including) this index.
            Default is None (load all available scales).
        load_discriminator : bool, optional
            Whether to load discriminator states for each scale
            (default is False).
        load_wells : bool, optional
            Whether to load well-conditioning data for each scale (default is False).

        Returns
        -------
        int
            The next scale index after the last successfully loaded scale.
        """
        scale = 0

        while os.path.exists(os.path.join(path, str(scale))):
            if until_scale is not None and scale > until_scale:
                break

            scale_path = os.path.join(path, str(scale))

            # Load generator if a checkpoint exists for this scale
            if self.has_generator_checkpoint(scale_path):
                self.init_generator_for_scale(scale)
                self.load_generator_state(scale_path, scale)

            if load_discriminator and self.has_discriminator_checkpoint(scale_path):
                self.init_discriminator_for_scale(scale)
                self.load_discriminator_state(scale_path, scale)

            # Load amplitude
            if self.has_amp_file(scale_path):
                self.load_amp(scale_path)

            # Load shapes
            if load_shapes and self.has_shape_file(scale_path):
                self.load_shape(scale_path)

            if load_wells and self.has_wells_file(scale_path):
                self.load_wells(scale_path)

            scale += 1

        return scale

    def move_to_device(self, obj: Any) -> Any:
        """Optional hook to move framework objects to `device`.

        Default implementation is a no-op. Framework subclasses (e.g.
        PyTorch) should override to call `.to(device)` on modules.

        Parameters
        ----------
        obj : Any
            Framework-specific object to move to device.

        Returns
        -------
        Any
            The same object, moved to the target device.
        """
        return obj

    def optimize_discriminator(
        self,
        indexes: list[int],
        facies_pyramid: tuple[TTensor, ...],
        wells_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
    ) -> tuple[DiscriminatorMetrics[TTensor], ...] | IterableMetrics[TTensor]:
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
        facies_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to real tensor samples.
        wells_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to well-conditioning tensors.
        seismic_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to seismic-conditioning tensors.
        Returns
        -------
        tuple[DiscriminatorMetrics[TTensor], ...] | IterableMetrics[TTensor]
            Tuple of computed discriminator metrics for each scale.
        """
        step_metrics: list[DiscriminatorMetrics[TTensor]] = []

        for _ in range(self.discriminator_steps):

            # Compute metrics for this discriminator step only
            step_metrics = []
            step_gradients: list[dict[str, TTensor]] = []
            losses: list[TTensor] = []
            for scale in self.active_scales:

                metrics, gradients = self.compute_discriminator_metrics(
                    indexes,
                    scale,
                    facies_pyramid[scale],
                    wells_pyramid,
                    seismic_pyramid,
                )

                # Delegate the optimization step to subclass
                updates = self.update_discriminator_weights(
                    scale,
                    metrics.total,
                    gradients,
                )
                if updates:
                    step_gradients.append(updates)

                step_metrics.append(metrics)
                losses.extend(metrics.as_tuple())

        return tuple(step_metrics)

    def optimize_generator(
        self,
        indexes: list[int],
        facies_pyramid: tuple[TTensor, ...],
        rec_in_pyramid: tuple[TTensor, ...],
        wells_pyramid: tuple[TTensor, ...] = (),
        masks_pyramid: tuple[TTensor, ...] = (),
        seismic_pyramid: tuple[TTensor, ...] = (),
    ) -> tuple[GeneratorMetrics[TTensor], ...] | IterableMetrics[TTensor]:
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
        facies_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to real tensor samples.
        rec_in_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to reconstruction inputs.
        wells_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to well-conditioning tensors.
        masks_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to well/mask tensors.
        seismic_pyramid : tuple[TTensor, ...]
            Tuple mapping scale indices to seismic-conditioning tensors.

        Returns
        -------
        tuple[GeneratorMetrics[TTensor], ...] | IterableMetrics[TTensor]
            Tuple of computed generator metrics for each scale.
        """

        step_metrics: list[GeneratorMetrics[TTensor]] = []

        for _ in range(self.generator_steps):

            # Compute per-scale metrics using subclass hook
            step_gradients: list[dict[str, TTensor]] = []
            losses: list[Any] = []
            step_metrics = []
            for scale in self.active_scales:
                if scale >= len(facies_pyramid):
                    continue

                real = facies_pyramid[scale]
                rec_in = rec_in_pyramid[scale]
                mask = masks_pyramid[scale]

                # Ensure noise amplitudes have been initialized for this scale.
                # In normal training `noise_amp` is populated during noise
                # initialization; missing values indicate setup wasn't run.
                if len(self.noise_amps) < scale + 1:
                    raise RuntimeError(
                        f"noise_amp not initialized for scale {scale}. "
                        "Call the project's noise initialization before training."
                    )

                metrics, gradients = self.compute_generator_metrics(
                    indexes,
                    scale,
                    real,
                    rec_in,
                    wells_pyramid,
                    seismic_pyramid,
                    mask,
                )

                # Delegate the optimization step to subclass
                updates = self.update_generator_weights(scale, metrics.total, gradients)
                if updates:
                    step_gradients.append(updates)

                step_metrics.append(metrics)
                losses.extend(metrics.as_tuple())

        return tuple(step_metrics)

    def save_amp(self, scale_path: str, scale: int) -> None:
        """Save amplitude (noise_amp) for `scale` into `scale_path` directory.

        This default implementation writes the float value of `self.noise_amp[scale]`
        to a small text file named by `AMP_FILE`. Subclasses may override if they
        need different semantics.

        Parameters
        ----------
        scale_path : str
            Directory path where amplitude file should be saved.
        scale : int
            Pyramid scale index being saved.
        """
        if scale < len(self.noise_amps):
            amp_path = os.path.join(scale_path, AMP_FILE)
            with open(amp_path, "w") as f:
                f.write(str(self.noise_amps[scale]))

    def save_scale(self, scale: int, path: str) -> None:
        """Save generator/discriminator and auxiliary files for a scale.

        The base implementation delegates framework-specific model saves to
        the concrete subclass hooks so the base remains agnostic about file
        formats and serialization APIs.

        Parameters
        ----------
        scale : int
            Pyramid scale index being saved.
        path : str
            Directory path where scale data should be saved.
        """
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)

        # Framework-specific model state
        self.save_generator_state(path, scale)
        self.save_discriminator_state(path, scale)

        # Save amplitude and shape via subclass hooks (formats chosen by subclass)
        self.save_amp(path, scale)
        self.save_shape(path, scale)

    def schedulers_step(
        self,
        scales: tuple[int, ...],
    ) -> None:
        """Step the learning-rate schedulers for the provided scales.

        Parameters
        ----------
        generator_schedulers : dict[int, LRScheduler]
            Generator learning-rate schedulers per scale.
        discriminator_schedulers : dict[int, LRScheduler]
            Discriminator learning-rate schedulers per scale.
        scales : tuple[int, ...]
            Tuple of scale indices to step the schedulers for.
        """
        for scale in scales:
            self.generator_schedulers[scale].step()
            self.discriminator_schedulers[scale].step()

    def setup_framework(self) -> None:
        """Create framework-specific objects and assign them to the instance.

        This generic helper calls the concrete `build_generator` and
        `create_discriminators_container` hooks and then moves the created
        generator to `device` using `move_to_device`. Subclasses may
        optionally override `move_to_device` to support framework-specific
        device placement.
        """
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    @abstractmethod
    def generate_noise(
        self,
        scale: int,
        indexes: list[int],
        well: TTensor | None = None,
        seismic: TTensor | None = None,
    ) -> TTensor:
        """Generate a noise tensor of given shape and batch size.

        Parameters
        ----------
        scale : int
            Pyramid scale index used to select the noise shape.
        indexes : tuple[int, ...]
            Tuple of batch/sample indices used to generate noise.
        well : TTensor | None, optional
            Well-conditioning tensor for the current scale.
        seismic : TTensor | None, optional
            Seismic-conditioning tensor for the current scale.

        Returns
        -------
        TTensor
            Generated noise tensor of shape (num_samp, *shape).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement generate_noise")
