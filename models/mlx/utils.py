from typing import TypeVar, cast
import mlx.core as mx
import mlx.nn as nn  # type: ignore

from collections.abc import Callable, Iterable


"""Utility helpers for MLX-based models used by the SPADE implementation.

This module contains helpers for noise generation, interpolation,
gradient penalty computation for WGAN-GP, parameter initialization and
weight loading. Functions are written to operate on MLX arrays in the
project's NHWC tensor layout (B, H, W, C) unless otherwise noted.
"""


def calc_gradient_penalty(
    discriminator: nn.Module | Callable[[mx.array], mx.array],
    real_data: mx.array,
    fake_data: mx.array,
    LAMBDA: float,
) -> mx.array:
    """Calculate gradient penalty for WGAN-GP training in MLX.

    Implements the gradient penalty term to enforce the 1-Lipschitz constraint.
    MLX computes gradients with respect to inputs using mx.grad.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator model.
    real_data : mx.array
        Real data samples (B, H, W, C).
    fake_data : mx.array
        Generated fake data samples (B, H, W, C).
    LAMBDA : float
        Gradient penalty coefficient.

    Returns
    -------
    mx.array
        Calculated gradient penalty scalar.
    """
    # Random interpolation factor
    batch_size = real_data.shape[0]
    alpha = mx.random.uniform(shape=(batch_size, 1, 1, 1))

    interpolates = alpha * real_data + (1 - alpha) * fake_data

    def grad_fn(x: mx.array) -> mx.array:
        # Discriminator output for interpolates
        out: mx.array = cast(mx.array, discriminator(x))
        return mx.sum(out)

    # Calculate gradients of the output w.r.t. the interpolates
    gradients = cast(mx.array, mx.grad(grad_fn)(interpolates))  # type: ignore

    # Compute L2 norm of gradients along the (H, W, C) dimensions
    # In MLX, we flatten the non-batch dimensions to calculate the norm per sample
    gradients_flat = gradients.reshape(batch_size, -1)  # type: ignore
    grad_norm = mx.sqrt(mx.sum(mx.square(gradients_flat), axis=1) + 1e-12)

    gradient_penalty = mx.mean(mx.square(grad_norm - 1.0)) * LAMBDA
    return gradient_penalty


def generate_noise(
    size: tuple[int, ...], num_samp: int = 1, scale: float = 1.0
) -> mx.array:
    """Generate random noise in NHWC format.

    Parameters
    ----------
    size : tuple[int, int, int]
        Shape as (height, width, channels).
    num_samp : int
        Batch size.
    scale : float
        Downscaling factor for spatial dimensions.

    Returns
    -------
    mx.array
        Random noise array.
    """
    h, w, c = size
    noise_shape = (num_samp, round(h / scale), round(w / scale), c)
    noise = mx.random.normal(shape=noise_shape)

    if scale != 1.0:
        noise = cast(mx.array, mx.image.resize(noise, h, w))  # type: ignore

    return noise


def interpolate(tensor: mx.array, size: tuple[int, ...]) -> mx.array:
    """Resize array using bilinear interpolation.

    Parameters
    ----------
    tensor : mx.array
        Input array (B, H, W, C).
    size : tuple[int, int]
        Target (height, width).

    Returns
    -------
    mx.array
        Resized array.
    """
    upsample_layer = nn.Upsample((size[0], size[1]), mode="linear", align_corners=True)
    resized_tensor = upsample_layer(tensor)
    return resized_tensor


def init_weights(model: nn.Module | Callable[[mx.array], mx.array]) -> None:
    """Initialize model parameters in-place.

    Applies standard GAN-style initialization to convolution and
    normalization layers:

    - Convolutional weights: normal distribution N(0, 0.02)
    - Biases: zeros
    - Batch/Instance norm weights: normal distribution with mean=1.0

    Parameters
    ----------
    model : nn.Module | Callable[[mx.array], mx.array]
        MLX model whose parameters will be updated in-place.

    Notes
    -----
    This function mutates fields on MLX module objects (e.g. ``weight``
    and ``bias``) rather than returning a new model object.
    """

    def initialize(m: nn.Module | Callable[[mx.array], mx.array]) -> None:
        if isinstance(m, nn.Conv2d):
            # Normal distribution N(0, 0.02)
            m.weight = mx.random.normal(shape=m.weight.shape, scale=0.02)
            m.bias = mx.zeros_like(m.bias)
        elif isinstance(m, (nn.BatchNorm, nn.InstanceNorm)):
            m.weight = mx.random.normal(shape=m.weight.shape, loc=1.0, scale=0.02)
            m.bias = mx.zeros_like(m.bias)

    # Walk through modules and initialize
    for _, module in cast(
        Iterable[tuple[str, nn.Module]],
        model.named_modules(),  # type: ignore
    ):
        initialize(module)


T = TypeVar("T")


def load(
    path: str,
    as_type: type[T] | None = None,
) -> T:
    """Load a PyTorch object from `path` and optionally validate its type.

    Parameters
    ----------
    path : str
        Filesystem path to the saved PyTorch object (state dict, tensors, list, etc.).
    device : torch.device, optional
        Device to map the loaded tensors to (defaults to CPU).
    as_type : type[T] | None, optional
        Optional runtime class used only for a non-fatal ``isinstance`` check
        to help callers detect mismatches early (for example, ``dict`` or
        ``list``). This parameter does not modify how ``torch.load`` behaves.

    Returns
    -------
    T
        The loaded object, cast to the requested generic return type.
    """
    obj = mx.load(path)  # type: ignore
    if as_type is not None:
        # `as_type` should be a concrete runtime class (e.g., `dict` or `list`).
        # For typing constructs like `Mapping[str, Any]` you should annotate the
        # receiving variable at the callsite instead.
        try:
            if not isinstance(obj, as_type):
                # non-fatal warning to help catch mismatches early
                print(f"Warning: loaded object is not an instance of {as_type}")
        except TypeError:
            # `as_type` may not be a valid runtime-checkable type (e.g., typing
            # constructs). Ignore the check in that case.
            pass
    return cast(T, obj)  # type: ignore
