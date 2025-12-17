"""Small array/tensor utilities and training helpers.

This module provides lightweight utilities used across training and
visualization code: device resolution, tensor<->numpy conversions, image
loading, random seeds, and common neural-network helpers such as
gradient-penalty and noise generation.
"""

import math
import random
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from options import TrainningOptions


def set_seed(seed: int = 42) -> None:
    """Set seeds for torch, numpy and python.random at module level.

    Keeping a module-level `set_seed` makes it easy for other modules to
    call it without constructing a `NeuralSmoother` instance. The
    `NeuralSmoother.set_seed` method remains as a thin wrapper that
    delegates to this function to preserve backward compatibility.
    """
    seed_int = int(seed)
    torch.manual_seed(seed_int)  # pyright: ignore
    np.random.seed(seed_int)
    random.seed(seed_int)


def resolve_device() -> torch.device:
    """Return the preferred device: MPS (Apple), CUDA, or CPU.

    Exposed at module level so other modules can determine the best device
    without constructing a `NeuralSmoother` instance.
    """
    if (
        getattr(torch.backends, "mps", None) is not None
        and getattr(torch.backends.mps, "is_available", lambda: False)()
        and getattr(torch.backends.mps, "is_built", lambda: True)()
    ):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mask_resize(mask: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Resize well mask tensor using bicubic interpolation with thresholding.

    This function resizes well location masks to match target pyramid scales.
    It amplifies the mask values before interpolation, then applies thresholding
    to preserve well locations at column positions.

    Parameters
    ----------
    mask : torch.Tensor
        Input well mask tensor of shape (B, C, H, W).
    size : tuple[int, ...]
        Target size (height, width) for the resized mask.

    Returns
    -------
    torch.Tensor
        Resized mask tensor with wells positioned at appropriate columns.
    """
    mask = (mask > 0).float() * 1000
    interpolated_mask = nn.functional.interpolate(
        mask, size=size, mode="bicubic", align_corners=False, antialias=True
    )
    mask_index = torch.argmax(torch.sum(interpolated_mask, dim=2), dim=-1)
    interpolated_mask.zero_()
    interpolated_mask[:, :, :, mask_index] = 1
    return interpolated_mask


def generate_scales(options: TrainningOptions) -> tuple[tuple[int, ...], ...]:
    """Generate multi-scale pyramid resolutions for progressive training.

    Creates a tuple of shapes representing different scales from coarse to fine
    resolution. Each scale is computed using exponential scaling between
    min_size and max_size parameters.

    Parameters
    ----------
    options : TrainningOptions
        Training configuration containing:
        - min_size: Minimum (coarsest) resolution
        - max_size: Maximum (finest) resolution
        - crop_size: Crop size for training
        - stop_scale: Number of pyramid scales
        - batch_size: Batch size for each scale
        - num_channels: Number of input channels

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Tuple of (batch_size, channels, height, width) tuples, one for each
        pyramid scale, arranged from coarsest to finest resolution.
    """
    shapes: list[tuple[int, ...]] = []
    scale_factor = math.pow(
        options.min_size / (min(options.max_size, options.crop_size)),
        1 / options.stop_scale,
    )
    for i in range(options.stop_scale + 1):
        scale = math.pow(scale_factor, options.stop_scale - i)
        out_shape = cast(
            Sequence[int],
            np.uint(
                np.round(
                    np.array(
                        [
                            min(options.max_size, options.crop_size),
                            min(options.max_size, options.crop_size),
                        ]
                    )
                    * scale
                )
            ).tolist(),
        )
        if out_shape[0] % 2 != 0:
            out_shape = [int(shape + 1) for shape in out_shape]
        shapes.append((options.batch_size, options.num_img_channels, *out_shape))

    return tuple(shapes)


def norm(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0, 1] to [-1, 1] range.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values in [0, 1] range.

    Returns
    -------
    torch.Tensor
        Normalized tensor with values clamped to [-1, 1].
    """
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def denorm(tensor: torch.Tensor, ceiling: bool = False) -> torch.Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1] range.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with values in [-1, 1] range.
    ceiling : bool, optional
        Whether to set all positive values to 1. Currently not implemented.
        Defaults to False.

    Returns
    -------
    torch.Tensor
        Denormalized tensor with values clamped to [0, 1].
    """
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    return tensor


def torch2np(
    tensor: torch.Tensor, denormalize: bool = False, ceiling: bool = False
) -> NDArray[np.float32]:
    """Convert PyTorch tensor to NumPy array with optional denormalization.

    Transforms tensor from (B, C, H, W) format to (B, H, W, C) NumPy array,
    optionally denormalizing from [-1, 1] to [0, 1] range.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in (B, C, H, W) format.
    denormalize : bool, optional
        If True, denormalize from [-1, 1] to [0, 1]. Defaults to False.
    ceiling : bool, optional
        If True, set positive values to 1 during denormalization. Defaults to False.

    Returns
    -------
    NDArray[np.float32]
        NumPy array with shape (B, H, W, C) and values clipped to [0, 1].
    """
    if denormalize:
        tensor = denorm(tensor, ceiling)
    np_array = tensor.detach().cpu().numpy()
    # Support both batch tensors (B, C, H, W) and single samples (C, H, W).
    if np_array.ndim == 4:
        # (B, C, H, W) -> (B, H, W, C)
        np_array = np.transpose(np_array, (0, 2, 3, 1))
    elif np_array.ndim == 3:
        # (C, H, W) -> (H, W, C)
        np_array = np.transpose(np_array, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported tensor ndim for torch2np: {np_array.ndim}")

    np_array = np.clip(np_array, 0, 1)
    return np_array.astype(np.float32)


def np2torch(np_array: NDArray[np.float32], normalize: bool = False) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor with optional normalization.

    Parameters
    ----------
    np_array : NDArray[np.float32]
        Input NumPy array to convert.
    normalize : bool, optional
        Whether to normalize the tensor to [-1, 1] range. Defaults to False.

    Returns
    -------
    torch.Tensor
        Converted tensor, always normalized to [-1, 1] range.

    Notes
    -----
    BUG: The function always applies normalization at the end, regardless of
    the normalize parameter value. The normalize parameter causes double
    normalization when True.
    """
    tensor = torch.from_numpy(np_array).float()  # type: ignore
    if normalize:
        tensor = norm(tensor)
    # BUG: Always normalizes regardless of the normalize parameter
    return norm(tensor)


def range_transform(
    facie: np.ndarray,
    in_range: tuple[int, int] = (0, 255),
    out_range: tuple[int, int] = (-1, 1),
) -> np.ndarray:
    """Transform array values from one range to another using linear scaling.

    Parameters
    ----------
    facie : np.ndarray
        Input facies array to transform.
    in_range : tuple[int, int], optional
        Input range (min, max) of the array. Defaults to (0, 255).
    out_range : tuple[int, int], optional
        Output range (min, max) for the transformed array. Defaults to (-1, 1).

    Returns
    -------
    np.ndarray
        Transformed array with values scaled to output range.
    """
    if in_range != out_range:
        scale = np.float32(out_range[1] - out_range[0]) / np.float32(
            in_range[1] - in_range[0]
        )
        bias = np.float32(out_range[0]) - np.float32(in_range[0]) * scale
        facie = facie * scale + bias
    return facie


def reset_grads(model: nn.Module, require_grad: bool = False) -> nn.Module:
    """Set requires_grad attribute for all parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be updated.
    require_grad : bool, optional
        Value to set for requires_grad attribute. Defaults to False.

    Returns
    -------
    nn.Module
        The model with updated requires_grad attributes.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(require_grad)

    return model


def calc_diversity_loss(
    fake_samples: list[torch.Tensor],
    noise_samples: list[list[torch.Tensor]],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate multi-scale diversity loss to encourage output variation.

    Penalizes the generator when different noise inputs produce similar outputs.
    Uses mode seeking loss: maximizes ratio of output distance to input distance.

    Loss = -mean(||G(z1) - G(z2)|| / ||z1 - z2||)

    This encourages the generator to produce diverse outputs for different
    noise inputs while respecting well conditioning.

    Parameters
    ----------
    fake_samples : list[torch.Tensor]
        List of generated images from different noise inputs.
        Each tensor has shape (1, C, H, W).
    noise_samples : list[torch.Tensor]
        List of noise tensors used to generate the fake samples.
        Each is a list of tensors per scale; we use the coarsest scale.
    eps : float, optional
        Small constant for numerical stability. Defaults to 1e-8.

    Returns
    -------
    torch.Tensor
        Scalar diversity loss (negative, to be minimized).
    """
    if len(fake_samples) < 2:
        return torch.tensor(0.0, device=fake_samples[0].device)

    # Vectorized pairwise distance calculation
    n = len(fake_samples)

    # Stack all samples and flatten for easier computation
    stacked_fakes = torch.stack([f.flatten() for f in fake_samples])
    stacked_noises = torch.stack([noise_samples[i][0].flatten() for i in range(n)])

    # Compute pairwise L1 distances using broadcasting
    # (n, 1, -1) - (1, n, -1) -> (n, n, -1)
    output_diffs = torch.abs(
        stacked_fakes.unsqueeze(1) - stacked_fakes.unsqueeze(0)
    ).mean(dim=2)
    input_diffs = (
        torch.abs(stacked_noises.unsqueeze(1) - stacked_noises.unsqueeze(0)).mean(dim=2)
        + eps
    )

    # Extract upper triangular part (excluding diagonal) for unique pairs
    mask = torch.triu(
        torch.ones(n, n, device=fake_samples[0].device), diagonal=1
    ).bool()
    ratios = -(output_diffs / input_diffs)[mask]

    diversity_loss = (
        ratios.mean()
        if ratios.numel() > 0
        else torch.tensor(0.0, device=fake_samples[0].device)
    )

    return diversity_loss
