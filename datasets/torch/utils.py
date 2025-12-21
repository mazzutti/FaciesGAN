"""Create cached multi-scale pyramids for facies, seismic and well data.

This module exposes helpers that build multi-resolution tensors for the
training pipeline. The functions are cached via ``joblib.Memory`` so repeated
calls with the same scale list are fast. All pyramid functions accept a
``scale_list`` that is a tuple of shape descriptors produced by
``ops.generate_scales`` (each element is ``(batch, channels, height, width)``)
and return a list of PyTorch tensors, one tensor per scale, with shape
``(N, C, H, W)``.
"""

import torch
from joblib import Memory  # type: ignore

import datasets.torch.utils as torch_utils
import datasets.utils as data_utils
from datasets.data_files import DataFiles
from interpolators.config import InterpolatorConfig
from interpolators.nearest import NearestInterpolator
from interpolators.neural import NeuralSmoother
from interpolators.well import WellInterpolator

# Create a cache directory
memory = Memory("./.cache", verbose=0)


def _stack_and_format(
    pyramids_list: list[list[torch.Tensor]], channels_last: bool
) -> list[torch.Tensor]:
    """Stack per-scale lists into properly ordered tensors.

    Returns a list of stacked tensors. If `channels_last` is True the
    returned tensors have shape `(N, H, W, C)` (squeezing a trailing
    singleton channel). Otherwise they are returned in `(N, C, H, W)`
    by permuting the stacked result.
    """
    if channels_last:
        return [torch.stack(pyramid, dim=0).squeeze(-1) for pyramid in pyramids_list]
    return [
        torch.stack(pyramid, dim=0).squeeze(1).permute(0, 3, 1, 2)
        for pyramid in pyramids_list
    ]


def _empty_pyramid_tensor(scale: tuple[int, ...], channels_last: bool) -> torch.Tensor:
    """Return an empty per-scale tensor with the correct layout.

    If `channels_last` is True the returned tensor has shape
    `(0, H, W, C)`, otherwise `(0, C, H, W)`.
    """
    if channels_last:
        _, height, width, channels = scale
        return torch.empty((0, height, width, channels), dtype=torch.float32)
    else:
        _, channels, height, width = scale
        return torch.empty((0, channels, height, width), dtype=torch.float32)


@memory.cache  # type: ignore
def to_facies_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for facies images using a neural interpolator.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors as produced by ``ops.generate_scales``. Each
        element must be a 4-tuple ``(batch, channels, height, width)`` describing
        the target resolution.
    channels_last : bool, optional
            Whether the channel dimension is last in the tensor shape, by default False.

    Returns
    -------
    tuple[torch.Tensor, ...]
        A tuple where each element is a PyTorch tensor containing all facies at
        that scale with shape ``(N, C, H, W)`` (N images, C channels).
    """
    facies_paths = data_utils.as_image_file_list(DataFiles.FACIES)
    models_paths = data_utils.as_model_file_list(DataFiles.FACIES)

    # If no facies files or models are available, return per-scale empty tensors
    if len(facies_paths) == 0 or len(models_paths) == 0:
        return tuple(
            _empty_pyramid_tensor(scale, channels_last) for scale in scale_list
        )

    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    for facie_path, model_path in zip(facies_paths, models_paths):
        neural_smoother = NeuralSmoother(
            model_path, InterpolatorConfig(channels_last=channels_last)
        )
        pyramid = neural_smoother.interpolate(facie_path, scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(torch_utils.norm(pyramid[i]))
    pyramids = _stack_and_format(pyramids_list, channels_last)
    return tuple(pyramids)


@memory.cache  # type: ignore
def to_seismic_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for seismic images using nearest interpolation.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, channels, height, width)``.
    channels_last : bool, optional
        Whether the channel dimension is last in the tensor shape, by default False.

    Returns
    -------
    tuple[torch.Tensor, ...]
        A tuple of tensors (one per scale) with shape ``(N, C, H, W)``.

    Notes
    -----
    Uses NearestInterpolator and caches results in ``./.cache``.
    """
    seismic_paths = data_utils.as_image_file_list(DataFiles.SEISMIC)
    # If no seismic files available return empty per-scale tensors
    if len(seismic_paths) == 0:
        return tuple(
            _empty_pyramid_tensor(scale, channels_last) for scale in scale_list
        )

    seismic_interpolator = NearestInterpolator(
        InterpolatorConfig(channels_last=channels_last)
    )
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    for seismic_path in seismic_paths:
        pyramid = seismic_interpolator.interpolate(seismic_path, scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(norm(pyramid[i]))

    pyramids = _stack_and_format(pyramids_list, channels_last)
    return tuple(pyramids)


@memory.cache  # type: ignore
def to_wells_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
    channels_last: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for well location data.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, channels, height, width)``.
    channels_last : bool, optional
        Whether the channel dimension is last in the tensor shape, by default False.

    Returns
    -------
    tuple[torch.Tensor, ...]
        A tuple of tensors (one per scale) with shape ``(N, C, H, W)``. Well
        locations are represented as sparse vertical traces at the appropriate
        column indices.

    Notes
    -----
    Uses ``WellInterpolator`` and caches results in ``./.cache``.
    """
    wells_interpolator = WellInterpolator(
        InterpolatorConfig(channels_last=channels_last)
    )
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    wells_paths = data_utils.as_image_file_list(DataFiles.WELLS)

    # If no wells files available return empty per-scale tensors
    if len(wells_paths) == 0:
        return tuple(
            _empty_pyramid_tensor(scale, channels_last) for scale in scale_list
        )

    for facie_path in wells_paths:
        pyramid = wells_interpolator.interpolate(facie_path, scale_list)  # type: ignore
        for i in range(len(scale_list)):
            # Normalize wells but preserve zeros (sparse structure)
            # Only normalize non-zero pixels, keep zeros as zeros
            well_data = pyramid[i]
            mask = (well_data.abs() > 0.001).float()
            normalized = norm(well_data) * mask
            pyramids_list[i].append(normalized)

    pyramids = _stack_and_format(pyramids_list, channels_last)
    return tuple(pyramids)


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
