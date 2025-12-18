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


@memory.cache  # type: ignore
def to_facies_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for facies images using a neural interpolator.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors as produced by ``ops.generate_scales``. Each
        element must be a 4-tuple ``(batch, channels, height, width)`` describing
        the target resolution.

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
        pyramids: list[torch.Tensor] = []
        for scale in scale_list:
            _, channels, height, width = scale
            pyramids.append(
                torch.empty((0, channels, height, width), dtype=torch.float32)
            )
        return tuple(pyramids)

    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    for facie_path, model_path in zip(facies_paths, models_paths):
        neural_smoother = NeuralSmoother(model_path, InterpolatorConfig())
        pyramid = neural_smoother.interpolate(facie_path, scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(torch_utils.norm(pyramid[i]))
    pyramids = [
        torch.stack(pyramid, dim=0).squeeze(1).permute(0, 3, 1, 2)
        for pyramid in pyramids_list
    ]
    return tuple(pyramids)


@memory.cache  # type: ignore
def to_seismic_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for seismic images using nearest interpolation.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, channels, height, width)``.

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
        pyramids: list[torch.Tensor] = []
        for scale in scale_list:
            _, channels, height, width = scale
            pyramids.append(
                torch.empty((0, channels, height, width), dtype=torch.float32)
            )
        return tuple(pyramids)

    seismic_interpolator = NearestInterpolator(InterpolatorConfig())
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    for seismic_path in seismic_paths:
        pyramid = seismic_interpolator.interpolate(seismic_path, scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(norm(pyramid[i]))
    pyramids = [
        torch.stack(pyramid, dim=0).squeeze(1).permute(0, 3, 1, 2)
        for pyramid in pyramids_list
    ]
    return tuple(pyramids)


@memory.cache  # type: ignore
def to_wells_pyramids(
    scale_list: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    """Generate multi-scale pyramid tensors for well location data.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, channels, height, width)``.

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
    wells_interpolator = WellInterpolator(InterpolatorConfig())
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    wells_paths = data_utils.as_image_file_list(DataFiles.WELLS)

    # If no wells files available return empty per-scale tensors
    if len(wells_paths) == 0:
        pyramids: list[torch.Tensor] = []
        for scale in scale_list:
            _, channels, height, width = scale
            pyramids.append(
                torch.empty((0, channels, height, width), dtype=torch.float32)
            )
        return tuple(pyramids)

    for facie_path in wells_paths:
        pyramid = wells_interpolator.interpolate(facie_path, scale_list)  # type: ignore
        for i in range(len(scale_list)):
            # Normalize wells but preserve zeros (sparse structure)
            # Only normalize non-zero pixels, keep zeros as zeros
            well_data = pyramid[i]
            mask = (well_data.abs() > 0.001).float()
            normalized = norm(well_data) * mask
            pyramids_list[i].append(normalized)
    pyramids = [
        torch.stack(pyramid, dim=0).squeeze(1).permute(0, 3, 1, 2)
        for pyramid in pyramids_list
    ]
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
