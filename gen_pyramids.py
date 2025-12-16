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

from data_files import DataFiles
from interpolators.config import InterpolatorConfig
from interpolators.nearest import NearestInterpolator
from interpolators.neural import NeuralSmoother
from interpolators.well import WellInterpolator
from ops import as_image_file_list, as_model_file_list, norm

# Create a cache directory
memory = Memory("./.cache", verbose=0)


@memory.cache  # type: ignore
def to_facies_pyramids(scale_list: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    """Generate multi-scale pyramid tensors for facies images using a neural interpolator.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors as produced by ``ops.generate_scales``. Each
        element must be a 4-tuple ``(batch, channels, height, width)`` describing
        the target resolution.

    Returns
    -------
    list[torch.Tensor]
        A list where each element is a PyTorch tensor containing all facies at
        that scale with shape ``(N, C, H, W)`` (N images, C channels).
    """
    facies_paths = as_image_file_list(DataFiles.FACIES)
    models_paths = as_model_file_list(DataFiles.FACIES)
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    for facie_path, model_path in zip(facies_paths, models_paths):
        neural_smoother = NeuralSmoother(model_path, InterpolatorConfig())
        pyramid = neural_smoother.interpolate(facie_path, scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(norm(pyramid[i]))
    pyramids = [
        torch.stack(pyramid, dim=0).squeeze(1).permute(0, 3, 1, 2)
        for pyramid in pyramids_list
    ]
    return pyramids


@memory.cache  # type: ignore
def to_seismic_pyramids(scale_list: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    """Generate multi-scale pyramid tensors for seismic images using nearest interpolation.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, channels, height, width)``.

    Returns
    -------
    list[torch.Tensor]
        A list of tensors (one per scale) with shape ``(N, C, H, W)``.

    Notes
    -----
    Uses NearestInterpolator and caches results in ``./.cache``.
    """
    seismic_paths = as_image_file_list(DataFiles.SEISMIC)
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
    return pyramids


@memory.cache  # type: ignore
def to_wells_pyramids(scale_list: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    """Generate multi-scale pyramid tensors for well location data.

    Parameters
    ----------
    scale_list : tuple[tuple[int, ...], ...]
        Tuple of scale descriptors ``(batch, channels, height, width)``.

    Returns
    -------
    list[torch.Tensor]
        A list of tensors (one per scale) with shape ``(N, C, H, W)``. Well
        locations are represented as sparse vertical traces at the appropriate
        column indices.

    Notes
    -----
    Uses ``WellInterpolator`` and caches results in ``./.cache``.
    """
    wells_interpolator = WellInterpolator(InterpolatorConfig())
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    facies_paths = as_image_file_list(DataFiles.WELLS)
    for facie_path in facies_paths:
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
    return pyramids
