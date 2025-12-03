import torch

from data_files import DataFiles
from interpolators.config import InterpolatorConfig
from interpolators.nearest import NearestInterpolator
from interpolators.neural import NeuralSmoother
from interpolators.well import WellInterpolator
from ops import as_image_file_list, as_model_file_list

from joblib import Memory # type: ignore

# Create a cache directory
memory = Memory("./.cache", verbose=0)

@memory.cache # type: ignore
def to_facies_pyramids(scale_list: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    """Generate multi-scale pyramid representations of facies images using neural 
    interpolation.

    For each facies image and its corresponding model checkpoint, this function
    creates a NeuralSmoother instance and generates interpolated images at all
    requested scales. The results are organized into pyramids where each scale
    level contains tensors for all facies images.

    Parameters
    ----------
    scale_list : list[tuple[int, ...]]
        List of target resolutions as (height, width) tuples. Each resolution
        defines one level in the output pyramid.

    Returns
    -------
    list[torch.Tensor]
        List of stacked tensors, one per scale level. Each tensor has shape
        (N, H, W, 3) where N is the number of facies images, and (H, W)
        matches the corresponding resolution in scale_list.
    """
    facies_paths = as_image_file_list(DataFiles.FACIES)
    models_paths = as_model_file_list(DataFiles.FACIES)
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]

    for facie_path, model_path in zip(facies_paths, models_paths):
        neural_smoother = NeuralSmoother(model_path, InterpolatorConfig())
        pyramid = neural_smoother.interpolate(facie_path, scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(pyramid[i])
    pyramids = [
        torch.stack(pyramid, dim=0).squeeze(1) for pyramid in pyramids_list
    ]
    return pyramids

@memory.cache # type: ignore
def to_seismic_pyramids(scale_list: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    seismic_paths = as_image_file_list(DataFiles.SEISMIC)
    seismic_interpolator = NearestInterpolator(InterpolatorConfig())
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    for seismic_path in seismic_paths:
        pyramid = seismic_interpolator.interpolate(seismic_path, scale_list)
        for i in range(len(scale_list)):
            pyramids_list[i].append(pyramid[i])
    pyramids = [
        torch.stack(pyramid, dim=0).squeeze(1) for pyramid in pyramids_list
    ]
    return pyramids
    
@memory.cache # type: ignore
def to_wells_pyramids(scale_list: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    wells_interpolator = WellInterpolator(InterpolatorConfig())
    pyramids_list: list[list[torch.Tensor]] = [[] for _ in range(len(scale_list))]
    facies_paths = as_image_file_list(DataFiles.WELLS) # Ensure facies pyramids are cached
    for facie_path in facies_paths:
        pyramid = wells_interpolator.interpolate(facie_path, scale_list) # type: ignore
        for i in range(len(scale_list)):
            pyramids_list[i].append(pyramid[i])
    pyramids = [
        torch.stack(pyramid, dim=0).squeeze(1) for pyramid in pyramids_list
    ]
    return pyramids

