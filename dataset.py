
from typing import TypeAlias

import torch
from torch.utils.data import Dataset

from gen_pyramids import (
    to_facies_pyramids, 
    to_seismic_pyramids, 
    to_wells_pyramids, 
    memory,
)
from ops import generate_scales
from options import TrainningOptions


Pyramid: TypeAlias = tuple[torch.Tensor, ...]
Masks: TypeAlias = tuple[Pyramid, Pyramid, Pyramid]


class PyramidsDataset(Dataset[Masks]):
    """PyTorch Dataset for loading geological facies, wells, and seismic data at multiple scales.

    This dataset generates multi-scale pyramid representations of facies images,
    well log data, and seismic data using cached interpolation. Each pyramid
    level contains tensors at different resolutions for progressive GAN training.

    Attributes
    ----------
    data_dir : str
        Directory containing the input data files.
    scales_list : tuple[tuple[int, ...], ...]
        Tuple of (batch, channels, height, width) tuples for each pyramid scale.
    facies_pyramids : list[torch.Tensor]
        List of tensors containing facies data at different scales.
    wells_pyramids : list[torch.Tensor]
        List of tensors containing well log data at different scales.
    seismic_pyramids : list[torch.Tensor]
        List of tensors containing seismic data at different scales.

    Parameters
    ----------
    options : TrainningOptions
        Training configuration containing input path, batch size, and scale parameters.
    shuffle : bool, optional
        Whether to shuffle the dataset after loading. Defaults to False.
    regenerate : bool, optional
        Whether to clear the cache and regenerate pyramids. Defaults to False.
    """

    def __init__(
        self,
        options: TrainningOptions,
        shuffle: bool = False,
        regenerate: bool = False,
    ) -> None:
        self.data_dir = options.input_path
        self.scales_list = generate_scales(options)
    
        self.wells_pyramids = [
            torch.empty((0, 1, *scale), dtype=torch.int32) for scale in self.scales_list
        ]
        self.seismic_pyramids = [
            torch.empty((0, 1, *scale), dtype=torch.float32) for scale in self.scales_list
        ]
        if regenerate:
            memory.clear(warn=False)

        self.facies_pyramids = to_facies_pyramids(self.scales_list)
        self.wells_pyramids = to_wells_pyramids(self.scales_list)   
        self.seismic_pyramids = to_seismic_pyramids(self.scales_list)

        if shuffle:
            self.shuffle()

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of facies samples across all pyramid scales.
        """
        return self.facies_pyramids[0].shape[0]

    def __getitem__(self, idx: int) -> Masks:
        """Retrieve facies, wells, and seismic data at the specified index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve from the dataset.

        Returns
        -------
        Masks
            Tuple of three pyramids: (facies_pyramid, wells_pyramid, seismic_pyramid),
            where each pyramid is a tuple of tensors at different scales.
        """
        return (
            tuple(facies[idx] for facies in self.facies_pyramids), 
            tuple(wells[idx] for wells in self.wells_pyramids),
            tuple(seismic[idx] for seismic in self.seismic_pyramids),
        )

    def shuffle(self) -> None:
        """Shuffle all pyramids using the same random permutation.

        Applies a random permutation to facies, wells, and seismic pyramids
        at all scales, ensuring that corresponding samples remain aligned
        across the three data types.
        """
        idxs = torch.randperm(self.__len__())
        for i in range(len(self.scales_list)):
            self.facies_pyramids[i] = self.facies_pyramids[i][idxs]
            self.wells_pyramids[i] = self.wells_pyramids[i][idxs]
            self.seismic_pyramids[i] = self.seismic_pyramids[i][idxs]
