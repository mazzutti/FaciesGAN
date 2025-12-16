from typing import TypeAlias

import torch
from torch.utils.data import Dataset

from gen_pyramids import (memory, to_facies_pyramids, to_seismic_pyramids,
                          to_wells_pyramids)
from ops import generate_scales
from options import TrainningOptions

Pyramid: TypeAlias = tuple[torch.Tensor, ...]
Masks: TypeAlias = tuple[Pyramid, Pyramid, Pyramid]


class PyramidsDataset(Dataset[Masks]):
    """
    A PyTorch Dataset class for loading geological facies and their corresponding masks at multiple scales.

    Attributes:
        data_dir (str): Directory containing the facie files.
        scales_list (list): List of scales for resizing facies and masks.
        facies_pyramid (list): List of tensors containing facies at different scales.
        masks_pyramid (list): List of tensors containing masks at different scales.
        ceiling (bool, optional): Whether to set all positive values to 1. Defaults to False.
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
            torch.empty((0, 1, *scale), dtype=torch.float32)
            for scale in self.scales_list
        ]
        if regenerate:
            memory.clear(warn=False)

        self.facies_pyramids = to_facies_pyramids(self.scales_list)
        if options.use_wells:
            self.wells_pyramids = to_wells_pyramids(self.scales_list)
        if options.use_seismic:
            self.seismic_pyramids = to_seismic_pyramids(self.scales_list)

        if shuffle:
            self.shuffle()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.facies_pyramids[0].shape[0]

    def __getitem__(self, idx: int) -> Masks:
        """
        Returns the facies and masks at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple of facies and masks at different scales.
        """
        return (
            tuple(facies[idx] for facies in self.facies_pyramids),
            (
                tuple(wells[idx] for wells in self.wells_pyramids)
                if self.wells_pyramids[0].shape[0]
                else ()
            ),
            (
                tuple(seismic[idx] for seismic in self.seismic_pyramids)
                if self.seismic_pyramids[0].shape[0]
                else ()
            ),
        )

    def shuffle(self) -> None:
        """
        Shuffles the dataset.
        """
        idxs = torch.randperm(self.__len__())
        for i in range(len(self.scales_list)):
            self.facies_pyramids[i] = self.facies_pyramids[i][idxs]
            if self.wells_pyramids[0].shape[0]:
                self.wells_pyramids[i] = self.wells_pyramids[i][idxs]
            if self.seismic_pyramids[0].shape:
                self.seismic_pyramids[i] = self.seismic_pyramids[i][idxs]
