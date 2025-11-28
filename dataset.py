import gzip
import os
from enum import Enum
from typing import TypeAlias

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore

from ops import (
    facie_resize,
    generate_scales,
    mask_resize,
    np2torch,
)
from options import TrainningOptions


class DataFiles(Enum):
    """Constants for dataset filenames stored in the data directory."""

    FACIES = "facies"
    WELLS = "wells"
    SEISMIC = "seismic"

    def __init__(self,  value: str, data_dir: str = "./data") -> None:
        self.filename = f'{value}.npy.gz'
        self.data_dir = data_dir

    def as_data_path(self, data_dir: str | None = None) -> str:
        """Return the full filesystem path for this data file inside `data_dir`.

        Parameters
        ----------
        data_dir : str
            Directory where the dataset files are stored.

        Returns
        -------
        str
            Full path to the file represented by this enum member.
        """
        return os.path.join(data_dir or self.data_dir, self.filename)


# A type alias for (facies_pyramid, masks_pyramid, seismic_pyramid) returned by the dataset
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
        resizers (list): List of torchvision transforms for resizing facies and masks.
        ceiling (bool, optional): Whether to set all positive values to 1. Defaults to False.
    """

    def __init__(
        self,
        options: TrainningOptions,
        shuffle: bool = False,
        ceiling: bool = True,
        regenerate: bool = False,
    ) -> None:
        self.data_dir = options.input_path
        self.scales_list = generate_scales(options)
        self.facies_pyramid = [
            torch.empty((0, 1, *scale), dtype=torch.int32) for scale in self.scales_list
        ]
        self.wells_pyramid = [
            torch.empty((0, 1, *scale), dtype=torch.int32) for scale in self.scales_list
        ]
        self.seismic_pyramid = [
            torch.empty((0, 1, *scale), dtype=torch.float32) for scale in self.scales_list
        ]
        self.resizers = [transforms.Resize(scale[2:]) for scale in self.scales_list[:-1]]
        self.ceiling = ceiling
        self.regenerate = regenerate

        self._ensure_compressed_archives()

        facies = np2torch(
            np.load(gzip.open(DataFiles.FACIES.as_data_path(self.data_dir), "rb"))
        )

        wells = np2torch(
            np.load(gzip.open(DataFiles.WELLS.as_data_path(self.data_dir), "rb"))
        )

        seismic = np2torch(
            np.load(gzip.open(DataFiles.SEISMIC.as_data_path(self.data_dir), "rb"))
        )

        for i, scale in enumerate(self.scales_list):
            self.facies_pyramid[i] = torch.stack(
                [facie_resize(facie.unsqueeze(0), scale[2:], self.ceiling) for facie in facies],
                dim=0,
            ).squeeze(1)
            self.wells_pyramid[i] = torch.stack(
                [mask_resize(mask.unsqueeze(0), scale[2:]) for mask in wells], dim=0
            ).squeeze(1)
            self.seismic_pyramid[i] = torch.stack(
                [facie_resize(s.unsqueeze(0), scale[2:], False) for s in seismic], dim=0
            ).squeeze(1)

        if shuffle:
            self.shuffle()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.facies_pyramid[0].shape[0]

    def __getitem__(self, idx: int) -> Masks:
        """
        Returns the facies and masks at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple of facies and masks at different scales.
        """
        return (
            tuple(facies[idx] for facies in self.facies_pyramid), 
            tuple(masks[idx] for masks in self.wells_pyramid),
            tuple(seismic[idx] for seismic in self.seismic_pyramid),
        )

    def shuffle(self) -> None:
        """
        Shuffles the dataset.
        """
        idxs = torch.randperm(self.__len__())
        for i in range(len(self.scales_list)):
            self.facies_pyramid[i] = self.facies_pyramid[i][idxs]
            self.wells_pyramid[i] = self.wells_pyramid[i][idxs]
            self.seismic_pyramid[i] = self.seismic_pyramid[i][idxs]

    def _ensure_compressed_archives(self) -> None:
        """Ensure dataset files are present as compressed archives.

        Iterates over known `DataFiles` members and (re)writes a compressed
        archive for each. The current implementation loads the file and
        writes a compressed archive using `np.savez_compressed` at the same
        path.
        """

        for data_file in DataFiles:
            if self.regenerate or not os.path.exists(
                data_file.as_data_path(self.data_dir)
            ):
                self._write_compressed_archive(data_file) 
            

    def _write_compressed_archive(self, data_file: DataFiles) -> None:
        """Write a compressed archive for the given data file.

        Parameters
        ----------
        data_file : DataFiles
            The data file enum member to (re)write as a compressed archive.
        """

        uncompressed_path = data_file.as_data_path(self.data_dir)

        print(
            f"Regenerating compressed archive for {data_file.name} "
            f"at {data_file.as_data_path(self.data_dir)}"
        )

        data = np.load(uncompressed_path)

        with gzip.open(data_file.as_data_path(self.data_dir), "wb") as f:
            np.save(f, data)