"""Dataset of multi-scale pyramids for facies, wells and seismic data.

This module provides :class:`PyramidsDataset` which loads precomputed
multi-resolution tensors (pyramids) for facies images and optional
conditioning channels (wells, seismic). Each dataset item is a tuple of
per-scale tensors used by the training pipeline.
"""

from typing import TypeAlias

import torch
from torch.utils.data import Dataset

import gen_pyramids as gp
import ops
from options import TrainningOptions

Pyramid: TypeAlias = tuple[torch.Tensor, ...]
Masks: TypeAlias = tuple[Pyramid, Pyramid, Pyramid]


class PyramidsDataset(Dataset[Masks]):
    """Dataset of multi-scale facies and optional conditioning data.

    This dataset loads precomputed pyramids of facies (categorical images) and
    optional conditioning channels (wells and seismic) at multiple spatial
    resolutions. Each item is a tuple containing:

    - a tuple of facies tensors (one tensor per scale),
    - a tuple of wells tensors (one per scale) or empty tuple when wells are
      not used, and
    - a tuple of seismic tensors (one per scale) or empty tuple when seismic
      data is not used.

    Parameters
    ----------
    options : TrainningOptions
        Configuration object that provides `input_path`, booleans indicating
        whether wells/seismic are used, and the scale generation parameters.
    shuffle : bool, optional
        Whether to shuffle the dataset on creation, by default False.
    regenerate : bool, optional
        If True, forces regeneration of pyramid memory caches, by default False.

    Attributes
    ----------
    data_dir : str
        Root directory containing input facies files.
    scales_list : list[tuple]
        List of pyramid shapes used by the generator and dataset.
    facies_pyramids : list[torch.Tensor]
        Per-scale facies tensors shaped (N, C, H, W).
    wells_pyramids : list[torch.Tensor]
        Per-scale well-conditioning tensors (may be empty tensors when unused).
    seismic_pyramids : list[torch.Tensor]
        Per-scale seismic-conditioning tensors (may be empty when unused).
    """

    def __init__(
        self,
        options: TrainningOptions,
        shuffle: bool = False,
        regenerate: bool = False,
    ) -> None:
        """Initialize dataset and optionally shuffle or regenerate caches.

        Parameters
        ----------
        options : TrainningOptions
            Training options that include `input_path`, `use_wells`,
            `use_seismic` and scale generation parameters.
        shuffle : bool, optional
            Shuffle dataset order immediately after loading, by default False.
        regenerate : bool, optional
            Clear in-memory pyramid cache and regenerate precomputed pyramids,
            by default False.
        """
        self.data_dir = options.input_path
        self.scales_list = ops.generate_scales(options)

        self.wells_pyramids = [
            torch.empty((0, 1, *scale), dtype=torch.int32) for scale in self.scales_list
        ]
        self.seismic_pyramids = [
            torch.empty((0, 1, *scale), dtype=torch.float32)
            for scale in self.scales_list
        ]
        if regenerate:
            gp.memory.clear(warn=False)

        self.facies_pyramids = gp.to_facies_pyramids(self.scales_list)
        if options.use_wells:
            self.wells_pyramids = gp.to_wells_pyramids(self.scales_list)
        if options.use_seismic:
            self.seismic_pyramids = gp.to_seismic_pyramids(self.scales_list)

        if shuffle:
            self.shuffle()

    def __len__(self):
        """Return the number of samples (pyramids) available.

        Returns
        -------
        int
            Number of top-level pyramid samples (length of the first scale).
        """
        return int(self.facies_pyramids[0].shape[0])

    def __getitem__(self, idx: int) -> Masks:
        """Fetch the pyramid sample at index ``idx``.

        Parameters
        ----------
        idx : int
            Index of the pyramid sample to retrieve.

        Returns
        -------
        Masks
            A 3-tuple: (facies_pyramid, wells_pyramid, seismic_pyramid), where
            each element is itself a tuple of per-scale tensors. If wells or
            seismic data are not available the corresponding element is an
            empty tuple.
        """
        facies_tuple = tuple(facies[idx] for facies in self.facies_pyramids)
        wells_tuple = (
            tuple(wells[idx] for wells in self.wells_pyramids)
            if self.wells_pyramids[0].shape[0]
            else ()
        )
        seismic_tuple = (
            tuple(seismic[idx] for seismic in self.seismic_pyramids)
            if self.seismic_pyramids[0].shape[0]
            else ()
        )

        return facies_tuple, wells_tuple, seismic_tuple

    def shuffle(self) -> None:
        """Shuffle all per-scale tensors in the dataset in the same order.

        The shuffle is performed in-place and keeps facies/wells/seismic
        correspondences aligned across scales.
        """
        idxs = torch.randperm(self.__len__())
        for i in range(len(self.scales_list)):
            self.facies_pyramids[i] = self.facies_pyramids[i][idxs]
            if self.wells_pyramids[0].shape[0]:
                self.wells_pyramids[i] = self.wells_pyramids[i][idxs]
            if self.seismic_pyramids[0].shape:
                self.seismic_pyramids[i] = self.seismic_pyramids[i][idxs]
