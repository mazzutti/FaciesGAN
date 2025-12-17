"""Dataset of multi-scale pyramids for facies, wells and seismic data.

This module provides :class:`PyramidsDataset` which loads precomputed
multi-resolution tensors (pyramids) for facies images and optional
conditioning channels (wells, seismic). Each dataset item is a tuple of
per-scale tensors used by the training pipeline.
"""

import torch

import datasets.torch.utils as torch_utils
import datasets.utils as utils
from datasets.base import BasePyramidsDataset
from options import TrainningOptions


class TorchPyramidsDataset(BasePyramidsDataset[torch.Tensor]):
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
        super().__init__(options, shuffle=shuffle, regenerate=regenerate)

    def generate_pyramids(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Generate the scales list used by the dataset.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Tuple of scale descriptors as produced by ``self.generate_scales``.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        facies_pyramids: tuple[torch.Tensor, ...] = torch_utils.to_facies_pyramids(
            self.scales_list
        )
        wells_pyramids: tuple[torch.Tensor, ...] = torch_utils.to_wells_pyramids(
            self.scales_list
        )
        seismic_pyramids: tuple[torch.Tensor, ...] = torch_utils.to_seismic_pyramids(
            self.scales_list
        )

        return facies_pyramids, wells_pyramids, seismic_pyramids

    def generate_scales(self, options: TrainningOptions) -> tuple[tuple[int, ...], ...]:
        """Generate the scales list used by the dataset.

        Parameters
        ----------
        options : TrainningOptions
            Training options that include scale generation parameters.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Tuple of scale descriptors as produced by ``utils.generate_scales``.
        """
        return utils.generate_scales(options)

    def shuffle(self) -> None:
        """Shuffle the dataset samples in-place."""
        idxs = torch.randperm(len(self.batches))
        self.batches = [self.batches[i] for i in idxs]

    def clean_cache(self) -> None:
        """Clear any in-memory or on-disk cache used by the dataset."""
        torch_utils.memory.clear(warn=False)
