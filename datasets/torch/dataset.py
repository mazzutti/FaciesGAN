"""Dataset of multi-scale pyramids for facies, wells and seismic data.

This module provides :class:`PyramidsDataset` which loads precomputed
multi-resolution tensors (pyramids) for facies images and optional
conditioning channels (wells, seismic). Each dataset item is a tuple of
per-scale tensors used by the training pipeline.
"""

import torch

import datasets.torch.utils as torch_utils
import datasets.utils as data_utils
from datasets.base import PyramidsDataset
from options import TrainningOptions


class TorchPyramidsDataset(PyramidsDataset[torch.Tensor]):
    """Torch-specific dataset for multi-scale facies with optional conditioning.

    Loads precomputed multi-resolution pyramids for facies images and optional
    conditioning channels (wells, seismic). Each dataset item (a "batch")
    exposes three tuple attributes: ``facies``, ``wells`` and ``seismic``,
    where each is a tuple containing one tensor per scale.

    Layout and shapes
    -----------------
    - Per-sample facies tensors have shape ``(C, H, W)``. When stacked into a
      batch they form a tensor of shape ``(N, C, H, W)`` (channel-first /
      NCHW). The dataset preserves the channel-first layout; downstream
      helpers may convert to channels-last when requested.
    - ``wells`` and ``seismic`` follow the same spatial layout and channel
      conventions as the preprocessed inputs. If a conditioning type is not
      used, the corresponding tuple will be empty for each batch.

    Parameters
    ----------
    options : TrainningOptions
        Training options containing ``input_path``, flags for using wells/
        seismic, and scale-generation parameters.
    shuffle : bool, optional
        Shuffle the dataset immediately after loading, by default ``False``.
    regenerate : bool, optional
        If ``True``, force regeneration of any in-memory pyramid caches,
        by default ``False``.

    Attributes
    ----------
    data_dir : str
        Root directory containing input facies files.
    scales : tuple[tuple[int, ...], ...]
        Pyramid descriptors (one tuple per scale) produced by
        :func:`datasets.utils.generate_scales`.
    batches : list[Batch[torch.Tensor]]
        In-memory list of per-sample :class:`typedefs.Batch` objects created
        from the generated pyramids. Use :meth:`get_scale_data` to obtain
        stacked per-scale tensors (facies, wells, seismic) shaped
        ``(N, C, H, W)``.

    Notes on pyramid generators
    ---------------------------
    The low-level pyramid helper :meth:`generate_pyramids` returns a tuple of
    per-scale tensors (facies, wells, seismic). Those generator results are
    used to populate ``self.batches`` during initialization but are not kept
    as persistent attributes on the instance; callers that need per-scale
    stacked tensors should call :meth:`get_scale_data`.

    Notes
    -----
    - The dataset returns tensors in channel-first format. Use
      :func:`datasets.torch.utils.to_device` with ``channels_last=True`` if a
      channels-last layout is required by the model or trainer.
    - Empty conditioning tuples are represented as ``()``; callers should
      handle that case when composing inputs for the generator/discriminator.
    """

    def __init__(
        self,
        options: TrainningOptions,
        shuffle: bool = False,
        regenerate: bool = False,
        channels_last: bool = False,
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
        super().__init__(
            options,
            shuffle=shuffle,
            regenerate=regenerate,
            channels_last=channels_last,
        )

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
            self.scales
        )
        wells_pyramids: tuple[torch.Tensor, ...] = torch_utils.to_wells_pyramids(
            self.scales
        )
        seismic_pyramids: tuple[torch.Tensor, ...] = torch_utils.to_seismic_pyramids(
            self.scales
        )

        return facies_pyramids, wells_pyramids, seismic_pyramids

    def generate_scales(
        self, options: TrainningOptions, channels_last: bool = False
    ) -> tuple[tuple[int, ...], ...]:
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
        return data_utils.generate_scales(options, channels_last)

    def shuffle(self) -> None:
        """Shuffle the dataset samples in-place."""
        idxs = torch.randperm(len(self.batches))
        self.batches = [self.batches[i] for i in idxs]

    def clean_cache(self) -> None:
        """Clear any in-memory or on-disk cache used by the dataset."""
        torch_utils.memory.clear(warn=False)

    def get_scale_data(self, scale: int | None = None) -> tuple[torch.Tensor, ...]:
        """Return facies, wells and seismic tensors for a given scale.

        This torch-specific implementation will construct per-scale tensors
        from ``self.batches`` if the cached ``self.facies_pyramids`` etc.
        are not available. This allows callers to access scale data even
        when the dataset was created from precomputed batch tuples.
        """
        if scale is None:
            scale = len(self.scales) - 1

        # Stack per-sample tensors for the requested scale
        facies_list = [batch.facies[scale] for batch in self.batches]
        facies_scale = torch.stack(facies_list, dim=0)

        # Wells may be an empty tuple on each batch; handle gracefully
        if len(self.batches[0].wells) == 0:
            wells_scale = torch.empty((0,), dtype=facies_scale.dtype)
        else:
            wells_list = [batch.wells[scale] for batch in self.batches]
            wells_scale = torch.stack(wells_list, dim=0)

        if len(self.batches[0].seismic) == 0:
            seismic_scale = torch.empty((0,), dtype=facies_scale.dtype)
        else:
            seismic_list = [batch.seismic[scale] for batch in self.batches]
            seismic_scale = torch.stack(seismic_list, dim=0)

        return facies_scale, wells_scale, seismic_scale
