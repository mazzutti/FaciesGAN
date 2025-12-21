"""Dataset of multi-scale pyramids for facies, wells and seismic data.

This module provides :class:`PyramidsDataset` which loads precomputed
multi-resolution tensors (pyramids) for facies images and optional
conditioning channels (wells, seismic). Each dataset item is a tuple of
per-scale tensors used by the training pipeline.
"""

import mlx.core as mx
import datasets.mlx.utils as mlx_utils
import datasets.torch.utils as torch_utils
import datasets.utils as data_utils
from datasets.base import PyramidsDataset
from options import TrainningOptions
import random


class MLXPyramidsDataset(PyramidsDataset[mx.array]):
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
    scales : tuple[tuple[int, ...], ...]
        Pyramid descriptors (one tuple per scale) produced by
        :func:`datasets.utils.generate_scales`.
    batches : list[Batch[mx.array]]
        In-memory list of per-sample :class:`typedefs.Batch` objects created
        from the generated pyramids. Use :meth:`get_scale_data` to obtain
        stacked per-scale tensors (facies, wells, seismic) shaped
        ``(N, H, W, C)``.

    Notes on pyramid generators
    ---------------------------
    The low-level pyramid helper :meth:`generate_pyramids` returns a tuple of
    per-scale tensors (facies, wells, seismic). Those generator results are
    used to populate ``self.batches`` during initialization but are not kept
    as persistent attributes on the instance; callers that need per-scale
    stacked tensors should call :meth:`get_scale_data`.

    Notes
    -----
    - The dataset returns tensors in channel-last format.
    - Empty conditioning tuples are represented as ``()``; callers should
      handle that case when composing inputs for the generator/discriminator.
    """

    def __init__(
        self,
        options: TrainningOptions,
        shuffle: bool = False,
        regenerate: bool = False,
        channels_last: bool = True,
    ) -> None:
        """Initialize dataset and optionally shuffle or regenerate caches.

        Parameters
        ----------
        options : TrainningOptions
            Training options that include `input_path`, `use_wells`,
            `use_seismic` and scale generation parameters.
        shuffle : bool, optional
            Shuffle dataset order immediately after loading, by default False.
        regenerate : bool, optional`
            Clear in-memory pyramid cache and regenerate precomputed pyramids,
            by default False.
        channels_last : bool, optional
            Whether the channel dimension is last in the tensor shape, by
        """
        super().__init__(
            options,
            shuffle=shuffle,
            regenerate=regenerate,
            channels_last=channels_last,
        )

    def generate_pyramids(
        self, channels_last: bool = True
    ) -> tuple[tuple[mx.array, ...], ...]:
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
        facies_pyramids = mlx_utils.to_facies_pyramids(
            self.scales, channels_last=channels_last
        )
        wells_pyramids = mlx_utils.to_wells_pyramids(
            self.scales, channels_last=channels_last
        )
        seismic_pyramids = mlx_utils.to_seismic_pyramids(
            self.scales, channels_last=channels_last
        )

        return facies_pyramids, wells_pyramids, seismic_pyramids

    def generate_scales(
        self, options: TrainningOptions, channels_last: bool = True
    ) -> tuple[tuple[int, ...], ...]:
        """Generate the scales list used by the dataset.

        Parameters
        ----------
        options : TrainningOptions
            Training options that include scale generation parameters.
        channels_last : bool, optional
            Whether the channel dimension is last in the tensor shape, by default True.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Tuple of scale descriptors as produced by ``utils.generate_scales``.
        """
        return data_utils.generate_scales(options, channels_last)

    def shuffle(self) -> None:
        """Shuffle the dataset samples in-place."""
        random.shuffle(self.batches)

    def clean_cache(self) -> None:
        """Clear any in-memory or on-disk cache used by the dataset."""
        torch_utils.memory.clear(warn=False)

    def get_scale_data(self, scale: int | None = None) -> tuple[mx.array, ...]:
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
        facies_scale = mx.stack(facies_list, axis=0)

        # Wells may be an empty tuple on each batch; handle gracefully
        if len(self.batches[0].wells) == 0:
            wells_scale = mx.array([], dtype=facies_scale.dtype)
        else:
            wells_list = [batch.wells[scale] for batch in self.batches]
            wells_scale = mx.stack(wells_list, axis=0)

        if len(self.batches[0].seismic) == 0:
            seismic_scale = mx.array([], dtype=facies_scale.dtype)
        else:
            seismic_list = [batch.seismic[scale] for batch in self.batches]
            seismic_scale = mx.stack(seismic_list, axis=0)

        return facies_scale, wells_scale, seismic_scale
