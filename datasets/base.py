"""Base dataset utilities for pyramid datasets.

Provides a `BasePyramidsDataset` that encapsulates common loading and
shuffling behaviour used by the framework-specific dataset implementations
(e.g. `datasets.torch.dataset.TorchPyramidsDataset`).
"""

from abc import abstractmethod
from itertools import repeat

from torch.utils.data import Dataset

from options import TrainningOptions
from typedefs import Batch, TTensor


class PyramidsDataset(Dataset[Batch[TTensor]]):
    """Common functionality for multi-scale pyramid datasets.

    This class centralises the shared logic for loading precomputed
    pyramids of facies, wells and seismic data so backend-specific
    dataset subclasses can focus on framework-specific behaviour.


    Parameters
    ----------
    options : TrainningOptions
        Configuration object that provides `input_path`, booleans indicating
        whether wells/seismic are used, and the scale generation parameters.
    shuffle : bool, optional
        Whether to shuffle the dataset on creation, by default False.
    regenerate : bool, optional
        If True, forces regeneration of pyramid memory caches, by default False.
    """

    def __init__(
        self,
        options: TrainningOptions,
        shuffle: bool = False,
        regenerate: bool = False,
        channels_last: bool = False,
    ) -> None:
        self.data_dir = options.input_path
        self.batches: list[Batch[TTensor]] = []
        self.scales = self.generate_scales(options, channels_last)
        if regenerate:
            self.clean_cache()

        fp, wp, sp = self.generate_pyramids()

        n_samples = 0
        if fp and fp[0].shape[0] > 0:
            n_samples = int(fp[0].shape[0])

        if n_samples:
            has_wells = bool(wp and wp[0].shape[0] > 0)
            has_seismic = bool(sp and sp[0].shape[0] > 0)
            facies_iter = zip(*fp)
            wells_iter = zip(*wp) if has_wells else repeat(())
            seismic_iter = zip(*sp) if has_seismic else repeat(())

            for facies, wells, seismic in zip(facies_iter, wells_iter, seismic_iter):
                wells = wells if has_wells else ()
                seismic = seismic if has_seismic else ()
                self.batches.append(Batch(facies=facies, wells=wells, seismic=seismic))

        if shuffle:
            self.shuffle()

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> Batch[TTensor]:
        return self.batches[idx]

    @abstractmethod
    def generate_pyramids(self) -> tuple[tuple[TTensor, ...], ...]:
        """Generate the scales list used by the dataset.

        Returns
        -------
        tuple[tuple[TTensor, ...], ...]
            Tuple of scale descriptors as produced by ``self.generate_scales``.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "Subclasses must implement the generate_scales method."
        )

    @abstractmethod
    def generate_scales(
        self, options: TrainningOptions, channels_last: bool = False
    ) -> tuple[tuple[int, ...], ...]:
        """Generate the scales list used by the dataset.

        Parameters
        ----------
        options : TrainningOptions
            Training options that include scale generation parameters.
        channels_last : bool, optional
            Whether the channel dimension is last in the tensor shape, by default False.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Tuple of scale descriptors as produced by ``ops.generate_scales``.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "Subclasses must implement the generate_scales method."
        )

    @abstractmethod
    def shuffle(self) -> None:
        """Shuffle the dataset samples in-place.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the shuffle method.")

    @abstractmethod
    def clean_cache(self) -> None:
        """Clear any in-memory or on-disk cache used by the dataset.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the clean_cache method.")

    @abstractmethod
    def get_scale_data(self, scale: int) -> tuple[TTensor, ...]:
        """Return the facies, wells and seismic tensors for a given scale.

        Parameters
        ----------
        scale : int | None
            Index of the scale to return. If ``None`` the finest (last)
            scale is returned.

        Returns
        -------
        tuple[TTensor, TTensor, TTensor]
            Facies, wells and seismic tensors for the requested scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        Notes
        -----
        Subclasses are expected to populate ``self.facies_pyramids``,
        ``self.wells_pyramids`` and ``self.seismic_pyramids`` when
        generating pyramids (see :meth:`generate_pyramids`). If those
        attributes are not present this method will call
        ``self.generate_pyramids()`` and cache the results on the
        instance before returning the requested scale.
        """
        raise NotImplementedError(
            "Subclasses must implement the get_scale_data method."
        )
