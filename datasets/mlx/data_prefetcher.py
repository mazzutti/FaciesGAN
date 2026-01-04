from typing import Iterator
from torch.utils.data import DataLoader
from datasets.data_prefetcher import DataPrefetcher, PyramidsBatch

import mlx.core as mx
from typedefs import Batch


class MLXDataPrefetcher(DataPrefetcher[mx.array]):
    """
    Prefetches data to the CPU stream to overlap data preparation with
    GPU computation.

    Parameters
    ----------
    loader : DataLoader
        The underlying data loader.
    trainer : MLXTrainer
        The trainer instance (needed for configuration).
    scales : list[int]
        List of scales to prepare data for.`
    """

    def __init__(
        self,
        loader: DataLoader[Batch[mx.array]],
        scales: tuple[int, ...],
        device: mx.Device = mx.cpu,  # type: ignore
    ) -> None:
        super().__init__(loader, scales)
        self.stream = mx.new_stream(device)  # type: ignore

    @property
    def stream(self) -> mx.Stream | None:
        return self._stream

    @stream.setter
    def stream(self, value: mx.Stream | None) -> None:
        self._stream = value
        self.next_batch = None
        self.next_prepared = None
        self.preload()

    def preload(self) -> None:
        """Preload the next batch and queue preparation on the stream."""
        try:
            # Fetch indices/batch from the torch dataloader
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            self.next_prepared = None
            return

        # Queue the MLX data preparation on the CPU stream.
        assert self.stream is not None
        with mx.stream(self.stream):
            if self.next_batch is not None:
                self.next_prepared = self.prepare_batch_async(self.next_batch)
            else:
                self.next_prepared = None

    def prepare_batch_async(self, batch: Batch[mx.array]) -> PyramidsBatch[mx.array]:
        """Perform batch preparation logic asynchronously (vectorized where possible)."""
        facies, wells, seismic = batch

        # Vectorized conversion for facies
        facies_pyramid = tuple(mx.array(facies[scale]) for scale in self.scales)

        # Vectorized wells and masks if wells are present
        if len(wells) > 0:
            wells_pyramid = tuple(mx.array(wells[scale]) for scale in self.scales)
            # Compute masks for all scales at once (vectorized)
            masks_pyramid = tuple(
                mx.greater(
                    mx.sum(mx.abs(wells_pyramid[scale]), axis=3, keepdims=True), 0
                )
                for scale in self.scales
            )
        else:
            wells_pyramid = ()
            masks_pyramid = ()

        # Vectorized seismic if present
        if len(seismic) > 0:
            seismic_pyramid = tuple(mx.array(seismic[scale]) for scale in self.scales)
        else:
            seismic_pyramid = ()

        return (facies_pyramid, wells_pyramid, masks_pyramid, seismic_pyramid)

    def next(self) -> PyramidsBatch[mx.array] | None:
        """Return the next batch and trigger loading of the subsequent one.

        Returns
        -------
        DictBatch[mx.array] | None
            The prepared batch, or ``None`` if no more batches are available.
        """
        batch = self.next_batch
        prepared = self.next_prepared

        if batch is not None:
            self.preload()

        return prepared

    def __iter__(self) -> Iterator[PyramidsBatch[mx.array] | None]:
        """Iterator over ``(raw_batch, prepared_batch)`` pairs.
        Subclasses should provide a concrete iteration strategy, typically
        by repeatedly calling ``self.next()`` until no batch remains.

        Returns
        -------
        An iterator over tuples of raw and prepared batches.
        """
        prepared = self.next()
        while prepared is not None:
            yield prepared
            prepared = self.next()
