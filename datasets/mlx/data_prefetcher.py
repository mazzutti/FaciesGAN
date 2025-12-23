from typing import Any, Iterator
from torch.utils.data import DataLoader
from datasets.data_prefetcher import DataPrefetcher

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
        loader: DataLoader[mx.array],
        scales: list[int],
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

    def prepare_batch_async(self, batch: Batch[mx.array]) -> tuple[
        dict[int, mx.array],
        dict[int, mx.array],
        dict[int, mx.array],
        dict[int, mx.array],
    ]:
        """Perform batch preparation logic asynchronously (vectorized where possible)."""
        facies, wells, seismic = batch

        # Vectorized conversion for facies
        real_facies_dict = {scale: mx.array(facies[scale]) for scale in self.scales}

        # Vectorized wells and masks if wells are present
        if len(wells) > 0:
            wells_dict = {scale: mx.array(wells[scale]) for scale in self.scales}
            # Compute masks for all scales at once
            masks_dict = {
                scale: mx.greater(
                    mx.sum(mx.abs(wells_dict[scale]), axis=3, keepdims=True), 0
                ).astype(mx.int32)
                for scale in self.scales
            }
        else:
            wells_dict = {}
            masks_dict = {}

        # Vectorized seismic if present
        if len(seismic) > 0:
            seismic_dict = {scale: mx.array(seismic[scale]) for scale in self.scales}
        else:
            seismic_dict = {}

        return (real_facies_dict, masks_dict, wells_dict, seismic_dict)

    def next(self) -> tuple[Batch[mx.array] | None, Any | None]:
        """Return the next batch and trigger loading of the subsequent one.

        Returns
        -------
        A tuple ``(raw_batch, prepared_batch)`` where ``raw_batch`` is the
        next raw batch from the loader (or ``None`` if no more batches are
        available), and ``prepared_batch`` is the corresponding prepared batch
        (or ``None`` if no more batches are available).
        """
        batch = self.next_batch
        prepared = self.next_prepared

        if batch is not None:
            self.preload()

        return batch, prepared

    def __iter__(self) -> Iterator[tuple[Batch[mx.array] | None, Any | None]]:
        """Iterator over ``(raw_batch, prepared_batch)`` pairs.
        Subclasses should provide a concrete iteration strategy, typically
        by repeatedly calling ``self.next()`` until no batch remains.

        Returns
        -------
        An iterator over tuples of raw and prepared batches.
        """
        batch, prepared = self.next()
        while batch is not None:
            yield batch, prepared
            batch, prepared = self.next()
