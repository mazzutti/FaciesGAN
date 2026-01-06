from typing import Iterator
import torch
from torch.utils.data import DataLoader

from datasets.data_prefetcher import DataPrefetcher, PyramidsBatch
from typedefs import Batch
import utils


class TorchDataPrefetcher(DataPrefetcher[torch.Tensor]):
    """Prefetches and prepares batches on the GPU asynchronously.

    This class wraps a DataLoader and provides an iterator that yields
    tuples of (original_batch, prepared_batch). The preparation (moving
    tensors to device, creating masks) happens on a separate CUDA stream
    while the previous batch is being processed.

    Parameters
    ----------
    loader : DataLoader[Batch[torch.Tensor]]
        The underlying data loader.
    scales : tuple[int, ...]
        Tuple of scales to prepare data for.
    """

    def __init__(
        self,
        loader: DataLoader[Batch[torch.Tensor]],
        scales: tuple[int, ...],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(loader, scales)
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    @property
    def stream(self) -> torch.cuda.Stream | None:
        return self._stream

    @stream.setter
    def stream(self, value: torch.cuda.Stream | None) -> None:
        self._stream = value
        self.next_batch = None
        self.next_prepared = None
        self.preload()

    def preload(self) -> None:
        """Preload the next batch and queue preparation on the stream."""
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            self.next_prepared = None
            return

        # Queue preparation logic on the stream (if available)
        if self.next_batch:
            if self.stream:
                with torch.cuda.stream(self.stream):
                    self.next_prepared = self.prepare_batch_async(self.next_batch)
            else:
                self.next_prepared = self.prepare_batch_async(self.next_batch)
        else:
            self.next_prepared = None

    def prepare_batch_async(
        self, batch: Batch[torch.Tensor]
    ) -> PyramidsBatch[torch.Tensor]:
        """Perform batch preparation logic identical to the old prepare_scale_batch."""
        facies, wells, seismic = batch

        # Build per-scale pyramids as dictionaries indexed by scale number
        facies_pyramid: dict[int, torch.Tensor] = {
            scale: utils.to_device(facies[scale], self.device, channels_last=True)
            for scale in self.scales
        }

        if len(wells) > 0:
            wells_pyramid: dict[int, torch.Tensor] = {
                scale: utils.to_device(wells[scale], self.device, channels_last=True)
                for scale in self.scales
            }
            # Masks are computed from the device-resident well tensors
            masks_pyramid: dict[int, torch.Tensor] = {
                scale: (w.abs().sum(dim=1, keepdim=True) > 0).int()
                for scale, w in wells_pyramid.items()
            }
        else:
            wells_pyramid = {}
            masks_pyramid = {}

        if len(seismic) > 0:
            seismic_pyramid: dict[int, torch.Tensor] = {
                scale: utils.to_device(seismic[scale], self.device, channels_last=True)
                for scale in self.scales
            }
        else:
            seismic_pyramid = {}

        return (facies_pyramid, wells_pyramid, masks_pyramid, seismic_pyramid)

    def next(self) -> PyramidsBatch[torch.Tensor] | None:
        """Return the next batch and trigger loading of the subsequent one.

        Waits for the stream to finish before returning if CUDA is enabled.
        """
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)  # type: ignore

        batch = self.next_batch
        prepared = self.next_prepared

        if batch is not None:
            self.preload()

        return prepared

    def __iter__(self) -> Iterator[PyramidsBatch[torch.Tensor] | None]:
        """Iterate over (raw_batch, prepared_batch) pairs until the loader
        is exhausted.

        Returns
        -------
        Iterator[DictBatch[torch.Tensor] | None]
            An iterator over prepared batches.
        """
        prepared = self.next()
        while prepared is not None:
            yield prepared
            prepared = self.next()
