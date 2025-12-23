from typing import Any, Iterator
import torch
from torch.utils.data import DataLoader

from datasets.data_prefetcher import DataPrefetcher
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
    loader : DataLoader
        The underlying data loader.
    scales : list[int]
        List of scales to prepare data for.
    """

    def __init__(
        self,
        loader: DataLoader[torch.Tensor],
        scales: list[int],
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

    def prepare_batch_async(self, batch: Batch[torch.Tensor]) -> tuple[
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
    ]:
        """Perform batch preparation logic identical to the old prepare_scale_batch."""
        facies, wells, seismic = batch

        real_facies_dict: dict[int, torch.Tensor] = {}
        masks_dict: dict[int, torch.Tensor] = {}
        wells_dict: dict[int, torch.Tensor] = {}
        seismic_dict: dict[int, torch.Tensor] = {}

        for scale in self.scales:
            # Real facies
            facies_batch = facies[scale]
            # When inside a stream context, to() calls are asynchronous
            real_facies_dict[scale] = utils.to_device(
                facies_batch, self.device, channels_last=True
            )

            # Wells
            if len(wells) > 0:
                wells_batch = wells[scale]
                wells_dev = utils.to_device(
                    wells_batch, self.device, channels_last=True
                )
                # Compute mask on device
                masks_dev = (wells_dev.abs().sum(dim=1, keepdim=True) > 0).int()
                masks_dev = utils.to_device(masks_dev, self.device, channels_last=True)
                wells_dict[scale] = wells_dev
                masks_dict[scale] = masks_dev

            # Seismic
            if len(seismic) > 0:
                seismic_batch = seismic[scale]
                seismic_dev = utils.to_device(
                    seismic_batch, self.device, channels_last=True
                )
                seismic_dict[scale] = seismic_dev

        return (real_facies_dict, masks_dict, wells_dict, seismic_dict)

    def next(self) -> tuple[Batch[torch.Tensor] | None, Any | None]:
        """Return the next batch and trigger loading of the subsequent one.

        Waits for the stream to finish before returning if CUDA is enabled.
        """
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)  # type: ignore

        batch = self.next_batch
        prepared = self.next_prepared

        if batch is not None:
            self.preload()

        return batch, prepared

    def __iter__(self) -> Iterator[tuple[Batch[torch.Tensor] | None, Any | None]]:
        """Iterate over (raw_batch, prepared_batch) pairs until the loader
        is exhausted.

        Returns
        -------
        An iterator over tuples of raw and prepared batches.
        """
        batch, prepared = self.next()
        while batch is not None:
            yield batch, prepared
            batch, prepared = self.next()
