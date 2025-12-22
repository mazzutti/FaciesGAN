"""Generic DataPrefetcher utility for datasets.

This module provides a lightweight, generic `DataPrefetcher` that wraps a
PyTorch `DataLoader` and optionally prepares batches asynchronously on a
CUDA stream using a user-supplied `prepare_fn`.

The implementation is intentionally generic so it can be reused by both
torch-specific trainers and other code that needs async batch preparation.
"""

from __future__ import annotations

from typing import Any, Iterator
from typing_extensions import Generic
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from typedefs import Batch, TTensor


class DataPrefetcher(Generic[TTensor], ABC):
    """Wraps a :class:`torch.utils.data.DataLoader` and preloads the next
    batch while the current one is being processed.

    Parameters
    ----------
    loader:
        A PyTorch ``DataLoader`` instance to iterate over.
    prepare_fn:
        Optional callable that accepts a raw batch and returns a prepared
        batch (for example, moving tensors to a device). If ``None`` the
        raw batch is returned as the prepared value.
    device:
        Optional ``torch.device`` used to decide whether to use a CUDA
        stream. If not provided, defaults to ``torch.device('cpu')``.

    Behavior
    --------
    If ``device`` is a CUDA device and ``prepare_fn`` is provided, the
    preparation call will be enqueued on a dedicated CUDA stream so it can
    overlap with host/GPU work on the current stream.
    """

    def __init__(self, loader: DataLoader[TTensor], scales: list[int]) -> None:
        self.loader = iter(loader)
        self.scales = scales
        self.next_batch: Any | None = None
        self.next_prepared: Any | None = None

    @abstractmethod
    def preload(self) -> None:
        """Preload the next batch into ``self.next_batch`` and set
        ``self.next_prepared`` accordingly. Implementations should handle
        stream-based asynchronous preparation when appropriate.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement preload method")

    @abstractmethod
    def prepare_batch_async(self, batch: Batch[TTensor]) -> tuple[
        dict[int, TTensor],
        dict[int, TTensor],
        dict[int, TTensor],
        dict[int, TTensor],
    ]:
        """Perform batch preparation logic asynchronously.

        Parameters
        ----------
        batch:
            The raw batch to prepare.
        Returns
        -------
        A tuple of prepared batch components.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement prepare_batch_async method"
        )

    @abstractmethod
    def next(self) -> tuple[Batch[TTensor] | None, Any | None]:
        """Return the next batch and trigger loading of the subsequent one.

        Returns
        -------
        A tuple ``(raw_batch, prepared_batch)`` where ``raw_batch`` is the
        next raw batch from the loader (or ``None`` if no more batches are
        available), and ``prepared_batch`` is the corresponding prepared batch
        (or ``None`` if no more batches are available).

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement next method")

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[Batch[TTensor] | None, Any | None]]:
        """Iterator over ``(raw_batch, prepared_batch)`` pairs.
        Subclasses should provide a concrete iteration strategy, typically
        by repeatedly calling ``self.next()`` until no batch remains.

        Returns
        -------
        An iterator over tuples of raw and prepared batches.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        """
        raise NotImplementedError("Subclasses must implement __iter__ method")

    @property
    @abstractmethod
    def stream(self) -> Any | None:
        """Abstract property for the optional async stream used to prepare
        batches. Implementations should provide a getter and setter.
        """
        raise NotImplementedError("Subclasses must implement stream property")
