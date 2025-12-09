"""Background worker helpers to offload CPU-bound visualization and quantization.

This module exposes a small :class:`concurrent.futures.ProcessPoolExecutor`
and helper ``submit_*`` functions that run heavy image processing and
saving in separate processes so the main training loop stays responsive.

Design notes:
- Tasks accept ``torch.Tensor`` objects. Callers should pass CPU tensors
    (e.g. moved to CPU with ``tensor.detach().cpu()``) so the worker process
    can safely serialize and operate on them. Avoid passing CUDA tensors
    directly into the process pool to prevent pickling errors.
- Uses ``multiprocessing.get_context('spawn')`` to be compatible across
    platforms (macOS).
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import threading
import torch
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Optional

logger = logging.getLogger(__name__)


def _save_plot_task(
    fake_list: list[torch.Tensor],
    real_arr: torch.Tensor,
    masks_arr: torch.Tensor,
    stage: int,
    index: int,
    out_dir: str,
) -> bool:
    # Import locally so the worker process has its own module imports
    from utils import plot_generated_facies

    # NOTE: This worker function expects the caller to pass `torch.Tensor`
    # objects (preferably already moved to CPU). Do not pass CUDA tensors
    # directly into the process pool; move them to CPU first using
    # `tensor.detach().cpu()` to avoid pickling errors when spawning child
    # processes.
    #
    # The plotting helper will accept the tensors and perform any
    # conversions internally as needed.
    plot_generated_facies(
        fake_list, real_arr, masks_arr, stage, index, out_dir, save=True
    )
    return True


class BackgroundWorker:
    """Singleton manager for a process pool that offloads CPU-bound tasks.

    Features:
    - Singleton instance (calling BackgroundWorker() returns same instance).
    - Bounded pending-job queue to avoid unbounded memory growth.
    - Tracks pending futures and exposes wait/shutdown helpers.
    - Logs exceptions raised by background tasks.
    """

    _instance: Optional["BackgroundWorker"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, max_workers: int = 2, max_pending: int = 32) -> "BackgroundWorker":
        # Double-checked locking to safely initialize singleton
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 2, max_pending: int = 32) -> None:
        # Initialize only once
        if getattr(self, "_initialized", False):
            return

        ctx = mp.get_context("spawn")
        self._executor: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx
        )

        # Pending futures tracking and coordination
        self._pending: set[Future[bool]] = set()
        self._pending_cond = threading.Condition()
        self._max_pending = int(max_pending)

        atexit.register(self.shutdown)
        self._initialized = True

    def _on_done(self, fut: Future[bool]) -> None:
        # Callback executed in the main process thread when a Future completes
        try:
            exc = fut.exception()
            if exc is not None:
                logger.exception("Background task failed", exc_info=exc)
        except Exception:
            logger.exception("Error checking future result")
        finally:
            with self._pending_cond:
                self._pending.discard(fut)
                self._pending_cond.notify_all()

    def submit_plot_generated_facies(
        self,
        fake_list: list[torch.Tensor],
        real: torch.Tensor,
        masks: torch.Tensor,
        stage: int,
        index: int,
        out_dir: str,
        wait_if_full: bool = True,
        timeout: Optional[float] = None,
    ) -> Future[bool]:
        """Submit a plot job to the process pool (non-blocking by default).

        If the number of pending jobs reaches ``max_pending``, the call will
        block until space is available when ``wait_if_full=True``. If
        ``wait_if_full=False`` a completed ``Future`` with a ``False`` result
        is returned immediately.

        Important: this API accepts ``torch.Tensor`` inputs. Convert tensors
        to CPU (``tensor.detach().cpu()``) before calling this method rather
        than passing CUDA tensors directly into the executor to avoid
        pickling/serialization issues with GPU-backed storage.
        """
        # Wait for available slot if configured
        with self._pending_cond:
            if self._max_pending > 0:
                if not wait_if_full and len(self._pending) >= self._max_pending:
                    fut: Future[bool] = Future()
                    fut.set_result(False)
                    return fut

                if timeout is None:
                    # Wait indefinitely until there's space
                    while len(self._pending) >= self._max_pending:
                        self._pending_cond.wait()
                else:
                    import time

                    # Compute deadline once and wait with the remaining time
                    end = time.time() + timeout
                    while len(self._pending) >= self._max_pending:
                        remaining = end - time.time()
                        if remaining <= 0:
                            fut: Future[bool] = Future()
                            fut.set_result(False)
                            return fut
                        self._pending_cond.wait(timeout=remaining)
            fut = self._executor.submit(
                _save_plot_task,
                fake_list,
                real,
                masks,
                int(stage),
                int(index),
                str(out_dir),
            )
            # Track and attach callback
            self._pending.add(fut)
            fut.add_done_callback(self._on_done)
            return fut

    def pending_count(self) -> int:
        with self._pending_cond:
            return len(self._pending)

    def wait_pending(self, timeout: Optional[float] = None) -> None:
        """Block until all pending tasks complete or timeout elapses."""
        with self._pending_cond:
            if timeout is None:
                while self._pending:
                    self._pending_cond.wait()
            else:
                import time

                end = time.time() + timeout
                while self._pending and time.time() < end:
                    self._pending_cond.wait(timeout=end - time.time())

    def shutdown(self, wait: bool = False) -> None:
        try:
            # Optionally wait for pending jobs to complete
            if wait:
                self.wait_pending()
            self._executor.shutdown(wait=wait)
        except Exception:
            logger.exception("Error shutting down BackgroundWorker")


def submit_plot_generated_facies(
    fake_list: list[torch.Tensor],
    real: torch.Tensor,
    masks: torch.Tensor,
    stage: int,
    index: int,
    out_dir: str,
) -> Future[bool]:
    """Convenience wrapper that uses the module-level BackgroundWorker.

    This keeps backward compatibility with existing import sites that call
    `submit_plot_generated_facies(...)` directly.
    """
    # BackgroundWorker is a singleton — calling the constructor returns the
    # shared instance. Use it to submit the job.
    worker = BackgroundWorker()
    return worker.submit_plot_generated_facies(
        fake_list, real, masks, stage, index, out_dir
    )
