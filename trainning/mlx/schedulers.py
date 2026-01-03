import bisect

from collections.abc import Iterable

import mlx.core as mx

from mlx.optimizers.optimizers import Optimizer as MLXOptimizer  # type: ignore


class MultiStepLR:
    """MultiStep learning rate scheduler for MLX.

    A lightweight MLX-compatible implementation that mirrors the semantics
    of ``torch.optim.lr_scheduler.MultiStepLR`` but without depending on a
    Torch optimizer. This scheduler can be used as a callable that returns
    the current learning rate for a given step, and also exposes a
    stateful :meth:`step` method.

    Args:
        init_lr: Initial learning rate or list of initial learning rates
            (for multiple parameter-groups compatibility).
        milestones: Iterable of increasing integers at which to decay the
            learning rate.
        gamma: Multiplicative factor of learning rate decay.
        last_step: The index of last step. Default: -1 (before first step).
        verbose: If True, prints a message to stdout on lr update (noop).
    """

    def __init__(
        self,
        init_lr: float | Iterable[float],
        milestones: Iterable[int],
        gamma: float = 0.1,
        last_step: int = -1,
        optimizer: MLXOptimizer | None = None,
        verbose: bool = False,
    ) -> None:
        if isinstance(init_lr, Iterable):
            self.base_lrs: list[float] = list(init_lr)
        else:
            self.base_lrs = [float(init_lr)]

        self.milestones = sorted(int(m) for m in milestones)
        if any(x >= y for x, y in zip(self.milestones, self.milestones[1:])):
            raise ValueError(
                "Milestones should be a strictly increasing list of integers"
            )

        self.gamma = float(gamma)
        self.last_step = int(last_step)
        self.verbose = bool(verbose)

        # attached optimizer (optional). If provided, the scheduler will
        # update `optimizer.learning_rate` on each `step()` call.
        self.optimizer = optimizer

        # If optimizer provided, use its current learning rate as base.
        if self.optimizer is not None:
            if len(self.base_lrs) > 1:
                raise ValueError(
                    "MLX optimizer exposes a single learning rate; init_lr must "
                    "be a scalar when providing an optimizer"
                )
            if isinstance(self.optimizer.learning_rate, mx.array):  # type: ignore
                lr_val = self.optimizer.learning_rate.item()
            else:
                lr_val = self.optimizer.learning_rate  # type: ignore
            self.base_lrs = [lr_val]

        # cache of last computed lrs
        self._last_lr: list[float] = self._compute_lrs()

    def _decay_count(self, step: int) -> int:
        # number of milestones <= step (behaves like bisect_right)
        return bisect.bisect_right(self.milestones, step)

    def _compute_lrs(self, step: int | None = None) -> list[float]:
        if step is None:
            step = self.last_step
        count = self._decay_count(int(step))
        mul = self.gamma**count
        return [base_lr * mul for base_lr in self.base_lrs]

    def step(self, step: int | None = None) -> list[float]:
        """Advance scheduler to ``step`` (or increment by 1 if None) and
        return the new learning rates as a list.
        """
        if step is None:
            self.last_step += 1
        else:
            self.last_step = int(step)

        self._last_lr = self._compute_lrs(self.last_step)

        # If attached to an MLX optimizer, update its learning rate state
        if self.optimizer is not None:
            # MLX optimizers expose a single `learning_rate` state.
            self.optimizer.learning_rate = mx.array(self._last_lr[0])

        return list(self._last_lr)

    def get_lr(self) -> list[float]:
        """Return current learning rates (list)."""
        return list(self._last_lr)

    def __call__(self, step: int) -> float | list[float]:
        """Return learning rate(s) for the provided ``step``.

        If the scheduler was constructed with a single initial LR, a float
        is returned. Otherwise a list of floats is returned.
        """
        lrs = self._compute_lrs(step)
        return lrs[0] if len(lrs) == 1 else lrs

    def state_dict(self) -> dict[str, object]:
        """Return scheduler state as a dict (optimizer excluded).

        The returned dict contains all scheduler attributes except the
        attached optimizer object. This is suitable for checkpointing the
        scheduler independently of optimizer state.
        """
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load scheduler state from ``state``.

        Notes
        -----
        - The provided ``state`` should be a dict produced by
            :meth:`state_dict`.
        - This will update the scheduler attributes in-place. If an
            optimizer is attached to the scheduler, its `learning_rate`
            will be synchronized to the scheduler's current value after
            loading.
        """
        # Update internal state (optimizer key is intentionally excluded)
        self.__dict__.update(state)

        # Ensure base_lrs and _last_lr are lists of floats
        self.base_lrs = [float(x) for x in self.base_lrs]
        self._last_lr = list(self._last_lr)

        # If an optimizer is attached, sync its learning rate to scheduler
        if self.optimizer is not None and len(self._last_lr) > 0:
            self.optimizer.learning_rate = mx.array(self._last_lr[0])
