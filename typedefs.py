from __future__ import annotations

import platform as _platform
from typing import Any, Generic, NamedTuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mlx_core
    import mlx.nn as mlx_nn  # type: ignore
    import mlx.optimizers as mlx_optim  # type: ignore
elif _platform.system() == "Darwin":
    try:
        import mlx.core as mlx_core
        import mlx.nn as mlx_nn  # type: ignore
        import mlx.optimizers as mlx_optim  # type: ignore
    except ImportError:
        mlx_core = None  # type: ignore[assignment]
        mlx_nn = None  # type: ignore[assignment]
        mlx_optim = None  # type: ignore[assignment]
else:
    mlx_core = None  # type: ignore[assignment]
    mlx_nn = None  # type: ignore[assignment]
    mlx_optim = None  # type: ignore[assignment]

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from trainning.mlx.schedulers import MultiStepLR
else:
    # Avoid importing training internals at module import time to prevent
    # circular imports when other modules (e.g., `trainning.base`) import
    # `typedefs`. Use a permissive fallback for runtime typing.
    MultiStepLR = object  # type: ignore

# Generic type variables for type hinting
T = TypeVar("T")

# Tensor type variable (e.g., torch.Tensor or mx.array)
if TYPE_CHECKING:
    TTensor = TypeVar("TTensor", torch.Tensor, mlx_core.array, np.ndarray)
elif mlx_core is not None:
    TTensor = TypeVar("TTensor", torch.Tensor, mlx_core.array, np.ndarray)
else:
    TTensor = TypeVar("TTensor", torch.Tensor, np.ndarray)  # type: ignore[misc]

PyramidsBatch = tuple[
    dict[int, TTensor],
    dict[int, TTensor],
    dict[int, TTensor],
    dict[int, TTensor],
]


# Module type variable (e.g., torch.nn.Module or mlx.nn.Module)
TDiscriminator = TypeVar("TDiscriminator")

# Noise type variable (e.g., torch.Tensor or mx.array)
TNoise = TypeVar("TNoise")


if TYPE_CHECKING:
    TModule = TypeVar("TModule", torch.nn.Module, mlx_nn.Module)
elif mlx_nn is not None:
    TModule = TypeVar("TModule", torch.nn.Module, mlx_nn.Module)
else:
    TModule = TypeVar("TModule", bound=torch.nn.Module)  # type: ignore[misc]

# Optimizer type variable (e.g., torch.optim or mlx.optim)
if TYPE_CHECKING:
    TOptimizer = TypeVar(
        "TOptimizer",
        torch.optim.Optimizer,
        mlx_optim.Optimizer,
    )
elif mlx_optim is not None:
    TOptimizer = TypeVar(
        "TOptimizer",
        torch.optim.Optimizer,
        mlx_optim.Optimizer,
    )
else:
    TOptimizer = TypeVar("TOptimizer", bound=torch.optim.Optimizer)  # type: ignore[misc]

# Scheduler type variable (e.g., torch.optim.lr_scheduler or a callable for mlx)
TScheduler = TypeVar(
    "TScheduler",
    torch.optim.lr_scheduler.LRScheduler,
    MultiStepLR,
)


# Container for a training batch of multi-scale pyramids.
class Batch(NamedTuple, Generic[TTensor]):
    """Container for a training batch of multi-scale pyramids.

    Fields are per-scale tuples of tensors. `Batch` is a NamedTuple so it
    subclasses `tuple` and is compatible with PyTorch's default collate.


    Parameters
    ----------
    facies : tuple[TTensor]
        Per-scale facies tensors.
    wells : tuple[TTensor]
        Per-scale well-conditioning tensors (may be empty when unused).
    seismic : tuple[TTensor]
        Per-scale seismic-conditioning tensors (may be empty when unused).
    """

    facies: tuple[TTensor, ...]
    wells: tuple[TTensor, ...]
    seismic: tuple[TTensor, ...]


# Type variable for a data loader that yields batches of tensors
IDataLoader = TypeVar("IDataLoader", DataLoader[Batch[torch.Tensor]], Any)
