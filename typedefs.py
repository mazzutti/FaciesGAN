from __future__ import annotations

from typing import Any, Generic, NamedTuple, TypeVar

import mlx.core as mlx_core
import mlx.nn as mlx_nn  # type: ignore
import mlx.optimizers as mlx_optim  # type: ignore

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainning.mlx.schedulers import MultiStepLR

# Generic type variables for type hinting
T = TypeVar("T")

# Tensor type variable (e.g., torch.Tensor or mx.array)
TTensor = TypeVar("TTensor", torch.Tensor, mlx_core.array, np.ndarray)

PyramidsBatch = tuple[
    tuple[TTensor, ...],
    tuple[TTensor, ...],
    tuple[TTensor, ...],
    tuple[TTensor, ...],
]


# Module type variable (e.g., torch.nn.Module or mlx.nn.Module)
TDiscriminator = TypeVar("TDiscriminator")

# Noise type variable (e.g., torch.Tensor or mx.array)
TNoise = TypeVar("TNoise")


TModule = TypeVar("TModule", torch.nn.Module, mlx_nn.Module)

# Optimizer type variable (e.g., torch.optim or mlx.optim)
TOptimizer = TypeVar(
    "TOptimizer",
    torch.optim.Optimizer,
    mlx_optim.Optimizer,
)

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
