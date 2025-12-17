from collections.abc import Callable
from typing import TypeVar

import mlx.core as mlx_core
import mlx.nn as mlx_nn  # type: ignore
import mlx.optimizers as mlx_optim  # type: ignore
import torch

# Generic type variables for type hinting
T = TypeVar("T")


# Tensor type variable (e.g., torch.Tensor or mx.array)
TTensor = TypeVar("TTensor", torch.Tensor, mlx_core.array)

# Module type variable (e.g., torch.nn.Module or mlx.nn.Module)
TDiscriminator = TypeVar("TDiscriminator")

# Noise type variable (e.g., torch.Tensor or mx.array)
TNoise = TypeVar("TNoise")

# Module type variable (e.g., torch.nn.Module or mlx.nn.Module)
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
    Callable[[float, float], Callable[[int], float]],
)

# Pyramid type variable
TorchPyramid = tuple[torch.Tensor, ...]
MLXPyramid = tuple[mlx_core.array, ...]

# Masks type variable
Pyramids = TypeVar(
    "Pyramids",
    tuple[TorchPyramid, TorchPyramid, TorchPyramid],
    tuple[MLXPyramid, MLXPyramid, MLXPyramid],
)
