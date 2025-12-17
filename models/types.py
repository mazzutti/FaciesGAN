from typing import TypeVar

import mlx.core as mx
import mlx.nn as mx_nn  # type: ignore
import torch
import torch.nn as nn

# Generic type variables for type hinting
T = TypeVar("T")

# Tensor type variable (e.g., torch.Tensor or mx.array)
TTensor = TypeVar("TTensor", torch.Tensor, mx.array)

# Module type variable (e.g., torch.nn.Module or mlx.nn.Module)
TDiscriminator = TypeVar("TDiscriminator")

# Noise type variable (e.g., torch.Tensor or mx.array)
TNoise = TypeVar("TNoise")

# Module type variable (e.g., torch.nn.Module or mlx.nn.Module)
TModule = TypeVar("TModule", nn.Module, mx_nn.Module)
