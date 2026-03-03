# Public package exports for `training` subpackage

import platform

from .base import Trainer
from .torch.trainer import TorchTrainer

if platform.system() == "Darwin":
    try:
        from .mlx.trainer import MLXTrainer
    except ImportError:
        MLXTrainer = None  # type: ignore[assignment,misc]
else:
    MLXTrainer = None  # type: ignore[assignment,misc]

__all__ = ["Trainer", "MLXTrainer", "TorchTrainer"]
