"""Public dataset symbols used across the project.

This module exposes a small, stable public API for the `datasets` package
so other modules can import `Batch`, the framework-specific dataset and any
helper factories without reaching into submodules.
"""

from .base import PyramidsDataset
from .torch.dataset import TorchPyramidsDataset

__all__ = [
    "PyramidsDataset",
    "TorchPyramidsDataset",
]
