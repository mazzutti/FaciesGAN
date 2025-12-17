"""Torch-specific dataset exports.

Expose the framework-specific `TorchPyramidsDataset` for simple imports like
``from datasets.torch import TorchPyramidsDataset``.
"""

from .dataset import TorchPyramidsDataset

__all__ = ["TorchPyramidsDataset"]
