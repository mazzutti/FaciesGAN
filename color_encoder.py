"""Color encoder for converting RGB images to label indices and vice versa.

This module provides the ColorEncoder class for managing color palettes
and conversions between RGB and categorical label representations.
"""

import logging
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ColorEncoder:
    def __init__(
        self, img_array: NDArray[Any], device: torch.device) -> None:
        # Ensure we work with float32 NumPy arrays to avoid creating
        # torch.float64 tensors which are not supported on MPS devices.
        pixels = img_array.reshape(-1, 3).astype(np.float32, copy=False)
        self.palette = np.unique(pixels, axis=0).astype(np.float32, copy=False)
        self.num_classes = len(self.palette)
        self.device = device
        # Use torch.from_numpy to preserve dtype (float32) and avoid creating
        # float64 tensors that MPS cannot handle.
        try:
            self.palette_tensor = torch.from_numpy(  # pyright: ignore
                self.palette,
            ).to(self.device)
        except Exception:
            # Fall back to generic constructor but force float32
            self.palette_tensor = torch.tensor(self.palette, dtype=torch.float32).to(
                self.device
            )
        logger.info(f"Detected {self.num_classes} unique facies classes.")

    def rgb_to_labels(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Convert an [N,3] RGB tensor to label indices using the encoder palette.

        Args:
            img_tensor: Tensor of shape [N, 3] with RGB values (0..1) on same device
                        as the encoder.palette_tensor.
        Returns:
            torch.Tensor: Long tensor with shape [N] with the index of closest palette
                          color for each input pixel.
        """
        dists = torch.cdist(img_tensor, self.palette_tensor)
        return torch.argmin(dists, dim=1)

    def labels_to_rgb(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """Map label indices back to RGB values from the palette.

        Args:
            label_tensor: Tensor of shape [N] containing class indices (ints).
        Returns:
            Tensor of shape [N, 3] with RGB colors as floats.
        """
        return self.palette_tensor[label_tensor.long()]

    def get_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Automatically calculates inverse frequency weights.
        Rare classes (thin beds) get higher weights.
        """
        # Work with torch tensors directly to avoid numpy type inference issues
        # Move labels to CPU and ensure integer dtype
        labels_cpu = labels.detach().cpu().long()

        # Use torch.bincount to get per-class counts (safe and faster)
        counts = torch.bincount(labels_cpu, minlength=self.num_classes).float()
        total = labels_cpu.numel()

        # Weight = Total / (Num_Classes * Count)
        # Add small epsilon to avoid division by zero and keep float math
        eps = 1e-6
        weights = total / (self.num_classes * (counts + eps))

        # Normalize so mean is roughly 1.0
        weights = weights / weights.mean()

        # Log rounded weights for user information
        try:
            # Prefer a numpy -> list conversion with explicit float dtype
            rounded_arr = (  # pyright: ignore
                weights.cpu().numpy().round(2).astype(float)  # pyright: ignore
            )
            rounded: list[float] = rounded_arr.tolist()
        except Exception:
            rounded = weights.tolist()  # pyright: ignore
        logger.info(f"Auto-calculated Class Weights: {rounded}")

        return weights.to(self.device).float()
