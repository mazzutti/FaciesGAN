"""Seismic interpolator module.

Provides a Lanczos-based seismic interpolator for direct ``.npy`` slices.
The finest scale preserves detail and lower resolutions are progressively
low-pass filtered to mimic noisier, bandwidth-limited seismic conditioning.
"""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

logger = logging.getLogger(__name__)

LANCZOS_RADIUS = 3


class SeismicInterpolator(BaseInterpolator):
    """Multi-scale seismic interpolator with Lanczos upsampling and blur.

    The finest scale uses a Lanczos resize to preserve coherent reflectors.
    Coarser scales are additionally low-pass filtered before resizing to
    simulate reduced bandwidth and a dirtier seismic appearance.
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def interpolate(
        self,
        seismic_path: Path,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:  # type: ignore[override]
        """Create a seismic pyramid from a direct ``.npy`` seismic slice.

        Parameters
        ----------
        seismic_path : Path
                Path to the input seismic data slice stored as ``.npy``.
        resolutions : tuple[tuple[int, ...], ...]
                Target pyramid shapes, ordered from coarse to fine.

        Returns
        -------
        list[torch.Tensor]
                One float32 tensor per target resolution, with values in
                ``[0, 1]`` and shape ``(H, W, 3)``.
        """
        seismic_data = np.load(seismic_path)
        if seismic_data.ndim == 2:
            seismic_data = np.repeat(seismic_data[:, :, None], 3, axis=2)
        seismic_data = seismic_data.astype(np.float32, copy=False)
        seismic_min = float(seismic_data.min())
        seismic_max = float(seismic_data.max())
        if seismic_max > seismic_min:
            seismic_data = (seismic_data - seismic_min) / (seismic_max - seismic_min)
        else:
            seismic_data = np.zeros_like(seismic_data, dtype=np.float32)
        smooth_seismic: list[torch.Tensor] = []

        logger.info("Rendering with Lanczos seismic interpolation and blur...")

        for resolution in resolutions:
            if self.config.channels_last:
                _, target_h, target_w, _ = resolution
            else:
                _, _, target_h, target_w = resolution

            resized = self._resize_with_band_limiting(
                seismic_data,
                target_h,
                target_w,
            )
            smooth_seismic.append(torch.from_numpy(resized))  # type: ignore[arg-type]

        return smooth_seismic

    @staticmethod
    def _resize_with_band_limiting(
        img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Resize seismic data with Lanczos resampling and scale-dependent blur."""
        src_h, src_w = img.shape[:2]

        # The coarser the target scale, the more aggressive the low-pass.
        downscale = max(src_h / max(target_h, 1), src_w / max(target_w, 1))
        blur_radius = 0.0
        if downscale > 1.0:
            blur_radius = min(3.0, 0.45 * (downscale - 1.0) ** 1.2)

        work_img = img.clip(0.0, 1.0)
        if blur_radius > 0.0:
            # Apply a gentle, scale-aware low-pass before shrinking to mimic dirty seismic.
            work_img = SeismicInterpolator._gaussian_blur(work_img, blur_radius)

        resized = SeismicInterpolator._lanczos_resize(work_img, target_h, target_w)
        return resized.clip(0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _gaussian_kernel(sigma: float) -> NDArray[np.float32]:
        """Build a normalized 1D Gaussian kernel from a scale factor."""
        sigma = max(float(sigma), 0.5)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0.0:
            kernel /= kernel_sum
        return kernel.astype(np.float32, copy=False)

    @staticmethod
    def _convolve_axis(
        img: NDArray[np.float32], kernel: NDArray[np.float32], axis: int
    ) -> NDArray[np.float32]:
        """Apply a 1D convolution along one axis with reflect padding."""
        if kernel.size == 1:
            return img

        work = np.moveaxis(img, axis, -1)
        pad = kernel.size // 2
        padded = np.pad(work, [(0, 0)] * (work.ndim - 1) + [(pad, pad)], mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=kernel.size, axis=-1
        )
        filtered = np.tensordot(windows, kernel, axes=([-1], [0]))
        return np.moveaxis(filtered, -1, axis)

    @staticmethod
    def _gaussian_blur(img: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
        """Blur a seismic volume with a separable Gaussian kernel."""
        kernel = SeismicInterpolator._gaussian_kernel(sigma)
        blurred = SeismicInterpolator._convolve_axis(img, kernel, axis=0)
        blurred = SeismicInterpolator._convolve_axis(blurred, kernel, axis=1)
        return blurred.astype(np.float32, copy=False)

    @staticmethod
    def _lanczos_kernel(
        x: NDArray[np.float32], radius: int = LANCZOS_RADIUS
    ) -> NDArray[np.float32]:
        """Evaluate the Lanczos kernel for a vector of offsets."""
        ax = np.abs(x)
        out = np.sinc(x) * np.sinc(x / radius)
        out = np.where(ax < radius, out, 0.0)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _resample_axis(
        img: NDArray[np.float32],
        target_len: int,
        axis: int,
        radius: int = LANCZOS_RADIUS,
    ) -> NDArray[np.float32]:
        """Resample a single axis with separable Lanczos interpolation."""
        if img.shape[axis] == target_len:
            return img

        work = np.moveaxis(img, axis, -1)
        src_len = work.shape[-1]
        scale = src_len / max(target_len, 1)
        positions = (np.arange(target_len, dtype=np.float32) + 0.5) * scale - 0.5
        base = np.floor(positions).astype(np.int64)
        window = np.arange(-radius + 1, radius + 1, dtype=np.int64)
        indices = base[:, None] + window[None, :]
        distances = positions[:, None] - indices.astype(np.float32)
        weights = SeismicInterpolator._lanczos_kernel(distances, radius=radius)
        valid = (indices >= 0) & (indices < src_len)
        weights = np.where(valid, weights, 0.0)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weights = np.divide(
            weights, weight_sum, out=np.zeros_like(weights), where=weight_sum != 0
        )

        indices = np.clip(indices, 0, src_len - 1)
        gathered = np.take(work, indices, axis=-1)
        resampled = np.sum(gathered * weights[None, ...], axis=-1)
        return np.moveaxis(resampled, -1, axis)

    @staticmethod
    def _lanczos_resize(
        img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Resize a seismic volume with separable Lanczos interpolation."""
        resized_w = SeismicInterpolator._resample_axis(img, target_w, axis=1)
        resized_hw = SeismicInterpolator._resample_axis(resized_w, target_h, axis=0)
        return resized_hw
