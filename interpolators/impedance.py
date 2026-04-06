"""Impedance data loader and interpolator.

Reads a GSLIB-format 3D acoustic impedance volume, extracts XZ crossline
slices, normalizes values to [0, 1], and produces multi-scale pyramid
tensors using nearest-neighbor trace-wise interpolation (identical to the
seismic pathway).
"""

import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import zoom  # type: ignore

from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

logger = logging.getLogger(__name__)

# Stanford VI-E grid: nx=150, ny=200 (crosslines), nz=200 (depth)
_NX, _NY, _NZ = 150, 200, 200


def load_impedance_volume(dat_path: Path) -> NDArray[np.float32]:
    """Read a GSLIB .dat file and return the 3D impedance volume.

    The file is expected to have a 3-line header (title, number of
    variables, variable name) followed by ``NX * NY * NZ`` float values
    with x varying fastest (Fortran/GSLIB order).

    Returns
    -------
    NDArray[np.float32]
        Volume with shape ``(nz, ny, nx)`` — depth × crossline × inline.
    """
    with open(dat_path, "r") as fh:
        for _ in range(3):
            fh.readline()
        values = np.loadtxt(fh, dtype=np.float32)

    expected = _NX * _NY * _NZ
    if values.size != expected:
        raise ValueError(
            f"Expected {expected} values in {dat_path}, got {values.size}"
        )
    # GSLIB order: x fastest, then y, then z → reshape as (nz, ny, nx)
    return values.reshape(_NZ, _NY, _NX)


def extract_crosslines(
    volume: NDArray[np.float32],
    target_height: int = 120,
) -> list[NDArray[np.float32]]:
    """Extract XZ crossline slices and resize the depth axis.

    Parameters
    ----------
    volume : NDArray[np.float32]
        3-D impedance volume with shape ``(nz, ny, nx)``.
    target_height : int
        Number of depth pixels to keep (cropped from the top of the
        flipped slice, i.e. the shallowest rows).  No interpolation is
        performed — this matches how facies and seismic images are
        extracted.

    Returns
    -------
    list[NDArray[np.float32]]
        List of ``ny`` images, each with shape ``(target_height, nx)``
        and values globally normalized to [0, 1].
    """
    _, ny, _ = volume.shape
    vmin, vmax = volume.min(), volume.max()
    normed = (volume - vmin) / (vmax - vmin + 1e-8)

    slices: list[NDArray[np.float32]] = []
    for y_idx in range(ny):
        xz_slice = normed[:, y_idx, :]  # shape (nz, nx)
        # Flip depth axis so shallow is at top (matches facies PNG convention)
        xz_slice = xz_slice[::-1, :]
        # Crop to target_height (same as facies/seismic — no interpolation)
        xz_slice = xz_slice[:target_height, :]
        slices.append(xz_slice)
    return slices


class ImpedanceInterpolator(BaseInterpolator):
    """Multi-scale pyramid generator for a single impedance crossline image.

    Works like :class:`NearestInterpolator` but accepts a 2-D numpy array
    (already extracted and normalized) instead of loading a PNG from disk.
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def interpolate_array(
        self,
        img: NDArray[np.float32],
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Produce a multi-scale pyramid from a 2-D grayscale array.

        The grayscale slice is broadcast to 3 identical RGB channels so that
        the downstream pipeline (which expects 3-channel images) works
        without modification.

        Parameters
        ----------
        img : NDArray[np.float32]
            Grayscale image with shape ``(H, W)`` and values in [0, 1].
        resolutions : tuple[tuple[int, ...], ...]
            Target shapes as ``(batch, channels, height, width)`` or
            ``(batch, height, width, channels)`` when ``channels_last``.

        Returns
        -------
        list[torch.Tensor]
            One float32 tensor per resolution, each with shape ``(H, W, 3)``.
        """
        # Expand grayscale → RGB
        img_rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)

        super_height, super_width = (
            resolutions[-1][1:3]
            if self.config.channels_last
            else resolutions[-1][2:]
        )

        high_res = self._resize(img_rgb, super_height, super_width)
        smooth_imgs: list[torch.Tensor] = []

        for resolution in resolutions[:-1]:
            if self.config.channels_last:
                _, new_h, new_w, _ = resolution
            else:
                _, _, new_h, new_w = resolution
            resized = self._resize(high_res, new_h, new_w)
            smooth_imgs.append(torch.from_numpy(resized.clip(0.0, 1.0)))  # type: ignore

        smooth_imgs.append(torch.from_numpy(high_res.astype(np.float32)))  # type: ignore
        return smooth_imgs

    # -- private helpers --------------------------------------------------

    @staticmethod
    def _resize(
        img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        h, w, _ = img.shape
        z_h = target_h / h
        z_w = target_w / w
        result: NDArray[np.float32] = np.asarray(
            zoom(img, (z_h, z_w, 1), order=1), dtype=np.float32
        )
        return result
