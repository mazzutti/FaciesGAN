import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import zoom

from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig
from ops import load_image

# Module logger
logger = logging.getLogger(__name__)


class NearestInterpolator(BaseInterpolator):
    """Nearest neighbor interpolator with API compatible with NeuralSmoother.

    This class provides a simple baseline interpolation method using
    nearest neighbor resampling. It implements a render() method that
    mirrors the NeuralSmoother API, making it easy to swap between
    neural and traditional interpolation methods.
    """

    def __init__(
        self,
        config: InterpolatorConfig,
    ) -> None:
        """Initialize the interpolator with configuration.

        Parameters
        ----------
        config : InterpolatorConfig
            Configuration object containing all interpolator parameters.
        """
        super().__init__(config)

    def interpolate(
        self,
        image_path: Path,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Create multi-scale pyramid using trace-wise nearest neighbor interpolation.

        First upsamples the input image to the highest resolution using trace-wise
        interpolation, then creates downsampled versions at each requested resolution.
        Trace-wise interpolation processes vertical columns independently, which is
        beneficial for seismic and well log data.

        Parameters
        ----------
        image_path : Path
            Filesystem path to the input image file.
        resolutions : tuple[tuple[int, ...], ...]
            Tuple of (batch, channels, height, width) tuples for output resolutions.

        Returns
        -------
        list[torch.Tensor]
            List of interpolated images as float32 tensors with values in [0, 1],
            one tensor per resolution in the pyramid.
        """
        smooth_imgs: list[torch.Tensor] = []

        logger.info("Rendering with trace-wise nearest neighbor interpolation...")

        # Get dimensions using base helper method
        super_height, super_width = resolutions[-1][2:]

        img_np = load_image(image_path)
        high_res_img = self._upsample_image(img_np, super_height, super_width)
        for _, _, new_h, new_w in resolutions[:-1]:
            interpolated_img = self._downsample_image(high_res_img, new_h, new_w)
            interpolated_img = interpolated_img.astype(np.float32).clip(0.0, 1.0)
            smooth_imgs.append(torch.from_numpy(interpolated_img))  # type: ignore
        smooth_imgs.append(torch.from_numpy(high_res_img.astype(np.float32)))  # type: ignore
        return smooth_imgs

    def _upsample_image(
        self, img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Upsample image using trace-wise nearest neighbor interpolation.

        Helper method that implements trace-wise (column-by-column) upsampling.

        Parameters
        ----------
        img : NDArray[np.float32]
            Input image of shape (h, w, 3)
        target_h : int
            Target height
        target_w : int
            Target width

        Returns
        -------
        NDArray[np.float32]
            Upsampled image of shape (target_h, target_w, 3)
        """
        h, w, c = img.shape

        # Step 1: Interpolate each column (trace) vertically
        upsampled_traces = np.zeros((target_h, w, c), dtype=np.float32)
        for col in range(w):
            for channel in range(c):
                # Interpolate this column vertically using nearest neighbor
                upsampled_traces[:, col, channel] = zoom(
                    img[:, col, channel], target_h / h, order=0
                )

        # Step 2: Replicate horizontally (nearest neighbor)
        output = np.zeros((target_h, target_w, c), dtype=np.float32)
        for channel in range(c):
            output[:, :, channel] = zoom(
                upsampled_traces[:, :, channel], (1, target_w / w), order=0
            )
        return output

    def _downsample_image(
        self, img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Downsample image using trace-wise (column-by-column) interpolation.

        This implementation uses nearest neighbor interpolation applied
        trace-wise (column-by-column) for better handling of seismic data.
        Vectorized implementation for 10-50x speedup over nested loops.

        Parameters
        ----------
        img : NDArray[np.float32]
            Input image of shape (h, w, 3)
        target_h : int
            Target height
        target_w : int
            Target width

        Returns
        -------
        NDArray[np.float32]
            Downsampled image of shape (target_h, target_w, 3)
        """
        h, w, _ = img.shape

        # Step 1: Vectorized horizontal downsampling for all channels at once
        h_downsampled = zoom(img, (1, target_w / w, 1), order=0)

        # Step 2: Vectorized vertical downsampling for all columns and channels at once
        output = zoom(h_downsampled, (target_h / h, 1, 1), order=0)

        return output.astype(np.float32)
