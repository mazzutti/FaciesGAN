"""Well-based interpolation utilities.

This module implements :class:`WellInterpolator`, which generates
multi-scale images by extracting vertical well traces from facies
images and positioning them at scaled column locations across
resolutions. The output is suitable for tasks that require preserving
vertical trace consistency (e.g. well logs and seismic traces).
"""

import logging
from pathlib import Path

import numpy as np
import torch

import datasets.utils as data_utils
from datasets.data_files import DataFiles
from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

# Module logger
logger = logging.getLogger(__name__)


class WellInterpolator(BaseInterpolator):
    """Well-based interpolator that extracts and scales vertical well traces.

    This class creates multi-scale pyramid representations by extracting
    vertical well traces from facies images and positioning them at
    proportionally scaled column locations across different resolutions.

    The interpolator uses well mapping data to identify well column positions,
    then creates downsampled versions while maintaining the vertical trace
    structure at each scale.
    """

    def __init__(
        self,
        config: InterpolatorConfig,
    ) -> None:
        """Initialize the well interpolator with configuration.

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
        """Create multi-scale well representations by extracting and scaling vertical traces.

        Extracts the vertical well trace from the input facies image at the
        well column position, then creates downsampled versions at each
        requested resolution with proportionally scaled column positions.

        Parameters
        ----------
        image_path : Path
            Path to the input facies image file.
        resolutions : tuple[tuple[int, ...], ...]
            Tuple of (batch, channels, height, width) tuples for output resolutions.

        Returns
        -------
        list[torch.Tensor]
            List of well trace tensors at different resolutions, with black
            backgrounds and the well trace positioned at the scaled column.
        """
        smooth_imgs: list[torch.Tensor] = []

        logger.info("Rendering with trace-wise nearest neighbor interpolation...")

        well_mapping = data_utils.as_wells_mapping(DataFiles.WELLS)

        facie = data_utils.load_image(image_path)

        height, width = facie.shape[:2]
        well_column = well_mapping[image_path.stem][0]
        well_trace = facie[:, well_column, :]

        # Process each resolution
        for resolution in resolutions:
            if self.config.channels_last:
                _, new_h, new_w, _ = resolution
            else:
                _, _, new_h, new_w = resolution
            # Scale the column position proportionally
            scaled_column = int(well_column * new_w / width)
            scaled_column = min(new_w - 1, max(0, scaled_column))

            # Create output image (black background)
            downsampled = np.zeros((new_h, new_w, 3), dtype=np.float32)

            # Downsample the vertical trace using nearest neighbor
            step = height // new_h
            for i in range(new_h):
                src_row = min(i * step, height - 1)
                downsampled[i, scaled_column, :] = well_trace[src_row, :]

            smooth_imgs.append(torch.from_numpy(downsampled))  # type: ignore

        return smooth_imgs
