import logging
from pathlib import Path
import numpy as np
import torch
from data_files import DataFiles
from interpolators.config import InterpolatorConfig
from interpolators.nearest import BaseInterpolator
from ops import as_wells_mapping, load_image

# Module logger
logger = logging.getLogger(__name__)


class WellInterpolator(BaseInterpolator):
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
        """Render images at specified resolutions using trace-wise nearest neighbor interpolation.

        Parameters
        ----------
        loader : CustomLoader
            Data loader containing the original image and encoder
        resolutions : list[tuple[int, int]]
            List of (height, width) tuples for output resolutions

        Returns
        -------
        tuple[list[NDArray[np.float32]], NDArray[np.float32]]
            (smooth_imgs, high_res_img) where:
            - smooth_imgs: list of images at requested resolutions (float32, 0-1)
            - high_res_img: upsampled high-resolution image (float32, 0-1)
        """

        smooth_imgs: list[torch.Tensor] = []
        
        logger.info("Rendering with trace-wise nearest neighbor interpolation...")

        well_mapping = as_wells_mapping(DataFiles.WELLS)
        
        facie = load_image(image_path)

        height, width = facie.shape[:2]
        well_column = well_mapping[image_path.stem][0]
        well_trace = facie[:, well_column, :]

        # Process each resolution
        for _, _, new_h, new_w in resolutions:
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

            smooth_imgs.append(torch.from_numpy(downsampled)) # type: ignore

        return smooth_imgs

    