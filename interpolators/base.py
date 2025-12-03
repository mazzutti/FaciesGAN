
from __future__ import annotations
import logging
from pathlib import Path


import torch

from interpolators.config import InterpolatorConfig

logger = logging.getLogger(__name__)

class BaseInterpolator:
    """Base class providing common functionality for all interpolators.

    This class provides the process_and_save method that can be used by
    both NearestInterpolator and NeuralSmoother without code duplication.

    Subclasses must define:
    - config: InterpolatorConfig attribute
    - render(loader, resolutions) -> tuple method
    """

    config: InterpolatorConfig

    def __init__(self, config: InterpolatorConfig) -> None:
        self.config = config

    
    def get_target_dimensions(self) -> tuple[int, ...]:
        """Calculate native and target (super-resolution) dimensions.

        Uses the geometry stored in the interpolator configuration and applies
        the upsampling factor to compute the super-resolution dimensions.

        Returns
        -------
        tuple[int, int, int, int]
            A 4-tuple of integers: (native_height, native_width, super_height,
            super_width), where super dimensions are native dimensions multiplied
            by ``self.config.upsample``.
        """
        native_height, native_width = self.config.geometry
        super_height = native_height * self.config.upsample
        super_width = native_width * self.config.upsample
        return native_height, native_width, super_height, super_width


    def interpolate(
        self, 
        image_path: Path,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Render interpolated images at specified resolutions.

        This abstract method must be implemented by subclasses to perform
        interpolation/smoothing at multiple output resolutions. Subclasses
        should load the input image, apply their specific interpolation
        algorithm, and return smoothed images at each requested resolution.

        Parameters
        ----------
        image_path : Path
            Filesystem path to the input image file to be interpolated. The
            image is loaded and used to generate interpolated outputs at the
            specified resolutions.
        resolutions : list[tuple[int, int]]
            List of (height, width) tuples specifying the desired output
            resolutions for the interpolated images.

        Returns
        -------
        list[torch.Tensor]
            List of interpolated images, one per requested resolution, as
            float32 arrays with values in [0, 1] and shape (H, W, 3)
            representing RGB color intensities.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement render")


