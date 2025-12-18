import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from datasets.data_files import DataFiles
from options import TrainningOptions


def load_image(image_path: Path) -> NDArray[np.float32]:
    """Load an image from disk and convert it to a normalized float32 numpy array.

    Parameters
    ----------
    image_path : Path
        Filesystem path to the image file to load.

    Returns
    -------
    NDArray[np.float32]
        RGB image as a float32 array with shape (H, W, 3) and values
        normalized to the range [0, 1].
    """
    img_pil: Image.Image = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil).astype(np.float32, copy=False) / 255.0
    return img_np


def generate_scales(options: TrainningOptions) -> tuple[tuple[int, ...], ...]:
    """Generate multi-scale pyramid resolutions for progressive training.

    Creates a tuple of shapes representing different scales from coarse to fine
    resolution. Each scale is computed using exponential scaling between
    min_size and max_size parameters.

    Parameters
    ----------
    options : TrainningOptions
        Training configuration containing:
        - min_size: Minimum (coarsest) resolution
        - max_size: Maximum (finest) resolution
        - crop_size: Crop size for training
        - stop_scale: Number of pyramid scales
        - batch_size: Batch size for each scale
        - num_channels: Number of input channels

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Tuple of (batch_size, channels, height, width) tuples, one for each
        pyramid scale, arranged from coarsest to finest resolution.
    """
    shapes: list[tuple[int, ...]] = []
    scale_factor = math.pow(
        options.min_size / (min(options.max_size, options.crop_size)),
        1 / options.stop_scale,
    )
    for i in range(options.stop_scale + 1):
        scale = math.pow(scale_factor, options.stop_scale - i)
        out_shape = cast(
            Sequence[int],
            np.uint(
                np.round(
                    np.array(
                        [
                            min(options.max_size, options.crop_size),
                            min(options.max_size, options.crop_size),
                        ]
                    )
                    * scale
                )
            ).tolist(),
        )
        if out_shape[0] % 2 != 0:
            out_shape = [int(shape + 1) for shape in out_shape]
        shapes.append((options.batch_size, options.num_img_channels, *out_shape))

    return tuple(shapes)


def as_image_file_list(data_file: DataFiles) -> list[Path]:
    """Return a sorted list of image file paths for the given data file type.

    Parameters
    ----------
    data_file : DataFiles
        The data file type (FACIES, WELLS, or SEISMIC) specifying which
        directory and file pattern to use.

    Returns
    -------
    list[Path]
        Sorted list of Path objects pointing to image files matching the
        pattern for the specified data file type.
    """
    data_dir = Path(data_file.as_data_path())
    return list(sorted(data_dir.glob(data_file.image_file_pattern)))


def as_model_file_list(data_file: DataFiles) -> list[Path]:
    """Return a sorted list of model checkpoint file paths for the given data file type.

    Parameters
    ----------
    data_file : DataFiles
        The data file type (FACIES, WELLS, or SEISMIC) specifying which
        directory and model file pattern to use.

    Returns
    -------
    list[Path]
        Sorted list of Path objects pointing to model checkpoint files
        matching the pattern for the specified data file type.
    """
    data_dir = Path(data_file.as_data_path())
    return list(sorted(data_dir.glob(data_file.model_file_pattern)))


@lru_cache(maxsize=1)
def as_wells_mapping(data_file: DataFiles) -> dict[str, tuple[int, int]]:
    """Load wells mapping from cache file.

    Returns
    -------
    dict[str, tuple[int, int]]
        Dictionary mapping image name to (column, non_black_pixels)
    """
    data_dir = Path(data_file.as_data_path())
    mapping_file = data_dir / data_file.mapping_file_pattern
    mapping_file = next(data_dir.glob(data_file.mapping_file_pattern))
    if not mapping_file.exists():
        raise FileNotFoundError(f"Wells mapping file not found: {mapping_file}")
    try:
        data = np.load(mapping_file, allow_pickle=True)
        columns = data["columns"]
        counts = data["counts"]
        image_names = data["image_names"]

        mapping = {
            name: (int(col), int(count))
            for name, col, count in zip(image_names, columns, counts)
        }
        return mapping
    except Exception as e:
        raise RuntimeError(f"Failed to load wells mapping: {e}")
