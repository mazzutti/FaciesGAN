import math
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import random
from typing import TypeVar, cast

from PIL import Image
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from data_files import DataFiles
from options import TrainningOptions

def set_seed(seed: int = 42) -> None:
    """Set seeds for torch, numpy and python.random at module level.

    Keeping a module-level `set_seed` makes it easy for other modules to
    call it without constructing a `NeuralSmoother` instance. The
    `NeuralSmoother.set_seed` method remains as a thin wrapper that
    delegates to this function to preserve backward compatibility.
    """
    seed_int = int(seed)
    torch.manual_seed(seed_int)  # pyright: ignore
    np.random.seed(seed_int)
    random.seed(seed_int)


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


def resolve_device() -> torch.device:
    """Return the preferred device: MPS (Apple), CUDA, or CPU.

    Exposed at module level so other modules can determine the best device
    without constructing a `NeuralSmoother` instance.
    """
    if (
        getattr(torch.backends, "mps", None) is not None
        and getattr(torch.backends.mps, "is_available", lambda: False)()
        and getattr(torch.backends.mps, "is_built", lambda: True)()
    ):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def mask_resize(mask: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Resize well mask tensor using bicubic interpolation with thresholding.

    This function resizes well location masks to match target pyramid scales.
    It amplifies the mask values before interpolation, then applies thresholding
    to preserve well locations at column positions.

    Parameters
    ----------
    mask : torch.Tensor
        Input well mask tensor of shape (B, C, H, W).
    size : tuple[int, ...]
        Target size (height, width) for the resized mask.

    Returns
    -------
    torch.Tensor
        Resized mask tensor with wells positioned at appropriate columns.
    """
    mask = (mask > 0).float() * 1000
    interpolated_mask = nn.functional.interpolate(
        mask, size=size, mode="bicubic", align_corners=False, antialias=True
    )
    mask_index = torch.argmax(torch.sum(interpolated_mask, dim=2), dim=-1)
    interpolated_mask.zero_()
    interpolated_mask[:, :, :, mask_index] = 1
    return interpolated_mask


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
        - facie_num_channels: Number of input channels

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
        shapes.append((options.batch_size, options.facie_num_channels, *out_shape))
        
    return tuple(shapes)


def weights_init(m: nn.Module) -> None:
    """Initialize neural network layer weights using normal distributions.

    Applies standard weight initialization strategies for convolutional and
    normalization layers. Conv2d layers use N(0, 0.02), while BatchNorm2d
    and InstanceNorm2d use N(1, 0.02) for weights and zero for biases.

    Parameters
    ----------
    m : nn.Module
        The neural network module to initialize.
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def norm(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0, 1] to [-1, 1] range.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with values in [0, 1] range.

    Returns
    -------
    torch.Tensor
        Normalized tensor with values clamped to [-1, 1].
    """
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def denorm(tensor: torch.Tensor, ceiling: bool = False) -> torch.Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1] range.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with values in [-1, 1] range.
    ceiling : bool, optional
        Whether to set all positive values to 1. Currently not implemented.
        Defaults to False.

    Returns
    -------
    torch.Tensor
        Denormalized tensor with values clamped to [0, 1].
    """
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    # if ceiling: tensor[torch.where(tensor > 0)] = 1
    return tensor


def torch2np(
    tensor: torch.Tensor, denormalize: bool = False, ceiling: bool = False
) -> NDArray[np.float32]:
    """Convert PyTorch tensor to NumPy array with optional denormalization.

    Transforms tensor from (B, C, H, W) format to (B, H, W, C) NumPy array,
    optionally denormalizing from [-1, 1] to [0, 1] range.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in (B, C, H, W) format.
    denormalize : bool, optional
        If True, denormalize from [-1, 1] to [0, 1]. Defaults to False.
    ceiling : bool, optional
        If True, set positive values to 1 during denormalization. Defaults to False.

    Returns
    -------
    NDArray[np.float32]
        NumPy array with shape (B, H, W, C) and values clipped to [0, 1].
    """
    if denormalize:
        tensor = denorm(tensor, ceiling)
    np_array = tensor.detach().cpu().numpy()
    np_array = np.permute_dims(np_array, (0, 2, 3, 1))
    np_array = np.clip(np_array, 0, 1)
    return np_array.astype(np.float32)


def np2torch(np_array: NDArray[np.float32], normalize: bool = False) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor with optional normalization.

    Parameters
    ----------
    np_array : NDArray[np.float32]
        Input NumPy array to convert.
    normalize : bool, optional
        Whether to normalize the tensor to [-1, 1] range. Defaults to False.

    Returns
    -------
    torch.Tensor
        Converted tensor, normalized to [-1, 1] range.

    Notes
    -----
    The function always normalizes the output, regardless of the normalize
    parameter value. This appears to be a bug in the implementation.
    """
    tensor = torch.from_numpy(np_array).float()  # type: ignore
    if normalize:
        tensor = norm(tensor)
    return norm(tensor)


def range_transform(
    facie: np.ndarray,
    in_range: tuple[int, int] = (0, 255),
    out_range: tuple[int, int] = (-1, 1),
) -> np.ndarray:
    """Transform array values from one range to another using linear scaling.

    Parameters
    ----------
    facie : np.ndarray
        Input facies array to transform.
    in_range : tuple[int, int], optional
        Input range (min, max) of the array. Defaults to (0, 255).
    out_range : tuple[int, int], optional
        Output range (min, max) for the transformed array. Defaults to (-1, 1).

    Returns
    -------
    np.ndarray
        Transformed array with values scaled to output range.
    """
    if in_range != out_range:
        scale = np.float32(out_range[1] - out_range[0]) / np.float32(in_range[1] - in_range[0])
        bias = np.float32(out_range[0]) - np.float32(in_range[0]) * scale
        facie = facie * scale + bias
    return facie


def reset_grads(model: nn.Module, require_grad: bool = False) -> nn.Module:
    """Set requires_grad attribute for all parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be updated.
    require_grad : bool, optional
        Value to set for requires_grad attribute. Defaults to False.

    Returns
    -------
    nn.Module
        The model with updated requires_grad attributes.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(require_grad)

    return model


def interpolate(tensor: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Resize the input noise tensor to the given size using bilinear interpolation.

    Parameters
    ----------
    facie : torch.Tensor
        The input noise tensor to be resized.
    size : tuple[int, ...]
        The target spatial dimensions for the resized tensor (height, width).
    ceiling : bool, optional
        If True, set all positive values in the interpolated tensor to 1.
        Defaults to False.

    Returns
    -------
    torch.Tensor
        The resized tensor tensor with the specified dimensions.
    """
    interpolated_tensor = nn.functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=True
    )
    return interpolated_tensor

def generate_noise(
    size: tuple[int, ...], device: torch.device, num_samp: int = 1, scale: float = 1.0
) -> torch.Tensor:
    """Generate a random noise tensor with specified dimensions.

    Parameters
    ----------
    size : tuple[int, ...]
        Shape of the noise tensor as (channels, height, width).
    device : torch.device
        Device on which to generate the tensor (CPU, CUDA, or MPS).
    num_samp : int, optional
        Number of samples (batch size) to generate. Defaults to 1.
    scale : float, optional
        Scale factor applied to spatial dimensions (height, width).
        Dimensions are divided by scale. Defaults to 1.0.

    Returns
    -------
    torch.Tensor
        Random tensor sampled from standard normal distribution with shape
        (num_samp, channels, height/scale, width/scale).
    """
    noise = torch.randn(num_samp, size[0], *[round(s / scale) for s in size[1:]], device=device)
    # if scale != 1:
    #     noise = to_facie_pyramid(noise, size[1:])
    return noise


def calc_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    LAMBDA: float,
    device: torch.device,
) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP training.

    Implements the gradient penalty term used in Wasserstein GAN with
    Gradient Penalty (WGAN-GP) to enforce the Lipschitz constraint.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator model.
    real_data : torch.Tensor
        Real data samples from the dataset.
    fake_data : torch.Tensor
        Generated fake data samples.
    LAMBDA : float
        Gradient penalty coefficient (typically 10.0).
    device : torch.device
        Device to perform calculations on.

    Returns
    -------
    torch.Tensor
        Calculated gradient penalty scalar value.
    """

    alpha = torch.rand(1, 1).expand(real_data.size()).to(device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    disc_interpolates = discriminator(interpolates)

    gradients: torch.Tensor = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # compute the L2 norm of gradients for each sample and apply the penalty
    gradients = cast(torch.Tensor, gradients.norm(2, dim=1) - 1)  # type: ignore
    gradient_penalty = (gradients**2).mean() * LAMBDA

    return gradient_penalty


def create_dirs(path: str) -> None:
    """Create directory and all parent directories if they don't exist.

    Parameters
    ----------
    path : str
        Directory path to create.

    Raises
    ------
    RuntimeError
        If directory creation fails.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        msg = "Error creating directory:"
        raise RuntimeError(msg, path, e)


T = TypeVar("T")


def load(
    path: str,
    device: torch.device = torch.device("cpu"),
    as_type: type[T] | None = None,
) -> T:
    """
    Load a PyTorch object from a file and return it as the requested type.

    Callers can pass a runtime class in `as_type` (for example `dict` or
    `list`) to indicate the expected shape of the loaded object. The
    function will `cast` the loaded object to the generic return type `T` so
    static type checkers can use the annotation. Note that `as_type` is only
    used for a runtime `isinstance` check (when possible) and for clarity; it
    does not change how `torch.load` behaves.

    Examples:
        state: Mapping[str, Any] = ops.load(path, as_type=dict)
        noises: list[torch.Tensor] = ops.load(path, as_type=list)

    Args:
        path (str): The file path to load the object from.
        device (torch.device): The device to map the loaded object to. Default is CPU.
        as_type (Type[T] | None): Optional runtime class to assert the loaded object's type.

    Returns:
        T: The loaded object, cast to the requested generic type.
    """
    obj = torch.load(path, map_location=device, weights_only=True)
    if as_type is not None:
        # `as_type` should be a concrete runtime class (e.g., `dict` or `list`).
        # For typing constructs like `Mapping[str, Any]` you should annotate the
        # receiving variable at the callsite instead.
        try:
            if not isinstance(obj, as_type):
                # non-fatal warning to help catch mismatches early
                print(f"Warning: loaded object is not an instance of {as_type}")
        except TypeError:
            # `as_type` may not be a valid runtime-checkable type (e.g., typing
            # constructs). Ignore the check in that case.
            pass
    return cast(T, obj)
