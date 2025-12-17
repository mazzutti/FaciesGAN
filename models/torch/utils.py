from typing import cast

import torch
import torch.nn as nn

from models.types import T


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
    noise = torch.randn(
        num_samp, size[0], *[round(s / scale) for s in size[1:]], device=device
    )
    if scale != 1:
        noise = interpolate(noise, size[1:])
    return noise


def interpolate(tensor: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Resize the input tensor to the given size using bilinear interpolation.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be resized.
    size : tuple[int, ...]
        The target spatial dimensions for the resized tensor (height, width).

    Returns
    -------
    torch.Tensor
        The resized tensor with the specified dimensions.
    """
    interpolated_tensor = nn.functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=True
    )
    return interpolated_tensor


def weights_init(m: nn.Module) -> None:
    """Initialize neural network layer weights using normal distributions.

    Applies standard weight initialization strategies for convolutional and
    normalization layers. Conv2d layers use N(0, 0.02), while BatchNorm2d
    and InstanceNorm2d use N(1, 0.02) for weights and zero for biases.

    Skips normalization layers without affine parameters (e.g., InstanceNorm2d
    with affine=False).

    Parameters
    ----------
    m : nn.Module
        The neural network module to initialize.
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        # Only initialize if affine parameters exist
        if getattr(m, "weight", None) is not None:
            m.weight.data.normal_(1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0)


def load(
    path: str,
    device: torch.device = torch.device("cpu"),
    as_type: type[T] | None = None,
) -> T:
    """Load a PyTorch object from `path` and optionally validate its type.

    Parameters
    ----------
    path : str
        Filesystem path to the saved PyTorch object (state dict, tensors, list, etc.).
    device : torch.device, optional
        Device to map the loaded tensors to (defaults to CPU).
    as_type : type[T] | None, optional
        Optional runtime class used only for a non-fatal ``isinstance`` check
        to help callers detect mismatches early (for example, ``dict`` or
        ``list``). This parameter does not modify how ``torch.load`` behaves.

    Returns
    -------
    T
        The loaded object, cast to the requested generic return type.
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
