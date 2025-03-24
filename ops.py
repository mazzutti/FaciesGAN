import math
import os
from typing import Tuple, List

import numpy as np
import torch
from torch import nn

def facie_resize(facie: torch.Tensor, size: tuple[int, ...], ceiling=False) -> torch.Tensor:
    """
    Resize the input  tensor to the given size using bilinear interpolation.

    Args:
        facie (torch.Tensor): The input  tensor.
        size (tuple[int, ...]): The target size for the tensor.
        ceiling (bool, optional): Whether to set all positive values to 1. Defaults to False.

    Returns:
        torch.Tensor: The resized  tensor.
    """
    interpolated_facie = nn.functional.interpolate(facie, size=size, mode="bilinear", align_corners=True)
    if ceiling: interpolated_facie[torch.where(interpolated_facie > 0)] = 1
    return interpolated_facie


def mask_resize(mask: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """
    Resize the input masked_facie tensor to the given size using bicubic interpolation and apply a threshold.

    Args:
        mask (torch.Tensor): The input masked_facie tensor.
        size (tuple[int, ...]): The target size for the masked_facie.

    Returns:
        torch.Tensor: The resized masked_facie tensor with applied threshold.
    """
    mask = (mask > 0).float() * 1000
    interpolated_mask = nn.functional.interpolate(mask, size=size, mode="bicubic", align_corners=False, antialias=True)
    mask_index = torch.argmax(torch.sum(interpolated_mask, dim=2), dim=-1)
    interpolated_mask.zero_()
    interpolated_mask[:, :, :, mask_index] = 1
    return interpolated_mask

def generate_scales(options) -> List[Tuple[int, ...]]:
    """
    Generate a list of shapes for different scales based on the given options.

    Args:
        options: An object containing the options for generating scales. It should have the following attributes:
            - min_size (int): The minimum size for the scales.
            - max_size (int): The maximum size for the scales.
            - crop_size (int): The crop size for the scales.
            - stop_scale (int): The number of scales to generate.
            - batch_size (int): The batch size for each scale.
            - facie_num_channels (int): The number of  channels.

    Returns:
        list: A list of tuples representing the shapes for each scale.
    """
    shapes = []
    scale_factor = math.pow(options.min_size / (min(options.max_size, options.crop_size)), 1 / options.stop_scale)
    for i in range(options.stop_scale + 1):
        scale = math.pow(scale_factor, options.stop_scale - i)
        out_shape = np.uint(np.round(np.array([
            min(options.max_size, options.crop_size),
            min(options.max_size, options.crop_size)]) * scale)).tolist()
        if out_shape[0] % 2 != 0:
            out_shape = [int(shape + 1) for shape in out_shape]
        shapes.append((options.batch_size, options.facie_num_channels, *out_shape))
    return shapes

def weights_init(m):
    """
    Initialize the weights of the given module.

    Args:
        m (torch.nn.Module): The module to initialize.

    If the module is a Conv2d layer, its weights are initialized with a normal distribution
    with mean 0.0 and standard deviation 0.02. If the module is a BatchNorm2d or InstanceNorm2d
    layer, its weights are initialized with a normal distribution with mean 1.0 and standard deviation 0.02,
    and its biases are initialized to 0.
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def norm(x):
    """
    Normalize the input tensor to the range [-1, 1].

    Args:
        x (torch.Tensor): The input tensor to normalize.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

def denorm(tensor: torch.Tensor, ceiling: bool = False) -> torch.Tensor:
    """
    Denormalize the input tensor from the range [-1, 1] to [0, 1].

    Args:
        tensor (torch.Tensor): The input tensor to denormalize.
        ceiling (bool, optional): Whether to set all positive values to 1. Defaults to False.

    Returns:
        torch.Tensor: The denormalized tensor.
    """
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    # if ceiling: tensor[torch.where(tensor > 0)] = 1
    return tensor


def torch2np(tensor: torch.Tensor, denormalize: bool = False, ceiling: bool = False) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The input tensor to convert.
        denormalize (bool, optional): Whether to denormalize the tensor from the range [-1, 1] to [0, 1]. Defaults to False.
        ceiling (bool, optional): Whether to set all positive values to 1. Defaults to False.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    if denormalize: tensor = denorm(tensor, ceiling)
    tensor = np.permute_dims(tensor.cpu().detach().numpy(), (0, 2, 3, 1))
    tensor = np.clip(tensor, 0, 1)
    return tensor.astype(np.float32)

def np2torch(tensor: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor and normalize it to the range [-1, 1].

    Args:
        tensor (np.ndarray): The input NumPy array to convert.
        normalize (bool): Whether to normalize the tensor to the range [-1, 1].

    Returns:
        torch.Tensor: The converted and normalized PyTorch tensor.
    """
    tensor = torch.from_numpy(tensor).float()
    if normalize: tensor = norm(tensor)
    return norm(tensor)


def range_transform(facie: np.ndarray,
                    in_range: Tuple[int, int] = (0, 255),
                    out_range: Tuple[int, int] = (-1, 1)) -> np.ndarray:
    """
    Transforms the input  from one range to another.

    Args:
        facie (np.ndarray): The input  array.
        in_range (Tuple[int, int]): The input range of the .
        out_range (Tuple[int, int]): The output range for the .

    Returns:
        np.ndarray: The transformed  array.
    """
    if in_range != out_range:
        scale = np.float32(out_range[1] - out_range[0]) / np.float32(in_range[1] - in_range[0])
        bias = np.float32(out_range[0]) - np.float32(in_range[0]) * scale
        facie = facie * scale + bias
    return facie

def reset_grads(model: nn.Module, require_grad: bool = False) -> nn.Module:
    """
    Set the requires_grad attribute of all parameters in the models.

    Args:
        model (nn.Module): The models whose parameters' requires_grad attribute will be set.
        require_grad (bool): The value to set for the requires_grad attribute. Default is False.

    Returns:
        nn.Module: The models with updated requires_grad attributes.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(require_grad)

    return model

def generate_noise(size: Tuple[int, ...], device: torch.device, num_samp: int = 1, scale: float = 1.0) -> torch.Tensor:
    """
    Generate random noise tensor.

    Args:
        size (Tuple[int]): The size of the noise tensor.
        device (torch.device): The device to generate the noise tensor on.
        num_samp (int): The number of samples to generate. Default is 1.
        scale (float): The scale factor for resizing the noise tensor. Default is 1.0.

    Returns:
        torch.Tensor: The generated noise tensor.
    """
    noise = torch.randn(num_samp, size[0], *[round(s / scale) for s in size[1:]], device=device)
    if scale != 1: noise = facie_resize(noise, size[1:])
    return noise


def calc_gradient_penalty(discriminator, real_data, fake_data, LAMBDA, device):
    """
    Calculate the gradient penalty for WGAN-GP.

    Args:
        discriminator (nn.Module): The discriminator models.
        real_data (torch.Tensor): The real data samples.
        fake_data (torch.Tensor): The generated fake data samples.
        LAMBDA (float): The gradient penalty coefficient.
        device (torch.device): The device to perform the calculations on.

    Returns:
        torch.Tensor: The calculated gradient penalty.
    """

    alpha = torch.rand(1, 1).expand(real_data.size()).to(device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

def create_dirs(path: str) -> None:
    """
    Create directories if they do not exist.

    Args:
        path (str): The directory path to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Error creating directory {path}: {e}")


def load(path: str, device: torch.device = torch.device("cpu")) -> dict | torch.nn.Module | torch.Tensor:
    """
    Load a PyTorch models state dictionary from a file.

    Args:
        path (str): The file path to load the models state dictionary from.
        device (torch.device): The device to map the loaded state dictionary to. Default is CPU.

    Returns:
        dict: The loaded state dictionary.
    """
    return torch.load(path, map_location=device, weights_only=True)