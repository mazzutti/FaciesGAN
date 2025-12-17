from .base import FaciesGAN
from .discriminator import Discriminator
from .generator import Generator
from .torch.facies_gan import TorchFaciesGAN

__all__ = [
    "Discriminator",
    "Generator",
    "FaciesGAN",
    "TorchFaciesGAN",
]
