"""Models package initialization.

This package exposes the project's model modules: generator, discriminator
and helper layers. Importing this package makes it clear you are working
with model objects, e.g. ``from models import generator``. This file is
kept minimal on purpose — all public APIs live in the submodules.
"""

__all__ = ["generator", "discriminator", "custom_layer", "facies_gan"]
