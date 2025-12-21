"""Custom dataset and data loader for image-based neural network training.

This module provides dataset and loader implementations for processing images
with optional transformations, color encoding, and batching support.
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import datasets.utils as data_utils
import utils
from interpolators.color_encoder import ColorEncoder
from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

FileLike: TypeAlias = str | os.PathLike[str] | IO[bytes]

# Module logger
logger = logging.getLogger(__name__)


def get_mgrid(height: int, width: int) -> torch.Tensor:
    """Create a flattened meshgrid of normalized coordinates in [-1, 1].

    This is the module-level variant extracted from `NeuralSmoother.get_mgrid`.
    Keeping a top-level function makes it easier for other modules to import
    and reuse the grid logic without constructing a `NeuralSmoother`.
    """
    tensors = (
        torch.linspace(-1, 1, steps=height),
        torch.linspace(-1, 1, steps=width),
    )
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, 2)


class NeuralSmoother(BaseInterpolator):
    """Encapsulates utility helpers and the main training/rendering engine.

    The original module-level functions `set_seed`, `get_mgrid` and
    `train` have been moved here. Thin module-level wrappers
    below keep API compatibility.
    """

    def __init__(
        self,
        model_path: Path,
        config: InterpolatorConfig,
    ) -> None:
        """Initialize a NeuralSmoother instance.

        Parameters
        ----------
        model_path : Path
                Filesystem path to a model checkpoint. If the file exists it will be
                loaded and the model weights restored. If the path does not point to
                an existing checkpoint the current implementation raises
                ``FileNotFoundError`` and training must be performed separately.
        config : InterpolatorConfig
                Configuration object containing interpolator parameters such as
                ``scale``, ``upsample`` and ``chunk_size`` which influence model
                construction and inference behavior.

        Notes
        -----
        - The constructor resolves a runtime device and constructs a
            :class:`ResidualMLP` moved to that device.
        - Model compilation via ``torch.compile`` is attempted in
            :meth:`_compile_model` where supported; failures are logged and
            execution continues with the uncompiled model.
        - Side effects: sets ``self.device``, ``self.num_classes`` and
            ``self.model``, and attempts to load weights from ``model_path``.
        """
        super().__init__(config)
        # resolved device for the instance (MPS/CUDA/CPU)
        self.device: torch.device = utils.resolve_device()
        self.num_classes: int = int(config.num_classes or 4)
        self.model = ResidualMLP(
            num_classes=self.num_classes,
            scale=config.scale,
        ).to(self.device)
        self._load_model(model_path)

    def _state_from_checkpoint(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        """Normalize common checkpoint shapes into a dict[str, Any].

        This is intentionally short and permissive: we accept Mapping checkpoints
        that contain `model_state` or `state_dict`, mapping-like state dicts
        (name->tensor), or any iterable convertible to `dict()`.
        """
        try:
            ms = state.get("model_state", state.get("state_dict", state))
            normalized = {str(k): v for k, v in dict(ms).items()}
            return normalized
        except Exception:
            return None

    def _compile_model(self) -> None:
        """Attempt to JIT/compile the model when supported (skip on MPS)."""
        try:
            if self.device.type == "mps":
                logger.info("Skipping torch.compile() on MPS (known Inductor issues)")
            else:
                try:
                    self.model = torch.compile(self.model)  # pyright: ignore
                    logger.info("Model compiled with torch.compile()")
                except Exception:
                    logger.info("torch.compile() not available or failed; continuing")
        except Exception:
            logger.info("Failed checking device for compilation; continuing")

    def _load_model(self, model_Path: Path) -> None:
        """Orchestrate model compilation, optional restore, and optimizer setup.

        This method delegates detailed work to small helpers to keep the
        responsibilities clear and testable.
        """
        self._compile_model()
        if model_Path.exists():
            state = torch.load(str(model_Path), map_location=self.device)
            ms: dict[str, Any] | None = self._state_from_checkpoint(state)
            if ms is None:
                raise RuntimeError(
                    "Unable to interpret checkpoint state as model state"
                )
            self.model.load_state_dict(ms)  # pyright: ignore
            logger.info(
                f"Loaded model checkpoint from {model_Path}; skipping training."
            )
        else:
            raise FileNotFoundError("No checkpoint found; training model from scratch.")

    def interpolate(
        self,
        image_path: Path,
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Render smoothed facies images at multiple resolutions using neural interpolation.

        This method performs inference with the trained ResidualMLP model at an
        upsampled super-resolution, evaluating coordinates in chunks to control
        memory usage. It loads the specified image to build a ColorEncoder palette,
        computes class probabilities via softmax, then bilinearly interpolates
        these probabilities to each requested resolution and converts them to RGB.

        Parameters
        ----------
        image_path : Path
            Filesystem path to the input image file. This image is loaded to
            construct a ColorEncoder that provides the color palette for
            converting model predictions (class labels) to RGB values.
        resolutions : list[tuple[int, ...]]
            List of (height, width) tuples specifying the desired output
            resolutions. Each resolution produces an interpolated image by
            bilinearly resampling the model's probability maps.

        Returns
        -------
        list[NDArray[np.float32]]
            A list of smoothed images as numpy arrays, one per requested resolution,
            each with shape (H, W, 3) and dtype float32 with values in [0, 1]
            representing RGB color intensities.

        Notes
        -----
        - The model is set to evaluation mode and inference runs under
          ``torch.inference_mode`` for efficiency and reproducibility.
        - The coordinate grid spans the super-resolution dimensions computed
          from ``self.config.geometry`` and ``self.config.upsample``.
        - Coordinates are processed in batches of size ``self.config.chunk_size``
          to limit peak GPU/CPU memory. Increase ``chunk_size`` for better
          throughput at the cost of higher memory usage.
        - A ColorEncoder is created from ``image_path`` during this call and
          stored in ``self.encoder`` for palette-based RGB conversion.
        """
        logger.info("Rendering facies pyramid...")
        # Get dimensions using base helper method
        _, _, super_height, super_width = self.get_target_dimensions()

        self.model.eval()  # pyright: ignore

        with torch.inference_mode():
            coords = get_mgrid(height=super_height, width=super_width).to(self.device)

            logits_chunks: list[torch.Tensor] = []
            for i in range(0, coords.shape[0], self.config.chunk_size):
                chunk = coords[i : i + self.config.chunk_size]
                logits_chunks.append(self.model(chunk))

            logits = torch.cat(logits_chunks, dim=0)

            probs = torch.softmax(logits, dim=1)
            probs = (
                probs.reshape(super_height, super_width, -1)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            labels = torch.argmax(probs.squeeze(0), dim=0)
            labels = labels.to(self.device)
            img_np = data_utils.load_image(image_path)
            self.encoder = ColorEncoder(img_np, device=self.device)
            pred_rgb = self.encoder.labels_to_rgb(labels)
            pred_rgb = pred_rgb.detach().cpu().reshape(super_height, super_width, 3)
            palette = self.encoder.palette_tensor.to(self.device).float()
            smooth_imgs: list[torch.Tensor] = []

            for resolution in resolutions:
                if self.config.channels_last:
                    _, new_h, new_w, _ = resolution
                else:
                    _, _, new_h, new_w = resolution

                inter_probs = F.interpolate(
                    probs,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

                inter_probs = (
                    inter_probs.squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(-1, inter_probs.shape[1])
                )

                inter_probs = inter_probs.to(self.device)
                pred_rgb = torch.matmul(inter_probs, palette)
                smooth_img = (  # pyright: ignore
                    pred_rgb.detach().cpu().reshape(new_h, new_w, 3)  # pyright: ignore
                )
                smooth_imgs.append(smooth_img)  # pyright: ignore

        return smooth_imgs


# ==========================================
# 2. IMPROVED ARCHITECTURE: Residual MLP
# ==========================================
class FourierFeatureTransform(nn.Module):
    """Fourier feature mapping used to embed 2D coordinates.

    This module projects 2D coordinates into a higher-dimensional
    sinusoidal feature space using a random Gaussian mapping matrix.

    Parameters
    ----------
    mapping_size : int
        Number of Fourier features per sine/cosine pair.
    scale : float
        Frequency scaling (sigma) applied to the random projection.
    """

    def __init__(self, mapping_size: int = 256, scale: float = 10.0) -> None:
        """Initialize Fourier feature projection and register buffers."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        # 'scale' is the "sigma". Higher = sharper/noisier. Lower = smoother/blurrier.
        # store as a buffer (not a trainable parameter) to avoid showing up in optimizer
        B = torch.randn(2, mapping_size) * scale
        self.register_buffer("B", B, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier features to input coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Concatenated sin/cos feature tensor of shape (..., mapping_size*2).
        """
        # Ensure numeric constant is a Python float so the result of the
        # multiplication with a torch.Tensor is a torch.Tensor. This helps
        # static type-checkers resolve the expression type (avoid 'Unknown').
        factor: float = float(2.0 * np.pi)
        # Use a locally-cast buffer to satisfy static type-checkers which may
        # infer an incorrect union type for registered buffers (e.g. Tensor | Module)
        B_tensor: torch.Tensor = cast(torch.Tensor, getattr(self, "B"))
        # Use torch.matmul to make the tensor operation explicit for static type checkers
        x_proj: torch.Tensor = torch.matmul(x * factor, B_tensor)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualMLP(nn.Module):
    """Residual MLP with Fourier features and skip connections.

    This architecture uses a Fourier feature embedding, residual/skip
    connections and LayerNorm to produce stable per-coordinate class
    predictions.
    """

    def __init__(
        self,
        num_classes: int,
        mapping_size: int = 128,
        scale: float = 1.0,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the ResidualMLP network components.

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        mapping_size : int
            Size of Fourier mapping features.
        scale : float
            Frequency scaling for Fourier features.
        hidden_dim : int
            Hidden layer dimension size.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.fourier = FourierFeatureTransform(mapping_size, scale)
        input_dim = mapping_size * 2

        # Standard layers
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # Smoother than ReLU
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Skip connection point: We concat input_dim + hidden_dim
        self.skip_layer = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute per-coordinate class logits from input coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Input coordinate tensor of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Unnormalized class logits for each input coordinate.
        """
        # Embed coordinates
        x_emb = self.fourier(coords)

        # First block
        h = self.layer1(x_emb)
        h = self.layer2(h)

        # Skip connection
        h = torch.cat([h, x_emb], dim=-1)
        h = self.skip_layer(h)

        # Final block
        h = self.layer3(h)
        return self.output(h)
