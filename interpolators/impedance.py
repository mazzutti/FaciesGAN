"""Impedance data loader and interpolator.

Reads a GSLIB-format 3D acoustic impedance volume, extracts XZ crossline
slices, normalizes values to [0, 1], and produces multi-scale pyramid
tensors using nearest-neighbor trace-wise interpolation (identical to the
seismic pathway).
"""

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.ndimage import zoom  # type: ignore

import matplotlib.cm as _mcm

from interpolators.base import BaseInterpolator
from interpolators.config import InterpolatorConfig

_viridis = _mcm.get_cmap("viridis")


def _to_viridis(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Map a (H, W) float32 array in [0, 1] to (H, W, 3) viridis RGB float32."""
    rgba: NDArray[np.float32] = _viridis(arr).astype(np.float32)  # type: ignore[attr-defined]
    return rgba[..., :3]  # drop alpha

logger = logging.getLogger(__name__)

# Stanford VI-E grid: nx=150, ny=200 (crosslines), nz=200 (depth)
_NX, _NY, _NZ = 150, 200, 200


def load_impedance_volume(dat_path: Path) -> NDArray[np.float32]:
    """Read a GSLIB .dat file and return the 3D impedance volume.

    The file is expected to have a 3-line header (title, number of
    variables, variable name) followed by ``NX * NY * NZ`` float values
    with x varying fastest (Fortran/GSLIB order).

    Returns
    -------
    NDArray[np.float32]
        Volume with shape ``(nz, ny, nx)`` — depth × crossline × inline.
    """
    with open(dat_path, "r") as fh:
        for _ in range(3):
            fh.readline()
        values = np.loadtxt(fh, dtype=np.float32)

    expected = _NX * _NY * _NZ
    if values.size != expected:
        raise ValueError(
            f"Expected {expected} values in {dat_path}, got {values.size}"
        )
    # GSLIB order: x fastest, then y, then z → reshape as (nz, ny, nx)
    return values.reshape(_NZ, _NY, _NX)


def extract_crosslines(
    volume: NDArray[np.float32],
    target_height: int = 120,
) -> list[NDArray[np.float32]]:
    """Extract XZ crossline slices and resize the depth axis.

    Parameters
    ----------
    volume : NDArray[np.float32]
        3-D impedance volume with shape ``(nz, ny, nx)``.
    target_height : int
        Number of depth pixels to keep (cropped from the top of the
        flipped slice, i.e. the shallowest rows).  No interpolation is
        performed — this matches how facies and seismic images are
        extracted.

    Returns
    -------
    list[NDArray[np.float32]]
        List of ``ny`` images, each with shape ``(target_height, nx)``
        and values globally normalized to [0, 1].
    """
    _, ny, _ = volume.shape
    vmin, vmax = volume.min(), volume.max()
    normed = (volume - vmin) / (vmax - vmin + 1e-8)

    slices: list[NDArray[np.float32]] = []
    for y_idx in range(ny):
        xz_slice = normed[:, y_idx, :]  # shape (nz, nx)
        # Flip depth axis so shallow is at top (matches facies PNG convention)
        xz_slice = xz_slice[::-1, :]
        # Crop to target_height (same as facies/seismic — no interpolation)
        xz_slice = xz_slice[:target_height, :]
        slices.append(xz_slice)
    return slices


class ImpedanceInterpolator(BaseInterpolator):
    """Multi-scale pyramid generator for a single impedance crossline image.

    Works like :class:`NearestInterpolator` but accepts a 2-D numpy array
    (already extracted and normalized) instead of loading a PNG from disk.
    """

    def __init__(self, config: InterpolatorConfig) -> None:
        super().__init__(config)

    def interpolate_array(
        self,
        img: NDArray[np.float32],
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Produce a multi-scale pyramid from a 2-D grayscale array.

        The grayscale slice is broadcast to 3 identical RGB channels so that
        the downstream pipeline (which expects 3-channel images) works
        without modification.

        Parameters
        ----------
        img : NDArray[np.float32]
            Grayscale image with shape ``(H, W)`` and values in [0, 1].
        resolutions : tuple[tuple[int, ...], ...]
            Target shapes as ``(batch, channels, height, width)`` or
            ``(batch, height, width, channels)`` when ``channels_last``.

        Returns
        -------
        list[torch.Tensor]
            One float32 tensor per resolution, each with shape ``(H, W, 3)``.
        """
        # Apply viridis colormap: (H, W) → (H, W, 3)
        img_rgb = _to_viridis(img)

        super_height, super_width = (
            resolutions[-1][1:3]
            if self.config.channels_last
            else resolutions[-1][2:]
        )

        high_res = self._resize(img_rgb, super_height, super_width)
        smooth_imgs: list[torch.Tensor] = []

        for resolution in resolutions[:-1]:
            if self.config.channels_last:
                _, new_h, new_w, _ = resolution
            else:
                _, _, new_h, new_w = resolution
            resized = self._resize(high_res, new_h, new_w)
            smooth_imgs.append(torch.from_numpy(resized.clip(0.0, 1.0)))  # type: ignore

        smooth_imgs.append(torch.from_numpy(high_res.astype(np.float32)))  # type: ignore
        return smooth_imgs

    # -- private helpers --------------------------------------------------

    @staticmethod
    def _resize(
        img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        h, w, _ = img.shape
        z_h = target_h / h
        z_w = target_w / w
        result: NDArray[np.float32] = np.asarray(
            zoom(img, (z_h, z_w, 1), order=1), dtype=np.float32
        )
        return result


# ── Neural regression smoother for impedance ────────────────────────────────

class _ImpedanceMLP(torch.nn.Module):
    """Tiny MLP with Fourier features that regresses a scalar value per (x, y)."""

    def __init__(self, mapping_size: int = 128, scale: float = 1.0, hidden_dim: int = 256) -> None:
        super().__init__()  # type: ignore[call-arg]
        B = torch.randn(2, mapping_size) * scale
        self.register_buffer("B", B, persistent=True)
        input_dim = mapping_size * 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        B_tensor: torch.Tensor = self.B  # type: ignore[assignment]
        x_proj = torch.matmul(x * float(2.0 * math.pi), B_tensor)
        feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(feats)  # (N, 1)


class ImpedanceNeuralSmoother(BaseInterpolator):
    """Neural regression smoother for a single impedance crossline.

    Trains a coordinate-MLP that maps (x, y) ∈ [-1, 1]² → scalar ∈ [0, 1]
    (normalized impedance value).  The scalar is replicated to 3 channels so
    that the downstream pipeline receives the same shape as the facies path.

    The checkpoint format is ``{"model_state": state_dict}`` — identical to
    :class:`NeuralSmoother` so that :class:`datasets.utils.as_model_file_list`
    can discover them by the ``*.pt`` glob.

    Parameters
    ----------
    model_path : Path
        Path to an existing ``.pt`` checkpoint or where one will be saved by
        :meth:`train`.
    config : InterpolatorConfig
        Shared configuration (geometry, upsample, chunk_size, scale).
    """

    def __init__(self, model_path: Path, config: InterpolatorConfig) -> None:
        super().__init__(config)
        import utils as _utils
        self.device: torch.device = _utils.resolve_device()
        self.model = _ImpedanceMLP(scale=config.scale).to(self.device)
        self._model_path = model_path
        if model_path.exists():
            self._load_checkpoint(model_path)

    # -- checkpoint helpers -----------------------------------------------

    def _load_checkpoint(self, path: Path) -> None:
        raw: Mapping[str, Any] = torch.load(str(path), map_location=self.device)
        ms = raw.get("model_state", raw)
        try:
            self.model.load_state_dict(dict(ms))  # type: ignore[arg-type]
        except RuntimeError as exc:
            logger.warning("Strict load failed (%s); retrying with strict=False", exc)
            self.model.load_state_dict(dict(ms), strict=False)  # type: ignore[arg-type]
        logger.info("Loaded impedance model from %s", path)

    # -- training ----------------------------------------------------------

    @classmethod
    def train(
        cls,
        arr: NDArray[np.float32],
        out_model_path: Path,
        config: InterpolatorConfig | None = None,
        epochs: int = 2000,
        lr: float = 3e-4,
    ) -> None:
        """Train a regression MLP on a single normalised crossline array.

        Parameters
        ----------
        arr : NDArray[np.float32]
            Grayscale array with shape ``(H, W)`` and values in ``[0, 1]``.
        out_model_path : Path
            Destination path for the saved ``{"model_state": ...}`` checkpoint.
        config : InterpolatorConfig, optional
            Configuration; defaults to ``InterpolatorConfig()``.
        epochs : int
            Number of Adam steps.
        lr : float
            Learning rate.
        """
        import utils as _utils
        from interpolators.neural import get_mgrid

        if config is None:
            config = InterpolatorConfig()

        device = _utils.resolve_device()
        model = _ImpedanceMLP(scale=config.scale).to(device)

        native_h, native_w = config.geometry
        # Resize arr to native geometry if needed
        if arr.shape != (native_h, native_w):
            arr = np.asarray(zoom(arr, (native_h / arr.shape[0], native_w / arr.shape[1]), order=1), dtype=np.float32)

        coords = get_mgrid(height=native_h, width=native_w).to(device)     # (H*W, 2)
        targets = torch.from_numpy(arr.ravel()).float().unsqueeze(-1).to(device)  # type: ignore # (H*W, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = torch.nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(coords)          # (H*W, 1)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step() # type: ignore
            scheduler.step()
            if (epoch + 1) % 500 == 0:
                logger.info("  [imp] epoch %d/%d  loss=%.6f", epoch + 1, epochs, loss.item())

        out_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, str(out_model_path))
        logger.info("Saved impedance checkpoint to %s", out_model_path)

    # -- inference ----------------------------------------------------------

    def interpolate_array(
        self,
        arr: NDArray[np.float32],
        resolutions: tuple[tuple[int, ...], ...],
    ) -> list[torch.Tensor]:
        """Produce a multi-scale pyramid from a 2-D grayscale array using the neural model.

        Falls back to bilinear upsampling (from the neural super-resolution
        output) instead of nearest-neighbour zoom.

        Parameters
        ----------
        arr : NDArray[np.float32]
            Grayscale image ``(H, W)`` in ``[0, 1]``.
        resolutions : tuple[tuple[int, ...], ...]
            Target shapes as ``(batch, C, H, W)`` or NHWC when channels_last.

        Returns
        -------
        list[torch.Tensor]
            One ``(H, W, 3)`` float32 tensor per resolution.
        """
        from interpolators.neural import get_mgrid

        _, _, super_h, super_w = self.get_target_dimensions()

        self.model.eval()
        with torch.inference_mode():
            coords = get_mgrid(height=super_h, width=super_w).to(self.device)
            chunks: list[torch.Tensor] = []
            for i in range(0, coords.shape[0], self.config.chunk_size):
                chunks.append(self.model(coords[i : i + self.config.chunk_size]))
            values = torch.cat(chunks, dim=0)  # (H*W, 1)

        # shape (1, 1, super_h, super_w) for F.interpolate
        prob_map = values.reshape(1, 1, super_h, super_w)

        smooth_imgs: list[torch.Tensor] = []
        for resolution in resolutions:
            if self.config.channels_last:
                _, new_h, new_w, _ = resolution
            else:
                _, _, new_h, new_w = resolution

            resized = F.interpolate(
                prob_map, size=(new_h, new_w), mode="bilinear",
                align_corners=False, antialias=True,
            )  # (1, 1, H, W)
            gray = resized.squeeze(0).squeeze(0).cpu().numpy()  # (H, W) np
            # Apply viridis colormap → (H, W, 3)
            rgb = torch.from_numpy(_to_viridis(gray)) # type: ignore
            smooth_imgs.append(rgb)

        return smooth_imgs
