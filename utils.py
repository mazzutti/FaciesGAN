from typing import Any, cast

import hashlib
from collections import OrderedDict
from typing_extensions import Self
import numpy as np
import scipy.stats as st
import torch
from PIL import Image, ImageDraw, ImageFont

from config import RESULTS_DIR
from ops import torch2np


class ExtractUniqueColors:
    """Callable class to extract unique colors from facies tensors with caching.

    Usage:
        extract_unique_colors = ExtractUniqueColors()
        palette = extract_unique_colors(facies_tensor, tolerance=0.01)

    The cache is keyed by `(tensor.shape, tolerance)`. Use
    `extract_unique_colors.clear_cache()` to invalidate.
    """

    # Singleton instance holder
    _instance: Self | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ExtractUniqueColors":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, max_cache_size: int = 128, device: torch.device = torch.device("cpu")
    ) -> None:
        # initialize cache once - safe even if __init__ called multiple times
        if getattr(self, "_cache", None) is None:
            # use OrderedDict for simple LRU eviction
            self._cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self.max_cache_size = max_cache_size
        # Optional device for torch operations (e.g. torch.device('cuda') or 'mps')
        self.device = device

    def __call__(
        self, facies_tensor: torch.Tensor, tolerance: float = 0.01
    ) -> np.ndarray:
        # Compute a small thumbnail fingerprint to include in the cache key.
        # This avoids returning stale results when the image content changes
        # but shapes remain the same.
        try:
            # If a torch tensor on CUDA, move a small downsampled tensor to CPU
            if facies_tensor.dim() >= 3:
                t0 = facies_tensor
                if t0.dim() == 4:
                    t0 = t0[0]
                # t0 shape: (C,H,W)
                # convert to HWC numpy quickly by permuting and taking a small thumbnail
                # Optionally move tensor to configured device for interpolation
                if (
                    getattr(self, "device", None) is not None
                    and t0.device != self.device
                ):
                    try:
                        t0 = t0.to(self.device)
                    except Exception:
                        pass

                small = torch.nn.functional.interpolate(
                    t0.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False
                )[0]
                # Ensure thumbnail is on CPU before converting to numpy
                try:
                    small_cpu = small.detach().cpu()
                except Exception:
                    small_cpu = small.detach()
                thumb_np = np.clip(torch2np(small_cpu), 0.0, 1.0)
            else:
                # Fallback: convert via torch2np
                facies_np_full = torch2np(
                    facies_tensor[0] if facies_tensor.dim() == 4 else facies_tensor
                )
                img = Image.fromarray(
                    (np.clip(facies_np_full, 0.0, 1.0) * 255).astype("uint8")
                )
                thumb = img.resize((16, 16), Image.BILINEAR)
                thumb_np = np.array(thumb).astype(np.float32) / 255.0

            try:
                thumb_hash = hashlib.sha1(thumb_np.tobytes()).hexdigest()[:12]
            except Exception:
                thumb_hash = "nohash"
        except Exception:
            thumb_hash = "nohash"

        key = (tuple(facies_tensor.shape), thumb_hash, float(tolerance))
        if key in self._cache:
            # move to end to mark recent use (LRU)
            val = self._cache.pop(key)
            self._cache[key] = val
            return val

        # Convert to numpy and reshape to (N, 3)
        if facies_tensor.dim() == 4:
            # Batch dimension present, use first sample
            facies_np = torch2np(facies_tensor[0])
        else:
            facies_np = torch2np(facies_tensor)

        # Reshape to (H*W, 3)
        pixels = facies_np.reshape(-1, 3)

        # Vectorized unique-color extraction using tolerance
        if float(tolerance) <= 0.0:
            unique_colors_array = np.unique(pixels, axis=0).astype(np.float32)
        else:
            # Scale and round to cluster colors within tolerance
            tol = float(tolerance)
            scaled = np.round(pixels / tol).astype(np.int64)
            uniq_scaled = np.unique(scaled, axis=0)
            unique_colors_array = (uniq_scaled.astype(np.float32)) * tol

        if unique_colors_array.size == 0:
            unique_colors_array = np.zeros((0, 3), dtype=np.float32)
        else:
            brightness = unique_colors_array.sum(axis=1)
            sorted_indices = np.argsort(brightness)
            unique_colors_array = unique_colors_array[sorted_indices]

        # Cache with simple LRU eviction
        self._cache[key] = unique_colors_array
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return unique_colors_array

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    @property
    def cache(self) -> dict[tuple[Any, ...], np.ndarray]:
        """Expose the internal cache dictionary for compatibility."""
        return self._cache


# Note: module-level helper instances were removed. Instantiate helpers
# at the call site (e.g., in `plot_generated_facies`) to allow explicit
# device and configuration control.


class PreprocessWellMask:
    """Callable class that preprocesses well masks and caches results.

    Usage:
        preprocess_well_mask = PreprocessWellMask()
        mask_2d, mask_bool, well_columns = preprocess_well_mask(mask, target_shape)

    The cache is keyed by (mask.shape, target_shape, sha1(mask.tobytes())).
    Call `preprocess_well_mask.clear_cache()` to invalidate.
    """

    def __init__(
        self, max_cache_size: int = 128, device: torch.device = torch.device("cpu")
    ) -> None:
        # OrderedDict for LRU eviction
        self._cache: OrderedDict[
            tuple[tuple[int, ...], tuple[int, int], str],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.device = device

    # Singleton instance holder
    _instance: Any | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "PreprocessWellMask":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(
        self, mask: np.ndarray, target_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask_2d = np.squeeze(mask)

        # Compute a stable content hash for the mask using the full mask bytes.
        # Using a thumbnail could cause collisions for small/resampled masks
        # (different scales appearing identical once downsampled). Hashing the
        # full mask bytes guarantees uniqueness across mask content and is
        # acceptably fast for mask arrays of typical size.
        try:
            content_hash = hashlib.sha1(mask_2d.tobytes()).hexdigest()[:12]
        except Exception:
            content_hash = "nohash"

        cache_key = (mask_2d.shape, target_shape, content_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if mask_2d.size > 0:
            # Resize mask to match target dimensions if needed (use PIL for speed)
            if mask_2d.shape != target_shape:
                from PIL import Image

                mask_img = Image.fromarray((mask_2d > 0.5).astype("uint8") * 255)
                # PIL resize expects (width, height)
                resized = mask_img.resize(
                    (target_shape[1], target_shape[0]), Image.NEAREST
                )
                mask_2d = np.array(resized) > 127

            # Find well column locations by summing mask vertically
            mask_bool = mask_2d > 0.5
            well_columns = np.where(np.sum(mask_bool, axis=0) > 0)[0]

            result = (mask_2d, mask_bool, well_columns)
            self._cache[cache_key] = result
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
            return result

        result = (mask_2d, np.zeros_like(mask_2d, dtype=bool), np.array([], dtype=int))
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()


class QuantizeToPureColors:
    """Singleton callable that quantizes an RGB array to a set of pure colors.

    Caches results by (image shape, content hash, palette signature, tolerance)
    to avoid recomputing nearest-color distances for repeated inputs.
    """

    _instance: Any | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "QuantizeToPureColors":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, max_cache_size: int = 128, device: torch.device = torch.device("cpu")
    ) -> None:
        if getattr(self, "_cache", None) is None:
            self._cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.device = device

    def _palette_signature(self, pure_colors: np.ndarray | None) -> tuple[Any, ...]:
        if pure_colors is None:
            # Default palette signature
            return (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
        arr = np.asarray(pure_colors, dtype=np.float32).ravel()
        # Round to reduce tiny float variations and convert to tuple for hashing
        return tuple(np.round(arr, 6).tolist())

    def __call__(
        self,
        rgb_array: np.ndarray | torch.Tensor,
        pure_colors: np.ndarray | None = None,
        tolerance: float = 0.0,
    ) -> np.ndarray:
        # Handle torch.Tensor input with optional GPU acceleration
        if isinstance(rgb_array, torch.Tensor):
            t = rgb_array
            if t.dim() == 4:
                t = t[0]
            # Convert CHW -> HWC
            arr_torch = t.permute(1, 2, 0).contiguous().float()
            # Use the device provided at construction time.
            if arr_torch.device != self.device:
                arr_torch = arr_torch.to(self.device)

            # Prepare palette as torch tensor on the chosen device
            if pure_colors is None:
                palette_t = torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=arr_torch.dtype,
                    device=self.device,
                )
            else:
                palette_t = torch.tensor(
                    np.asarray(pure_colors, dtype=np.float32), device=self.device
                )

            h0, w0, _ = arr_torch.shape
            pixels_t = arr_torch.view(-1, 3)
            # compute squared distances via broadcasting on torch
            dists = torch.sum(
                (pixels_t[:, None, :] - palette_t[None, :, :]) ** 2, dim=2
            )
            idx = torch.argmin(dists, dim=1)
            quantized_t = palette_t[idx].view(h0, w0, 3)
            # Always return a CPU numpy array for consistency
            quantized = quantized_t.detach().cpu().numpy()
            return quantized

        # Validate numpy input shape
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            return rgb_array

        # Use a short content hash for the image to avoid hashing very large data repeatedly
        arr = np.ascontiguousarray(rgb_array, dtype=np.float32)
        try:
            # Create a small thumbnail to fingerprint the image (faster than hashing full data)
            img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype("uint8"))
            thumb = img.resize((16, 16), Image.BILINEAR)
            # Convert thumbnail to a numpy array and use ndarray.tobytes() which has a known signature
            thumb_arr = np.asarray(thumb)
            h = hashlib.sha1(thumb_arr.tobytes()).hexdigest()[:12]
        except Exception:
            # Fallback to shape-only key if hashing fails
            h = "nohash"

        key = (arr.shape, h, self._palette_signature(pure_colors), float(tolerance))
        if key in self._cache:
            return self._cache[key]

        # Prepare palette
        if pure_colors is None:
            pure_colors_np = np.array(
                [
                    [0.0, 0.0, 0.0],  # Black
                    [1.0, 0.0, 0.0],  # Red
                    [0.0, 1.0, 0.0],  # Green
                    [0.0, 0.0, 1.0],  # Blue
                ],
                dtype=np.float32,
            )
        else:
            pure_colors_np = np.asarray(pure_colors, dtype=np.float32)

        h0, w0, _ = arr.shape
        pixels = arr.reshape(-1, 3)

        # Compute squared distances and assign nearest
        distances = np.sum(
            (pixels[:, np.newaxis, :] - pure_colors_np[np.newaxis, :, :]) ** 2, axis=2
        )
        nearest_indices = np.argmin(distances, axis=1)
        quantized = pure_colors_np[nearest_indices].reshape(h0, w0, 3)

        # Cache and return (LRU eviction)
        self._cache[key] = quantized
        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return quantized

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache(self) -> dict[tuple[Any, ...], np.ndarray]:
        return self._cache


# Quantizer instances should be created where needed (see note above).


def apply_well_mask(
    facies_array: np.ndarray,
    mask: np.ndarray,
    well_facies_array: np.ndarray,
    preprocessed_mask: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Overlay well pixels onto facies image, replacing black with white in well columns.

    Parameters
    ----------
    facies_array : np.ndarray
        The facies image array in shape (H, W) or (H, W, C).
    mask : np.ndarray
        Well mask array indicating which pixels are well locations.
    well_facies_array : np.ndarray
        The original well facies with true colors.
    preprocessed_mask : tuple[np.ndarray, np.ndarray, np.ndarray] | None, optional
        Precomputed (mask_2d, mask_bool, well_columns) from preprocess_well_mask.
        If provided, mask processing is skipped for performance.

    Returns
    -------
    np.ndarray
        Modified facies array with well pixels overlaid.
        use_gpu: bool = False,
    """
    # Ensure we're working with a copy
    result = facies_array.copy()

    # Use preprocessed mask if provided, otherwise compute it
    if preprocessed_mask is not None:
        mask_2d, mask_bool, well_columns = preprocessed_mask
        well_2d = np.squeeze(well_facies_array)

        # Resize well data if needed
        if well_2d.size > 0 and well_2d.shape != result.shape[:2]:
            from PIL import Image

            target_h, target_w = result.shape[0], result.shape[1]
            if well_2d.ndim == 2:
                img = Image.fromarray((well_2d * 255).astype("uint8"))
                resized = img.resize((target_w, target_h), Image.NEAREST)
                well_2d = np.array(resized).astype(np.float32) / 255.0
            else:
                # RGB case
                img = Image.fromarray((well_2d * 255).astype("uint8"), mode="RGB")
                resized = img.resize((target_w, target_h), Image.NEAREST)
                well_2d = np.array(resized).astype(np.float32) / 255.0
    else:
        # Legacy path: compute mask processing inline
        mask_2d = np.squeeze(mask)
        well_2d = np.squeeze(well_facies_array)

        if mask_2d.size == 0 or well_2d.size == 0:
            return result

        # Resize mask and well data to match facies dimensions if needed (use PIL for speed)
        if mask_2d.shape != result.shape[:2]:
            from PIL import Image

            target_h, target_w = result.shape[0], result.shape[1]
            # Resize mask
            mask_img = Image.fromarray((mask_2d > 0.5).astype("uint8") * 255)
            resized_mask = mask_img.resize((target_w, target_h), Image.NEAREST)
            mask_2d = np.array(resized_mask) > 127

            # Resize well data
            if well_2d.ndim == 2:
                img = Image.fromarray((well_2d * 255).astype("uint8"))
                resized = img.resize((target_w, target_h), Image.NEAREST)
                well_2d = np.array(resized).astype(np.float32) / 255.0
            else:
                img = Image.fromarray((well_2d * 255).astype("uint8"), mode="RGB")
                resized = img.resize((target_w, target_h), Image.NEAREST)
                well_2d = np.array(resized).astype(np.float32) / 255.0

        # Find well column locations by summing mask vertically
        mask_bool = mask_2d > 0.5
        well_columns = np.where(np.sum(mask_bool, axis=0) > 0)[0]

    # Apply well pixels where mask is non-zero
    if well_2d.size > 0 and len(well_columns) > 0:

        # Replace entire well columns with white first (vectorized)
        if len(well_columns) > 0:
            if result.ndim == 3:
                result[:, well_columns, :] = 1.0
            else:
                result[:, well_columns] = 1.0

        # Now apply well pixels ONLY where they are NOT black (to preserve white background)
        # Identify which well pixels are NOT black
        if well_2d.ndim == 3:
            # RGB case - calculate distance to black
            well_distances = np.sqrt(np.sum(well_2d**2, axis=-1))
            well_not_black = well_distances >= 0.3  # NOT near black
        elif well_2d.ndim == 2:
            # Grayscale case
            well_distances = np.abs(well_2d)
            well_not_black = well_distances >= 0.3  # NOT near black
        else:
            well_not_black = np.ones(well_2d.shape[:2], dtype=bool)

        # Combine mask with "not black" condition - only apply non-black well pixels
        final_mask = mask_bool & well_not_black

        # Apply well pixels where final_mask indicates
        if result.ndim == 3 and well_2d.ndim == 3:
            result[final_mask] = well_2d[final_mask]
        elif result.ndim == 3 and well_2d.ndim == 2:
            # Broadcast grayscale well to RGB
            for c in range(result.shape[2]):
                result[final_mask, c] = well_2d[final_mask]
        elif result.ndim == 2:
            if well_2d.ndim == 2:
                result[final_mask] = well_2d[final_mask]
            else:
                # Take first channel if well is RGB
                result[final_mask] = well_2d[final_mask, 0]

    return result


def draw_well_arrows(
    mask: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
    draw: ImageDraw.ImageDraw,
    x_offset: int,
    y_offset: int,
    cell_size: int,
) -> None:
    """Draw red arrow markers above the plot area to indicate well positions.

    Parameters
    ----------
    mask : np.ndarray
        Well mask array indicating well column positions.
    draw : ImageDraw.ImageDraw
        PIL ImageDraw object for drawing on the main canvas.
    x_offset : int
        X position of the subplot on the main canvas.
    y_offset : int
        Y position where arrows should be drawn (above the subplot).
    cell_size : int
        Width of the subplot in pixels.
    """
    # Sum vertically to find columns that contain well pixels
    # If caller provided a preprocessed mask tuple (mask_2d, mask_bool, well_columns)
    # use the `well_columns` directly since they are already resized to the
    # `cell_size` resolution; this yields exact pixel alignment.
    if isinstance(mask, tuple):
        _, _, well_columns = mask
        if well_columns.size == 0:
            return
        # well_columns are indices in [0, cell_size-1]; center the arrow on the
        # middle of the column by adding 0.5 before integer conversion.
        center_x = float(np.mean(well_columns)) + 0.5
        arrow_x = x_offset + int(round(center_x))

        arrow_y = y_offset + 3
        arrow_size = 10
        draw.polygon(
            [
                (arrow_x, arrow_y + arrow_size),
                (arrow_x - arrow_size // 2, arrow_y),
                (arrow_x + arrow_size // 2, arrow_y),
            ],
            fill=(255, 0, 0),
        )
        return

    # Fallback: accept a raw mask array and map its column indices into the
    # cell pixel coordinates (for backward compatibility with callers that
    # still pass the raw mask).
    mask_sum = np.sum(np.squeeze(mask), axis=0)
    well_cols = np.where(mask_sum > 0)[0]
    if well_cols.size == 0:
        return

    center_col = float(np.mean(well_cols))
    num_columns = len(mask_sum)
    center_x = (center_col + 0.5) * (cell_size / num_columns)
    arrow_x = x_offset + int(round(center_x))

    arrow_y = y_offset + 3
    arrow_size = 10
    draw.polygon(
        [
            (arrow_x, arrow_y + arrow_size),
            (arrow_x - arrow_size // 2, arrow_y),
            (arrow_x + arrow_size // 2, arrow_y),
        ],
        fill=(255, 0, 0),
    )


def plot_generated_facies(
    fake_facies: list[torch.Tensor],
    real_facies: torch.Tensor,
    masks: torch.Tensor,
    stage: int,
    index: int,
    out_dir: str = RESULTS_DIR,
    save: bool = False,
    cell_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Plot and optionally save generated facies using PIL (50-100x faster than matplotlib).

    Creates a grid visualization comparing generated facies realizations with
    real facies and well constraint overlays.

    Parameters
    ----------
    fake_facies : list[torch.Tensor]
        List of generated facies tensors, one tensor per real facie containing
        multiple generated samples.
    real_facies : torch.Tensor
        Tensor of real facies images.
    masks : torch.Tensor
        Tensor of well location masks corresponding to the real facies.
    stage : int
        Current training stage/scale for labeling.
    index : int
        Iteration index for filename when saving.
    out_dir : str, optional
        Directory to save the plot. Defaults to RESULTS_DIR.
    save : bool, optional
        Whether to save the plot to disk. Defaults to False.
    cell_size : int
        Size of each cell in pixels (default 256).
    device : torch.device
        Optional device to run torch-based helpers on (e.g. `torch.device('cuda')`).
    """
    if not save:
        return

    # Support both torch.Tensor inputs and already-converted numpy arrays.
    # If inputs are torch tensors, convert them to numpy using torch2np.

    num_real_facies = int(real_facies.shape[0])
    num_generated_per_real = int(fake_facies[0].shape[0])
    # Convert to numpy
    fake_facies_arr = [
        torch2np(fake_facie, denormalize=True) for fake_facie in fake_facies
    ]
    np_real_facies = torch2np(real_facies, denormalize=True, ceiling=True)
    np_masks = torch2np(masks)

    # Calculate grid dimensions with spacing and margins
    spacing = 20  # pixels between subplots
    title_height = 20  # height for subplot titles
    main_title_height = 30  # height for main title at top
    arrow_height = 15  # height for arrow markers above plots
    margin = 20  # margin around the entire figure
    cols = num_generated_per_real + 1
    rows = num_real_facies
    grid_width = cols * (cell_size + spacing) - spacing + 2 * margin
    grid_height = (
        main_title_height
        + arrow_height
        + rows * (cell_size + title_height + spacing)
        - spacing
        + 2 * margin
    )

    # Create output image (RGB)
    output_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    main_draw: ImageDraw.ImageDraw = ImageDraw.Draw(output_img)

    # Try to use a better font, fall back to default if unavailable
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        main_title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except (IOError, OSError):
        title_font = ImageFont.load_default()
        main_title_font = ImageFont.load_default()

    # Draw main title
    main_title = f"Stage {stage} - Well Log, Real vs Generated Facies"
    # Get text bounding box for centering
    bbox = main_draw.textbbox((0, 0), main_title, font=main_title_font)
    text_width = bbox[2] - bbox[0]
    main_title_x = (grid_width - text_width) // 2
    cast(Any, main_draw).text(
        (main_title_x, margin), main_title, fill=(0, 0, 0), font=main_title_font
    )

    # Extract unique colors from all real facies (do this once for all plots)
    # Instantiate helpers locally so callers can control device/config
    extractor = ExtractUniqueColors(device=device)
    preprocessor = PreprocessWellMask(device=device)
    quantizer = QuantizeToPureColors(device=device)

    pure_colors: np.ndarray = extractor(real_facies, 0.01)
    pure_colors = np.asarray(pure_colors)  # Ensure pure_colors is always a ndarray

    for i in range(num_real_facies):
        # Preprocess well mask once for this row (reusable for real and all generated facies)
        # Always use cell_size since all images will be resized to this dimension
        preprocessed_mask = preprocessor(np_masks[i], (cell_size, cell_size))

        # Real facies (first column) - use RGB directly without colormap
        real_arr = np.squeeze(np_real_facies[i])

        # Resize real_arr to cell_size first if needed (before applying mask)
        if real_arr.shape[:2] != (cell_size, cell_size):
            # Use PIL for faster nearest/bilinear resizing
            if real_arr.ndim == 2:
                img = Image.fromarray((real_arr * 255).astype("uint8"))
                resized = img.resize((cell_size, cell_size), Image.NEAREST)
                real_arr = np.array(resized).astype(np.float32) / 255.0
            else:
                img = Image.fromarray((real_arr * 255).astype("uint8"), mode="RGB")
                resized = img.resize((cell_size, cell_size), Image.BICUBIC)
                real_arr = np.array(resized).astype(np.float32) / 255.0

        # Apply well mask to show well pixels with original colors and replace black with white
        real_arr_with_wells = apply_well_mask(
            real_arr, np_masks[i], real_arr, preprocessed_mask
        )

        # Handle both grayscale (H, W) and RGB (H, W, C) arrays
        if real_arr_with_wells.ndim == 2:
            # Grayscale - convert to RGB by stacking
            real_rgb = np.stack(
                [real_arr_with_wells, real_arr_with_wells, real_arr_with_wells], axis=-1
            )
        else:
            # Already RGB
            real_rgb = real_arr_with_wells

        # Convert to uint8
        real_rgb = (real_rgb * 255).astype(np.uint8)

        h, w = real_rgb.shape[:2]
        real_img = Image.fromarray(real_rgb, mode="RGB")
        if h != cell_size or w != cell_size:
            real_img = real_img.resize((cell_size, cell_size), Image.BICUBIC)

        # Paste into grid with spacing, margin, and title
        x_offset = margin
        y_offset = (
            main_title_height
            + arrow_height
            + margin
            + i * (cell_size + title_height + spacing)
        )

        # Draw subplot title
        subplot_title = f"Well {i + 1}"
        bbox = main_draw.textbbox((0, 0), subplot_title, font=title_font)
        text_width = bbox[2] - bbox[0]
        title_x = x_offset + (cell_size - text_width) // 2
        # Shift subplot title 5 pixels up for improved spacing
        cast(Any, main_draw).text(
            (title_x, y_offset - 5), subplot_title, fill=(0, 0, 0), font=title_font
        )

        # Draw arrow above the plot (use preprocessed mask for exact pixel alignment)
        arrow_y = y_offset + title_height - arrow_height
        draw_well_arrows(preprocessed_mask, main_draw, x_offset, arrow_y, cell_size)

        output_img.paste(real_img, (x_offset, y_offset + title_height))

        # Generated facies (remaining columns) - use RGB directly without colormap
        for j in range(num_generated_per_real):
            gen_arr = fake_facies_arr[i][j]
            # Ensure gen_arr is a numpy float array in [0, 1]
            gen_arr_np = np.asarray(gen_arr, dtype=np.float32)
            if gen_arr_np.size == 0:
                gen_arr_np = gen_arr_np.reshape((cell_size, cell_size))

            # Resize to cell_size if needed (before applying mask)
            if gen_arr_np.shape[:2] != (cell_size, cell_size):
                # Resize generated array using PIL for speed and visual quality
                if gen_arr_np.ndim == 2:
                    img = Image.fromarray((gen_arr_np * 255).astype("uint8"))
                    resized = img.resize((cell_size, cell_size), Image.NEAREST)
                    gen_arr_np = np.array(resized).astype(np.float32) / 255.0
                else:
                    img = Image.fromarray(
                        (gen_arr_np * 255).astype("uint8"), mode="RGB"
                    )
                    resized = img.resize((cell_size, cell_size), Image.BICUBIC)
                    gen_arr_np = np.array(resized).astype(np.float32) / 255.0

            # Handle both grayscale (H, W) and RGB (H, W, C) arrays BEFORE quantization
            if gen_arr_np.ndim == 2:
                # Grayscale - convert to RGB by stacking
                gen_rgb = np.stack([gen_arr_np, gen_arr_np, gen_arr_np], axis=-1)
            else:
                # Already RGB
                gen_rgb = gen_arr_np

            # Normalize if values are outside [0,1]
            max_val = float(gen_rgb.max()) if gen_rgb.size > 0 else 1.0
            if max_val > 1.0:
                gen_rgb = gen_rgb / max_val

            # Quantize to pure colors extracted from real facies to remove noise
            # DO THIS BEFORE applying well mask to avoid corrupting well colors
            # Ensure pure_colors is a numpy ndarray of shape (N, 3) and dtype float
            pure_colors_np = np.asarray(pure_colors, dtype=np.float32)
            gen_rgb = quantizer(gen_rgb, pure_colors=pure_colors_np)

            # Now apply well mask using preprocessed mask and the already-processed real facies
            # Use real_arr_with_wells which already has white background in well columns
            gen_rgb = apply_well_mask(
                gen_rgb, np_masks[i], real_arr_with_wells, preprocessed_mask
            )

            # Convert to uint8
            gen_rgb = (gen_rgb * 255).astype(np.uint8)

            h, w = gen_rgb.shape[:2]
            gen_img = Image.fromarray(gen_rgb, mode="RGB")
            if h != cell_size or w != cell_size:
                gen_img = gen_img.resize((cell_size, cell_size), Image.BICUBIC)

            x_offset = margin + (j + 1) * (cell_size + spacing)

            # Draw subplot title for generated facies
            gen_title = f"Gen {j + 1}"
            bbox = main_draw.textbbox((0, 0), gen_title, font=title_font)
            text_width = bbox[2] - bbox[0]
            title_x = x_offset + (cell_size - text_width) // 2
            # Shift generated subplot title 5 pixels up to match real facies titles
            cast(Any, main_draw).text(
                (title_x, y_offset - 5), gen_title, fill=(0, 0, 0), font=title_font
            )

            # Draw arrow above the plot (use preprocessed mask for exact pixel alignment)
            arrow_y = y_offset + title_height - arrow_height
            draw_well_arrows(preprocessed_mask, main_draw, x_offset, arrow_y, cell_size)

            output_img.paste(gen_img, (x_offset, y_offset + title_height))

    # Save directly
    output_img.save(f"{out_dir}/gen_{stage}_{index}.png", optimize=True)


def get_best_distribution(data: np.ndarray) -> tuple[str, float, tuple[Any, ...]]:
    """
    Identify the best fitting distribution for the given data.

    Args:
        data (np.ndarray): The data to fit the distributions to.

    Returns:
        tuple[str, float, tuple[Any, ...]]: The name of the best fitting distribution, the p-value, and the parameters
        of the best fit.
    """
    dist_names = ["norm", "exponweib", "pareto", "genextreme"]
    # Explicitly type these containers so static analysis can infer types for methods like .append
    dist_results: list[tuple[str, float]] = []
    params: dict[str, tuple[Any, ...]] = {}

    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data.flatten())
        params[dist_name] = param

        # Applying the Kolmogorov-Smirnov test
        # Use flattened data for the KS test to match the fit input
        result = st.kstest(data.flatten(), dist_name, args=param)
        # pvalue may be a scalar or array-like; coerce to float
        pval = float(np.sum(result.pvalue))
        dist_results.append((dist_name, pval))

    # Select the best fitted distribution
    best_dist, best_p = max(dist_results, key=lambda item: item[1])

    print(f"Best fitting distribution: {best_dist}")
    print(f"Best p value: {best_p}")
    print(f"Parameters for the best fit: {params[best_dist]}")

    return best_dist, best_p, params[best_dist]
