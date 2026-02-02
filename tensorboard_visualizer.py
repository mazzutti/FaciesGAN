"""TensorBoard-based training visualizer for parallel LAPGAN training.

This provides a clean, real-time, non-blocking visualization using TensorBoard.
Much more responsive than matplotlib with better interactivity.
"""

from __future__ import annotations

# pyright: reportUnknownMemberType=false
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter  # pyright: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainning.metrics import ScaleMetrics
    from typedefs import TTensor  # type: ignore
else:
    # Use string-based/type-agnostic annotations at runtime to avoid importing
    # the `trainning` package here and creating a circular import during
    # package initialization. The actual types are only needed for static
    # type checking.
    ScaleMetrics = object  # type: ignore
    TTensor = object  # type: ignore


class TensorBoardVisualizer:
    """Real-time training visualization using TensorBoard.

    This helper writes per-scale scalar metrics and generated sample images to
    a `tensorboardX.SummaryWriter`. It accepts either a `ScaleMetrics`
    dataclass (with tensor-valued fields) or a pre-flattened mapping of
    floats and converts values appropriately before logging.

    Attributes
    ----------
    num_scales : int
        Number of scales being trained in parallel.
    output_dir : str
        Directory where visualization images are saved.
    update_interval : int
        Frequency (in epochs) at which `update()` will emit logs.
    writer : SummaryWriter
        TensorBoard writer used to record scalars and images.
    start_time : float
        Timestamp when the visualizer was created (used to compute elapsed time).
    """

    def __init__(
        self,
        num_scales: int,
        output_dir: str,
        log_dir: str | None = None,
        update_interval: int = 1,
        dataset_info: str | None = None,
    ):
        """Initialize the TensorBoard visualizer.

        Parameters
        ----------
        num_scales : int
            Number of scales being trained in parallel.
        output_dir : str
            Directory to save visualization images.
        log_dir : str, optional
            Directory for TensorBoard logs. If None, uses output_dir/tensorboard_logs
        update_interval : int
            How often to log metrics (in epochs).
        dataset_info : str, optional
            Information about the dataset being used.
        """
        self.num_scales = num_scales
        self.output_dir = output_dir
        self.update_interval = update_interval
        self.dataset_info = dataset_info or "Unknown dataset"

        # Setup TensorBoard logging
        if not log_dir:
            log_dir = os.path.join(output_dir, "tensorboard_logs")

        # Ensure directories exist; guard against empty strings and race conditions.
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        os.makedirs(output_dir or ".", exist_ok=True)

        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        # Training timing
        self.start_time = time.time()
        self.last_update_time = self.start_time

        # Write dataset info as text
        self.writer.add_text("Dataset/Info", self.dataset_info, 0)
        self.writer.add_text(
            "Training/Scales", f"Training {num_scales} scales in parallel", 0
        )

        print(f"✅ TensorBoard initialized. View at: tensorboard --logdir={log_dir}")
        print(f"   Or run: tensorboard --logdir={log_dir} --port=6006")

    def update(
        self,
        epoch: int,
        scale_metrics: ScaleMetrics[TTensor] | dict[int, dict[str, float]],
        generated_samples: tuple[TTensor, ...] | None = None,
        samples_processed: int = 0,
    ) -> None:
        """Update TensorBoard with per-scale metrics and optional images.

        Parameters
        ----------
        epoch : int
            Current epoch number (used as the global step in TensorBoard).
        scale_metrics : ScaleMetrics or tuple[dict[str, float], ...]
            Either a `ScaleMetrics` dataclass (with tensor fields) or a
            pre-flattened tuple of metric-name->float dictionaries, one per scale.
            When a `ScaleMetrics` is provided the method will extract scalar
            values via `tensor.item()` before logging.
        generated_samples : dict[int, TTensor] | None, optional
            Dictionary of generated sample tensors, one per scale. Expected
            tensor shapes: (B, C, H, W) or (C, H, W). The first sample in the
            batch is used for logging. Tensors are detached and moved to CPU
            before conversion to numpy arrays.
        samples_processed : int, optional
            Running count of processed samples used for progress logging.

        Notes
        -----
        This method normalizes images to the [0, 1] range when necessary and
        converts them to CHW format for TensorBoard. Mean scalars across
        scales are written under the `Mean/*` tags.
        """
        # Update at intervals
        if epoch % self.update_interval != 0 and epoch != 1:
            return

        # At runtime `ScaleMetrics` is an alias to `object` to avoid imports,
        # so `isinstance(..., ScaleMetrics)` will be true for any object
        # (including plain dicts). Use attribute detection instead to
        # determine whether we have a typed ScaleMetrics instance.
        if hasattr(scale_metrics, "as_tuple_of_dicts"):
            flat_metrics: dict[int, dict[str, float]] = {
                i: metrics
                for i, metrics in enumerate(scale_metrics.as_tuple_of_dicts())
            }
        else:
            flat_metrics: dict[int, dict[str, float]] = scale_metrics

        # Calculate timing info
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Log individual scale metrics
        for scale, metrics in flat_metrics.items():
            # Discriminator losses
            self.writer.add_scalar(f"Scale_{scale}/D_Total", metrics["d_total"], epoch)
            self.writer.add_scalar(f"Scale_{scale}/D_Real", metrics["d_real"], epoch)
            self.writer.add_scalar(f"Scale_{scale}/D_Fake", metrics["d_fake"], epoch)

            # Generator losses
            self.writer.add_scalar(f"Scale_{scale}/G_Total", metrics["g_total"], epoch)
            self.writer.add_scalar(
                f"Scale_{scale}/G_Adversarial", metrics["g_adv"], epoch
            )
            self.writer.add_scalar(
                f"Scale_{scale}/G_Reconstruction", metrics["g_rec"], epoch
            )

        # Compute and log mean losses across all scales
        if scale_metrics:
            scales = list(range(len(flat_metrics)))

            # Mean discriminator losses
            mean_d_total = np.mean([flat_metrics[s]["d_total"] for s in scales])
            mean_d_real = np.mean([flat_metrics[s]["d_real"] for s in scales])
            mean_d_fake = np.mean([flat_metrics[s]["d_fake"] for s in scales])

            self.writer.add_scalar("Mean/D_Total", mean_d_total, epoch)
            self.writer.add_scalar("Mean/D_Real", mean_d_real, epoch)
            self.writer.add_scalar("Mean/D_Fake", mean_d_fake, epoch)

            # Mean generator losses
            mean_g_total = np.mean([flat_metrics[s]["g_total"] for s in scales])
            mean_g_adv = np.mean([flat_metrics[s]["g_adv"] for s in scales])
            mean_g_rec = np.mean([flat_metrics[s]["g_rec"] for s in scales])

            self.writer.add_scalar("Mean/G_Total", mean_g_total, epoch)
            self.writer.add_scalar("Mean/G_Adversarial", mean_g_adv, epoch)
            self.writer.add_scalar("Mean/G_Reconstruction", mean_g_rec, epoch)

        # Log training progress
        self.writer.add_scalar("Training/Samples_Processed", samples_processed, epoch)
        self.writer.add_scalar("Training/Elapsed_Time_Minutes", elapsed / 60, epoch)

        # Log generated samples as images with color mapping
        if generated_samples:
            print(f"   Logging {len(generated_samples)} sample images...")
            for scale, sample in enumerate(generated_samples):
                # Convert to numpy
                if isinstance(sample, torch.Tensor):
                    img = np.array(sample.detach().cpu())
                else:
                    img = np.array(sample)

                # Handle different tensor formats
                if img.ndim == 4:  # (B, C, H, W)
                    img = img[0]

                # Handle tensor format: convert (C, H, W) to (H, W, C)
                if img.ndim == 3 and img.shape[0] in [1, 3]:  # Channel first format
                    img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)

                # Check if already RGB (3 channels with values that look like RGB)
                is_rgb = img.ndim == 3 and img.shape[2] == 3

                if is_rgb:
                    # Already RGB - normalize to [0, 1] if needed
                    if img.max() > 1.0:
                        img = img / 255.0
                    else:
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                else:
                    # Single channel or grayscale - apply colormap for visualization
                    if img.ndim == 3:
                        img = img[:, :, 0]  # Extract first channel

                    # Normalize to [0, 1]
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                # Ensure float32 and proper range
                img = np.clip(img, 0, 1).astype(np.float32)

                # Convert to CHW format for TensorBoard (expects C, H, W)
                img_chw = np.transpose(img, (2, 0, 1))
                self.writer.add_image(f"Samples/Scale_{scale}", img_chw, epoch)
                print(
                    f"      Scale {scale}: shape {img_chw.shape}, range [{img_chw.min():.3f}, {img_chw.max():.3f}]"
                )

        self.last_update_time = current_time

        # Flush to ensure data is written
        self.writer.flush()

        print(f"📊 TensorBoard updated: epoch {epoch}, elapsed {elapsed/60:.1f}m")

    def close(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer"):
            self.writer.close()
            print("✅ TensorBoard writer closed")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()
