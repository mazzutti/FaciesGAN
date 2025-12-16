"""Main entry point for parallel multi-scale FaciesGAN training.

This script provides the command-line interface and initialization for
training a FaciesGAN model with parallel scale processing. Multiple pyramid
scales can be trained simultaneously for faster overall training.
"""

import json
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import torch
from dateutil import tz

from config import CHECKPOINT_PATH, OPT_FILE
from log import init_output_logging
from options import TrainningOptions
from train import Trainer


def get_arguments() -> ArgumentParser:
    """Parse command-line arguments for parallel FaciesGAN training.

    Returns
    -------
    ArgumentParser
        Configured argument parser with all training options.
    """
    parser = ArgumentParser()

    # workspace:
    parser.add_argument("--use-cpu", action="store_true", help="use cpu")
    parser.add_argument("--gpu-device", type=int, help="which GPU to use", default=0)
    parser.add_argument("--input-path", help="input facie path", required=True)

    # load, input, save configurations:
    parser.add_argument("--manual-seed", type=int, help="manual seed")
    parser.add_argument(
        "--output-path", help="output folder path", default="facies_gan_parallel"
    )
    parser.add_argument("--stop-scale", type=int, help="stop scale", default=6)
    parser.add_argument(
        "--num-img-channels", type=int, help="facie number of channels", default=3
    )
    parser.add_argument(
        "--noise-channels",
        type=int,
        help="number of noise channels to generate per scale",
        default=3,
    )
    parser.add_argument(
        "--img-color-range",
        type=int,
        nargs=2,
        help="range of values in the input facie",
        default=[0, 255],
    )
    parser.add_argument(
        "--crop-size", type=int, help="crop size to train the facie", default=256
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Total batch size - e.g: num_gpus = 2, batch_size = 128 then, effectively, 64",
    )

    # networks hyperparameters:
    parser.add_argument(
        "--num-features",
        type=int,
        help="initial number of features in each layer",
        default=32,
    )
    parser.add_argument(
        "--min-num-features",
        type=int,
        help="minimal number of features in each layer",
        default=32,
    )
    parser.add_argument("--kernel-size", type=int, help="kernel size", default=3)
    parser.add_argument(
        "--num-layers", type=int, help="number of layers in each scale", default=5
    )
    parser.add_argument("--stride", help="stride", default=1)
    parser.add_argument("--padding-size", type=int, help="net pad size", default=0)

    # pyramid parameters:
    parser.add_argument(
        "--noise-amp", type=float, help="adaptive noise cont weight", default=0.1
    )
    parser.add_argument(
        "--min-noise-amp",
        type=float,
        help="minimum noise amplitude floor for diversity",
        default=0.5,
    )
    parser.add_argument(
        "--scale0-noise-amp",
        type=float,
        help="noise amplitude at scale 0 (controls structural diversity)",
        default=1.0,
    )

    # Parallel training specific parameters:
    parser.add_argument(
        "--num-parallel-scales",
        type=int,
        help="Number of scales to train in parallel (default: 2)",
        default=2,
    )

    # profiling
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        help="Enable PyTorch profiler and export a chrome trace to the output path",
    )

    # optimization hyperparameters:
    parser.add_argument(
        "--num-iter", type=int, default=2000, help="number of epochs to train per scale"
    )
    parser.add_argument("--gamma", type=float, help="scheduler gamma", default=0.9)
    parser.add_argument(
        "--lr-g", type=float, default=5e-5, help="learning rate, default=5e-5"
    )
    parser.add_argument(
        "--lr-d", type=float, default=5e-5, help="learning rate, default=5e-5"
    )
    parser.add_argument(
        "--lr-decay", type=int, default=1000, help="number of epochs before lr decay"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--generator-steps", type=int, help="Generator inner steps", default=3
    )
    parser.add_argument(
        "--discriminator-steps", type=int, help="Discriminator inner steps", default=3
    )
    parser.add_argument(
        "--lambda-grad", type=float, help="gradient penalty weight", default=0.1
    )
    parser.add_argument(
        "--alpha", type=float, help="reconstruction loss weight", default=10
    )
    parser.add_argument(
        "--lambda-diversity",
        type=float,
        help="diversity loss weight (encourages different outputs for different noise)",
        default=1.0,
    )
    parser.add_argument(
        "--save-interval", type=int, help="save log interval", default=100
    )
    parser.add_argument(
        "--num-real-facies",
        type=int,
        help="Number of real facies to use in the grid plot",
        default=5,
    )
    parser.add_argument(
        "--num-generated-per-real",
        type=int,
        help="Number of generated facies per real facies to use in the grid plot",
        default=5,
    )
    parser.add_argument(
        "--num-train-pyramids",
        type=int,
        help="Number of train pyramids to use in the FaciesGAN training",
        default=200,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading (0 = main process only)",
        default=0,
    )

    parser.add_argument(
        "--use-wells",
        action="store_true",
        help="enable using wells during data loading (filter by --wells-mask-columns if set)",
    )

    parser.add_argument(
        "--wells-mask-columns",
        type=int,
        help="list of well indices to train the model from",
        nargs="+",
        default=tuple(),
    )

    parser.add_argument(
        "--use-seismic",
        action="store_true",
        help="enable using seismic data during data loading",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="disable TensorBoard logging during training",
    )
    parser.add_argument(
        "--no-plot-facies",
        action="store_true",
        help="disable plot_generated_facies visualizations during training",
    )

    return parser


def main() -> None:
    """Run parallel FaciesGAN training.

    Handles argument parsing, environment setup, logger initialization, and
    constructs the :class:`Trainer` instance that performs the training
    loop. This entry point is intended for CLI usage and is also used in
    the ``if __name__ == '__main__'`` guard.
    """
    argument_parser = get_arguments()
    options = argument_parser.parse_args(namespace=TrainningOptions())

    # Handle --no-tensorboard flag
    if hasattr(options, "no_tensorboard") and options.no_tensorboard:
        options.enable_tensorboard = False

    # Handle --no-plot-facies flag
    if hasattr(options, "no_plot_facies") and options.no_plot_facies:
        options.enable_plot_facies = False

    if options.manual_seed is not None:
        random.seed(options.manual_seed)
        torch.manual_seed(options.manual_seed)  # type: ignore

    timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
    options.output_path = os.path.join(
        CHECKPOINT_PATH, f"{timestamp}_{options.output_path}"
    )
    options.start_scale = 0

    os.makedirs(options.output_path, exist_ok=True)

    # Save the input parameters options
    with open(os.path.join(options.output_path, OPT_FILE), "w") as file:
        json.dump(vars(options), file, indent=4)  # type: ignore

    init_output_logging(os.path.join(options.output_path, "log.txt"))

    device = torch.device(
        f"cuda:{options.gpu_device}"
        if torch.cuda.is_available()
        else f"mps:{options.gpu_device}" if torch.backends.mps.is_available() else "cpu"
    )

    print("\n" + "=" * 60)
    print("PARALLEL LAPGAN TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Training scales: {options.start_scale} to {options.stop_scale}")
    print(f"Parallel scales: {options.num_parallel_scales}")
    print(f"Iterations per scale: {options.num_iter}")
    print(f"Output path: {options.output_path}")
    print("=" * 60 + "\n")

    # Performance tuning: enable cuDNN autotuner and TF32 where available
    try:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # Allow TF32 for faster matmuls on compatible NVIDIA GPUs
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True  # type: ignore
            except Exception:
                pass
    except Exception:
        # If backend tuning isn't supported on this build, continue without failing
        pass

    # Reasonable default for intra-op threads to avoid oversubscription
    try:
        cpu_threads = min(4, max(1, (os.cpu_count() or 1) // 2))
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass

    trainer = Trainer(device, options)

    # Optionally run the trainer under the PyTorch profiler and export traces.
    # MPS backend uses a different profiler API (torch.mps.profiler) that
    # generates OS Signpost traces viewable in Xcode Instruments.
    if getattr(options, "use_profiler", False):
        if device.type == "mps":
            # Use MPS-specific profiler for Apple Silicon GPUs
            print("Starting MPS profiler (use Xcode Instruments to view traces)...")
            print(
                "Note: Launch Xcode Instruments with the Logging tool before starting training"
            )
            try:
                torch.mps.profiler.start(mode="interval", wait_until_completed=False)  # type: ignore
                trainer.train()
                torch.mps.profiler.stop()  # type: ignore
                print("MPS profiler stopped. Opening Xcode Instruments...")
                # Attempt to open Xcode Instruments
                import subprocess

                try:
                    subprocess.run(["open", "-a", "Instruments"], check=False)
                except Exception as open_err:
                    print(f"Could not automatically open Instruments: {open_err}")
                    print(
                        "Please open Xcode Instruments > Logging tool manually to view OS Signpost traces."
                    )
            except Exception as e:
                print(f"Warning: MPS profiler error: {e}")
                trainer.train()
        else:
            # Use standard torch.profiler for CPU/CUDA
            try:
                from torch.profiler import ProfilerActivity, profile  # type: ignore
            except Exception:
                print("Warning: PyTorch profiler not available in this environment.")
                trainer.train()
            else:
                trace_file = os.path.join(options.output_path, "profiler_trace.json")
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)

                with profile(
                    activities=activities, record_shapes=True, profile_memory=True
                ) as prof:
                    trainer.train()

                try:
                    prof.export_chrome_trace(trace_file)
                    print(f"Profiler trace saved to: {trace_file}")
                    print("View in Chrome at: chrome://tracing")
                except Exception as e:
                    print(f"Could not export profiler trace: {e}")
    else:
        trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import lovely_tensors as lt  # type: ignore

    lt.monkey_patch()
    main()
