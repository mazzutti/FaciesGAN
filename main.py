"""Main entry point for parallel multi-scale FaciesGAN training.

This script provides the command-line interface and initialization for
training a FaciesGAN model with parallel scale processing. Multiple pyramid
scales can be trained simultaneously for faster overall training.
"""

import atexit
import json
import os
import random
import signal
from argparse import ArgumentParser
from datetime import datetime

# Suppress torch.compile symbolic-shape C++ warnings (pow_by_natural etc.)
# Must be set before importing torch.  Apex AMP (O1) is compatible with
# torch.compile; Apex issues its own diagnostics separately.
os.environ.setdefault("TORCH_LOGS", "-dynamo")
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")


# ---------------------------------------------------------------------------
# Silence harmless resource_tracker KeyError tracebacks at exit.
#
# loky (used by joblib.Memory for pyramid caching) creates POSIX shared-
# memory segments and cleans them up on its own.  Python 3.12's
# multiprocessing.resource_tracker daemon still has them registered and
# prints noisy KeyError tracebacks when it later tries to double-clean.
#
# The tracker is a separate process so we cannot monkey-patch it.  Instead,
# we register an atexit handler that terminates the tracker daemon before
# it enters the cleanup-and-warn code path.  Because loky already unlinked
# every segment it created, nothing actually leaks.
# ---------------------------------------------------------------------------
def _silence_resource_tracker() -> None:
    try:
        from multiprocessing.resource_tracker import (
            _resource_tracker,  # type: ignore[attr-defined]cd
        )

        if _resource_tracker._pid is not None:  # type: ignore[union-attr]
            os.kill(_resource_tracker._pid, signal.SIGKILL)  # type: ignore[arg-type]
            os.waitpid(_resource_tracker._pid, 0)  # type: ignore[arg-type]
            _resource_tracker._pid = None  # type: ignore[assignment]
        if _resource_tracker._fd is not None:  # type: ignore[union-attr]
            os.close(_resource_tracker._fd)  # type: ignore[arg-type]
            _resource_tracker._fd = None  # type: ignore[assignment]
    except Exception:
        pass


atexit.register(_silence_resource_tracker)

import torch
import torch.distributed as dist
from dateutil import tz

from config import OPT_FILE
from log import init_output_logging
from options import TrainningOptions
from trainning import TorchTrainer


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
        "--output-path", help="output folder path", default="resuslts/py/"
    )
    parser.add_argument(
        "--output-fullpath",
        help="Set exact output path (overrides automatic timestamp prefix).",
        default=None,
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
        "--lr-g", type=float, default=5e-4, help="learning rate, default=5e-4"
    )
    parser.add_argument(
        "--lr-d", type=float, default=5e-4, help="learning rate, default=5e-4"
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
        "--gp-interval",
        type=int,
        help=(
            "Gradient penalty lazy regularization interval. Compute the "
            "expensive WGAN-GP penalty every N discriminator steps instead "
            "of every step (StyleGAN2-style). Weight is scaled by N to "
            "compensate. Default 16."
        ),
        default=16,
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
        help="Number of workers for data loading (0 = main process only). "
        "Defaults to min(4, cpu_count//2) for parallel data prep.",
        default=min(4, max(1, (os.cpu_count() or 1) // 2)),
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
        "--well-loss-penalty",
        type=float,
        help="weight multiplier for well/mask reconstruction loss",
        default=10.0,
    )

    parser.add_argument(
        "--use-seismic",
        action="store_true",
        help="enable using seismic data during data loading",
    )
    parser.add_argument(
        "--use-mlx",
        action="store_true",
        help="enable MLX backend/model implementations when available",
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

    parser.add_argument(
        "--compile-backend",
        action="store_true",
        help="enable JIT compilation for MLX training (can significantly speed up training)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile on CUDA (enabled by default for Ampere+ GPUs)",
    )
    parser.add_argument(
        "--hand-off-to-c",
        action="store_true",
        help="Hand off orchestration to compiled C library via ctypes (thin wrapper)",
    )
    parser.add_argument(
        "--gradient-checkpoint",
        action="store_true",
        help=(
            "Enable activation checkpointing on generator blocks.  Trades "
            "~30%% extra compute for significantly lower peak GPU memory, "
            "allowing larger batch sizes.  Incompatible with torch.compile "
            "(compile is automatically disabled when this flag is set)."
        ),
    )

    return parser


def _kill_child_processes() -> None:
    """Kill all descendant processes of the current PID.

    Walks ``/proc`` to find every process whose parent (ppid) matches our
    pid, then sends ``SIGKILL``.  This is more reliable than
    ``os.killpg`` when processes are spawned by debugpy or torchrun into
    separate process groups.
    """
    my_pid = os.getpid()
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    parts = f.read().split()
                # Field 3 (0-indexed) is ppid
                ppid = int(parts[3])
                child_pid = int(entry)
                if ppid == my_pid and child_pid != my_pid:
                    os.kill(child_pid, signal.SIGKILL)
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass


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

    # Handle --no-compile flag (torch.compile is on by default for CUDA)
    if hasattr(options, "no_compile") and options.no_compile:
        options.compile_backend = False
    elif not hasattr(options, "compile_backend") or not options.compile_backend:
        # Default to True for CUDA when --no-compile is not set
        if torch.cuda.is_available():
            options.compile_backend = True

    # Handle --gradient-checkpoint (incompatible with torch.compile)
    if hasattr(options, "gradient_checkpoint") and options.gradient_checkpoint:
        options.gradient_checkpointing = True
        # torch.compile reorders saved tensors in a way that breaks
        # checkpoint recomputation metadata — disable it when
        # checkpointing is active.
        options.compile_backend = False
    else:
        if not hasattr(options, "gradient_checkpointing"):
            options.gradient_checkpointing = False

    if options.manual_seed is not None:
        random.seed(options.manual_seed)
        torch.manual_seed(options.manual_seed)  # type: ignore

    # ── Detect distributed (torchrun) ────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0
    rank = 0

    # Ensure SIGTERM (sent by torchrun when a peer dies) actually kills us
    # instead of being swallowed by debugpy or hanging in NCCL collectives.
    signal.signal(signal.SIGTERM, lambda *_: os._exit(1))  # type: ignore

    # Set OMP_NUM_THREADS before torchrun can default it to 1.
    if "OMP_NUM_THREADS" not in os.environ:
        omp_threads = max(1, (os.cpu_count() or 1) // max(1, torch.cuda.device_count()))
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    if distributed:
        # Tell NCCL to surface errors asynchronously instead of blocking
        # forever when a peer dies or a collective is mismatched.  With
        # this flag an unrecoverable NCCL error raises a Python exception
        # rather than hanging the whole job.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        device_id = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        )
        # Use a 5-minute timeout so a DDP desync surfaces as an error
        # instead of hanging silently for the default 30 minutes.
        from datetime import timedelta

        dist.init_process_group(
            backend=backend,
            device_id=device_id,
            timeout=timedelta(minutes=5),
        )
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        # Keep model initialisation deterministic (same weights on every rank);
        # the DistributedSampler gives each rank different data.
        if options.manual_seed is not None:
            torch.manual_seed(options.manual_seed)  # type: ignore
            # Seed the per-device CUDA RNG so noise tensors generated on
            # each rank are reproducible (though intentionally different
            # across ranks for diversity).
            if torch.cuda.is_available():
                torch.cuda.manual_seed(options.manual_seed + rank)
    else:
        device = torch.device(
            f"cuda:{options.gpu_device}"
            if torch.cuda.is_available()
            else (
                f"mps:{options.gpu_device}"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )

    is_main = rank == 0

    timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
    options.output_path = os.path.join(options.output_path, timestamp)
    options.start_scale = 0

    if is_main:
        os.makedirs(options.output_path, exist_ok=True)

        # Save the input parameters options
        with open(os.path.join(options.output_path, OPT_FILE), "w") as file:
            json.dump(vars(options), file, indent=4)  # type: ignore

        init_output_logging(os.path.join(options.output_path, "log.txt"))

    # Synchronise so non-zero ranks wait for rank 0 to create output dir
    if distributed:
        dist.barrier()  # type: ignore[arg-type]
    if not is_main:
        os.makedirs(options.output_path, exist_ok=True)

    if is_main:
        print("\n" + "=" * 60)
        print("PARALLEL LAPGAN TRAINING")
        print("=" * 60)
        print(f"Device: {device}")
        if distributed:
            world_size = dist.get_world_size()
            print(f"DDP training: {world_size} processes")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
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
            # Use TF32 precision globally for matmuls (Ampere+ GPUs)
            try:
                torch.set_float32_matmul_precision("high")  # type: ignore
            except Exception:
                pass

            # Silence harmless torch.compile symbolic-shape warnings.

            # The pow_by_natural warnings come from torch.utils._sympy.interp
            # logger at WARNING level during shape guard compilation.
            import logging
            import warnings

            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._functorch").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", message=".*pow_by_natural.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            # Only call set_logs when TORCH_LOGS env var is not already set,
            # otherwise PyTorch ignores the call and emits a warning.
            if "TORCH_LOGS" not in os.environ:
                try:
                    torch._logging.set_logs(dynamo=logging.ERROR)  # type: ignore[attr-defined]
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

    if options.use_mlx:
        from trainning.mlx.trainer import MLXTrainer

        trainer = MLXTrainer(options)
        if is_main:
            print("Using MLX backend for training.")
    else:
        trainer = TorchTrainer(options, device=device, distributed=distributed)
        if is_main:
            print("Using Torch backend for training.")

    # Register cleanup so child processes are killed on any exit path
    # (unhandled exception, KeyboardInterrupt, normal exit, etc.)
    atexit.register(_kill_child_processes)

    # Optionally run the trainer under the PyTorch profiler and export traces.
    # MPS backend uses a different profiler API (torch.mps.profiler) that
    # generates OS Signpost traces viewable in Xcode Instruments.
    try:
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
                    from torch.profiler import ProfilerActivity  # type: ignore
                    from torch.profiler import profile
                except Exception:
                    print(
                        "Warning: PyTorch profiler not available in this environment."
                    )
                    trainer.train()
                else:
                    trace_file = os.path.join(
                        options.output_path, "profiler_trace.json"
                    )
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
    except Exception as exc:
        import sys
        import traceback

        # Print the traceback on EVERY rank so the actual error from the
        # failing worker is never silently swallowed.  Previously only
        # rank 0 printed, which hid OOM/NCCL errors from other ranks.
        rank_label = f"[rank {rank}] " if distributed else ""
        print(
            f"\n{rank_label}{'=' * 60}\n"
            f"{rank_label}TRAINING FAILED — cleaning up\n"
            f"{rank_label}{'=' * 60}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)

        # Surface CUDA memory stats when the failure looks like OOM.
        if torch.cuda.is_available():
            try:
                dev = torch.device(f"cuda:{local_rank}" if distributed else device)
                alloc = torch.cuda.memory_allocated(dev) / (1024**3)
                reserved = torch.cuda.memory_reserved(dev) / (1024**3)
                peak = torch.cuda.max_memory_allocated(dev) / (1024**3)
                total = torch.cuda.get_device_properties(dev).total_mem / (1024**3)  # type: ignore
                print(
                    f"{rank_label}CUDA memory: "
                    f"alloc={alloc:.2f}G  reserved={reserved:.2f}G  "
                    f"peak={peak:.2f}G  total={total:.2f}G",
                    file=sys.stderr,
                )
            except Exception:
                pass

        sys.stderr.flush()
        sys.stdout.flush()

        # Shut down background workers before hard exit
        try:
            from background_workers import BackgroundWorker

            BackgroundWorker().shutdown(wait=False)
        except Exception:
            pass

        # Under DDP, a crash on one rank leaves the other hanging in NCCL
        # collectives.  dist.destroy_process_group() itself can block when
        # the peer is already dead.  Force-exit so no orphan survives.
        if distributed:
            os._exit(1)

        raise exc
    finally:
        try:
            from background_workers import BackgroundWorker

            BackgroundWorker().shutdown(wait=True)
        except Exception:
            pass
        if distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    if is_main:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import lovely_tensors as lt  # type: ignore

    lt.monkey_patch()

    main()
