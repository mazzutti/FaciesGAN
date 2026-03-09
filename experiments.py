"""Run conditioning-ablation experiments: train & generate for 4 variants.

Each variant trains a FaciesGAN model with a different combination of
well and seismic conditioning, then generates facies samples and
comparison plots from the trained checkpoints.

Training is launched via ``torchrun --nproc_per_node=2`` to use both
available GPUs with DDP.

Variants
--------
1. wells + seismic  (``--use-wells --use-seismic``)
2. wells only       (``--use-wells``)
3. seismic only     (``--use-seismic``)
4. unconditional    (neither flag)

Usage
-----
    python experiments.py --input-path data --output-path results/experiments \\
        --num-iter 1000 --num-train-pyramids 10 --how-many 20

All training hyperparameters not listed below are forwarded verbatim to
each training run.  Add any ``main.py`` argument and it will be passed
through (e.g. ``--batch-size 40 --num-parallel-scales 7``).
"""

import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

from config import OPT_FILE
from datasets.torch.dataset import TorchPyramidsDataset
from gen_facies import plot_mds
from log import format_time
from models.torch.facies_gan import TorchFaciesGAN
from options import TrainningOptions

# ---------------------------------------------------------------------------
# Experiment variant descriptors
# ---------------------------------------------------------------------------

VARIANTS: list[dict[str, bool]] = [
    {"use_wells": True, "use_seismic": True},
    {"use_wells": True, "use_seismic": False},
    {"use_wells": False, "use_seismic": True},
    {"use_wells": False, "use_seismic": False},
]

VARIANT_NAMES: list[str] = [
    "wells_seismic",
    "wells_only",
    "seismic_only",
    "unconditional",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_arguments() -> ArgumentParser:
    """Build argument parser for the experiments runner.

    All unrecognised arguments are forwarded to each training run.
    """
    parser = ArgumentParser(
        description="Run conditioning-ablation experiments for FaciesGAN.",
    )
    # Required paths
    parser.add_argument(
        "--input-path", required=True, help="Path to the dataset root directory."
    )
    parser.add_argument(
        "--output-path",
        default="results/experiments",
        help="Base output directory for all experiment runs.",
    )

    # Generation options (applied after training)
    parser.add_argument(
        "--how-many",
        type=int,
        default=2000,
        help="Number of facies to generate per variant (default: 2000).",
    )

    # Convenience: allow skipping training if models already exist
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run generation using existing model paths.",
    )
    parser.add_argument(
        "--model-paths",
        nargs=4,
        metavar=("WELLS_SEISMIC", "WELLS_ONLY", "SEISMIC_ONLY", "UNCONDITIONAL"),
        help=(
            "Explicit model paths for generation-only mode (requires "
            "--skip-training). Provide 4 paths in order."
        ),
    )

    # Forwarded training hyper-parameters with sensible defaults
    parser.add_argument("--num-iter", type=int, default=2000)
    parser.add_argument("--num-train-pyramids", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-parallel-scales", type=int, default=7)
    parser.add_argument("--stop-scale", type=int, default=6)
    parser.add_argument("--discriminator-steps", type=int, default=3)
    parser.add_argument("--generator-steps", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--scale0-noise-amp", type=float, default=1.5)
    parser.add_argument("--min-noise-amp", type=float, default=0.3)
    parser.add_argument("--lambda-diversity", type=float, default=1.0)
    parser.add_argument("--manual-seed", type=int, default=None)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
        help="Number of GPUs for DDP training (default: 2).",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--no-plot-facies",
        action="store_true",
        help="Disable PNG sample plots during training.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_args(
    args: object,
    variant: dict[str, bool],
    variant_output: str,
) -> list[str]:
    """Build CLI argument list for ``main.py`` for one experiment variant."""
    cmd_args: list[str] = [
        "--input-path",
        str(getattr(args, "input_path")),
        "--output-fullpath",
        variant_output,
        "--num-iter",
        str(getattr(args, "num_iter")),
        "--num-train-pyramids",
        str(getattr(args, "num_train_pyramids")),
        "--batch-size",
        str(getattr(args, "batch_size")),
        "--num-workers",
        str(getattr(args, "num_workers")),
        "--num-parallel-scales",
        str(getattr(args, "num_parallel_scales")),
        "--stop-scale",
        str(getattr(args, "stop_scale")),
        "--discriminator-steps",
        str(getattr(args, "discriminator_steps")),
        "--generator-steps",
        str(getattr(args, "generator_steps")),
        "--alpha",
        str(getattr(args, "alpha")),
        "--scale0-noise-amp",
        str(getattr(args, "scale0_noise_amp")),
        "--min-noise-amp",
        str(getattr(args, "min_noise_amp")),
        "--lambda-diversity",
        str(getattr(args, "lambda_diversity")),
        "--compile-backend",
    ]

    seed = getattr(args, "manual_seed", None)
    if seed is not None:
        cmd_args.extend(["--manual-seed", str(seed)])

    if variant["use_wells"]:
        cmd_args.append("--use-wells")
    if variant["use_seismic"]:
        cmd_args.append("--use-seismic")
    if getattr(args, "no_tensorboard", False):
        cmd_args.append("--no-tensorboard")
    if getattr(args, "no_plot_facies", False):
        cmd_args.append("--no-plot-facies")
    else:
        # Plot 1 image per scale at the last epoch only: 5 real × 5 generated.
        # Setting save-interval to num_iter ensures only the final epoch
        # triggers (the condition also fires on epoch == num_iter - 1).
        num_iter = getattr(args, "num_iter")
        cmd_args.extend(
            [
                "--save-interval",
                str(max(num_iter - 1, 1)),
                "--num-real-facies",
                "5",
                "--num-generated-per-real",
                "5",
            ]
        )

    return cmd_args


def _resolve_device(gpu_device: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_device}")
    elif torch.backends.mps.is_available():
        return torch.device(f"mps:{gpu_device}")
    return torch.device("cpu")


def _train_variant(
    variant_args: list[str],
    nproc_per_node: int,
) -> None:
    """Train a single variant via ``torchrun`` with DDP on all GPUs."""
    main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    python = sys.executable

    cmd = [
        python,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        main_py,
        *variant_args,
    ]

    print(f"  Command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


def _generate_on_device(
    device: torch.device,
    model_path: str,
    opts: TrainningOptions,
    how_many: int,
    wells_pyramid: dict[int, torch.Tensor],
    seismic_pyramid: dict[int, torch.Tensor],
    noise_channels: int,
    gen_output: str,
    start_index: int,
    batch_size: int = 100,
) -> tuple[list[np.ndarray], list[int]]:
    """Generate facies on a single device in batches, writing .tif files."""
    import random as _rng

    import tifffile as tif

    import utils

    model = TorchFaciesGAN(options=opts, device=device, noise_channels=noise_channels)
    model.load(model_path, load_discriminator=False, load_wells=False)

    max_scale = len(model.noise_amps) - 1
    all_facies: list[np.ndarray] = []
    all_mi: list[int] = []

    for off in range(0, how_many, batch_size):
        chunk = min(batch_size, how_many - off)
        mi = [_rng.choice(opts.wells) for _ in range(chunk)]
        noises = model.get_pyramid_noise(
            max_scale, mi, wells_pyramid, seismic_pyramid, rec=opts.rec
        )
        with torch.no_grad():
            for j, g in enumerate(
                model.generator(noises, model.get_noise_aplitude(max_scale))
            ):
                arr = utils.torch2np(g.unsqueeze(0), denormalize=True)
                all_facies.append(arr)
                tif.imwrite(
                    os.path.join(
                        gen_output,
                        f"generated_facie_{start_index + off + j + 1}.tif",
                    ),
                    arr,
                )
        all_mi.extend(mi)
        print(f"    [{device}] {off + chunk}/{how_many}")

    return all_facies, all_mi


def _generate_variant(
    model_path: str,
    gen_output: str,
    how_many: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[int]]:
    """Load a trained model and generate facies samples using all GPUs.

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        Generated facies arrays and mask indexes.
    """
    # Load saved options, filtering out keys not accepted by TrainningOptions
    with open(os.path.join(model_path, OPT_FILE), "r") as f:
        json_data = json.load(f)

    import inspect

    _valid_keys = set(inspect.signature(TrainningOptions.__init__).parameters) - {
        "self"
    }
    opts = TrainningOptions(**{k: v for k, v in json_data.items() if k in _valid_keys})
    opts.wells = list(range(200))
    opts.rec = False
    opts.compile_backend = False  # no need to compile for one-shot generation

    os.makedirs(gen_output, exist_ok=True)

    # Build conditioning pyramids directly — avoids the expensive
    # NeuralSmoother interpolation of all 200 facies that
    # TorchPyramidsDataset would trigger (facies pyramids are unused
    # during generation).
    import datasets.torch.utils as torch_utils
    import datasets.utils as data_utils

    scales = data_utils.generate_scales(opts)
    wells_pyramid: dict[int, torch.Tensor] = {}
    seismic_pyramid: dict[int, torch.Tensor] = {}
    if opts.use_wells:
        wp = torch_utils.to_wells_pyramids(scales)
        for s, w in enumerate(wp):
            if w.numel() > 0:
                wells_pyramid[s] = w
    if opts.use_seismic:
        sp = torch_utils.to_seismic_pyramids(scales)
        for s, se in enumerate(sp):
            if se.numel() > 0:
                seismic_pyramid[s] = se

    noise_channels = (
        opts.noise_channels
        + (opts.num_img_channels if opts.use_wells else 0)
        + (opts.num_img_channels if opts.use_seismic else 0)
    )

    # Use all available CUDA GPUs for parallel generation
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        from concurrent.futures import Future, ThreadPoolExecutor

        chunk_per_gpu = how_many // num_gpus
        remainder = how_many % num_gpus
        futures: list[Future[tuple[list[np.ndarray], list[int]]]] = []
        with ThreadPoolExecutor(max_workers=num_gpus) as pool:
            start = 0
            for gpu_id in range(num_gpus):
                count = chunk_per_gpu + (1 if gpu_id < remainder else 0)
                dev = torch.device(f"cuda:{gpu_id}")
                futures.append(
                    pool.submit(
                        _generate_on_device,
                        dev,
                        model_path,
                        opts,
                        count,
                        wells_pyramid,
                        seismic_pyramid,
                        noise_channels,
                        gen_output,
                        start,
                    )
                )
                start += count
        facies: list[np.ndarray] = []
        mask_indexes: list[int] = []
        for fut in futures:
            f, mi = fut.result()
            facies.extend(f)
            mask_indexes.extend(mi)
    else:
        facies, mask_indexes = _generate_on_device(
            device,
            model_path,
            opts,
            how_many,
            wells_pyramid,
            seismic_pyramid,
            noise_channels,
            gen_output,
            0,
        )

    print(f"  Generated {len(facies)} facies -> {gen_output}")

    # MDS comparison plot (requires full dataset for real facies)
    dataset = TorchPyramidsDataset(opts)
    mds_path = os.path.join(gen_output, "mds_comparison.png")
    plot_mds(facies, mask_indexes, opts, dataset, save_path=mds_path)
    print(f"  MDS plot -> {mds_path}")

    return facies, mask_indexes


def _plot_combined_mds(
    all_facies: dict[str, list[np.ndarray]],
    all_mask_indexes: dict[str, list[int]],
    model_paths: dict[str, str],
    base_output: str,
) -> None:
    """Create a 2×2 combined MDS plot comparing all variants against real facies."""
    from matplotlib import pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics import euclidean_distances

    import utils
    from datasets.torch.dataset import TorchPyramidsDataset

    # Load real facies once (use the first available variant's options)
    first_name = next(iter(model_paths))
    first_path = model_paths[first_name]

    import inspect

    with open(os.path.join(first_path, OPT_FILE), "r") as f:
        json_data = json.load(f)
    _valid_keys = set(inspect.signature(TrainningOptions.__init__).parameters) - {
        "self"
    }
    ref_opts = TrainningOptions(
        **{k: v for k, v in json_data.items() if k in _valid_keys}
    )
    ref_opts.compile_backend = False

    dataset = TorchPyramidsDataset(ref_opts)
    real_facies_tensor, _, _ = dataset.get_scale_data(-1)
    real_facies_flat = np.reshape(
        utils.torch2np(real_facies_tensor, denormalize=True),
        [real_facies_tensor.shape[0], -1],
    )
    real_sim = euclidean_distances(real_facies_flat)

    mds = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=np.random.RandomState(seed=3),
        metric="precomputed",
        n_init=4,
        init="random",  # type: ignore
        n_jobs=1,
        normalized_stress="auto",
    )
    real_reduced = mds.fit((real_sim + real_sim.T) / 2).embedding_

    variant_labels = {
        "wells_seismic": "Wells + Seismic",
        "wells_only": "Wells Only",
        "seismic_only": "Seismic Only",
        "unconditional": "Unconditional",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # type: ignore
    fig.suptitle("MDS Comparison: Real vs Generated Facies", fontsize=14)  # type: ignore

    for ax, name in zip(axes.flat, VARIANT_NAMES):
        if name not in all_facies:
            ax.set_title(f"{variant_labels.get(name, name)} (no data)")
            ax.axis("off")
            continue

        fake_arr = np.stack(all_facies[name], 0)
        if fake_arr.shape[-1] == 1:
            fake_arr = fake_arr.squeeze(-1)
        fake_flat = np.reshape(fake_arr, [fake_arr.shape[0], -1])
        fake_sim = euclidean_distances(fake_flat)
        fake_reduced = mds.fit((fake_sim + fake_sim.T) / 2).embedding_

        mi = all_mask_indexes[name]
        real_subset = (
            real_reduced[mi] if len(mi) <= real_reduced.shape[0] else real_reduced
        )

        ax.scatter(real_subset[:, 0], real_subset[:, 1], alpha=0.6, label="Real")
        ax.scatter(fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.6, label="Generated")
        ax.set_title(variant_labels.get(name, name))
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    combined_path = os.path.join(base_output, "mds_comparison_all_variants.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    print(f"\nCombined MDS plot -> {combined_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = get_arguments()
    args = parser.parse_args()

    device = _resolve_device(args.gpu_device)
    base_output = args.output_path
    nproc = args.nproc_per_node

    # Map variant name -> model path (either from training or --model-paths)
    model_paths: dict[str, str] = {}

    total_start = time.time()

    if args.skip_training:
        if not args.model_paths:
            parser.error(
                "--skip-training requires --model-paths with 4 paths "
                "(wells_seismic, wells_only, seismic_only, unconditional)."
            )
        for name, path in zip(VARIANT_NAMES, args.model_paths):
            if not os.path.isdir(path):
                parser.error(f"Model path does not exist: {path}")
            model_paths[name] = path
        print("Skipping training, using provided model paths.")
    else:
        # ── Train all 4 variants ──
        print("=" * 70)
        print("FACIESGAN CONDITIONING-ABLATION EXPERIMENTS")
        print("=" * 70)
        print(f"Device: {device} (DDP with {nproc} GPUs)")
        print(f"compile_backend: ON")
        print(f"Variants: {', '.join(VARIANT_NAMES)}")
        print(f"Output:   {base_output}")
        print("=" * 70 + "\n")

        for name, variant in zip(VARIANT_NAMES, VARIANTS):
            variant_output = os.path.join(base_output, name)
            os.makedirs(variant_output, exist_ok=True)
            wells_flag = "ON" if variant["use_wells"] else "OFF"
            seismic_flag = "ON" if variant["use_seismic"] else "OFF"

            print(f"\n{'─' * 60}")
            print(f"Training variant: {name}")
            print(f"  wells={wells_flag}  seismic={seismic_flag}")
            print(f"  DDP: {nproc} GPUs  compile_backend: ON")
            print(f"{'─' * 60}")

            variant_args = _build_training_args(args, variant, variant_output)
            variant_start = time.time()
            _train_variant(variant_args, nproc)
            elapsed = format_time(int(time.time() - variant_start))

            # --output-fullpath places artifacts directly in variant_output
            model_paths[name] = variant_output
            print(f"  Training complete ({elapsed}) -> {variant_output}")

    # ── Generate facies from all trained models ──
    print(f"\n{'=' * 70}")
    print("GENERATING FACIES FROM TRAINED MODELS")
    print(f"{'=' * 70}\n")

    all_facies: dict[str, list[np.ndarray]] = {}
    all_mask_indexes: dict[str, list[int]] = {}

    for name in VARIANT_NAMES:
        model_path = model_paths[name]
        gen_output = os.path.join(base_output, name, "generated")

        print(f"Generating from variant: {name}")
        print(f"  model: {model_path}")

        variant_facies, variant_mi = _generate_variant(
            model_path=model_path,
            gen_output=gen_output,
            how_many=args.how_many,
            device=device,
        )
        all_facies[name] = variant_facies
        all_mask_indexes[name] = variant_mi

    # ── Combined MDS comparison plot across all variants ──
    _plot_combined_mds(all_facies, all_mask_indexes, model_paths, base_output)

    total_elapsed = format_time(int(time.time() - total_start))
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE  ({total_elapsed})")
    print(f"{'=' * 70}")
    print(f"\nResults in: {base_output}")
    for name in VARIANT_NAMES:
        print(f"  {name}: {model_paths[name]}")


if __name__ == "__main__":
    main()
