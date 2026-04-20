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
from log import format_time
from models.torch.facies_gan import TorchFaciesGAN
from options import TrainningOptions

# Type alias for shared embeddings: method -> (real_reduced, {variant: fake_reduced})
_SharedEmbeddings = dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]


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
    parser.add_argument(
        "--num-iter",
        type=int,
        default=2000,
        help="number of full dataset passes (each pass shuffles independently)",
    )
    parser.add_argument("--num-train-pyramids", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-parallel-scales", type=int, default=7)
    parser.add_argument("--stop-scale", type=int, default=6)
    parser.add_argument("--discriminator-steps", type=int, default=3)
    parser.add_argument(
        "--scale0-disc-steps-multiplier",
        type=int,
        default=1,
        help="Extra D-step multiplier for scale 0 only (default: 1).",
    )
    parser.add_argument(
        "--scale0-loss-multiplier",
        type=float,
        default=1.0,
        help="Extra loss multiplier for rec and impedance at scale 0 (default: 1.0).",
    )
    parser.add_argument("--generator-steps", type=int, default=3)
    parser.add_argument("--reconstruction-loss-penalty", type=float, default=10)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr-g", type=float, default=5e-4)
    parser.add_argument("--lr-d", type=float, default=5e-4)
    parser.add_argument("--lr-decay", type=int, default=999999)
    parser.add_argument(
        "--lr-decay-unit",
        type=str,
        choices=["epoch", "step", "batch"],
        default="epoch",
        help="unit for --lr-decay: 'epoch' resets per batch, 'step' uses global steps, 'batch' decays every N dataset batches (default: epoch)",
    )
    parser.add_argument("--lr-patience", type=int, default=400)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-smoothing-alpha", type=float, default=0.95)
    parser.add_argument("--lr-g-factor", type=float, default=0.8)
    parser.add_argument("--scale0-noise-amp", type=float, default=1.5)
    parser.add_argument("--min-noise-amp", type=float, default=0.3)
    parser.add_argument("--num-diversity-samples", type=int, default=3)
    parser.add_argument("--diversity-loss-penalty", type=float, default=1.0)
    parser.add_argument("--adversarial-loss-penalty", type=float, default=1.0)
    parser.add_argument("--well-loss-penalty", type=float, default=10.0)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="max gradient norm for generator clipping (default: 1.0; set to 0 to disable)",
    )
    parser.add_argument(
        "--gradient-loss-penalty",
        type=float,
        default=0.1,
        help="Gradient penalty weight (default: 0.1).",
    )
    parser.add_argument(
        "--gp-interval",
        type=int,
        default=8,
        help="Compute gradient penalty every N discriminator steps (default: 8).",
    )
    # parser.add_argument(
    #     "--wells-mask-columns",
    #     type=int,
    #     nargs="+",
    #     default=list(range(0, 200, 1)) * 100,
    #     help="Explicit pyramid indices to train on. Default: 50 evenly spaced (0,1,2,...,199).",
    # )
    parser.add_argument("--manual-seed", type=int, default=None)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
        help="Number of GPUs for DDP training (default: 2).",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Resume training from this epoch (default: 0).",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="disable dataset shuffling"
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--no-plot-outputs",
        action="store_true",
        help="Disable PNG sample plots during training.",
    )
    parser.add_argument(
        "--use-impedance",
        action="store_true",
        help="Train with acoustic impedance as an additional output channel.",
    )
    parser.add_argument(
        "--impedance-loss-penalty",
        type=float,
        default=1.0,
        help="Weight for the impedance MSE reconstruction loss (default: 1.0).",
    )

    # ── Embedding selection ────────────────────────────────────────────────
    _ALL_METHODS = ["isomap", "mds", "tsne", "umap"]
    parser.add_argument(
        "--embedding-methods",
        nargs="+",
        choices=_ALL_METHODS,
        default=_ALL_METHODS,
        metavar="METHOD",
        help=(
            "Embedding methods to compute and plot. Choose one or more of: "
            "isomap mds tsne umap (default: all four)."
        ),
    )
    parser.add_argument(
        "--embedding-data",
        nargs="+",
        choices=["facies", "impedance"],
        default=["facies", "impedance"],
        metavar="KIND",
        help=(
            "Data kinds to embed: 'facies', 'impedance', or both "
            "(default: facies impedance)."
        ),
    )
    parser.add_argument(
        "--embedding-per-facies",
        action="store_true",
        help=(
            "Also generate per-crossline embedding plots: for each unique "
            "conditioning crossline index, produce a separate plot showing "
            "the real facies vs the generated samples conditioned on that "
            "crossline (using the shared embedding space)."
        ),
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip all embedding computation and plots.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_last_completed_scale(variant_output: str) -> int:
    """Return the index of the last fully-completed scale, or -1 if none."""
    scale = 0
    while os.path.isfile(os.path.join(variant_output, str(scale), "generator.pth")):
        scale += 1
    return scale - 1


def _read_completed_epochs(variant_output: str, stop_scale: int) -> int:
    """Read the completed epoch count from the first scale's metadata.

    Returns 0 if no metadata file exists.
    """
    from config import COMPLETED_EPOCH_FILE

    meta_path = os.path.join(variant_output, "0", COMPLETED_EPOCH_FILE)
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return int(f.read().strip())
    return 0


def _build_training_args(
    args: object,
    variant: dict[str, bool],
    variant_output: str,
    start_scale: int = 0,
) -> list[str]:
    """Build CLI argument list for ``main.py`` for one experiment variant."""
    cmd_args: list[str] = [
        "--input-path",
        str(getattr(args, "input_path")),
        "--output-fullpath",
        variant_output,
        "--start-scale",
        str(start_scale),
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
        "--scale0-disc-steps-multiplier",
        str(getattr(args, "scale0_disc_steps_multiplier", 1)),
        "--scale0-loss-multiplier",
        str(getattr(args, "scale0_loss_multiplier", 1.0)),
        "--generator-steps",
        str(getattr(args, "generator_steps")),
        "--reconstruction-loss-penalty",
        str(getattr(args, "reconstruction_loss_penalty", 10)),
        "--gamma",
        str(getattr(args, "gamma")),
        "--lr-g",
        str(getattr(args, "lr_g")),
        "--lr-d",
        str(getattr(args, "lr_d")),
        "--lr-decay",
        str(getattr(args, "lr_decay")),
        "--lr-decay-unit",
        str(getattr(args, "lr_decay_unit", "epoch")),
        "--lr-patience",
        str(getattr(args, "lr_patience", 400)),
        "--lr-min",
        str(getattr(args, "lr_min", 1e-4)),
        "--lr-smoothing-alpha",
        str(getattr(args, "lr_smoothing_alpha", 0.95)),
        "--lr-g-factor",
        str(getattr(args, "lr_g_factor", 0.8)),
        "--scale0-noise-amp",
        str(getattr(args, "scale0_noise_amp")),
        "--min-noise-amp",
        str(getattr(args, "min_noise_amp")),
        "--num-diversity-samples",
        str(getattr(args, "num_diversity_samples", 3)),
        "--diversity-loss-penalty",
        str(getattr(args, "diversity_loss_penalty", 1.0)),
        "--adversarial-loss-penalty",
        str(getattr(args, "adversarial_loss_penalty", 1.0)),
        "--well-loss-penalty",
        str(getattr(args, "well_loss_penalty")),
        "--grad-clip-norm",
        str(getattr(args, "grad_clip_norm", 1.0)),
        "--gradient-loss-penalty",
        str(getattr(args, "gradient_loss_penalty", 0.1)),
        "--gp-interval",
        str(getattr(args, "gp_interval", 8)),
        "--compile-backend",
    ]

    wells_mask = getattr(args, "wells_mask_columns", None)
    if wells_mask:
        cmd_args.append("--wells-mask-columns")
        cmd_args.extend(str(i) for i in wells_mask)

    start_epoch = getattr(args, "start_epoch", 0)
    if start_epoch > 0:
        cmd_args.extend(["--start-epoch", str(start_epoch)])

    seed = getattr(args, "manual_seed", None)
    if seed is not None:
        cmd_args.extend(["--manual-seed", str(seed)])

    if variant["use_wells"]:
        cmd_args.append("--use-wells")
    if variant["use_seismic"]:
        cmd_args.append("--use-seismic")
    if getattr(args, "use_impedance", False):
        cmd_args.append("--use-impedance")
        penalty = getattr(args, "impedance_loss_penalty", 1.0)
        cmd_args.extend(["--impedance-loss-penalty", str(penalty)])
    if getattr(args, "no_shuffle", False):
        cmd_args.append("--no-shuffle")
    if getattr(args, "no_tensorboard", False):
        cmd_args.append("--no-tensorboard")
    if getattr(args, "no_plot_outputs", False):
        cmd_args.append("--no-plot-outputs")
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
        # torchrun may return non-zero if a worker crashes during
        # shutdown (e.g. std::bad_alloc in NCCL teardown) even though
        # training completed successfully.  Check whether model
        # artifacts were actually written before aborting.
        output_flag_idx = None
        for i, a in enumerate(variant_args):
            if a == "--output-fullpath" and i + 1 < len(variant_args):
                output_flag_idx = i + 1
                break
        output_dir = variant_args[output_flag_idx] if output_flag_idx else None
        if output_dir and _find_last_completed_scale(output_dir) >= 0:
            print(
                f"  Warning: torchrun exited with code {result.returncode} "
                f"(likely shutdown cleanup error); model artifacts exist, "
                f"continuing.",
            )
        else:
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Generate facies (and impedance when enabled) on a single device.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[int]]
        ``(facies_arrays, impedance_arrays, mask_indexes)``.
        *impedance_arrays* is empty when impedance is not enabled.
    """
    import random as _rng

    import tifffile as tif

    import utils

    model = TorchFaciesGAN(options=opts, device=device, noise_channels=noise_channels)
    model.load(model_path, load_discriminator=False, load_wells=False)

    has_impedance = getattr(opts, "use_impedance", False)
    facies_ch = model.orig_num_img_channels  # 3

    max_scale = len(model.noise_amps) - 1
    all_facies: list[np.ndarray] = []
    all_impedance: list[np.ndarray] = []
    all_mi: list[int] = []

    facies_dir = os.path.join(gen_output, "facies")
    impedance_dir = os.path.join(gen_output, "impedance")
    os.makedirs(facies_dir, exist_ok=True)
    if has_impedance:
        os.makedirs(impedance_dir, exist_ok=True)

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
                idx = start_index + off + j + 1

                if has_impedance:
                    facies_t = g[:facies_ch, ...]
                    imp_t = g[facies_ch : facies_ch + 3, ...]

                    fac_arr = utils.torch2np(facies_t.unsqueeze(0), denormalize=True)
                    imp_arr = utils.torch2np(imp_t.unsqueeze(0), denormalize=True)

                    all_facies.append(fac_arr)
                    all_impedance.append(imp_arr)

                    tif.imwrite(
                        os.path.join(facies_dir, f"generated_facie_{idx}.tif"),
                        fac_arr,
                    )
                    tif.imwrite(
                        os.path.join(impedance_dir, f"generated_impedance_{idx}.tif"),
                        imp_arr,
                    )
                else:
                    arr = utils.torch2np(g.unsqueeze(0), denormalize=True)
                    all_facies.append(arr)
                    tif.imwrite(
                        os.path.join(facies_dir, f"generated_facie_{idx}.tif"),
                        arr,
                    )
        all_mi.extend(mi)
        print(f"    [{device}] {off + chunk}/{how_many}")

    return all_facies, all_impedance, all_mi


def _generate_variant(
    model_path: str,
    gen_output: str,
    how_many: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Load a trained model and generate facies (and impedance) samples.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[int]]
        ``(facies_arrays, impedance_arrays, mask_indexes)``.
        *impedance_arrays* is empty when impedance is not enabled.
    """
    # Load saved options, filtering out keys not accepted by TrainningOptions
    with open(os.path.join(model_path, OPT_FILE), "r") as f:
        json_data = json.load(f)

    import inspect

    _valid_keys = set(inspect.signature(TrainningOptions.__init__).parameters) - {
        "self"
    }
    opts = TrainningOptions(**{k: v for k, v in json_data.items() if k in _valid_keys})
    opts.wells = (
        list(opts.wells_mask_columns) if opts.wells_mask_columns else list(range(200))
    )
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

        # Pre-initialize CUDA contexts on every GPU from the main thread
        # so worker threads don't hit "operation not permitted" errors.
        for gpu_id in range(num_gpus):
            torch.cuda.init()
            torch.zeros(1, device=torch.device(f"cuda:{gpu_id}"))

        chunk_per_gpu = how_many // num_gpus
        remainder = how_many % num_gpus
        futures: list[Future[tuple[list[np.ndarray], list[np.ndarray], list[int]]]] = []
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
        impedance: list[np.ndarray] = []
        mask_indexes: list[int] = []
        for fut in futures:
            f, imp, mi = fut.result()
            facies.extend(f)
            impedance.extend(imp)
            mask_indexes.extend(mi)
    else:
        facies, impedance, mask_indexes = _generate_on_device(
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
    if impedance:
        print(f"  Generated {len(impedance)} impedance -> {gen_output}")

    return facies, impedance, mask_indexes


_EMBEDDINGS_FILE = "shared_embeddings.npz"
_FACIES_FILE = "generated_facies.npz"
_MASK_INDEXES_FILE = "mask_indexes.npz"


def _save_shared_embeddings(
    shared: _SharedEmbeddings,
    all_facies: dict[str, list[np.ndarray]],
    base_output: str,
    num_iter: int,
    all_mask_indexes: dict[str, list[int]] | None = None,
) -> None:
    """Persist shared embeddings, generated facies, and mask indexes to disk."""
    data: dict[str, np.ndarray] = {"_num_iter": np.array(num_iter)}
    for method, (real_r, per_variant) in shared.items():
        data[f"{method}_real"] = real_r
        for vname, fake_r in per_variant.items():
            data[f"{method}_{vname}"] = fake_r
    np.savez(os.path.join(base_output, _EMBEDDINGS_FILE), **data)  # type: ignore

    facies_data: dict[str, np.ndarray] = {}
    for vname, flist in all_facies.items():
        facies_data[vname] = np.stack(flist, 0)
    np.savez(os.path.join(base_output, _FACIES_FILE), **facies_data)  # type: ignore

    if all_mask_indexes:
        mi_data: dict[str, np.ndarray] = {
            vname: np.array(idxs) for vname, idxs in all_mask_indexes.items()
        }
        np.savez(os.path.join(base_output, _MASK_INDEXES_FILE), **mi_data)  # type: ignore
    print(f"  Saved embeddings checkpoint -> {base_output}/{_EMBEDDINGS_FILE}")


def _load_shared_embeddings(
    base_output: str,
    num_iter: int,
) -> tuple[_SharedEmbeddings, dict[str, list[np.ndarray]], dict[str, list[int]]] | None:
    """Load previously saved shared embeddings if they match *num_iter*.

    Returns
    -------
    tuple | None
        ``(shared_embeddings, all_facies, all_mask_indexes)`` or *None* when
        the cache is absent / stale.
    """
    emb_path = os.path.join(base_output, _EMBEDDINGS_FILE)
    fac_path = os.path.join(base_output, _FACIES_FILE)
    if not (os.path.isfile(emb_path) and os.path.isfile(fac_path)):
        return None
    emb_data = np.load(emb_path)
    saved_iter = int(emb_data["_num_iter"])
    if saved_iter != num_iter:
        return None

    methods = {k.split("_")[0] for k in emb_data.files if k != "_num_iter"}
    shared: _SharedEmbeddings = {}
    for method in methods:
        real_key = f"{method}_real"
        if real_key not in emb_data:
            continue
        real_r = emb_data[real_key]
        per_variant: dict[str, np.ndarray] = {}
        for vname in VARIANT_NAMES:
            vkey = f"{method}_{vname}"
            if vkey in emb_data:
                per_variant[vname] = emb_data[vkey]
        shared[method] = (real_r, per_variant)

    fac_data = np.load(fac_path)
    all_facies: dict[str, list[np.ndarray]] = {}
    for vname in VARIANT_NAMES:
        if vname in fac_data:
            all_facies[vname] = list(fac_data[vname])

    # Load mask indexes (optional — absent in older caches)
    all_mask_indexes: dict[str, list[int]] = {}
    mi_path = os.path.join(base_output, _MASK_INDEXES_FILE)
    if os.path.isfile(mi_path):
        mi_data = np.load(mi_path)
        for vname in VARIANT_NAMES:
            if vname in mi_data:
                all_mask_indexes[vname] = mi_data[vname].tolist()

    return shared, all_facies, all_mask_indexes


def _compute_shared_embeddings(
    all_facies: dict[str, list[np.ndarray]],
    dataset: TorchPyramidsDataset,
    impedance_only: bool = False,
    methods: list[str] | None = None,
) -> _SharedEmbeddings:
    """Compute UMAP, Isomap, t-SNE, and MDS in a single shared space.

    Fits each reducer on the concatenation of real data and **all**
    variants' generated data so that the real-data coordinates are
    identical across subplots in both per-variant and combined plots.

    Parameters
    ----------
    all_facies : dict[str, list[np.ndarray]]
        Mapping variant -> list of generated arrays (facies or impedance).
    dataset : TorchPyramidsDataset
        Dataset with real samples.
    impedance_only : bool
        When *True*, use only the impedance channels from the real data
        (channels after ``num_img_channels``).
    methods : list[str] | None
        Subset of ``["isomap", "mds", "tsne", "umap"]`` to compute.
        *None* (default) computes all four.

    Returns
    -------
    _SharedEmbeddings
        Mapping *method* -> ``(real_reduced, {variant: fake_reduced})``.
    """
    import warnings

    import scipy.sparse as sparse  # type: ignore
    from sklearn.manifold import MDS, TSNE, Isomap  # type: ignore
    from sklearn.metrics import euclidean_distances  # type: ignore
    from umap import UMAP  # type: ignore

    import utils

    real_tensor, _, _ = dataset.get_scale_data(-1)
    real_np = utils.torch2np(real_tensor, denormalize=True)
    if impedance_only:
        facies_ch = dataset.options.num_img_channels  # 3
        real_np = real_np[..., facies_ch : facies_ch + 3]
    elif real_np.shape[-1] > dataset.options.num_img_channels:
        # When computing facies embeddings, strip impedance channels
        real_np = real_np[..., : dataset.options.num_img_channels]
    real_flat = np.reshape(real_np, [real_np.shape[0], -1])
    n_real = real_flat.shape[0]

    ordered = [n for n in VARIANT_NAMES if n in all_facies]
    fake_flats: list[np.ndarray] = []
    sizes: dict[str, int] = {}
    for name in ordered:
        arr = np.stack(all_facies[name], 0)
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        flat = np.reshape(arr, [len(all_facies[name]), -1])
        fake_flats.append(flat)
        sizes[name] = flat.shape[0]

    combined = np.concatenate([real_flat] + fake_flats, axis=0).astype(np.float32)

    def _split(emb: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        real_r = emb[:n_real]
        off = n_real
        per_variant: dict[str, np.ndarray] = {}
        for nm in ordered:
            n = sizes[nm]
            per_variant[nm] = emb[off : off + n]
            off += n
        return real_r, per_variant

    results: _SharedEmbeddings = {}

    n_samples = combined.shape[0]
    print(
        f"  Total samples: {n_samples} ({n_real} real + {n_samples - n_real} fake)",
        flush=True,
    )

    # Pre-compute the MDS distance matrix on the main thread only when
    # MDS is actually requested (shared read-only memory avoids a copy).
    _need_mds = methods is None or "mds" in (methods or [])
    if _need_mds:
        distances = euclidean_distances(combined)
        distances = (distances + distances.T) / 2
    else:
        distances = np.empty((0,), dtype=np.float32)  # placeholder, never used

    # ── Helper closures (one per method) ──────────────────────────
    def _fit_umap() -> tuple[str, np.ndarray]:
        print("  Computing shared UMAP embedding ...", flush=True)
        reducer = UMAP(  # type: ignore
            n_components=2,
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
            n_epochs=200,
            init="spectral",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="umap")
            emb: np.ndarray = reducer.fit_transform(combined)  # type: ignore
        print("    UMAP done.", flush=True)
        return "umap", emb  # type: ignore

    def _fit_isomap() -> tuple[str, np.ndarray]:
        print("  Computing shared Isomap embedding ...", flush=True)
        iso = Isomap(n_components=2, n_neighbors=min(10, n_samples - 1))
        emb: np.ndarray = iso.fit_transform(combined)  # type: ignore
        print("    Isomap done.", flush=True)
        return "isomap", emb

    def _fit_tsne() -> tuple[str, np.ndarray]:
        print("  Computing shared t-SNE embedding ...", flush=True)
        tsne = TSNE(
            n_components=2,
            perplexity=min(30.0, (n_samples - 1) / 3.0),
            random_state=42,
        )
        emb: np.ndarray = tsne.fit_transform(combined)  # type: ignore
        print("    t-SNE done.", flush=True)
        return "tsne", emb

    def _fit_mds() -> tuple[str, np.ndarray]:
        print("  Computing shared MDS embedding ...", flush=True)
        # Use PCA on the distance matrix for a good initialisation so
        # that a single SMACOF run converges fast.
        from sklearn.decomposition import PCA  # type: ignore

        pca_init = PCA(n_components=2, random_state=3).fit_transform(distances)
        mds = MDS(
            n_components=2,
            max_iter=300,
            eps=1e-6,
            random_state=np.random.RandomState(seed=3),
            metric="precomputed",  # type: ignore
            n_init=1,
            init="random",  # type: ignore  # overridden by fit_transform(init=)
            normalized_stress="auto",
        )
        emb: np.ndarray = mds.fit_transform(distances, init=pca_init)  # type: ignore
        print("    MDS done.", flush=True)
        return "mds", emb

    # ── Run all four in parallel ──────────────────────────────────
    # The underlying C/Cython/numba code releases the GIL, so threads
    # provide true parallelism without serialization overhead.
    from concurrent.futures import Future, ThreadPoolExecutor
    from typing import Callable

    _active_methods = (
        set(methods) if methods is not None else {"umap", "isomap", "tsne", "mds"}
    )
    _method_fns: dict[str, Callable[[], tuple[str, np.ndarray]]] = {
        "umap": _fit_umap,
        "isomap": _fit_isomap,
        "tsne": _fit_tsne,
        "mds": _fit_mds,
    }
    _selected = [(m, fn) for m, fn in _method_fns.items() if m in _active_methods]

    label_str = ", ".join(m.upper() for m, _ in _selected)
    print(f"  Running {label_str} in parallel ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, len(_selected))) as pool:
        futures: list[Future[tuple[str, np.ndarray]]] = [
            pool.submit(fn) for _, fn in _selected  # type: ignore[arg-type]
        ]
        for fut in futures:
            method_name, emb = fut.result()  # type: ignore[misc]
            results[method_name] = _split(emb)  # type: ignore[arg-type]

    return results


def _plot_sample_grid(
    all_generated: dict[str, list[np.ndarray]],
    real_samples: np.ndarray | None,
    base_output: str,
    data_kind: str,
    num_samples: int = 5,
) -> None:
    """Create a comparison grid of real vs generated samples per variant.

    Layout: one row per variant, first column shows a real sample,
    remaining columns show generated samples.

    Parameters
    ----------
    all_generated : dict[str, list[np.ndarray]]
        Mapping from variant name to generated sample arrays (H, W, C) or
        (1, H, W, C).
    real_samples : np.ndarray | None
        Real data tensor converted to numpy (B, H, W, C).  When *None*
        the "Real" column is left blank.
    base_output : str
        Root output directory; the combined PNG is written here.
    data_kind : str
        ``"facies"`` or ``"impedance"`` — used in titles and filenames.
    num_samples : int
        Number of generated samples to show per variant (columns 1+).
    """
    from matplotlib import pyplot as plt

    variant_labels = {
        "wells_seismic": "Wells + Seismic",
        "wells_only": "Wells Only",
        "seismic_only": "Seismic Only",
        "unconditional": "Unconditional",
    }

    # Build per-variant rows: (label, real_img_or_None, [generated_imgs])
    rows: list[tuple[str, np.ndarray | None, list[np.ndarray]]] = []
    for i, name in enumerate(VARIANT_NAMES):
        if name not in all_generated or not all_generated[name]:
            continue
        samples = all_generated[name][:num_samples]
        imgs = [s.squeeze(0) if s.ndim == 4 else s for s in samples]
        real_img: np.ndarray | None = None
        if real_samples is not None and i < len(real_samples):
            real_img = real_samples[i]
        rows.append((variant_labels.get(name, name), real_img, imgs))

    if not rows:
        print(f"  No {data_kind} data available; skipping comparison grid.")
        return

    # Columns: 1 (Real) + num generated
    max_gen = max(len(imgs) for _, _, imgs in rows)
    n_cols = 1 + max_gen
    n_rows = len(rows)
    fig, axes = plt.subplots(  # type: ignore
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    # Column headers
    axes[0][0].set_title("Real", fontsize=11, fontweight="bold")
    for c in range(1, n_cols):
        axes[0][c].set_title(f"Generated {c}", fontsize=11)

    for r, (label, real_img, gen_imgs) in enumerate(rows):
        # Column 0: real sample
        ax = axes[r][0]
        if real_img is not None:
            ax.imshow(real_img)
        else:
            ax.set_facecolor("#111")
        ax.axis("off")
        ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=10)

        # Columns 1+: generated samples
        for c in range(1, n_cols):
            ax = axes[r][c]
            gi = c - 1
            if gi < len(gen_imgs):
                ax.imshow(gen_imgs[gi])
            else:
                ax.set_facecolor("#111")
            ax.axis("off")

    kind_title = data_kind.capitalize()
    fig.suptitle(  # type: ignore
        f"Real vs Generated {kind_title} — All Variants", fontsize=14
    )
    fig.tight_layout()
    out_path = os.path.join(base_output, f"{data_kind}_comparison_all_variants.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    print(f"  {kind_title} comparison grid -> {out_path}")


def _plot_per_variant_embedding(
    method: str,
    real_reduced: np.ndarray,
    fake_reduced: np.ndarray,
    save_path: str,
) -> None:
    """Save a single per-variant embedding plot using pre-computed coordinates."""
    from matplotlib import pyplot as plt

    label = "t-SNE" if method == "tsne" else method.upper()
    plt.figure()  # type: ignore
    plt.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.6)  # type: ignore
    plt.scatter(fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.6)  # type: ignore
    plt.title(f"{label} Visualization of FaciesGAN generated facies")  # type: ignore
    plt.xlabel(f"{label} Dimension 1")  # type: ignore
    plt.ylabel(f"{label} Dimension 2")  # type: ignore
    plt.legend(("Real Facies", "Fake Facies"), loc="upper right")  # type: ignore
    plt.savefig(save_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close()  # type: ignore


def _plot_combined_embeddings(
    method: str,
    shared_embedding: tuple[np.ndarray, dict[str, np.ndarray]],
    base_output: str,
    num_iter: int = 0,
    data_kind: str = "facies",
) -> None:
    """Create a 2x2 combined embedding plot.

    Uses a single real-data projection paired with per-variant fake
    projections, so the real dots are identical across all four subplots.

    Parameters
    ----------
    method : str
        Embedding method name (``"mds"``, ``"umap"``, ``"isomap"``, or
        ``"tsne"``).
    shared_embedding : tuple[np.ndarray, dict[str, np.ndarray]]
        ``(real_reduced, {variant: fake_reduced})`` computed in a single
        shared embedding space.
    base_output : str
        Directory where the combined PNG will be saved.
    num_iter : int, optional
        Epoch number appended to the filename when > 0.
    data_kind : str
        ``"facies"`` or ``"impedance"`` — used in titles and filenames.
    """
    from matplotlib import pyplot as plt

    label = "t-SNE" if method == "tsne" else method.upper()

    variant_labels = {
        "wells_seismic": "Wells + Seismic",
        "wells_only": "Wells Only",
        "seismic_only": "Seismic Only",
        "unconditional": "Unconditional",
    }

    kind_title = data_kind.capitalize()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # type: ignore
    fig.suptitle(f"{label} Comparison: Real vs Generated {kind_title}", fontsize=14)  # type: ignore

    for ax, name in zip(axes.flat, VARIANT_NAMES):
        if name not in shared_embedding[1]:
            ax.set_title(f"{variant_labels.get(name, name)} (no data)")
            ax.axis("off")
            continue
        real_reduced = shared_embedding[0]
        fake_reduced = shared_embedding[1][name]

        ax.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.6, label="Real")
        ax.scatter(fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.6, label="Generated")
        ax.set_title(variant_labels.get(name, name))
        ax.set_xlabel(f"{label} Dimension 1")
        ax.set_ylabel(f"{label} Dimension 2")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    epoch_tag = f"_epoch{num_iter}" if num_iter > 0 else ""
    combined_path = os.path.join(
        base_output, f"{method}_{data_kind}_comparison_all_variants{epoch_tag}.png"
    )
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    print(f"\nCombined {label} {kind_title} plot -> {combined_path}")


def _plot_per_facies_embedding(
    method: str,
    real_reduced: np.ndarray,
    fake_reduced_for_facies: np.ndarray,
    facies_idx: int,
    save_path: str,
    data_kind: str = "facies",
) -> None:
    """Save an embedding plot for a single conditioning crossline index.

    Highlights the generated samples produced from crossline *facies_idx*
    against the full real-data embedding cloud.

    Parameters
    ----------
    method : str
        Embedding method name.
    real_reduced : np.ndarray
        Pre-computed 2-D coordinates for all real samples.
    fake_reduced_for_facies : np.ndarray
        Pre-computed 2-D coordinates for generated samples conditioned on
        this specific crossline.
    facies_idx : int
        Crossline index used for conditioning (for plot title / filename).
    save_path : str
        Destination file path.
    data_kind : str
        ``"facies"`` or ``"impedance"`` — used in the title.
    """
    from matplotlib import pyplot as plt

    label = "t-SNE" if method == "tsne" else method.upper()
    kind_title = data_kind.capitalize()
    plt.figure()  # type: ignore
    plt.scatter(  # type: ignore
        real_reduced[:, 0],
        real_reduced[:, 1],
        alpha=0.4,
        s=10,
        label="Real (all)",
        c="steelblue",
    )
    plt.scatter(  # type: ignore
        fake_reduced_for_facies[:, 0],
        fake_reduced_for_facies[:, 1],
        alpha=0.8,
        s=20,
        label=f"Generated (crossline {facies_idx})",
        c="tomato",
    )
    plt.title(f"{label} — {kind_title} conditioned on crossline {facies_idx}")  # type: ignore
    plt.xlabel(f"{label} Dimension 1")  # type: ignore
    plt.ylabel(f"{label} Dimension 2")  # type: ignore
    plt.legend(loc="upper right", fontsize=8)  # type: ignore
    plt.savefig(save_path, dpi=120, bbox_inches="tight")  # type: ignore
    plt.close()  # type: ignore


def _save_per_facies_embeddings(
    shared: _SharedEmbeddings,
    all_mask_indexes: dict[str, list[int]],
    base_output: str,
    methods: list[str],
    data_kind: str = "facies",
) -> None:
    """Generate per-crossline embedding plots for every variant and method.

    For each variant and each unique conditioning crossline index found in
    *all_mask_indexes*, this writes one PNG per method to::

        {base_output}/{variant}/generated/per_{data_kind}_embeddings/
            {method}_{data_kind}_crossline_{idx}.png

    Parameters
    ----------
    shared : _SharedEmbeddings
        Shared embedding coordinates (method -> (real, {variant: fake})).
    all_mask_indexes : dict[str, list[int]]
        Mapping variant -> list of crossline indices (one per generated sample).
    base_output : str
        Experiment root directory.
    methods : list[str]
        Methods to plot (subset of those in *shared*).
    data_kind : str
        ``"facies"`` or ``"impedance"`` — drives subdirectory naming and titles.
    """
    for name in VARIANT_NAMES:
        # Skip variants with no mask indexes or no computed embedding
        if name not in all_mask_indexes:
            continue
        if not methods or not any(
            name in shared.get(m, (None, {}))[1] for m in methods  # type: ignore[index]
        ):
            continue
        mi_list = np.array(all_mask_indexes[name])
        unique_idxs = sorted(set(mi_list.tolist()))

        per_emb_dir = os.path.join(
            base_output, name, "generated", f"per_{data_kind}_embeddings"
        )
        os.makedirs(per_emb_dir, exist_ok=True)

        for method in methods:
            if method not in shared:
                continue
            real_reduced, per_variant_fakes = shared[method]
            if name not in per_variant_fakes:
                continue
            fake_all = per_variant_fakes[name]  # (N_generated, 2)

            for idx in unique_idxs:
                mask = mi_list == idx
                fake_for_idx = fake_all[mask]
                if fake_for_idx.shape[0] == 0:
                    continue
                save_path = os.path.join(
                    per_emb_dir,
                    f"{method}_{data_kind}_crossline_{idx:04d}.png",
                )
                _plot_per_facies_embedding(
                    method,
                    real_reduced,
                    fake_for_idx,
                    idx,
                    save_path,
                    data_kind=data_kind,
                )
        n_plots = len(unique_idxs) * len(methods)
        print(
            f"  Per-{data_kind} embeddings ({name}): {n_plots} plots -> {per_emb_dir}"
        )


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
    any_retrained = False

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

            # Check for existing checkpoint
            last_done = _find_last_completed_scale(variant_output)
            start_epoch = getattr(args, "start_epoch", 0)

            # Auto-detect start_epoch from completed_epoch.txt when
            # resuming (--start-epoch > 0) or when all scales exist.
            completed_epochs = _read_completed_epochs(variant_output, args.stop_scale)

            # If the user specified --start-epoch, use the max of that
            # and what was actually completed.
            if start_epoch > 0:
                effective_start = max(start_epoch, completed_epochs)
            else:
                effective_start = completed_epochs

            # Skip if all scales are trained AND the model has already
            # reached (or exceeded) the requested num_iter.
            if last_done >= args.stop_scale and effective_start >= args.num_iter:
                model_paths[name] = variant_output
                print(f"\n{'─' * 60}")
                print(
                    f"Skipping variant: {name} "
                    f"(fully trained — {completed_epochs}/{args.num_iter} epochs)"
                )
                print(f"  {variant_output}")
                print(f"{'─' * 60}")
                continue

            # When resuming from a specific epoch, re-enter all scale
            # groups so the trainer can load existing checkpoints and
            # continue from that epoch.
            if effective_start > 0:
                resume_scale = 0
            else:
                resume_scale = last_done + 1 if last_done >= 0 else 0

            print(f"\n{'─' * 60}")
            print(f"Training variant: {name}")
            print(f"  wells={wells_flag}  seismic={seismic_flag}")
            if effective_start > 0:
                print(
                    f"  Resuming from epoch {effective_start} (all scales will be re-entered)"
                )
            elif resume_scale > 0:
                print(
                    f"  Resuming from scale {resume_scale} (scales 0-{last_done} already done)"
                )
            print(f"  DDP: {nproc} GPUs  compile_backend: ON")
            print(f"{'─' * 60}")

            variant_args = _build_training_args(
                args, variant, variant_output, start_scale=resume_scale
            )
            # Override --start-epoch with effective_start so the
            # subprocess always receives the actual resume point.
            # Remove any existing --start-epoch first.
            try:
                idx = variant_args.index("--start-epoch")
                variant_args[idx + 1] = str(effective_start)
            except ValueError:
                if effective_start > 0:
                    variant_args.extend(["--start-epoch", str(effective_start)])
            variant_start = time.time()
            _train_variant(variant_args, nproc)
            any_retrained = True
            elapsed = format_time(int(time.time() - variant_start))

            # --output-fullpath places artifacts directly in variant_output
            model_paths[name] = variant_output
            print(f"  Training complete ({elapsed}) -> {variant_output}")

    # ── Load base options & dataset (needed for plots and embeddings) ──
    import inspect

    first_model = model_paths[VARIANT_NAMES[0]]
    with open(os.path.join(first_model, OPT_FILE), "r") as f:
        _json = json.load(f)
    _valid = set(inspect.signature(TrainningOptions.__init__).parameters) - {"self"}
    _base_opts = TrainningOptions(**{k: v for k, v in _json.items() if k in _valid})
    # _base_opts.wells = (
    #     list(_base_opts.wells_mask_columns)
    #     if _base_opts.wells_mask_columns
    #     else list(range(1, 200, 4))
    # )
    _base_opts.rec = False
    _base_opts.compile_backend = True
    _dataset = TorchPyramidsDataset(_base_opts)

    # ── Try to load cached embeddings ──
    # Reuse only when no variant was retrained (all skipped); if any model
    # was retrained the old embeddings are stale and must be recomputed.
    cached = (
        _load_shared_embeddings(base_output, args.num_iter)
        if not any_retrained
        else None
    )
    if cached is not None:
        shared, all_facies, _cached_mi = cached
        all_impedance: dict[str, list[np.ndarray]] = {}
        all_mask_indexes: dict[str, list[int]] = _cached_mi
        print(f"Loaded cached embeddings for epoch {args.num_iter}")
    else:
        # ── Generate facies (and impedance) from all trained models ──
        print(f"\n{'=' * 70}")
        print("GENERATING FACIES FROM TRAINED MODELS")
        print(f"{'=' * 70}\n")

        all_facies: dict[str, list[np.ndarray]] = {}
        all_impedance: dict[str, list[np.ndarray]] = {}
        all_mask_indexes: dict[str, list[int]] = {}

        for name in VARIANT_NAMES:
            model_path = model_paths[name]
            gen_output = os.path.join(base_output, name, "generated")

            print(f"Generating from variant: {name}")
            print(f"  model: {model_path}")

            variant_facies, variant_imp, variant_mi = _generate_variant(
                model_path=model_path,
                gen_output=gen_output,
                how_many=args.how_many,
                device=device,
            )
            all_facies[name] = variant_facies
            if variant_imp:
                all_impedance[name] = variant_imp
            all_mask_indexes[name] = variant_mi

        if not args.no_embeddings:
            # ── Shared embeddings for all plots ──
            # Fit selected methods once on real + ALL variants' fakes so
            # the real-data coordinates are identical across all plots.
            emb_methods: list[str] = args.embedding_methods
            print(
                f"\nComputing shared embeddings "
                f"({', '.join(m.upper() for m in emb_methods)}) ...",
                flush=True,
            )
            shared = _compute_shared_embeddings(
                all_facies, _dataset, methods=emb_methods
            )

            # Persist for future resume (includes mask indexes)
            _save_shared_embeddings(
                shared, all_facies, base_output, args.num_iter, all_mask_indexes
            )
        else:
            shared = {}

    # ── Resolve effective embedding method list ────────────────────────────
    emb_methods = list(
        getattr(args, "embedding_methods", ["isomap", "mds", "tsne", "umap"])
    )
    emb_data_kinds: list[str] = list(
        getattr(args, "embedding_data", ["facies", "impedance"])
    )

    # ── Facies comparison grid (real vs generated per variant) ──
    print(f"\n{'-' * 70}")
    print("Generating facies comparison grid...")
    import utils as _utils

    real_tensor, _, _ = _dataset.get_scale_data(-1)
    real_facies_np = _utils.torch2np(real_tensor, denormalize=True)
    # If impedance channels are present, keep only facies channels
    facies_ch = _base_opts.num_img_channels  # 3
    if real_facies_np.shape[-1] > facies_ch:
        real_facies_np = real_facies_np[..., :facies_ch]
    _plot_sample_grid(all_facies, real_facies_np, base_output, "facies")

    # ── Impedance comparison grid (only when impedance data exists) ──
    if all_impedance:
        print(f"\n{'-' * 70}")
        print("Generating impedance comparison grid...")
        real_imp_np: np.ndarray | None = None
        real_full = _utils.torch2np(real_tensor, denormalize=True)
        if real_full.shape[-1] > facies_ch:
            real_imp_np = real_full[..., facies_ch : facies_ch + 3]
        _plot_sample_grid(all_impedance, real_imp_np, base_output, "impedance")

    if args.no_embeddings or not shared:
        total_elapsed = format_time(int(time.time() - total_start))
        print(f"\n{'=' * 70}")
        print(f"ALL EXPERIMENTS COMPLETE  ({total_elapsed})")
        print(f"{'=' * 70}")
        print(f"\nResults in: {base_output}")
        for name in VARIANT_NAMES:
            print(f"  {name}: {model_paths[name]}")
        return

    # ── Facies embedding plots (per-variant + combined) ────────────────────
    if "facies" in emb_data_kinds:
        print(f"\n{'-' * 70}")
        print("Generating facies embedding plots...")
        for method in emb_methods:
            if method not in shared:
                continue
            real_reduced, per_variant_fakes = shared[method]
            # Per-variant individual plots
            for name in VARIANT_NAMES:
                if name not in per_variant_fakes:
                    continue
                gen_output = os.path.join(base_output, name, "generated")
                plot_path = os.path.join(gen_output, f"{method}_comparison.png")
                _plot_per_variant_embedding(
                    method, real_reduced, per_variant_fakes[name], plot_path
                )
                print(f"  {method.upper()} plot -> {plot_path}")
            # Combined 2×2 grid
            _plot_combined_embeddings(
                method, shared[method], base_output, args.num_iter, data_kind="facies"
            )

        # Per-crossline facies embeddings
        if args.embedding_per_facies and all_mask_indexes:
            print(f"\n{'-' * 70}")
            print("Generating per-crossline facies embedding plots...")
            _save_per_facies_embeddings(
                shared, all_mask_indexes, base_output, emb_methods, data_kind="facies"
            )

    # ── Impedance embedding plots (only when impedance data exists) ────────
    if "impedance" in emb_data_kinds and all_impedance:
        imp_emb_methods: list[str] = [m for m in emb_methods]
        print(f"\n{'-' * 70}")
        print("Computing shared impedance embeddings for plots...", flush=True)
        shared_imp = _compute_shared_embeddings(
            all_impedance, _dataset, impedance_only=True, methods=imp_emb_methods
        )
        for method in imp_emb_methods:
            if method not in shared_imp:
                continue
            real_reduced, per_variant_fakes = shared_imp[method]
            # Per-variant individual plots
            for name in VARIANT_NAMES:
                if name not in per_variant_fakes:
                    continue
                gen_output = os.path.join(base_output, name, "generated")
                plot_path = os.path.join(
                    gen_output, f"{method}_impedance_comparison.png"
                )
                _plot_per_variant_embedding(
                    method, real_reduced, per_variant_fakes[name], plot_path
                )
                print(f"  {method.upper()} impedance plot -> {plot_path}")
            # Combined 2×2 grid
            _plot_combined_embeddings(
                method,
                shared_imp[method],
                base_output,
                args.num_iter,
                data_kind="impedance",
            )

        # Per-crossline impedance embeddings
        if args.embedding_per_facies and all_mask_indexes:
            print(f"\n{'-' * 70}")
            print("Generating per-crossline impedance embedding plots...")
            _save_per_facies_embeddings(
                shared_imp,
                all_mask_indexes,
                base_output,
                imp_emb_methods,
                data_kind="impedance",
            )

    total_elapsed = format_time(int(time.time() - total_start))
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE  ({total_elapsed})")
    print(f"{'=' * 70}")
    print(f"\nResults in: {base_output}")
    for name in VARIANT_NAMES:
        print(f"  {name}: {model_paths[name]}")


if __name__ == "__main__":
    main()
