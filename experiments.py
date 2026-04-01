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
    parser.add_argument("--num-iter", type=int, default=2000)
    parser.add_argument("--num-train-pyramids", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-parallel-scales", type=int, default=7)
    parser.add_argument("--stop-scale", type=int, default=6)
    parser.add_argument("--discriminator-steps", type=int, default=3)
    parser.add_argument("--generator-steps", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--lr-g", type=float, default=5e-4)
    parser.add_argument("--lr-d", type=float, default=5e-4)
    parser.add_argument("--lr-decay", type=int, default=1000)
    parser.add_argument("--scale0-noise-amp", type=float, default=1.5)
    parser.add_argument("--min-noise-amp", type=float, default=0.3)
    parser.add_argument("--lambda-diversity", type=float, default=1.0)
    parser.add_argument("--well-loss-penalty", type=float, default=10.0)
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
        "--generator-steps",
        str(getattr(args, "generator_steps")),
        "--alpha",
        str(getattr(args, "alpha")),
        "--lr-g",
        str(getattr(args, "lr_g")),
        "--lr-d",
        str(getattr(args, "lr_d")),
        "--lr-decay",
        str(getattr(args, "lr_decay")),
        "--scale0-noise-amp",
        str(getattr(args, "scale0_noise_amp")),
        "--min-noise-amp",
        str(getattr(args, "min_noise_amp")),
        "--lambda-diversity",
        str(getattr(args, "lambda_diversity")),
        "--well-loss-penalty",
        str(getattr(args, "well_loss_penalty")),
        "--compile-backend",
    ]

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
    opts.wells = list(range(0, 200, 4))
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

    return facies, mask_indexes


_EMBEDDINGS_FILE = "shared_embeddings.npz"
_FACIES_FILE = "generated_facies.npz"


def _save_shared_embeddings(
    shared: _SharedEmbeddings,
    all_facies: dict[str, list[np.ndarray]],
    base_output: str,
    num_iter: int,
) -> None:
    """Persist shared embeddings and generated facies to disk."""
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
    print(f"  Saved embeddings checkpoint -> {base_output}/{_EMBEDDINGS_FILE}")


def _load_shared_embeddings(
    base_output: str,
    num_iter: int,
) -> tuple[_SharedEmbeddings, dict[str, list[np.ndarray]]] | None:
    """Load previously saved shared embeddings if they match *num_iter*."""
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
    return shared, all_facies


def _compute_shared_embeddings(
    all_facies: dict[str, list[np.ndarray]],
    dataset: TorchPyramidsDataset,
) -> _SharedEmbeddings:
    """Compute UMAP, Isomap, t-SNE, and MDS in a single shared space.

    Fits each reducer on the concatenation of real data and **all**
    variants' generated data so that the real-data coordinates are
    identical across subplots in both per-variant and combined plots.

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
    real_flat = np.reshape(
        utils.torch2np(real_tensor, denormalize=True),
        [real_tensor.shape[0], -1],
    )
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

    # Pre-compute the MDS distance matrix on the main thread (shared
    # read-only memory avoids a copy in the thread pool).
    distances = euclidean_distances(combined)
    distances = (distances + distances.T) / 2

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
        return "umap", emb

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
    from concurrent.futures import ThreadPoolExecutor

    print("  Running UMAP, Isomap, t-SNE, MDS in parallel ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_fit_umap),
            pool.submit(_fit_isomap),
            pool.submit(_fit_tsne),
            pool.submit(_fit_mds),
        ]
        for fut in futures:
            method_name, emb = fut.result()
            results[method_name] = _split(emb)

    return results


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
    """
    from matplotlib import pyplot as plt

    label = "t-SNE" if method == "tsne" else method.upper()

    variant_labels = {
        "wells_seismic": "Wells + Seismic",
        "wells_only": "Wells Only",
        "seismic_only": "Seismic Only",
        "unconditional": "Unconditional",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # type: ignore
    fig.suptitle(f"{label} Comparison: Real vs Generated Facies", fontsize=14)  # type: ignore

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
        base_output, f"{method}_comparison_all_variants{epoch_tag}.png"
    )
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close(fig)
    print(f"\nCombined {label} plot -> {combined_path}")


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

    # ── Try to load cached embeddings ──
    # Reuse only when no variant was retrained (all skipped); if any model
    # was retrained the old embeddings are stale and must be recomputed.
    cached = (
        _load_shared_embeddings(base_output, args.num_iter)
        if not any_retrained
        else None
    )
    if cached is not None:
        shared, all_facies = cached
        print(f"Loaded cached embeddings for epoch {args.num_iter}")
    else:
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

        # ── Shared embeddings for all plots ──
        # Fit UMAP, Isomap, t-SNE, and MDS once on real + ALL variants'
        # fakes so the real dots are identical across all plots.
        import inspect

        first_model = model_paths[VARIANT_NAMES[0]]
        with open(os.path.join(first_model, OPT_FILE), "r") as f:
            _json = json.load(f)
        _valid = set(inspect.signature(TrainningOptions.__init__).parameters) - {"self"}
        _base_opts = TrainningOptions(**{k: v for k, v in _json.items() if k in _valid})
        _base_opts.wells = list(range(0, 200, 4))
        _base_opts.rec = False
        _base_opts.compile_backend = False
        _dataset = TorchPyramidsDataset(_base_opts)

        print("\nComputing shared embeddings for all plots...", flush=True)
        shared = _compute_shared_embeddings(all_facies, _dataset)

        # Persist for future resume
        _save_shared_embeddings(shared, all_facies, base_output, args.num_iter)

    # ── Per-variant plots (consistent real coordinates) ──
    for method in ("mds", "umap", "isomap", "tsne"):
        real_reduced, per_variant_fakes = shared[method]
        for name in VARIANT_NAMES:
            if name not in per_variant_fakes:
                continue
            gen_output = os.path.join(base_output, name, "generated")
            plot_path = os.path.join(gen_output, f"{method}_comparison.png")
            _plot_per_variant_embedding(
                method, real_reduced, per_variant_fakes[name], plot_path
            )
            print(f"  {method.upper()} plot -> {plot_path}")

    # ── Combined comparison plots (2x2 grid) ──
    for method in ("mds", "umap", "isomap", "tsne"):
        _plot_combined_embeddings(
            method,
            shared[method],
            base_output,
            args.num_iter,
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
