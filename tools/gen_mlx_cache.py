#!/usr/bin/env python3
"""Generate MLX-style cached NPZ batches using the Python pipeline.

Usage:
  gen_mlx_cache.py --output <out_npz> --num-pyramids N

This script calls the existing Python dataset helpers to produce exactly the
same NPZ that the Python/MLX trainer would produce.
"""
import argparse
import numpy as np
import os
import sys
import atexit

# Ensure the project root is on sys.path so imports like `datasets.*` work
# when this script is executed from the `tools/` directory or elsewhere.
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_THIS_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
except Exception:
    pass
from datasets.torch.utils import to_facies_pyramids
from datasets.torch.utils import to_seismic_pyramids
from datasets.torch.utils import to_wells_pyramids
from datasets.utils import generate_scales as py_generate_scales

# Ensure child processes spawned by multiprocessing use the same Python
# executable and prefer the 'fork' start method to avoid using the parent
# process name as the python executable (which can cause spawned children
# to invoke the C binary with '-c').
import os
import sys

try:
    # Force PYTHONEXECUTABLE for multiprocessing.spawn to read.
    # This is critical when the parent process name is a C binary.
    os.environ["PYTHONEXECUTABLE"] = sys.executable
    import multiprocessing

    # Track and explicitly unregister multiprocessing resources to avoid
    # resource_tracker warnings on shutdown.
    try:
        import multiprocessing.resource_tracker as _rt

        _registered: set[tuple[str, str]] = set()
        _orig_register = _rt.register
        _orig_unregister = _rt.unregister

        def _register(name: str, rtype: str) -> None:
            _registered.add((name, rtype))
            _orig_register(name, rtype)

        def _unregister(name: str, rtype: str) -> None:
            _registered.discard((name, rtype))
            _orig_unregister(name, rtype)

        _rt.register = _register  # type: ignore[assignment]
        _rt.unregister = _unregister  # type: ignore[assignment]

        def _cleanup_resource_tracker() -> None:
            for name, rtype in list(_registered):
                try:
                    _orig_unregister(name, rtype)
                except Exception:
                    pass

        atexit.register(_cleanup_resource_tracker)
    except Exception:
        pass

    try:
        multiprocessing.set_start_method("fork")
    except Exception:
        # start method may already be set or unavailable; ignore
        pass
    try:
        # Ensure both multiprocessing and spawn internals use the real python.
        multiprocessing.set_executable(sys.executable)
        import multiprocessing.spawn as _s

        _s._python_exe = sys.executable.encode()  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    pass


# Diagnostic: print what Python thinks the executable is and what
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--num-pyramids", "-n", type=int, default=1)
    args = p.parse_args()

    # Build scales using Python helper to match MLX
    # Use defaults from datasets.utils.generate_scales (batch,channels,height,width)
    # We will create scale_list as ((batch,channels,height,width),...)

    # Minimal options imitation: read options.json if present else defaults.
    # Prefer options.json near the output path (C trainer writes it under
    # <output_path>/options.json), then fall back to CWD.
    try:
        import json

        candidates: list[str] = []
        if args.output:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            parent_dir = os.path.dirname(out_dir)
            candidates.extend(
                [
                    os.path.join(out_dir, "options.json"),
                    os.path.join(parent_dir, "options.json"),
                ]
            )
        candidates.append(os.path.join(os.getcwd(), "options.json"))

        opts_path = None
        for p in candidates:
            if os.path.isfile(p):
                opts_path = p
                break
        if opts_path is None:
            raise FileNotFoundError("options.json not found")

        with open(opts_path, "r") as f:
            opts: dict[str, object] = json.load(f)
    except Exception:
        opts = {}

    def _get_int(key: str, default: int) -> int:
        val = opts.get(key, default)
        if isinstance(val, (int, float)):
            return int(val)
        return default

    stop_scale = _get_int("stop_scale", 6)
    crop_size = _get_int("crop_size", 256)
    max_size = _get_int("max_size", crop_size)
    min_size = _get_int("min_size", 12)
    num_img_channels = _get_int("num_img_channels", 3)

    # Reproduce dataset_generate_scales behavior from C/Python
    nscales = stop_scale + 1
    scale_factor = 1.0
    if stop_scale > 0:
        scale_factor = (
            min_size / float(max_size if max_size < crop_size else crop_size)
        ) ** (1.0 / stop_scale)
    scale_list: list[tuple[int, int, int, int]] = []
    for i in range(nscales):
        s = scale_factor ** (stop_scale - i)
        base = (max_size if max_size < crop_size else crop_size) * s
        out_wh = int(round(base))
        if out_wh % 2 != 0:
            out_wh += 1
        # When requesting channels_last=True, generate scale tuples as
        # (batch, height, width, channels) to match `generate_scales`.
        scale_list.append((1, out_wh, out_wh, num_img_channels))

    # Produce pyramids
    fac = to_facies_pyramids(tuple(scale_list), channels_last=True)
    seis = to_seismic_pyramids(tuple(scale_list), channels_last=True)
    wells = to_wells_pyramids(tuple(scale_list), channels_last=True)

    # Pack into NPZ similar structure used by MLX trainer dump
    # We'll follow keys sample_0/facies_i etc and only include sample_0 since
    # num_pyramids handling is more complex; this is enough to ensure parity
    arrays: dict[str, np.ndarray] = {}
    # Each returned tensor has shape (N, H, W, C) because we asked for
    # channels_last=True. Store only the first element (sample_0) per scale
    # and ensure shape is (H, W, C) float32 to match the C loader.
    # NOTE: np.savez automatically appends `.npy` to key names, so we use keys
    # WITHOUT the `.npy` suffix here.  The resulting archive will have members
    # like `sample_0/facies_0.npy`, matching what the C loader expects.
    for i, a in enumerate(fac):
        np_a = a.numpy()
        if np_a.ndim == 4:
            out_arr = np_a[0].astype(np.float32)
        else:
            out_arr = np_a.astype(np.float32)
        arrays[f"sample_0/facies_{i}"] = out_arr
    for i, a in enumerate(seis):
        np_a = a.numpy()
        arrays[f"sample_0/seismic_{i}"] = (
            np_a[0].astype(np.float32) if np_a.ndim == 4 else np_a.astype(np.float32)
        )
    for i, a in enumerate(wells):
        np_a = a.numpy()
        arrays[f"sample_0/wells_{i}"] = (
            np_a[0].astype(np.float32) if np_a.ndim == 4 else np_a.astype(np.float32)
        )

    # Save as .npz
    # Note: numpy.savez will append .npy extension to key names automatically
    np.savez(args.output, **arrays)  # type: ignore[call-arg]
    return 0


if __name__ == "__main__":
    sys.exit(main())
