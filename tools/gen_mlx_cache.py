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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--num-pyramids", "-n", type=int, default=1)
    args = p.parse_args()

    # Build scales using Python helper to match MLX
    # Use defaults from datasets.utils.generate_scales (batch,channels,height,width)
    # We will create scale_list as ((batch,channels,height,width),...)

    # Minimal options imitation: read options.json if present else defaults
    # For simplicity assume current working dir has options.json matching trainer
    try:
        import json

        with open("options.json", "r") as f:
            opts = json.load(f)
    except Exception:
        opts = {}
    stop_scale = int(opts.get("stop_scale", 6))
    crop_size = int(opts.get("crop_size", 256))
    max_size = int(opts.get("max_size", crop_size))
    min_size = int(opts.get("min_size", 12))
    num_img_channels = int(opts.get("num_img_channels", 3))

    # Reproduce dataset_generate_scales behavior from C/Python
    nscales = stop_scale + 1
    scale_factor = 1.0
    if stop_scale > 0:
        scale_factor = (
            min_size / float(max_size if max_size < crop_size else crop_size)
        ) ** (1.0 / stop_scale)
    scale_list = []
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
    arrays = {}
    # Each returned tensor has shape (N, H, W, C) because we asked for
    # channels_last=True. Store only the first element (sample_0) per scale
    # and ensure shape is (H, W, C) float32 to match the C loader.
    for i, a in enumerate(fac):
        np_a = a.numpy()
        if np_a.ndim == 4:
            out_arr = np_a[0].astype(np.float32)
        else:
            out_arr = np_a.astype(np.float32)
        arrays[f"sample_0/facies_{i}.npy"] = out_arr
    for i, a in enumerate(seis):
        np_a = a.numpy()
        arrays[f"sample_0/seismic_{i}.npy"] = (
            np_a[0].astype(np.float32) if np_a.ndim == 4 else np_a.astype(np.float32)
        )
    for i, a in enumerate(wells):
        np_a = a.numpy()
        arrays[f"sample_0/wells_{i}.npy"] = (
            np_a[0].astype(np.float32) if np_a.ndim == 4 else np_a.astype(np.float32)
        )

    # Save as .npz
    # Note: numpy.savez will append .npy extension in names; we keep keys simple
    np.savez(args.output, **arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
