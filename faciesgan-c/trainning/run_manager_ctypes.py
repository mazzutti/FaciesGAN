#!/usr/bin/env python3
import ctypes
from pathlib import Path

LIB = Path("build_dbg/libcustom_layer.dylib")
if not LIB.exists():
    raise SystemExit(f"C library not found: {LIB}")

lib = ctypes.CDLL(str(LIB), mode=ctypes.RTLD_GLOBAL)

# int mlx_run_manager_from_python(const char *output_path, int num_parallel_scales, int num_img_channels,
# int discriminator_steps, int generator_steps, int num_feature, int min_num_feature, int num_layer,
# int kernel_size, int padding_size, int num_diversity_samples, int epochs, int steps_per_epoch,
# const char *checkpoint_path, int checkpoint_every, int use_create_graph_gp)

func = lib.mlx_run_manager_from_python
func.restype = ctypes.c_int
func.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
]


def run():
    out = Path("results/replay_c_200")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(out / "checkpoint.bin")

    # parameters
    num_parallel_scales = 1
    num_img_channels = 3
    discriminator_steps = 1
    generator_steps = 1
    num_feature = 16
    min_num_feature = 8
    num_layer = 1
    kernel_size = 3
    padding_size = 1
    num_diversity_samples = 1
    epochs = 1
    steps_per_epoch = 200
    checkpoint_every = steps_per_epoch  # save at end
    use_create_graph_gp = 0

    args: list[object] = [
        str(out).encode("utf-8"),
        num_parallel_scales,
        num_img_channels,
        discriminator_steps,
        generator_steps,
        num_feature,
        min_num_feature,
        num_layer,
        kernel_size,
        padding_size,
        num_diversity_samples,
        epochs,
        steps_per_epoch,
        checkpoint_path.encode("utf-8"),
        checkpoint_every,
        use_create_graph_gp,
    ]

    print("Invoking C manager via ctypes...")
    rc = func(*args)
    print("C manager returned", rc)


if __name__ == "__main__":
    run()
