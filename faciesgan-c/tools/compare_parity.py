import numpy as np
import sys
import os


def load_npy_from_file(path):
    return np.load(path)


def load_npy_from_npz(npz_path, member):
    import zipfile, io

    with zipfile.ZipFile(npz_path, "r") as zf:
        data = zf.read(member)
    return np.load(io.BytesIO(data))


def stack_from_npz(npz_path, n_samples, scale):
    fac_list = []
    wells_present = True
    wells_list = []
    for i in range(n_samples):
        fac_name = f"sample_{i}/facies_{scale}.npy"
        fac = load_npy_from_npz(npz_path, fac_name)
        fac_list.append(fac)
        well_name = f"sample_{i}/wells_{scale}.npy"
        try:
            w = load_npy_from_npz(npz_path, well_name)
            wells_list.append(w)
        except KeyError:
            wells_present = False
            break

    fac_stack = np.stack(fac_list, axis=0)
    if not wells_present:
        wells_stack = np.empty((0,) + fac_stack.shape[1:], dtype=fac_stack.dtype)
    else:
        wells_stack = np.stack(wells_list, axis=0)

    # masks: try explicit
    masks_list = []
    masks_present = True
    for i in range(n_samples):
        try:
            m = load_npy_from_npz(npz_path, f"sample_{i}/masks_{scale}.npy")
            masks_list.append(m)
        except KeyError:
            masks_present = False
            break
    if masks_present:
        masks_stack = np.stack(masks_list, axis=0)
    else:
        if wells_stack.shape[0] == 0:
            masks_stack = np.empty((0,) + fac_stack.shape[1:], dtype=fac_stack.dtype)
        else:
            masks_stack = np.greater(
                np.sum(np.abs(wells_stack), axis=3, keepdims=True), 0
            )

    return fac_stack, wells_stack, masks_stack


def main():
    if len(sys.argv) < 5:
        print("usage: compare_parity.py <npz_path> <n_samples> <scale> <out_prefix>")
        return 2
    npz = sys.argv[1]
    n = int(sys.argv[2])
    scale = int(sys.argv[3])
    outp = sys.argv[4]

    py_fac, py_wells, py_masks = stack_from_npz(npz, n, scale)

    c_fac = load_npy_from_file(f"{outp}_facies.npy")
    equal_fac = np.allclose(py_fac, c_fac)
    print("facies equal:", equal_fac)

    if os.path.exists(f"{outp}_wells.npy"):
        c_wells = load_npy_from_file(f"{outp}_wells.npy")
        print("wells equal:", np.allclose(py_wells, c_wells))
    else:
        print("c wells missing; py_wells shape:", py_wells.shape)

    if os.path.exists(f"{outp}_masks.npy"):
        c_masks = load_npy_from_file(f"{outp}_masks.npy")
        print("masks equal:", np.allclose(py_masks, c_masks))
    else:
        print("c masks missing; py_masks shape:", py_masks.shape)


if __name__ == "__main__":
    main()
