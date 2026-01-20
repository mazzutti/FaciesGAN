import numpy as np
import zipfile
import io
import sys


def write_npy_to_zip(zf, name, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(name, buf.getvalue())


def main(out_path="/tmp/test_cache.npz", n_samples=2, h=1, w=1, c=1):
    # create simple test arrays
    with zipfile.ZipFile(out_path, "w") as zf:
        for i in range(n_samples):
            fac = np.array([[[float(i + 1.0)]]], dtype=np.float32).reshape(h, w, c)
            write_npy_to_zip(zf, f"sample_{i}/facies_0.npy", fac)
            # wells present with two channels for sample 0, single channel for others
            wells = (
                np.zeros((h, w, 2), dtype=np.float32)
                if c == 1
                else np.zeros((h, w, c), dtype=np.float32)
            )
            wells[:] = float(i)
            write_npy_to_zip(zf, f"sample_{i}/wells_0.npy", wells)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_cache.npz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    main(out, n)
