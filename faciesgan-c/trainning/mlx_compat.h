#ifndef MLX_COMPAT_H
#define MLX_COMPAT_H

#include <mlx/c/mlx.h>
#include "mem_debug.h"

/* Helper: detach from computation graph and free an mlx_array.
 * Uses mlx_stop_gradient to break gradient references before freeing,
 * which helps release memory from accumulated computation graphs.
static inline void detach_and_free(mlx_array arr) {
    if (arr.ctx) {
        /* Create a stopped version to break graph references */
        mlx_array stopped = mlx_array_new();
        mlx_stream s = mlx_default_gpu_stream_new();
        if (mlx_stop_gradient(&stopped, arr, s) == 0) {
            /* Evaluate to materialize and release graph */
            mlx_array_eval(stopped);
            mlx_array_free(stopped);
        }
        mlx_stream_free(s);
    }
    mlx_array_free(arr);
}

/* Prototypes for compatibility helpers around MLX conv variants.
 * Implementations are in mlx_compat.c to avoid duplicate definitions
 * when this header is included from multiple translation units. */
int safe_mlx_conv2d(mlx_array *res, const mlx_array input,
                    const mlx_array weight, int stride0, int stride1, int pad0,
                    int pad1, int dil0, int dil1, int groups,
                    const mlx_stream s);

int safe_mlx_conv_transpose2d(mlx_array *res, const mlx_array input,
                              const mlx_array weight, int stride0, int stride1,
                              int pad0, int pad1, int dil0, int dil1,
                              int groups, const mlx_stream s);

#endif
