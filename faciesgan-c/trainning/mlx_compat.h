#ifndef MLX_COMPAT_H
#define MLX_COMPAT_H

#include <mlx/c/mlx.h>

/* Helper: detach and free an mlx_array in one step.
 * CRITICAL: We MUST detach arrays before freeing to break computation graph
 * references. Without detach, freed arrays still have inputs[] pointing to
 * other arrays, keeping those alive and causing memory leaks in backward pass.
 */
static inline void detach_and_free(mlx_array arr) {
    if (arr.ctx) {
        _mlx_array_detach(arr);
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
