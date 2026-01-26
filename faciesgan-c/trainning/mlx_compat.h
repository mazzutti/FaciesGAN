#ifndef MLX_COMPAT_H
#define MLX_COMPAT_H

#include <mlx/c/mlx.h>

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
