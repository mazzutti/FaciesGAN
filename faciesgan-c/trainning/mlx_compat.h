#ifndef MLX_COMPAT_H
#define MLX_COMPAT_H

#include <mlx/c/mlx.h>
#include "mem_debug.h"
#include "array_helpers.h"

/* Helper: detach from computation graph and free an mlx_array.
 * Perf: Previously used mlx_stop_gradient + mlx_array_eval which added an
 * unnecessary sync barrier. MLX's reference counting correctly frees memory
 * when handles are released; the extra eval was not needed.
 */
static inline void detach_and_free(mlx_array arr) {
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

/* Layout-state values for cached_mlx_conv2d / cached_mlx_conv_transpose2d:
 *   0 = not yet detected (will inspect shapes and update *layout_state)
 *   1 = direct (no transpose needed)
 *   2 = needs transpose (axes 0↔3)
 */
#define CONV_LAYOUT_UNKNOWN  0
#define CONV_LAYOUT_DIRECT   1
#define CONV_LAYOUT_TRANSPOSE 2

/* Cached variant: detects layout once and stores the result in *layout_state.
 * Subsequent calls skip the shape inspection entirely. */
int cached_mlx_conv2d(mlx_array *res, const mlx_array input,
                      const mlx_array weight, int stride0, int stride1,
                      int pad0, int pad1, int dil0, int dil1, int groups,
                      int *layout_state, const mlx_stream s);

int cached_mlx_conv_transpose2d(mlx_array *res, const mlx_array input,
                                const mlx_array weight, int stride0,
                                int stride1, int pad0, int pad1, int dil0,
                                int dil1, int groups, int *layout_state,
                                const mlx_stream s);

#endif
