#ifndef MLX_SCALAR_POOL_H
#define MLX_SCALAR_POOL_H

#include <mlx/c/mlx.h>

/**
 * Cached scalar constant pool.
 *
 * These return lazily-initialized, program-lifetime mlx_array handles
 * for frequently used scalar constants.  The returned arrays must NOT
 * be freed by the caller — they are owned by the pool and live until
 * `mlx_scalar_pool_destroy()` is called (typically at program exit).
 */

mlx_array mlx_scalar_zero(void);      /* 0.0f  */
mlx_array mlx_scalar_one(void);       /* 1.0f  */
mlx_array mlx_scalar_neg_one(void);   /* -1.0f */
mlx_array mlx_scalar_eps(void);       /* 1e-5f (instance-norm epsilon) */
mlx_array mlx_scalar_neg_ten(void);   /* -10.0f (diversity loss coeff) */
mlx_array mlx_scalar_half(void);      /* 0.5f  */

/**
 * Release all cached scalars.  Call once at program shutdown if desired;
 * omitting the call is safe (OS reclaims the handles).
 */
void mlx_scalar_pool_destroy(void);

#endif
