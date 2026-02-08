#ifndef MLX_C_TRAIN_STEP_H
#define MLX_C_TRAIN_STEP_H

#include "facies_gan.h"
#include "base_manager.h"
#include "optimizer.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Apply SGD to a generator using provided gradients (array length == n).
 * Returns 0 on success, -1 on error.
 */
int mlx_faciesgan_apply_sgd_to_generator(MLXFaciesGAN *m, MLXOptimizer *opt, mlx_array **grads, int n);
/* Scale-specific version - use when gradients are collected for a specific scale only */
int mlx_faciesgan_apply_sgd_to_generator_for_scale(MLXFaciesGAN *m, MLXOptimizer *opt, mlx_array **grads, int n, int scale);

/* Apply SGD to a discriminator using provided gradients (array length == n).
 * Returns 0 on success, -1 on error.
 */
int mlx_faciesgan_apply_sgd_to_discriminator(MLXFaciesGAN *m, MLXOptimizer *opt, mlx_array **grads, int n);
/* Scale-specific version - use when gradients are collected for a specific scale only */
int mlx_faciesgan_apply_sgd_to_discriminator_for_scale(MLXFaciesGAN *m, MLXOptimizer *opt, mlx_array **grads, int n, int scale);

/* Convenience train-step that updates both generator and discriminator parameters.
 * Each grads array must match the parameter counts for its model.
 */
int mlx_faciesgan_train_step(MLXFaciesGAN *m, MLXOptimizer *opt_g, mlx_array **gen_grads, int gen_n, MLXOptimizer *opt_d, mlx_array **disc_grads, int disc_n);

/* Wrappers that accept an MLXBaseManager (adapter-backed) */
int mlx_base_apply_sgd_to_generator(MLXBaseManager *mgr, MLXOptimizer *opt, mlx_array **grads, int n);
int mlx_base_apply_sgd_to_discriminator(MLXBaseManager *mgr, MLXOptimizer *opt, mlx_array **grads, int n);
int mlx_base_train_step(MLXBaseManager *mgr, MLXOptimizer *opt_g, mlx_array **gen_grads, int gen_n, MLXOptimizer *opt_d, mlx_array **disc_grads, int disc_n);

/* Metrics & gradient packaging structures ---------------------------------*/
typedef struct MLXScaleMetrics
{
    /* Generator metrics */
    mlx_array *fake;  /* scalar - g_adv (adversarial) */
    mlx_array *well;  /* scalar - g_well (masked loss) */
    mlx_array *div;   /* scalar - g_div (diversity) */
    mlx_array *rec;   /* scalar - g_rec (recovery) */
    mlx_array *total; /* scalar - g_total */
    /* Discriminator metrics */
    mlx_array *d_real;  /* scalar - -mean(d_real) */
    mlx_array *d_fake;  /* scalar - mean(d_fake) */
    mlx_array *d_gp;    /* scalar - gradient penalty */
    mlx_array *d_total; /* scalar - d_real + d_fake + d_gp */
} MLXScaleMetrics;

typedef struct MLXScaleResults
{
    int scale;
    MLXScaleMetrics metrics;
    mlx_array **gen_grads; /* array of mlx_array* (length = gen_n) */
    int gen_n;
    mlx_array **disc_grads;
    int disc_n;
} MLXScaleResults;

typedef struct MLXResults
{
    int n_scales;
    MLXScaleResults *scales; /* malloc'd array of length n_scales */
} MLXResults;

/* Mode flags for collect_metrics_and_grads_native: control which
   gradient computations are performed per call.  Using DISC_ONLY in
   the discriminator loop and GEN_ONLY in the generator loop avoids
   wasted forward/backward passes and keeps the RNG state aligned
   with the Python implementation. */
#define MLX_COLLECT_BOTH      0
#define MLX_COLLECT_DISC_ONLY 1
#define MLX_COLLECT_GEN_ONLY  2

/* Collect per-scale metrics and gradient arrays for generator and
   discriminator using MLX's native value_and_grad.
   Caller receives malloc'd MLXResults and must free it via
   `mlx_results_free`. Returns 0 on success.
   `mode` selects which gradients to compute (see MLX_COLLECT_* above). */
int mlx_faciesgan_collect_metrics_and_grads_native(
    MLXFaciesGAN *m,
    const int *indexes,
    int n_indexes,
    const int *active_scales,
    int n_active_scales,
    mlx_array **facies_pyramid,
    mlx_array **rec_in_pyramid,
    mlx_array **wells_pyramid,
    mlx_array **masks_pyramid,
    mlx_array **seismic_pyramid,
    float lambda_diversity,
    float well_loss_penalty,
    float alpha,
    float lambda_grad,
    int mode,
    MLXResults **out_results);

void mlx_results_free(MLXResults *res);

#ifdef __cplusplus
}
#endif

#endif
