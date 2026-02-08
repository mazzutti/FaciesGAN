#ifndef MLX_C_OPTIMIZER_H
#define MLX_C_OPTIMIZER_H

#include "custom_layer.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct MLXOptimizer MLXOptimizer;
/* forward declare scheduler type to avoid circular includes */
struct MLXScheduler;

/* Adam optimizer (compatible with Python defaults). Creates an optimizer
 * configured with learning rate `lr` and Adam hyperparameters. */
/* Create Adam with global defaults */
MLXOptimizer *mlx_adam_create_with_defaults(float lr);
/* Create Adam with custom hyperparameters */
MLXOptimizer *mlx_adam_create(float lr, float beta1, float beta2, float eps);
/* Extended Adam create allowing bias correction and weight decay (AdamW).
 * Use this when you need per-instance control. */
MLXOptimizer *mlx_adam_create_ext(float lr, float beta1, float beta2, float eps, int bias_correction, float weight_decay);
/* Explicit Adam API (preferred): free and step functions */
void mlx_adam_free(MLXOptimizer *opt);
int mlx_adam_step(MLXOptimizer *opt, mlx_array **params, mlx_array **grads, int n);
/* Set global default Adam hyperparameters used by `mlx_adam_create_with_defaults`.
 * Call this from a launcher (e.g., main.c) to configure beta/eps via CLI.
 */
void mlx_optimizer_set_global_adam_params(float beta1, float beta2, float eps);

/* Set learning rate */
void mlx_optimizer_set_lr(MLXOptimizer *opt, float lr);
/* Get learning rate */
float mlx_optimizer_get_lr(MLXOptimizer *opt);
/* Configure global defaults used by `mlx_adam_create_with_defaults`. */
/* Configure whether Adam uses bias-correction by default (0 or 1). */
void mlx_optimizer_set_global_adam_bias_correction(int enabled);
/* Configure global default weight decay used for AdamW */
void mlx_optimizer_set_global_adam_weight_decay(float weight_decay);
/* Global getters for CLI-independent code to obtain current defaults. */
void mlx_optimizer_get_global_adam_params(float *beta1, float *beta2, float *eps);
int mlx_optimizer_get_global_adam_bias_correction(void);
float mlx_optimizer_get_global_adam_weight_decay(void);

/* Convenience: initialize optimizer internal state from a flat list of
 * parameters (length n). This lets callers (or a Python bridge) flatten
 * nested parameter trees and initialize the optimizer state in C. */
void mlx_optimizer_init_from_params(MLXOptimizer *opt, mlx_array **params, int n);
/* Convenience wrapper to apply step given flat param/grad arrays. */
int mlx_optimizer_apply_flat(MLXOptimizer *opt, mlx_array **params, mlx_array **grads, int n);

/* Save/load optimizer state to/from a NumPy .npz file.
 * The .npz will contain per-parameter `m_i.npy` and `v_i.npy` members
 * along with scalar hyperparameters (step, lr, beta1, beta2, eps,
 * weight_decay, bias_correction) saved as small .npy arrays.
 */
int mlx_optimizer_save_to_npz(MLXOptimizer *opt, const char *npz_path);
int mlx_optimizer_load_from_npz(MLXOptimizer *opt, const char *npz_path);
/* Attach a scheduler to an optimizer so the optimizer samples LR per-step. */
void mlx_optimizer_attach_scheduler(MLXOptimizer *opt, struct MLXScheduler *s);
struct MLXScheduler *mlx_optimizer_get_attached_scheduler(MLXOptimizer *opt);
/* After a step, retrieve the last LR value used by the optimizer. */
float mlx_optimizer_get_last_used_lr(MLXOptimizer *opt);
/* Evaluate all optimizer state arrays (m, v) to materialize computation graphs.
 * This is essential for memory management: without evaluating these arrays,
 * the MLX computation graph accumulates across iterations and leaks memory. */
void mlx_optimizer_eval_state(MLXOptimizer *opt);
/* Append all optimizer state arrays (m, v) to an external vector_array for
 * deferred batch evaluation (single-eval-per-epoch pattern). */
void mlx_optimizer_append_state_to_vec(MLXOptimizer *opt, mlx_vector_array vec);

#ifdef __cplusplus
}
#endif

#endif
