#ifndef MLX_C_SCHEDULER_H
#define MLX_C_SCHEDULER_H

#include "optimizer.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct MLXScheduler MLXScheduler;

    /* StepLR scheduler: multiply lr by gamma every step_size steps. */
    MLXScheduler *mlx_scheduler_step_lr_create(int step_size, float gamma);
    /* Create and set initial learning rates (scalar or array). If `init_lr` is
     * non-NULL and `n_init_lrs`>0 the scheduler will use those base_lrs. */
    MLXScheduler *mlx_scheduler_step_lr_create_with_init(int step_size, float gamma, const float *init_lr, int n_init_lrs);
    /* MultiStepLR scheduler: multiply lr by gamma when `step` reaches any milestone in the provided list. */
    MLXScheduler *mlx_scheduler_multistep_create(const int *milestones, int n_milestones, float gamma);
    MLXScheduler *mlx_scheduler_multistep_create_with_init(const int *milestones, int n_milestones, float gamma, const float *init_lr, int n_init_lrs);
    void mlx_scheduler_free(MLXScheduler *s);
    void mlx_scheduler_step(MLXScheduler *s, int step, MLXOptimizer *opt);
    /* Wrapper that performs a step and returns the computed learning rates.
     * Writes up to `max_out` floats into `out_lrs` and returns the number
     * of values written. */
    int mlx_scheduler_step_and_get_lr(MLXScheduler *s, int step, MLXOptimizer *opt, float *out_lrs, int max_out);
    /* Match Python step(None) semantics: pass NULL for `step` to increment by
     * one. Writes up to `max_out` LRs into `out_lrs` and returns the count. */
    int mlx_scheduler_step_nullable(MLXScheduler *s, const int *step, float *out_lrs, int max_out);

    /* Extended parity APIs (match Python `MultiStepLR` semantics) */
    void mlx_scheduler_set_base_lrs(MLXScheduler *s, const float *base_lrs, int n_base_lrs);
    /* Fill `out_lrs` up to `max_out` with current learning rates; returns number of lrs written. */
    int mlx_scheduler_get_lr(MLXScheduler *s, float *out_lrs, int max_out);
    /* Compute learning rates for an arbitrary step without changing internal state. */
    int mlx_scheduler_lr_for_step(MLXScheduler *s, int step, float *out_lrs, int max_out);

    /* Last-step accessor (tracks internal `last_step` like PyTorch scheduler) */
    void mlx_scheduler_set_last_step(MLXScheduler *s, int last_step);
    int mlx_scheduler_get_last_step(MLXScheduler *s);

    /* Attach an optimizer to the scheduler so `mlx_scheduler_step` updates it. */
    void mlx_scheduler_attach_optimizer(MLXScheduler *s, MLXOptimizer *opt);

    /* Serialize scheduler state into a heap-allocated string. Caller must free(*out_json).
     * Returns 0 on success. */
    int mlx_scheduler_serialize_state(MLXScheduler *s, char **out_json);

    /* Load scheduler state from a string produced by `mlx_scheduler_serialize_state`.
     * Returns 0 on success. */
    int mlx_scheduler_load_state_from_json(MLXScheduler *s, const char *json_str);

    /* Implicit step: increment internal `last_step` by 1 and apply. */
    void mlx_scheduler_step_auto(MLXScheduler *s, MLXOptimizer *opt);

    /* Callable-style API: compute learning rates for `step` without mutating state.
     * Returns number of learning rates written into `out_lrs`. */
    int mlx_scheduler_call(MLXScheduler *s, int step, float *out_lrs, int max_out);

    /* Apply computed per-group LRs to an array of optimizers. If `n_opts` differs
     * from the internal number of base_lrs, mapping truncates or repeats as needed. */
    void mlx_scheduler_apply_to_optimizers(MLXScheduler *s, MLXOptimizer **opts, int n_opts);

    /* Save scheduler state into an .npz file at `npz_path` (member: "state.json").
     * Returns 0 on success. */
    int mlx_scheduler_save_to_npz(MLXScheduler *s, const char *npz_path);

    /* Load scheduler state from an .npz file containing member "state.json". */
    int mlx_scheduler_load_from_npz(MLXScheduler *s, const char *npz_path);

#ifdef __cplusplus
}
#endif

#endif
