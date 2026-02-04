#ifndef MLX_C_TRAIN_MANAGER_H
#define MLX_C_TRAIN_MANAGER_H

#include "facies_gan.h"
#include "base_manager.h"
#include "optimizer.h"
#include "logger.h"
#include "scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Callback invoked per step to produce grads and metrics.
 * Should allocate `*out_gen_grads` (array of mlx_array* length *out_gen_n) and same for disc.
 * Return 0 on success, non-zero on error.
 */
typedef int (*MLXTrainStepCallback)(MLXBaseManager *mgr, int step, void *ctx, mlx_array ***out_gen_grads, int *out_gen_n, mlx_array ***out_disc_grads, int *out_disc_n, float *out_loss);
int mlx_faciesgan_train_manager(
    MLXBaseManager *mgr,
    MLXOptimizer *opt_g,
    MLXOptimizer *opt_d,
    MLXTrainStepCallback cb,
    void *cb_ctx,
    int epochs,
    int steps_per_epoch,
    MLXLogger *logger,
    MLXScheduler *sched,
    const char *checkpoint_path,
    int checkpoint_every
);

#ifdef __cplusplus
}
#endif

#endif
