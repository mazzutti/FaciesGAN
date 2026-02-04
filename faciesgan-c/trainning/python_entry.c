#include "array_helpers.h"
#include "base_manager.h"
#include "facies_gan.h"
#include "optimizer.h"
#include "options.h"
#include "train_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Lightweight C entrypoint callable from Python via ctypes.
 * This runs a basic training loop using existing MLX helpers.
 * Parameters are intentionally simple: path strings + numeric hyperparams.
/* Globals used by the callback installed below. */
static MLXOptimizer **g_opts_by_scale = NULL;
static int g_num_parallel_scales = 0;

/* Top-level callback used by the train manager. It uses globals set by the
 * caller before invoking the manager loop. */
int train_cb(MLXBaseManager *bm, int step, void *ctx,
             mlx_array ***out_gen_grads, int *out_gen_n,
             mlx_array ***out_disc_grads, int *out_disc_n, float *out_loss) {
    /* unused parameters for this lightweight callback */
    MLXFaciesGAN *fg = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(bm);
    if (!fg)
        return -1;
    if (!g_opts_by_scale || g_num_parallel_scales <= 0)
        return -1;

    int *active = NULL;
    if (mlx_alloc_int_array(&active, g_num_parallel_scales) != 0)
        return -1;
    for (int i = 0; i < g_num_parallel_scales; ++i)
        active[i] = i;

    mlx_faciesgan_optimize_discriminator_scales(fg, NULL, 0, g_opts_by_scale,
            NULL, NULL, NULL, active,
            g_num_parallel_scales);
    mlx_faciesgan_optimize_generator_scales(fg, NULL, 0, g_opts_by_scale, NULL,
                                            NULL, NULL, NULL, NULL, active,
                                            g_num_parallel_scales);

    mlx_free_int_array(&active, &g_num_parallel_scales);
    if (out_gen_n)
        *out_gen_n = 0;
    if (out_disc_n)
        *out_disc_n = 0;
    return 0;
}

int mlx_run_manager_from_python(const char *output_path,
                                int num_parallel_scales, int num_img_channels,
                                int discriminator_steps, int generator_steps,
                                int num_feature, int min_num_feature,
                                int num_layer, int kernel_size,
                                int padding_size, int num_diversity_samples,
                                int epochs, int steps_per_epoch,
                                const char *checkpoint_path,
                                int checkpoint_every, int use_create_graph_gp) {
    /* output_path intentionally unused in this example entrypoint */
    if (!num_parallel_scales)
        return -1;

    /* Obtain canonical C defaults and map them into the MLX train options. */
    TrainningOptions *t = mlx_options_new_trainning_defaults();
    if (!t)
        return -1;
    /* Explicitly override with parameters provided by the caller. */
    t->num_parallel_scales = num_parallel_scales;
    t->num_img_channels = num_img_channels;
    t->discriminator_steps = discriminator_steps;
    t->generator_steps = generator_steps;
    t->num_feature = num_feature;
    t->min_num_feature = min_num_feature;
    t->num_layer = num_layer;
    t->kernel_size = kernel_size;
    t->padding_size = padding_size;
    t->num_diversity_samples = num_diversity_samples;

    mlx_faciesgan_set_use_create_graph_gp(use_create_graph_gp);

    MLXBaseManager *mgr = mlx_base_manager_create_with_faciesgan(t);
    if (!mgr)
        return -2;

    /* initialize scales */
    mlx_base_manager_init_scales(mgr, 0, num_parallel_scales);

    /* create optimizer per-scale (simple SGD with fixed LR) */
    MLXOptimizer **opts_by_scale = NULL;
    if (mlx_alloc_ptr_array((void ***)&opts_by_scale, num_parallel_scales) != 0) {
        mlx_base_manager_free(mgr);
        return -3;
    }
    for (int i = 0; i < num_parallel_scales; ++i) {
        float gb1 = 0.0f, gb2 = 0.0f, geps = 0.0f;
        mlx_optimizer_get_global_adam_params(&gb1, &gb2, &geps);
        int gbc = mlx_optimizer_get_global_adam_bias_correction();
        float gwd = mlx_optimizer_get_global_adam_weight_decay();
        opts_by_scale[i] = mlx_adam_create_ext(5e-5f, gb1, gb2, geps, gbc, gwd);
    }

    /* Create a MultiStep scheduler and attach it to the first per-scale optimizer
     * so the scheduler has an optimizer to update. This demonstrates runtime
     * wiring of the new scheduler parity APIs. */
    int milestones_arr[2] = {100, 200};
    float base_lr = mlx_optimizer_get_lr(opts_by_scale[0]);
    MLXScheduler *sched = mlx_scheduler_multistep_create_with_init(
                              milestones_arr, 2, 0.1f, &base_lr, 1);
    if (sched && opts_by_scale[0]) {
        mlx_scheduler_attach_optimizer(sched, opts_by_scale[0]);
        /* start from last_step = -1 (before first step) */
        mlx_scheduler_set_last_step(sched, -1);
    }

    /* callback and globals are defined above */

    /* Run manager loop (logger and scheduler left NULL for simplicity) */
    /* Install globals for callback, invoke manager, then clear globals. */
    g_opts_by_scale = opts_by_scale;
    g_num_parallel_scales = num_parallel_scales;

    int rc = mlx_faciesgan_train_manager(
                 mgr, NULL, NULL, (MLXTrainStepCallback)train_cb, NULL, epochs,
                 steps_per_epoch, NULL, sched, checkpoint_path, checkpoint_every);

    g_opts_by_scale = NULL;
    g_num_parallel_scales = 0;

    for (int i = 0; i < num_parallel_scales; ++i) {
        if (opts_by_scale[i])
            mlx_adam_free(opts_by_scale[i]);
    }
    mlx_free_ptr_array((void ***)&opts_by_scale, num_parallel_scales);

    if (sched)
        mlx_scheduler_free(sched);

    /* free manager (adapter will free internal MLXFaciesGAN, if present) */
    mlx_base_manager_free(mgr);

    return rc;
}
