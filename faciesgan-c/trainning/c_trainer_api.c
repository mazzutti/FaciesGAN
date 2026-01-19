#include "trainning/c_trainer_api.h"
#include "trainning/c_trainer.h"
#include "trainning/train_manager.h"
#include "trainning/train_step.h"
#include "models/facies_gan.h"
#include "datasets/mlx_pyramids_dataset.h"
#include "datasets/prefetcher.h"
#include "optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

struct CTrainer
{
    TrainningOptions opts;
    MLXFaciesGAN *model;
    MLXOptimizer **gen_opts;  /* per-scale */
    MLXOptimizer **disc_opts; /* per-scale */
    MLXScheduler **gen_scheds;
    MLXScheduler **disc_scheds;
    int n_scales;
};

CTrainer *c_trainer_create_with_opts(const TrainningOptions *opts)
{
    if (!opts)
        return NULL;
    CTrainer *t = (CTrainer *)malloc(sizeof(CTrainer));
    if (!t)
        return NULL;
    memset(t, 0, sizeof(*t));
    t->opts = *opts; /* copy options */
    /* Create MLXBaseManager + MLXFaciesGAN using existing utilities so
     * scales/shapes are initialized consistently with other C trainers. */
    MLXTrainOptions train_opts = {0};
    mlx_options_to_mlx_train_opts(&t->opts, &train_opts);
    MLXBaseManager *mgr = mlx_base_manager_create_with_faciesgan(&train_opts);
    if (!mgr)
    {
        free(t);
        return NULL;
    }
    t->model = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
    if (!t->model)
    {
        mlx_base_manager_free(mgr);
        free(t);
        return NULL;
    }
    /* number of scales is driven by the model state/shapes */
    int *shapes = NULL;
    int n_shapes = 0;
    if (mlx_faciesgan_get_shapes_flat(t->model, &shapes, &n_shapes) == 0 && n_shapes > 0)
    {
        t->n_scales = n_shapes / 3; /* each scale has 3 ints (H,W,C) */
        free(shapes);
    }
    /* allocate arrays for per-scale optimizers/schedulers */
    t->gen_opts = (MLXOptimizer **)calloc((size_t)t->n_scales, sizeof(MLXOptimizer *));
    t->disc_opts = (MLXOptimizer **)calloc((size_t)t->n_scales, sizeof(MLXOptimizer *));
    t->gen_scheds = (MLXScheduler **)calloc((size_t)t->n_scales, sizeof(MLXScheduler *));
    t->disc_scheds = (MLXScheduler **)calloc((size_t)t->n_scales, sizeof(MLXScheduler *));
    return t;
}

void c_trainer_destroy(CTrainer *t)
{
    if (!t)
        return;
    if (t->model)
        mlx_faciesgan_free(t->model);
    free(t->gen_opts);
    free(t->disc_opts);
    free(t->gen_scheds);
    free(t->disc_scheds);
    free(t);
}

int c_trainer_setup_optimizers(CTrainer *t, const int *scales, int n_scales)
{
    if (!t || !scales || n_scales <= 0)
        return -1;
    for (int i = 0; i < n_scales; ++i)
    {
        int sc = scales[i];
        /* create Adam optimizers with defaults (mirrors Python defaults) */
        t->gen_opts[sc] = mlx_adam_create(t->opts.lr_g, t->opts.beta1, 0.999f, 1e-8f);
        t->disc_opts[sc] = mlx_adam_create(t->opts.lr_d, t->opts.beta1, 0.999f, 1e-8f);
        /* schedulers: multistep with single milestone (lr_decay) */
        int milestones[1] = {t->opts.lr_decay};
        t->gen_scheds[sc] = mlx_scheduler_multistep_create_with_init(milestones, 1, t->opts.gamma, (const float *)&t->opts.lr_g, 1);
        t->disc_scheds[sc] = mlx_scheduler_multistep_create_with_init(milestones, 1, t->opts.gamma, (const float *)&t->opts.lr_d, 1);
    }
    return 0;
}

int c_trainer_optimization_step(
    CTrainer *t,
    const int *indexes,
    int n_indexes,
    mlx_array **facies_pyramid,
    int n_facies,
    mlx_array **rec_in_pyramid,
    int n_rec,
    mlx_array **wells_pyramid,
    int n_wells,
    mlx_array **masks_pyramid,
    int n_masks,
    mlx_array **seismic_pyramid,
    int n_seismic,
    const int *active_scales,
    int n_active_scales)
{
    if (!t || !t->model)
        return -1;

    int rc = 0;
    /* First apply discriminator updates if optimizers are present */
    if (t->disc_opts)
    {
        rc = mlx_faciesgan_optimize_discriminator_scales(
            t->model,
            indexes,
            n_indexes,
            t->disc_opts,
            facies_pyramid,
            NULL, /* rec_in not used for discriminator */
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
            active_scales,
            n_active_scales);
        if (rc != 0)
            return rc;
    }

    /* Then apply generator updates */
    if (t->gen_opts)
    {
        rc = mlx_faciesgan_optimize_generator_scales(
            t->model,
            indexes,
            n_indexes,
            t->gen_opts,
            facies_pyramid,
            rec_in_pyramid,
            wells_pyramid,
            masks_pyramid,
            seismic_pyramid,
            active_scales,
            n_active_scales);
    }

    return rc;
}

int c_trainer_load_model(CTrainer *t, int scale, const char *checkpoint_dir)
{
    if (!t || !checkpoint_dir)
        return -1;
    /* Reuse existing per-scale state loaders (load_*_state). The expected
     * argument is a directory path for the scale; pass `checkpoint_dir/scale`. */
    char scale_dir[PATH_MAX];
    snprintf(scale_dir, PATH_MAX, "%s/%d", checkpoint_dir, scale);
    if (mlx_faciesgan_load_generator_state(t->model, scale_dir, scale) != 0)
        return -1;
    if (mlx_faciesgan_load_discriminator_state(t->model, scale_dir, scale) != 0)
        return -1;
    return 0;
}

int c_trainer_save_generated_facies(CTrainer *t, int scale, int epoch, const char *results_path)
{
    if (!t || !results_path)
        return -1;
    /* Generate noises and call numeric forward (similar to train_utils.c) */
    mlx_array **noises = NULL;
    int n_noises = 0;
    if (mlx_faciesgan_get_pyramid_noise(t->model, scale, NULL, 0, &noises, &n_noises, NULL, NULL, 0) != 0)
        return -1;

    /* use default amplitudes (all ones) */
    float *use_amps = (float *)malloc(sizeof(float) * (size_t)(scale + 1));
    if (!use_amps)
    {
        for (int i = 0; i < n_noises; ++i)
        {
            if (noises[i])
            {
                mlx_array_free(*noises[i]);
                free(noises[i]);
            }
        }
        free(noises);
        return -1;
    }
    for (int i = 0; i < scale + 1; ++i)
        use_amps[i] = 1.0f;

    mlx_array in_noise = mlx_array_new();
    mlx_array_t fake = mlx_faciesgan_generate_fake(t->model, (const mlx_array *)noises, n_noises, use_amps, scale + 1, in_noise, scale, scale);

    /* free noises */
    for (int i = 0; i < n_noises; ++i)
    {
        if (noises[i])
        {
            mlx_array_free(*noises[i]);
            free(noises[i]);
        }
    }
    free(noises);
    free(use_amps);

    char fname[PATH_MAX];
    snprintf(fname, PATH_MAX, "%s/scale_%d_epoch_%d.npy", results_path, scale, epoch);
    int rc = mlx_save(fname, fake);
    mlx_array_free(fake);
    mlx_array_free(in_noise);
    return rc;
}

void *c_trainer_get_model_ctx(CTrainer *t)
{
    if (!t)
        return NULL;
    return (void *)t->model;
}
