#include "options.h"

#include "datasets/data_files.h"
#include "models/base_manager.h"
#include "trainning/array_helpers.h"
#include <stdlib.h>
#include <string.h>

static char *dupstr(const char *s) {
    if (!s)
        return NULL;
    return strdup(s);
}

TrainningOptions *mlx_options_new_trainning_defaults(void) {
    TrainningOptions *o = NULL;
    if (mlx_alloc_pod((void **)&o, sizeof(TrainningOptions), 1) != 0)
        return NULL;
    o->alpha = 10;
    o->batch_size = 1;
    o->beta1 = 0.5;
    o->crop_size = 256;
    o->discriminator_steps = 3;
    o->num_img_channels = 3;
    o->gamma = 0.9;
    o->generator_steps = 3;
    o->gpu_device = 0;
    o->img_color_min = 0;
    o->img_color_max = 255;
    o->input_path = dupstr(DF_BASE_DATA_DIR);
    o->kernel_size = 3;
    o->lambda_grad = 0.1;
    o->lr_d = 5e-05;
    o->lr_decay = 1000;
    o->lr_g = 5e-05;
    o->manual_seed = -1;
    o->max_size = 1024;
    o->min_num_feature = 32;
    o->min_size = 12;
    o->noise_amp = 0.1;
    o->min_noise_amp = 0.1;
    o->scale0_noise_amp = 1.0;
    o->well_loss_penalty = 10.0;
    o->lambda_diversity = 1.0;
    o->num_diversity_samples = 3;
    o->num_feature = 32;
    o->num_generated_per_real = 5;
    o->num_iter = 2000;
    o->num_layer = 5;
    o->noise_channels = 3;
    o->num_real_facies = 5;
    o->num_train_pyramids = 200;
    o->num_parallel_scales = 2;
    o->num_workers = 4;
    o->output_path = dupstr("results");
    o->padding_size = 0;
    o->regen_npy_gz = false;
    o->save_interval = 100;
    o->start_scale = 0;
    o->stride = 1;
    o->stop_scale = 6;
    o->use_cpu = false;
    o->use_profiler = false;
    o->use_mlx = false;
    o->use_wells = false;
    o->use_seismic = false;
    o->hand_off_to_c = false;
    o->output_fullpath = NULL;
    o->enable_tensorboard = true;
    o->enable_plot_facies = true;
    o->compile_backend = true;  /* Perf: enable MLX JIT compilation by default for single-process */
    o->use_pybridge_plot = true;  /* Use Python for plotting by default */
    o->wells_mask_columns = NULL;
    o->wells_mask_count = 0;
    return o;
}

void mlx_options_free_trainning(TrainningOptions *opt) {
    if (!opt)
        return;
    if (opt->input_path)
        free(opt->input_path);
    if (opt->output_path)
        free(opt->output_path);
    if (opt->wells_mask_columns) {
        mlx_free_int_array(&opt->wells_mask_columns, NULL);
        opt->wells_mask_count = 0;
    }
    if (opt->output_fullpath)
        free(opt->output_fullpath);
    mlx_free_pod((void **)&opt);
}

ResumeOptions *mlx_options_new_resume_defaults(void) {
    ResumeOptions *r = NULL;
    if (mlx_alloc_pod((void **)&r, sizeof(ResumeOptions), 1) != 0)
        return NULL;
    r->fine_tuning = false;
    r->checkpoint_path = dupstr("");
    r->num_iter = -1;
    r->start_scale = 0;
    return r;
}

void mlx_options_free_resume(ResumeOptions *opt) {
    if (!opt)
        return;
    if (opt->checkpoint_path)
        free(opt->checkpoint_path);
    mlx_free_pod((void **)&opt);
}

void mlx_options_to_mlx_train_opts(const TrainningOptions *t,
                                   TrainningOptions *out) {
    if (!t || !out)
        return;
    /* Copy the subset of fields previously represented by MLXTrainOptions.
     * This keeps the helper useful for existing callsites while avoiding a
     * separate type. */
    out->num_parallel_scales = t->num_parallel_scales;
    out->num_img_channels = t->num_img_channels;
    out->discriminator_steps = t->discriminator_steps;
    out->generator_steps = t->generator_steps;
    out->num_feature = t->num_feature;
    out->min_num_feature = t->min_num_feature;
    out->num_layer = t->num_layer;
    out->kernel_size = t->kernel_size;
    out->padding_size = t->padding_size;
    out->num_diversity_samples = t->num_diversity_samples;
}

TrainningOptions *mlx_options_clone(const TrainningOptions *src) {
    if (!src)
        return NULL;
    TrainningOptions *c = NULL;
    if (mlx_alloc_pod((void **)&c, sizeof(TrainningOptions), 1) != 0)
        return NULL;
    memcpy(c, src, sizeof(TrainningOptions));
    /* Deep-copy string fields */
    c->input_path = src->input_path ? dupstr(src->input_path) : NULL;
    c->output_path = src->output_path ? dupstr(src->output_path) : NULL;
    if (src->wells_mask_count > 0 && src->wells_mask_columns) {
        if (mlx_alloc_int_array(&c->wells_mask_columns, src->wells_mask_count) ==
                0) {
            memcpy(c->wells_mask_columns, src->wells_mask_columns,
                   sizeof(int) * src->wells_mask_count);
            c->wells_mask_count = src->wells_mask_count;
        }
    } else {
        c->wells_mask_columns = NULL;
        c->wells_mask_count = 0;
    }
    return c;
}
