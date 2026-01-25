#include "options_builder.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

TrainningOptions *trainning_options_from_cli(CLIArgs *args) {
  if (!args)
    return NULL;

  TrainningOptions *topt = mlx_options_new_trainning_defaults();
  if (!topt)
    return NULL;

  if (args->input_path) {
    free(topt->input_path);
    topt->input_path = strdup(args->input_path);
  }
  if (args->manual_seed >= 0)
    topt->manual_seed = args->manual_seed;
  if (args->stop_scale >= 0)
    topt->stop_scale = args->stop_scale;
  if (args->num_img_channels > 0)
    topt->num_img_channels = args->num_img_channels;
  if (args->crop_size > 0)
    topt->crop_size = args->crop_size;
  if (args->batch_size > 0)
    topt->batch_size = args->batch_size;
  if (args->num_parallel_scales > 0)
    topt->num_parallel_scales = args->num_parallel_scales;
  if (args->num_train_pyramids > 0)
    topt->num_train_pyramids = args->num_train_pyramids;
  if (args->num_workers >= 0)
    topt->num_workers = args->num_workers;
  if (args->noise_channels > 0)
    topt->noise_channels = args->noise_channels;
  if (args->img_color_lo >= 0) {
    topt->img_color_min = args->img_color_lo;
    topt->img_color_max =
        args->img_color_hi >= 0 ? args->img_color_hi : args->img_color_lo;
  }
  if (args->num_features > 0)
    topt->num_feature = args->num_features;
  if (args->min_num_features > 0)
    topt->min_num_feature = args->min_num_features;
  if (args->kernel_size > 0)
    topt->kernel_size = args->kernel_size;
  if (args->num_layers > 0)
    topt->num_layer = args->num_layers;
  if (args->stride >= 0)
    topt->stride = args->stride;
  if (args->padding_size >= 0)
    topt->padding_size = args->padding_size;
  if (args->noise_amp >= 0.0)
    topt->noise_amp = args->noise_amp;
  if (args->min_noise_amp >= 0.0)
    topt->min_noise_amp = args->min_noise_amp;
  if (args->scale0_noise_amp >= 0.0)
    topt->scale0_noise_amp = args->scale0_noise_amp;
  if (args->generator_steps > 0)
    topt->generator_steps = args->generator_steps;
  if (args->discriminator_steps > 0)
    topt->discriminator_steps = args->discriminator_steps;
  if (args->lambda_grad >= 0.0)
    topt->lambda_grad = args->lambda_grad;
  if (args->alpha >= 0)
    topt->alpha = args->alpha;
  if (args->lambda_diversity >= 0.0)
    topt->lambda_diversity = args->lambda_diversity;
  if (args->beta1 >= 0.0)
    topt->beta1 = args->beta1;
  if (args->lr_g >= 0.0)
    topt->lr_g = args->lr_g;
  if (args->lr_d >= 0.0)
    topt->lr_d = args->lr_d;
  if (args->lr_decay >= 0)
    topt->lr_decay = args->lr_decay;
  if (args->num_real_facies > 0)
    topt->num_real_facies = args->num_real_facies;
  if (args->num_generated_per_real > 0)
    topt->num_generated_per_real = args->num_generated_per_real;
  if (args->num_iter > 0)
    topt->num_iter = args->num_iter;
  if (args->save_interval > 0)
    topt->save_interval = args->save_interval;

  topt->use_wells = args->use_wells;
  topt->use_seismic = args->use_seismic;
  topt->use_cpu = args->use_cpu;
  topt->use_mlx = args->use_mlx;
  topt->gpu_device = args->gpu_device;
  topt->enable_tensorboard = !args->no_tensorboard;
  topt->enable_plot_facies = !args->no_plot_facies;
  topt->compile_backend = args->compile_backend;
  topt->use_profiler = args->use_profiler;
  topt->hand_off_to_c = args->hand_off_to_c;
  topt->wells_mask_columns = args->wells_mask_columns;
  topt->wells_mask_count = args->wells_mask_count;
  args->wells_mask_columns = NULL;
  args->wells_mask_count = 0;

  return topt;
}
