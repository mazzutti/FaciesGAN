#ifndef FACIESGAN_CLI_ARGS_H
#define FACIESGAN_CLI_ARGS_H

#include <stdbool.h>
#include <stddef.h>

typedef struct CLIArgs {
  char *input_path;
  char *output_path;
  int manual_seed;
  int stop_scale;
  int num_img_channels;
  int crop_size;
  int batch_size;
  int num_parallel_scales;
  int num_train_pyramids;
  int num_workers;
  int save_interval;
  int noise_channels;
  int img_color_lo;
  int img_color_hi;
  int num_features;
  int min_num_features;
  int kernel_size;
  int num_layers;
  int stride;
  int padding_size;
  double noise_amp;
  double min_noise_amp;
  double scale0_noise_amp;
  int generator_steps;
  int discriminator_steps;
  double lambda_grad;
  int alpha;
  double lambda_diversity;
  double beta1;
  double lr_g;
  double lr_d;
  int lr_decay;
  int num_real_facies;
  int num_generated_per_real;
  int num_iter;

  bool use_wells;
  int *wells_mask_columns;
  size_t wells_mask_count;

  bool use_seismic;
  bool use_cpu;
  bool use_mlx;
  int gpu_device;
  bool no_tensorboard;
  bool no_plot_facies;
  bool compile_backend;
  bool no_compile_backend;
  bool use_profiler;
  bool hand_off_to_c;
} CLIArgs;

/* Initialize/cleanup helpers */
void CLIArgs_init(CLIArgs *args);
void CLIArgs_free(CLIArgs *args);

/* Helper setters */
void CLIArgs_set_input_path(CLIArgs *args, const char *path);
void CLIArgs_set_output_path(CLIArgs *args, const char *path);
int CLIArgs_add_well_mask_column(CLIArgs *args, int col);

/* Parse argc/argv into CLIArgs. Return 0 on success, 1 on error, 2 when
 * help was requested (caller should exit normally). */
int CLIArgs_parse_from_argv(CLIArgs *args, int argc, char **argv);

#endif
