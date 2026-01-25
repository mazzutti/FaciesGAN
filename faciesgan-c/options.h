#ifndef FACIESGAN_C_OPTIONS_H
#define FACIESGAN_C_OPTIONS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrainningOptions {
  int alpha;
  int batch_size;
  double beta1;
  int crop_size;
  int discriminator_steps;
  int num_img_channels;
  double gamma;
  int generator_steps;
  int gpu_device;
  int img_color_min;
  int img_color_max;
  char *input_path;
  char *output_fullpath;
  int kernel_size;
  double lambda_grad;
  double lr_d;
  int lr_decay;
  double lr_g;
  int manual_seed;
  int max_size;
  int min_num_feature;
  int min_size;
  double noise_amp;
  double min_noise_amp;
  double scale0_noise_amp;
  double well_loss_penalty;
  double lambda_diversity;
  int num_diversity_samples;
  int num_feature;
  int num_generated_per_real;
  int num_iter;
  int num_layer;
  int noise_channels;
  int num_real_facies;
  int num_train_pyramids;
  int num_parallel_scales;
  int num_workers;
  char *output_path;
  int padding_size;
  bool regen_npy_gz;
  int save_interval;
  int start_scale;
  int stride;
  int stop_scale;
  bool use_cpu;
  bool use_profiler;
  bool use_mlx;
  bool use_wells;
  bool use_seismic;
  bool hand_off_to_c;
  int *wells_mask_columns;
  size_t wells_mask_count;
  bool enable_tensorboard;
  bool enable_plot_facies;
  bool compile_backend;
} TrainningOptions;

typedef struct ResumeOptions {
  bool fine_tuning;
  char *checkpoint_path;
  int num_iter;
  int start_scale;
} ResumeOptions;

TrainningOptions *mlx_options_new_trainning_defaults(void);
void mlx_options_free_trainning(TrainningOptions *opt);

ResumeOptions *mlx_options_new_resume_defaults(void);
void mlx_options_free_resume(ResumeOptions *opt);

/* No separate MLXTrainOptions type is used any more. The conversion helper
 * is kept for source-compatibility but operates on `TrainningOptions`. */
void mlx_options_to_mlx_train_opts(const TrainningOptions *t,
                                   TrainningOptions *out);

/* Utilities */
TrainningOptions *mlx_options_clone(const TrainningOptions *src);

#ifdef __cplusplus
}
#endif

#endif /* FACIESGAN_C_OPTIONS_H */
