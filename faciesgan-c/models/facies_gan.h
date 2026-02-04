#ifndef MLX_C_FACIES_GAN_H
#define MLX_C_FACIES_GAN_H

#include "discriminator.h"
#include "generator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLXFaciesGAN MLXFaciesGAN;
typedef struct MLXOptimizer MLXOptimizer;

/* Lifecycle */
MLXFaciesGAN *mlx_faciesgan_create(int num_layer, int kernel_size,
                                   int padding_size, int num_img_channels,
                                   int num_feature, int min_num_feature,
                                   int discriminator_steps,
                                   int generator_steps);
void mlx_faciesgan_free(MLXFaciesGAN *m);

/* Builders */
MLXDiscriminator *mlx_faciesgan_build_discriminator(MLXFaciesGAN *m);
MLXGenerator *mlx_faciesgan_build_generator(MLXFaciesGAN *m);
int mlx_faciesgan_create_generator_scale(MLXFaciesGAN *m, int scale,
        int num_features,
        int min_num_features);
int mlx_faciesgan_create_discriminator_scale(MLXFaciesGAN *m,
        int num_features,
        int min_num_features);

/* Inference */
mlx_array_t mlx_faciesgan_generate_fake(MLXFaciesGAN *m,
                                        const mlx_array *z_list, int z_count,
                                        const float *amp, int amp_count,
                                        mlx_array_t in_noise, int start_scale,
                                        int stop_scale);

/* Helper: return noise amplitude list for given target scale (caller
   receives malloc'd array of floats; caller must free). */
int mlx_faciesgan_get_noise_amplitude(MLXFaciesGAN *m, int scale,
                                      float **out_amps, int *out_n);

/* Generate per-scale noise arrays up to `scale` (inclusive).
     - `indexes`/`n_indexes` may be used by future implementations to
         select slices from conditioning tensors; currently ignored.
     - `wells_pyramid` and `seismic_pyramid` are accepted for parity but
         not currently applied; they may be NULL.
     Caller receives a malloc'd array of `mlx_array*` (one per scale).
     Caller must free each mlx_array via `mlx_array_free` and free the
     returned pointer array.
int mlx_faciesgan_get_pyramid_noise(MLXFaciesGAN *m, int scale,
                                    const int *indexes, int n_indexes,
                                    mlx_array ***out_noises, int *out_n,
                                    mlx_array **wells_pyramid,
                                    mlx_array **seismic_pyramid, int rec);

/* Configure generator input channels (noise + wells + seismic conditioning). */
int mlx_faciesgan_set_gen_input_channels(MLXFaciesGAN *m, int channels);
int mlx_faciesgan_get_gen_input_channels(MLXFaciesGAN *m);

/* Get number of image channels (facies categories). */
int mlx_faciesgan_get_num_img_channels(MLXFaciesGAN *m);

/* Configure diversity sample count used by C-side metrics collection. */
int mlx_faciesgan_set_num_diversity_samples(MLXFaciesGAN *m, int n);
int mlx_faciesgan_get_num_diversity_samples(MLXFaciesGAN *m);

/* Helpers to configure per-scale shapes/noise used by C training loop. */
int mlx_faciesgan_set_shapes(MLXFaciesGAN *m, const int *shapes, int n_scales);
int mlx_faciesgan_set_noise_amps(MLXFaciesGAN *m, const float *amps, int n);
/* Get shapes/noise amps previously set (caller receives malloc'd array). */
int mlx_faciesgan_get_shapes_flat(MLXFaciesGAN *m, int **out_shapes,
                                  int *out_n_scales);
int mlx_faciesgan_get_noise_amps(MLXFaciesGAN *m, float **out_amps, int *out_n);

/* Checkpoint I/O: save/load generator/discriminator weights and shape/amp
   files. Return 0 on success, non-zero on error. */
int mlx_faciesgan_save_generator_state(MLXFaciesGAN *m, const char *scale_path,
                                       int scale);
int mlx_faciesgan_save_discriminator_state(MLXFaciesGAN *m,
        const char *scale_path, int scale);
int mlx_faciesgan_load_generator_state(MLXFaciesGAN *m, const char *scale_path,
                                       int scale);
int mlx_faciesgan_load_discriminator_state(MLXFaciesGAN *m,
        const char *scale_path, int scale);
int mlx_faciesgan_save_shape(MLXFaciesGAN *m, const char *scale_path,
                             int scale);
int mlx_faciesgan_load_shape(MLXFaciesGAN *m, const char *scale_path,
                             int scale);
int mlx_faciesgan_load_amp(MLXFaciesGAN *m, const char *scale_path);
int mlx_faciesgan_load_wells(MLXFaciesGAN *m, const char *scale_path);

/* High-level per-scale training orchestration (placeholders may be no-ops).
   These accept arrays indexed by scale and arrays of optimizers for each scale.
int mlx_faciesgan_optimize_discriminator_scales(
    MLXFaciesGAN *m, const int *indexes, int n_indexes,
    MLXOptimizer **optimizers_by_scale, mlx_array **facies_pyramid,
    mlx_array **wells_pyramid, mlx_array **seismic_pyramid,
    const int *active_scales, int n_active_scales);

int mlx_faciesgan_optimize_generator_scales(
    MLXFaciesGAN *m, const int *indexes, int n_indexes,
    MLXOptimizer **optimizers_by_scale, mlx_array **facies_pyramid,
    mlx_array **rec_in_pyramid, mlx_array **wells_pyramid,
    mlx_array **masks_pyramid, mlx_array **seismic_pyramid,
    const int *active_scales, int n_active_scales);

/* Utility losses implemented on top of MLX ops (non-AG numeric versions).
   The functions compute scalar loss values (as a malloc'd mlx_array) and
   return 0 on success. Caller must free the returned mlx_array via
   mlx_array_free() when done.
int mlx_faciesgan_compute_diversity_loss(MLXFaciesGAN *m,
        mlx_array **fake_samples,
        int n_samples, float lambda_diversity,
        mlx_array **out_loss);

int mlx_faciesgan_compute_masked_loss(MLXFaciesGAN *m, const mlx_array *fake,
                                      const mlx_array *real,
                                      const mlx_array *well,
                                      const mlx_array *mask,
                                      float well_loss_penalty,
                                      mlx_array **out_loss);

int mlx_faciesgan_compute_recovery_loss(
    MLXFaciesGAN *m, const int *indexes, int n_indexes, int scale,
    const mlx_array rec_in, const mlx_array real, mlx_array **wells_pyramid,
    mlx_array **seismic_pyramid, float alpha, mlx_array **out_loss);

/* Runtime option: toggle using create-graph (higher-order) GP computation.
   Set to non-zero to use create-graph; zero to use numeric finite-diff
   fallback. Default is create-graph (1). */
void mlx_faciesgan_set_use_create_graph_gp(int use);
int mlx_faciesgan_get_use_create_graph_gp(void);

/* Evaluate all model parameters (parity with Python mx.eval(self.model.state)).
   This forces lazy computation graphs to materialize and releases intermediate
   arrays, preventing memory accumulation during training. */
int mlx_faciesgan_eval_all_parameters(MLXFaciesGAN *m);

#ifdef __cplusplus
}
#endif

#endif /* MLX_C_FACIES_GAN_H */
