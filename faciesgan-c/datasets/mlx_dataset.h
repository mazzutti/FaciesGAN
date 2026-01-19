#ifndef FACIESGAN_DATASETS_MLX_DATASET_H
#define FACIESGAN_DATASETS_MLX_DATASET_H

#include <mlx/c/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Load MLX pyramids into vector-of-vector arrays from a function-level cache.
 * Returns 0 on success; out_num_samples set to number of loaded samples.
 */
int mlx_pyramids_dataset_load(const char *input_path, const char *cache_dir,
                              int desired_num, int stop_scale, int crop_size,
                              int num_img_channels, int use_wells, int use_seismic,
                              int manual_seed,
                              mlx_vector_vector_array *out_facies,
                              mlx_vector_vector_array *out_wells,
                              mlx_vector_vector_array *out_seismic,
                              int *out_num_samples);

#ifdef __cplusplus
}
#endif

#endif
