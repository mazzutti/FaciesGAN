#ifndef FACIESGAN_DATASETS_MLX_PYRAMIDS_DATASET_H
#define FACIESGAN_DATASETS_MLX_PYRAMIDS_DATASET_H

#include <mlx/c/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLXPyramidsDataset MLXPyramidsDataset;

/* Create a new dataset by loading pyramids from function cache (or generating).
 * - `input_path`, `cache_dir`: dataset root and cache directory
 * - `desired_num`, `stop_scale`, `crop_size`, `num_img_channels`, `use_wells`, `use_seismic`, `manual_seed`: generation params
 * - `shuffle`: whether to shuffle samples after load
 * Returns 0 on success and sets `*out`.
 */
int mlx_pyramids_dataset_new(MLXPyramidsDataset **out, const char *input_path, const char *cache_dir,
                            int desired_num, int stop_scale, int crop_size, int num_img_channels,
                            int use_wells, int use_seismic, int manual_seed, int shuffle);

void mlx_pyramids_dataset_free(MLXPyramidsDataset *ds);

/* Shuffle samples in-place (deterministic by seed when seed!=0). */
int mlx_pyramids_dataset_shuffle(MLXPyramidsDataset *ds, unsigned int seed);

/* Remove function-level cache files under cache_dir (non-recursive). */
int mlx_pyramids_dataset_clean_cache(const char *cache_dir);

/* Get stacked per-scale tensors as a single mlx_array with shape (N,H,W,C).
 * Caller must free the returned mlx_array with `mlx_array_free()`.
 * Returns 0 on success, non-zero on failure.
 */
int mlx_pyramids_dataset_get_scale_stack(MLXPyramidsDataset *ds, int scale, mlx_array *out);

#ifdef __cplusplus
}
#endif

#endif
