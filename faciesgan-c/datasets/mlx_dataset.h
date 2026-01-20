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
                              int num_img_channels, int use_wells,
                              int use_seismic, int manual_seed,
                              mlx_vector_vector_array *out_facies,
                              mlx_vector_vector_array *out_wells,
                              /* optional: output masks per-sample per-scale */
                              mlx_vector_vector_array *out_masks,
                              mlx_vector_vector_array *out_seismic,
                              int *out_num_samples);

/* -------------------------------------------------------------------------- */
/* MLXPyramidsDataset API (merged) */

typedef struct MLXPyramidsDataset MLXPyramidsDataset;

/* Create a new dataset by loading pyramids from function cache (or generating).
 * - `input_path`, `cache_dir`: dataset root and cache directory
 * - `desired_num`, `stop_scale`, `crop_size`, `num_img_channels`, `use_wells`,
 * `use_seismic`, `manual_seed`: generation params
 * - `shuffle`: whether to shuffle samples after load
 * Returns 0 on success and sets `*out`.
 */
int mlx_pyramids_dataset_new(MLXPyramidsDataset **out, const char *input_path,
                             const char *cache_dir, int desired_num,
                             int stop_scale, int crop_size,
                             int num_img_channels, int use_wells,
                             int use_seismic, int manual_seed, int shuffle);

void mlx_pyramids_dataset_free(MLXPyramidsDataset *ds);

/* Shuffle samples in-place (deterministic by seed when seed!=0). */
int mlx_pyramids_dataset_shuffle(MLXPyramidsDataset *ds, unsigned int seed);

/* Remove function-level cache files under cache_dir (non-recursive). */
int mlx_pyramids_dataset_clean_cache(const char *cache_dir);

/* Get stacked per-scale tensors as a single mlx_array with shape (N,H,W,C).
 * Caller must free the returned mlx_array with `mlx_array_free()`.
 * Returns 0 on success, non-zero on failure.
 */
int mlx_pyramids_dataset_get_scale_stack(MLXPyramidsDataset *ds, int scale,
                                         mlx_array *out);

/* Get stacked per-scale wells arrays (shape (N,H,W,C)). If the dataset has no
 * wells, returns an empty array with shape (0,H,W,C) to match Python semantics.
 */
int mlx_pyramids_dataset_get_wells_stack(MLXPyramidsDataset *ds, int scale,
                                         mlx_array *out);

/* Get stacked per-scale masks arrays (shape (N,H,W,C)). If dataset has no
 * masks, returns an empty array with shape (0,H,W,C).
 */
int mlx_pyramids_dataset_get_masks_stack(MLXPyramidsDataset *ds, int scale,
                                         mlx_array *out);

/* Get stacked per-scale seismic arrays (shape (N,H,W,C)). If dataset has no
 * seismic data, returns an empty array with shape (0,H,W,C).
 */
int mlx_pyramids_dataset_get_seismic_stack(MLXPyramidsDataset *ds, int scale,
                                           mlx_array *out);

#ifdef __cplusplus
}
#endif

#endif
