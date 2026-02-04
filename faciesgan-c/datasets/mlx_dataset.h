#ifndef FACIESGAN_DATASETS_MLX_DATASET_H
#define FACIESGAN_DATASETS_MLX_DATASET_H

#include "options.h"
#include <mlx/c/array.h>
#include <mlx/c/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Batch struct mirrors Python Batch(facies,wells,seismic). Members are
 * owned by the caller when returned via accessor functions. */
typedef struct MLXBatch {
  mlx_vector_array facies;
  mlx_vector_array wells;
  mlx_vector_array seismic;
} MLXBatch;

/* Full dataset struct exposed so callers can inspect metadata. Keep fields
 * minimal and preserve ownership semantics: callers must not free internals.
typedef struct MLXPyramidsDataset {
  mlx_vector_vector_array facies;
  mlx_vector_vector_array wells;
  mlx_vector_vector_array masks;
  mlx_vector_vector_array seismic;
  const TrainningOptions *options;
  const char *data_dir;
  MLXBatch *batches;
  int n_batches;
  int *scales;
  int n_scales;
  int n_samples;
} MLXPyramidsDataset;

/* Create a new dataset by loading pyramids from function cache (or generating).
 * - `input_path`, `cache_dir`: dataset root and cache directory
 * - `desired_num`, `stop_scale`, `crop_size`, `num_img_channels`, `use_wells`,
 * `use_seismic`, `manual_seed`: generation params
 * - `shuffle`: whether to shuffle samples after load
 * Returns 0 on success and sets `*out`.
int mlx_pyramids_dataset_new(MLXPyramidsDataset **out,
                             const TrainningOptions *options, int shuffle,
                             int regenerate, int channels_last);

void mlx_pyramids_dataset_free(MLXPyramidsDataset *ds);

/* Shuffle samples in-place (deterministic by seed when seed!=0). */
int mlx_pyramids_dataset_shuffle(MLXPyramidsDataset *ds, unsigned int seed);

/* Remove function-level cache files under cache_dir (non-recursive). */
int mlx_pyramids_dataset_clean_cache(const char *cache_dir);

/* Get stacked per-scale tensors as a single mlx_array with shape (N,H,W,C).
 * Caller must free the returned mlx_array with `mlx_array_free()`.
 * Returns 0 on success, non-zero on failure.
int mlx_pyramids_dataset_get_scale_stack(MLXPyramidsDataset *ds, int scale,
                                         mlx_array *out);

/* Get stacked per-scale wells arrays (shape (N,H,W,C)). If the dataset has no
 * wells, returns an empty array with shape (0,H,W,C) to match Python semantics.
int mlx_pyramids_dataset_get_wells_stack(MLXPyramidsDataset *ds, int scale,
                                         mlx_array *out);

/* Get stacked per-scale masks arrays (shape (N,H,W,C)). If dataset has no
 * masks, returns an empty array with shape (0,H,W,C).
int mlx_pyramids_dataset_get_masks_stack(MLXPyramidsDataset *ds, int scale,
                                         mlx_array *out);

/* Get stacked per-scale seismic arrays (shape (N,H,W,C)). If dataset has no
 * seismic data, returns an empty array with shape (0,H,W,C).
int mlx_pyramids_dataset_get_seismic_stack(MLXPyramidsDataset *ds, int scale,
                                           mlx_array *out);

/* Get per-sample batch vectors: facies, wells, seismic. Caller receives
 * ownership of the returned `mlx_vector_array` values and must free them
 * with `mlx_vector_array_free()` when done. Returns 0 on success. */
int mlx_pyramids_dataset_get_batch(MLXPyramidsDataset *ds, int index,
                                   mlx_vector_array *out_facies,
                                   mlx_vector_array *out_wells,
                                   mlx_vector_array *out_seismic);

/* Fill an `MLXBatch` struct with deep-copied per-scale vectors for the
 * specified sample index. The caller receives ownership of the returned
 * vector arrays and must free them with `mlx_vector_array_free()`.
 * Returns 0 on success. */
int mlx_pyramids_dataset_get_batch_struct(MLXPyramidsDataset *ds, int index,
                                          MLXBatch *out);

/* Generate scales as a flat int array in NHWC order (Batch, Height, Width,
 * Channels). On success returns 0 and sets `*out_shapes` to a malloc'd
 * array of length `4 * *out_n` which the caller must free. */
int mlx_generate_scales_flat(MLXPyramidsDataset *ds, int channels_last);

/* Generate per-scale stacked pyramids for facies, wells and seismic.
 * On success allocates three arrays of `mlx_array` of length `ds->n_scales`
 * and returns them via the `out_*` pointers. The caller is responsible for
 * freeing each `mlx_array` with `mlx_array_free()` and the arrays with
 * `free()` when done. If `ds->n_scales == 0` the function returns 0 and
 * sets the outputs to NULL. */
int mlx_pyramids_dataset_generate_pyramids(MLXPyramidsDataset *ds,
                                           mlx_array **out_facies,
                                           mlx_array **out_wells,
                                           mlx_array **out_seismic);

/* Dump all per-sample per-scale arrays (facies, wells, seismic) into an
 * .npz archive at `npz_path`. Member names follow the pattern
 * `sample_<i>/facies_<s>.npy`, `sample_<i>/wells_<s>.npy`, etc. Returns 0
 * on success. Caller provides path buffer. */
int mlx_pyramids_dataset_dump_batches_npz(MLXPyramidsDataset *ds,
                                          const char *npz_path);

#ifdef __cplusplus
}
#endif

#endif
