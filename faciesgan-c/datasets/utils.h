#ifndef FACIESGAN_DATASETS_UTILS_H
#define FACIESGAN_DATASETS_UTILS_H

#include "options.h"
#include <mlx/c/array.h>
#include <stddef.h>

typedef struct DatasetScale {
  int batch;
  int channels;
  int height;
  int width;
} DatasetScale;

/* Generate scales from TrainningOptions. Returns 0 on success and fills
 * `out` with a malloc'd array of DatasetScale; caller must free(*out).
 */
int dataset_generate_scales(const TrainningOptions *opts, int channels_last,
                            DatasetScale **out, int *out_count);

/* List image files in a data subdirectory (e.g., "facies", "wells", "seismic").
 * Returns 0 on success and fills `files` with a malloc'd array of C strings
 * (each strdup'd). Caller must free each string and then free(files).
 */
int datasets_list_image_files(const char *data_root, const char *subdir,
                              char ***files, int *count);

/* List model files (e.g., '*.pt') under a subdir. Similar ownership as above.
 */
int datasets_list_model_files(const char *data_root, const char *subdir,
                              char ***files, int *count);

/*
 * Convert a list of DatasetScale descriptors into per-scale MLX arrays for
 * facies images. On success this allocates an array of `mlx_array` of length
 * `*out_count` and stores it in `*out`. The caller must free each
 * `mlx_array` with `mlx_array_free()` and then free(*out). Returns 0 on
 * success or -1 on error.
 */
int to_facies_pyramids(const TrainningOptions *opts, int channels_last,
                       DatasetScale *scales, int n_scales, mlx_array **out,
                       int *out_count);

int to_seismic_pyramids(const TrainningOptions *opts, int channels_last,
                        DatasetScale *scales, int n_scales, mlx_array **out,
                        int *out_count);

int to_wells_pyramids(const TrainningOptions *opts, int channels_last,
                      DatasetScale *scales, int n_scales, mlx_array **out,
                      int *out_count);

/*
 * Generate image pyramids for a single image and save each level as
 * `{sample_dir}/{prefix}_{level}.npy`. Returns 0 on success, -1 on error.
 */
int datasets_generate_and_save_pyramids(const char *img_path,
                                        const char *sample_dir,
                                        const char *prefix,
                                        int num_img_channels, int stop_scale,
                                        int crop_size);

#endif
