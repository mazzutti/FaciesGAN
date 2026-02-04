#ifndef FACIESGAN_DATASETS_FUNC_CACHE_H
#define FACIESGAN_DATASETS_FUNC_CACHE_H

#include "utils.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ensure a function-level .npz cache exists for given generator parameters.
 * - `input_path`: dataset root
 * - `cache_dir`: directory to store cache (.npz) files
 * - `desired_num`: requested number of samples
 * - `stop_scale`, `crop_size`, `num_img_channels`, `use_wells`, `use_seismic`,
 * `manual_seed`: generator params
 * - `out_cache_npz`/`out_len`: output buffer for cache npz path
 * - `out_num_samples`: actual number of samples available (<= desired_num)
 * Returns 0 on success, non-zero on failure.
 */
int ensure_function_cache(const char *input_path, const char *cache_dir,
                          int desired_num, const struct DatasetScale *scales,
                          int n_scales, int num_img_channels, int use_wells,
                          int use_seismic, int manual_seed, char *out_cache_npz,
                          size_t out_len, int *out_num_samples);

/* In-tree C generator: generate pyramids cache into sample_<i> dirs. Returns 0
 * on success. Declared here so callers can use the in-tree generator when the
 * standalone `gen_cache` tool is not available. */
int generate_pyramids_cache(const char *input_path, const char *cache_dir,
                            int num_samples, int stop_scale, int crop_size,
                            int num_img_channels, int use_wells,
                            int use_seismic, int seed);

#ifdef __cplusplus
}
#endif

#endif
