#ifndef FACIESGAN_C_UTILS_H
#define FACIESGAN_C_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MLX types used by helpers */
#include <mlx/c/array.h>
#include <mlx/c/stream.h>

/* Project options type used by write_options_json. Include the definition
 * here so callers of utils.h see the TrainningOptions type. */
#include "options.h"

/* Create directory and parents (mkdir -p). Returns 0 on success, -1 on error.
int mlx_create_dirs(const char *path);

/* Set seed for C library RNGs. */
void mlx_set_seed(int seed);

/* Clamp MLX array values in range [min_val, max_val].
 * Equivalent to: min(max(a, min_val), max_val)
 * Returns 0 on success, non-zero MLX error code on failure.
int mlx_clamp(mlx_array *res, const mlx_array a, float min_val, float max_val,
              const mlx_stream s);

/* Path and timestamp buffer sizes used across the C launcher. */
#define PATH_BUFSZ 4096
#define TIMESTAMP_BUFSZ 128

#define OPT_FILE "options.json"

/* Fill `buf` with a timestamp in the form YYYY_MM_DD_HH_MM_SS. */
void format_timestamp(char *buf, size_t bufsz);

/* Return JSON boolean string for `v` ("true"/"false"). */
const char *bool_str(int v);

/* Join two path components into `dst` using '/' appropriately. */
void join_path(char *dst, size_t dstsz, const char *a, const char *b);

/* Create directory and parents like mkdir -p, tolerant of existing dirs. */
void ensure_dir(const char *path);

/* Write `options.json` under `topt->output_path` using the values inside
 * `topt`. `wells_mask_columns`/`wells_mask_count` remain optional and are
 * provided separately. This simplifies the call site: callers need only
 * pass the populated `TrainningOptions` and an optional wells mask. */
void write_options_json(const TrainningOptions *topt,
                        const int *wells_mask_columns, size_t wells_mask_count);

/* Additional MLX-host helpers (previously in utils_extra.h) */

/* Copy MLX array contents to a newly-allocated float buffer.
 * Caller must free(*out_buf) when done. Returns 0 on success, -1 on error.
int mlx_array_to_float_buffer(const mlx_array a, float **out_buf,
                              size_t *out_elems, int *out_ndim,
                              int **out_shape);

/* Create an MLX array from a host float buffer with given shape/dimensions.
 * The function copies `buf` into MLX internal storage. Returns 0 on success.
int mlx_array_from_float_buffer(mlx_array *out, const float *buf,
                                const int *shape, int ndim);

/* CPU helper: quantize `in_pixels` (H*W*C) to nearest colors in `palette`.
 * - `in_pixels` and `out_pixels` are length (h*w*c) floats.
 * - `palette` is (ncolors * c) floats.
void quantize_pixels_float(const float *in_pixels, float *out_pixels,
                           size_t npixels, int c, const float *palette,
                           int ncolors);

/* CPU helper: apply well mask (boolean mask) onto facies image.
 * - `facies` and `out` are length (h*w*c) floats.
 * - `well` is (h*w*wc) floats (wc may equal c or be 1).
 * - `mask` is length (h*w) bytes (0/1).
void apply_well_mask_cpu(const float *facies, float *out, int h, int w, int c,
                         const unsigned char *mask, const float *well, int wc);

/* MLX-aware helpers: operate on mlx_array inputs and produce mlx_array outputs.
 * Return 0 on success, -1 on error.
int mlx_denorm_array(const mlx_array in, mlx_array *out, int ceiling);
int mlx_quantize_array(const mlx_array in, mlx_array *out,
                       const mlx_array palette);
int mlx_apply_well_mask_array(const mlx_array facies, mlx_array *out,
                              const mlx_array mask, const mlx_array well);

/* Save an mlx_array image directly as PNG.
 * Array should be shape (H,W,C) or (1,H,W,C) with float32 values in [0,1].
 * Returns 0 on success, -1 on error.
int mlx_save_png(const char *path, mlx_array arr);

/* Save fake and real facies side-by-side as a single PNG image.
 * Both arrays should be (1,H,W,C) or (H,W,C) with float32 values in [0,1].
 * Creates a visualization with real on left, fake on right.
 * Returns 0 on success, -1 on error.
int mlx_save_facies_comparison_png(const char *path, mlx_array fake,
                                   mlx_array real);

/* Save facies grid visualization with multiple generated samples.
 * Creates a grid with rows=batch_size, cols=1+num_fake (real + fakes).
 * Applies color quantization to the 4-color facies palette.
 * - fake_samples: array of num_fake mlx_arrays (each: batch x H x W x C)
 * - real: single mlx_array (batch x H x W x C)
 * - cell_size: size of each cell in the grid (pixels)
 * - scale, epoch: used for labeling (reserved for future use)
 * Returns 0 on success, -1 on error.
int mlx_save_facies_grid_png(const char *path, mlx_array *fake_samples,
                             int num_fake, mlx_array real, int cell_size,
                             int scale, int epoch);

/* Save facies grid visualization with proper num_real_facies x num_generated_per_real layout.
 * Creates a grid matching Python's visualization:
 * - Rows = num_real (randomly selected real facies from batch)
 * - Cols = 1 (real) + num_gen_per_real (generated samples)
 * Applies color quantization to the 4-color facies palette.
 * - all_fakes: array of total_gen mlx_arrays (total_gen = num_real * num_gen_per_real)
 * - real: single mlx_array (batch x H x W x C)
 * - selected_indices: array of num_real indices into the batch
 * - masks: optional mlx_array for well masks (batch x H x W x 1)
 * Returns 0 on success, -1 on error.
int mlx_save_facies_grid_png_v2(const char *path, mlx_array *all_fakes,
                                int total_gen, mlx_array real,
                                const int *selected_indices, int num_real,
                                int num_gen_per_real, int cell_size,
                                int scale, int epoch, mlx_array masks);

#ifdef __cplusplus
}
#endif

#endif /* FACIESGAN_C_UTILS_H */
