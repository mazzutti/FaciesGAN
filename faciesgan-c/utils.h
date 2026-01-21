#ifndef FACIESGAN_C_UTILS_H
#define FACIESGAN_C_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* MLX types used by helpers */
#include <mlx/c/array.h>
#include <mlx/c/stream.h>

/* Project options type used by write_options_json. Include the definition
 * here so callers of utils.h see the TrainningOptions type. */
#include "options.h"

    /* Create directory and parents (mkdir -p). Returns 0 on success, -1 on error. */
    int mlx_create_dirs(const char *path);

    /* Set seed for C library RNGs. */
    void mlx_set_seed(int seed);

    /* Clamp MLX array values in range [min_val, max_val].
     * Equivalent to: min(max(a, min_val), max_val)
     * Returns 0 on success, non-zero MLX error code on failure.
     */
    int mlx_clamp(mlx_array *res, const mlx_array a, float min_val, float max_val, const mlx_stream s);

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

    /* Write a detailed options JSON file under `outdir/options.json`.
     * `topt` provides most values; `orig_output_path` is the user-supplied
     * output path before timestamping. `wells_mask_columns`/`wells_mask_count`
     * provide the optional wells mask list.
     */
    /* Write `options.json` under `topt->output_path` using the values inside
     * `topt`. `wells_mask_columns`/`wells_mask_count` remain optional and are
     * provided separately. This simplifies the call site: callers need only
     * pass the populated `TrainningOptions` and an optional wells mask. */
    void write_options_json(const TrainningOptions *topt,
                            const int *wells_mask_columns,
                            size_t wells_mask_count);

#ifdef __cplusplus
}
#endif

#endif /* FACIESGAN_C_UTILS_H */
