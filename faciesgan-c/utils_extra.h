#ifndef FACIESGAN_C_UTILS_EXTRA_H
#define FACIESGAN_C_UTILS_EXTRA_H

#include <stddef.h>
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* Copy MLX array contents to a newly-allocated float buffer.
     * Caller must free(*out_buf) when done.
     * Returns 0 on success, -1 on error.
     */
    int mlx_array_to_float_buffer(const mlx_array a, float **out_buf, size_t *out_elems, int *out_ndim, int **out_shape);

    /* Create an MLX array from a host float buffer with given shape/dimensions.
     * The function copies `buf` into MLX internal storage. Returns 0 on success.
     */
    int mlx_array_from_float_buffer(mlx_array *out, const float *buf, const int *shape, int ndim);

    /* CPU helper: quantize `in_pixels` (H*W*C) to nearest colors in `palette`.
     * - `in_pixels` and `out_pixels` are length (h*w*c) floats.
     * - `palette` is (ncolors * c) floats.
     */
    void quantize_pixels_float(const float *in_pixels, float *out_pixels, size_t npixels, int c, const float *palette, int ncolors);

    /* CPU helper: apply well mask (boolean mask) onto facies image.
     * - `facies` and `out` are length (h*w*c) floats.
     * - `well` is (h*w*wc) floats (wc may equal c or be 1).
     * - `mask` is length (h*w) bytes (0/1).
     */
    void apply_well_mask_cpu(const float *facies, float *out, int h, int w, int c, const unsigned char *mask, const float *well, int wc);

#ifdef __cplusplus
}
#endif

/* MLX-aware helpers: operate on mlx_array inputs and produce mlx_array outputs.
 * These call the CPU helpers after copying data to host where necessary.
 * Return 0 on success, -1 on error.
 */
int mlx_denorm_array(const mlx_array in, mlx_array *out, int ceiling);
int mlx_quantize_array(const mlx_array in, mlx_array *out, const mlx_array palette);
int mlx_apply_well_mask_array(const mlx_array facies, mlx_array *out, const mlx_array mask, const mlx_array well);

#endif /* FACIESGAN_C_UTILS_EXTRA_H */
