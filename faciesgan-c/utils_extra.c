#include "utils_extra.h"

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <mlx/c/array.h>

int mlx_array_to_float_buffer(const mlx_array a, float **out_buf, size_t *out_elems, int *out_ndim, int **out_shape)
{
    if (!out_buf || !out_elems)
        return -1;
    /* Ensure data is available on host */
    bool ok_avail = false;
    if (_mlx_array_is_available(&ok_avail, a) != 0 || !ok_avail)
        return -1;
    if (mlx_array_dtype(a) != MLX_FLOAT32)
        return -1;
    size_t elems = (size_t)mlx_array_size(a);
    const float *data = mlx_array_data_float32(a);
    if (!data)
        return -1;
    float *buf = (float *)malloc(sizeof(float) * elems);
    if (!buf)
        return -1;
    memcpy(buf, data, sizeof(float) * elems);
    *out_buf = buf;
    *out_elems = elems;
    if (out_ndim)
        *out_ndim = mlx_array_ndim(a);
    if (out_shape)
    {
        int ndim = mlx_array_ndim(a);
        int *shape = NULL;
        if (ndim > 0)
        {
            shape = (int *)malloc(sizeof(int) * ndim);
            if (!shape)
            {
                free(buf);
                return -1;
            }
            const int *s = mlx_array_shape(a);
            for (int i = 0; i < ndim; ++i)
                shape[i] = s[i];
        }
        *out_shape = shape;
    }
    return 0;
}

int mlx_array_from_float_buffer(mlx_array *out, const float *buf, const int *shape, int ndim)
{
    if (!out || !buf)
        return -1;
    /* mlx_array_set_data copies the buffer, so we can pass a malloc'd copy then free it */
    size_t elems = 1;
    for (int i = 0; i < ndim; ++i)
        elems *= (size_t)shape[i];
    float *tmp = (float *)malloc(sizeof(float) * elems);
    if (!tmp)
        return -1;
    memcpy(tmp, buf, sizeof(float) * elems);
    int rc = mlx_array_set_data(out, tmp, shape, ndim, MLX_FLOAT32);
    free(tmp);
    return rc == 0 ? 0 : -1;
}

void quantize_pixels_float(const float *in_pixels, float *out_pixels, size_t npixels, int c, const float *palette, int ncolors)
{
    if (!in_pixels || !out_pixels || !palette)
        return;
    for (size_t i = 0; i < npixels; ++i)
    {
        const float *px = in_pixels + (size_t)i * c;
        int best = 0;
        float best_dist = FLT_MAX;
        for (int p = 0; p < ncolors; ++p)
        {
            const float *col = palette + (size_t)p * c;
            float d = 0.0f;
            for (int k = 0; k < c; ++k)
            {
                float diff = px[k] - col[k];
                d += diff * diff;
            }
            if (d < best_dist)
            {
                best_dist = d;
                best = p;
            }
        }
        const float *sel = palette + (size_t)best * c;
        for (int k = 0; k < c; ++k)
            out_pixels[i * c + k] = sel[k];
    }
}

void apply_well_mask_cpu(const float *facies, float *out, int h, int w, int c, const unsigned char *mask, const float *well, int wc)
{
    if (!facies || !out || !mask || !well)
        return;
    size_t npixels = (size_t)h * (size_t)w;
    /* Copy input to output */
    for (size_t i = 0; i < npixels * (size_t)c; ++i)
        out[i] = facies[i];

    /* Compute well_columns */
    unsigned char *col_has = (unsigned char *)malloc((size_t)w);
    if (!col_has)
        return;
    memset(col_has, 0, (size_t)w);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            if (mask[y * w + x])
                col_has[x] = 1;
        }
    }

    /* Replace entire well columns with 1.0 */
    for (int x = 0; x < w; ++x)
    {
        if (col_has[x])
        {
            for (int y = 0; y < h; ++y)
            {
                for (int ch = 0; ch < c; ++ch)
                {
                    out[(y * w + x) * c + ch] = 1.0f;
                }
            }
        }
    }

    /* Now apply well pixels where mask true and well pixel not near-black */
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int idx = y * w + x;
            if (!mask[idx])
                continue;
            /* compute well pixel brightness */
            float brightness = 0.0f;
            if (wc == c)
            {
                for (int ch = 0; ch < c; ++ch)
                    brightness += fabsf(well[idx * wc + ch]);
                brightness = brightness / (float)c;
            }
            else if (wc == 1)
            {
                brightness = fabsf(well[idx]);
            }
            if (brightness < 0.3f)
                continue; /* near black, skip */
            /* apply well pixel */
            if (wc == c)
            {
                for (int ch = 0; ch < c; ++ch)
                    out[idx * c + ch] = well[idx * wc + ch];
            }
            else if (wc == 1)
            {
                for (int ch = 0; ch < c; ++ch)
                    out[idx * c + ch] = well[idx];
            }
        }
    }

    free(col_has);
}

int mlx_denorm_array(const mlx_array in, mlx_array *out, int ceiling)
{
    if (!out)
        return -1;
    /* Ensure evaluated and float32 */
    bool ok_avail = false;
    if (_mlx_array_is_available(&ok_avail, in) != 0 || !ok_avail)
        return -1;
    if (mlx_array_dtype(in) != MLX_FLOAT32)
        return -1;
    size_t elems = mlx_array_size(in);
    const float *data = mlx_array_data_float32(in);
    if (!data)
        return -1;
    float *buf = (float *)malloc(sizeof(float) * elems);
    if (!buf)
        return -1;
    for (size_t i = 0; i < elems; ++i)
    {
        float v = (data[i] + 1.0f) * 0.5f;
        if (v < 0.0f)
            v = 0.0f;
        if (v > 1.0f)
            v = 1.0f;
        if (ceiling && v > 0.0f)
            v = 1.0f;
        buf[i] = v;
    }
    int ndim = mlx_array_ndim(in);
    const int *shape = mlx_array_shape(in);
    /* Create new MLX array from host data */
    mlx_array out_arr = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
    *out = out_arr;
    free(buf);
    return 0;
}

int mlx_quantize_array(const mlx_array in, mlx_array *out, const mlx_array palette)
{
    if (!out)
        return -1;
    /* Convert input and palette to host buffers */
    float *in_buf = NULL;
    size_t in_elems = 0;
    int in_ndim = 0;
    int *in_shape = NULL;
    if (mlx_array_to_float_buffer(in, &in_buf, &in_elems, &in_ndim, &in_shape) != 0)
        return -1;
    float *pal_buf = NULL;
    size_t pal_elems = 0;
    int pal_ndim = 0;
    int *pal_shape = NULL;
    if (mlx_array_to_float_buffer(palette, &pal_buf, &pal_elems, &pal_ndim, &pal_shape) != 0)
    {
        free(in_buf);
        if (in_shape)
            free(in_shape);
        return -1;
    }
    /* Determine channels and pixels */
    int c = 1;
    int h = 1, w = 1;
    if (in_ndim == 3)
    {
        h = in_shape[0];
        w = in_shape[1];
        c = in_shape[2];
    }
    else if (in_ndim == 2)
    {
        h = in_shape[0];
        w = in_shape[1];
        c = 1;
    }
    else
    {
        /* Unsupported shape */
        free(in_buf);
        free(pal_buf);
        if (in_shape)
            free(in_shape);
        if (pal_shape)
            free(pal_shape);
        return -1;
    }
    size_t npixels = (size_t)h * (size_t)w;
    int ncolors = (int)(pal_elems / (size_t)c);
    float *out_buf = (float *)malloc(sizeof(float) * npixels * (size_t)c);
    if (!out_buf)
    {
        free(in_buf);
        free(pal_buf);
        if (in_shape)
            free(in_shape);
        if (pal_shape)
            free(pal_shape);
        return -1;
    }
    quantize_pixels_float(in_buf, out_buf, npixels, c, pal_buf, ncolors);
    /* Create output MLX array */
    int shape_out[3];
    int ndim_out = 0;
    if (c == 1)
    {
        shape_out[0] = h;
        shape_out[1] = w;
        ndim_out = 2;
    }
    else
    {
        shape_out[0] = h;
        shape_out[1] = w;
        shape_out[2] = c;
        ndim_out = 3;
    }
    mlx_array out_arr = mlx_array_new_data(out_buf, shape_out, ndim_out, MLX_FLOAT32);
    *out = out_arr;
    free(in_buf);
    free(pal_buf);
    if (in_shape)
        free(in_shape);
    if (pal_shape)
        free(pal_shape);
    free(out_buf);
    return 0;
}

int mlx_apply_well_mask_array(const mlx_array facies, mlx_array *out, const mlx_array mask, const mlx_array well)
{
    if (!out)
        return -1;
    /* Convert facies to float buffer */
    float *fac_buf = NULL;
    size_t fac_elems = 0;
    int fac_ndim = 0;
    int *fac_shape = NULL;
    if (mlx_array_to_float_buffer(facies, &fac_buf, &fac_elems, &fac_ndim, &fac_shape) != 0)
        return -1;
    /* Convert mask to uint8 host buffer */
    bool ok_mask = false;
    if (_mlx_array_is_available(&ok_mask, mask) != 0 || !ok_mask)
    {
        return -1;
    }
    int m_dtype = mlx_array_dtype(mask);
    size_t mask_elems = mlx_array_size(mask);
    unsigned char *mask_buf = (unsigned char *)malloc(mask_elems);
    if (!mask_buf)
    {
        free(fac_buf);
        if (fac_shape)
            free(fac_shape);
        return -1;
    }
    if (m_dtype == MLX_BOOL)
    {
        const bool *mb = mlx_array_data_bool(mask);
        for (size_t i = 0; i < mask_elems; ++i)
            mask_buf[i] = mb[i] ? 1 : 0;
    }
    else if (m_dtype == MLX_UINT8)
    {
        const uint8_t *mb = mlx_array_data_uint8(mask);
        for (size_t i = 0; i < mask_elems; ++i)
            mask_buf[i] = mb[i];
    }
    else
    {
        free(fac_buf);
        if (fac_shape)
            free(fac_shape);
        free(mask_buf);
        return -1;
    }
    /* Convert well to float buffer */
    float *well_buf = NULL;
    size_t well_elems = 0;
    int well_ndim = 0;
    int *well_shape = NULL;
    if (mlx_array_to_float_buffer(well, &well_buf, &well_elems, &well_ndim, &well_shape) != 0)
    {
        free(fac_buf);
        if (fac_shape)
            free(fac_shape);
        free(mask_buf);
        return -1;
    }
    /* Determine dimensions */
    int h = 1, w = 1, c = 1, wc = 1;
    if (fac_ndim == 3)
    {
        h = fac_shape[0];
        w = fac_shape[1];
        c = fac_shape[2];
    }
    else if (fac_ndim == 2)
    {
        h = fac_shape[0];
        w = fac_shape[1];
        c = 1;
    }
    if (well_ndim == 3)
    {
        wc = well_shape[2];
    }
    else if (well_ndim == 2)
    {
        wc = 1;
    }
    float *out_buf = (float *)malloc(sizeof(float) * fac_elems);
    if (!out_buf)
    {
        free(fac_buf);
        if (fac_shape)
            free(fac_shape);
        free(mask_buf);
        free(well_buf);
        if (well_shape)
            free(well_shape);
        return -1;
    }
    apply_well_mask_cpu(fac_buf, out_buf, h, w, c, mask_buf, well_buf, wc);
    /* Create MLX array from out_buf */
    int ndim_out = fac_ndim;
    const int *shape_in = mlx_array_shape(facies);
    mlx_array out_arr = mlx_array_new_data(out_buf, shape_in, ndim_out, MLX_FLOAT32);
    *out = out_arr;
    /* Cleanup */
    free(fac_buf);
    if (fac_shape)
        free(fac_shape);
    free(mask_buf);
    free(well_buf);
    if (well_shape)
        free(well_shape);
    free(out_buf);
    return 0;
}
