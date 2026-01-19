/*
 * Pure-C port implementation for custom layers. This file is intentionally
 * conservative: it contains implementations that call into the expected
 * mlx-c C API. The exact symbol names for creating layers / operators
 * may differ depending on the mlx-c release; adapt the function calls
 * below to the installed mlx-c public headers.
 *
 * The implementations allocate and return new mlx_array_t* objects for
 * forward calls. Memory ownership and lifetime semantics follow a simple
 * convention: forward functions return a newly retained array pointer
 * (or the same pointer passed through when not transformed). The user
 * of this library must adapt these to their memory management model.
 */

#include "custom_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mlx/c/mlx.h>

/* Ensure common string prototypes are visible */
#include <stddef.h>

/* Project helpers */
#include "utils_extra.h"

/* (Instrumentation removed) */

struct MLXLeakyReLU
{
    float negative_slope;
    /* cached scalar arrays to avoid reallocating per-forward call */
    mlx_array slope_scalar; /* holds negative_slope */
    mlx_array two_scalar;   /* holds constant 2.0 */
    int has_cached_scalars;
};

MLXLeakyReLU *mlx_leakyrelu_create(float negative_slope)
{
    MLXLeakyReLU *m = (MLXLeakyReLU *)malloc(sizeof(MLXLeakyReLU));
    if (!m)
        return NULL;
    m->negative_slope = negative_slope;
    m->has_cached_scalars = 0;
    /* create scalar caches */
    m->slope_scalar = mlx_array_new_float(negative_slope);
    m->two_scalar = mlx_array_new_float(2.0f);
    m->has_cached_scalars = 1;
    return m;
}

void mlx_leakyrelu_free(MLXLeakyReLU *m)
{
    if (!m)
        return;
    if (m->has_cached_scalars)
    {
        mlx_array_free(m->slope_scalar);
        mlx_array_free(m->two_scalar);
    }
    free(m);
}

mlx_array_t mlx_leakyrelu_forward(MLXLeakyReLU *m, mlx_array_t x)
{
    /* Implement leaky-relu using mlx-c elementwise ops.
     * out = max(x, x * negative_slope) via (a + b + sqrt((a-b)^2)) / 2
     */
    if (!m)
        return x;

    mlx_stream s = mlx_default_cpu_stream_new();

    mlx_array scaled = mlx_array_new();
    mlx_array diff = mlx_array_new();
    mlx_array diff_sq = mlx_array_new();
    mlx_array absdiff = mlx_array_new();
    mlx_array sum = mlx_array_new();
    mlx_array tmp = mlx_array_new();
    mlx_array out = mlx_array_new();

    int err = 0;

    if (mlx_multiply(&scaled, x, m->slope_scalar, s) != 0)
    {
        err = 1;
        goto cleanup;
    }

    if (mlx_subtract(&diff, x, scaled, s) != 0)
    {
        err = 1;
        goto cleanup;
    }

    if (mlx_square(&diff_sq, diff, s) != 0)
    {
        err = 1;
        goto cleanup;
    }

    if (mlx_sqrt(&absdiff, diff_sq, s) != 0)
    {
        err = 1;
        goto cleanup;
    }

    if (mlx_add(&sum, x, scaled, s) != 0)
    {
        err = 1;
        goto cleanup;
    }

    if (mlx_add(&tmp, sum, absdiff, s) != 0)
    {
        err = 1;
        goto cleanup;
    }

    if (mlx_divide(&out, tmp, m->two_scalar, s) != 0)
    {
        err = 1;
        goto cleanup;
    }

cleanup:
    /* Free temporaries if they were allocated (ctx==0 indicates empty) */
    if (scaled.ctx)
        mlx_array_free(scaled);
    if (diff.ctx)
        mlx_array_free(diff);
    if (diff_sq.ctx)
        mlx_array_free(diff_sq);
    if (absdiff.ctx)
        mlx_array_free(absdiff);
    if (sum.ctx)
        mlx_array_free(sum);
    if (tmp.ctx)
        mlx_array_free(tmp);

    /* Free the stream we created */
    mlx_stream_free(s);

    if (err)
    {
        /* If something failed, return input unchanged (caller owns input) */
        if (out.ctx)
            mlx_array_free(out);
        return x;
    }

    /* success: return computed output (caller must free it) */
    return out;
}

/* Lightweight InstanceNorm handle used when code expects an mlx_nn-style API.
 * This mirrors the Python InstanceNorm behavior by storing optional affine
 * parameters (scale and bias). The actual normalization used in forward
 * implementations in this file is done manually, so this handle only
 * provides allocation/free and optional parameter storage for later use.
 */
typedef struct mlx_nn_instancenorm_
{
    int num_features;
    int affine;
    mlx_array *weight; /* 1-D array of length num_features (scale), may be NULL */
    mlx_array *bias;   /* 1-D array of length num_features (offset), may be NULL */
} mlx_nn_instancenorm;

void *mlx_nn_instancenorm_create(int num_features, int affine)
{
    mlx_nn_instancenorm *h = (mlx_nn_instancenorm *)malloc(sizeof(mlx_nn_instancenorm));
    if (!h)
        return NULL;
    h->num_features = num_features;
    h->affine = affine ? 1 : 0;
    h->weight = NULL;
    h->bias = NULL;

    if (h->affine)
    {
        /* allocate weight initialized to 1.0 and bias initialized to 0.0 */
        int shape[1] = {num_features};
        float *wbuf = (float *)malloc(sizeof(float) * (size_t)num_features);
        float *bbuf = (float *)malloc(sizeof(float) * (size_t)num_features);
        if (!wbuf || !bbuf)
        {
            free(wbuf);
            free(bbuf);
            free(h);
            return NULL;
        }
        for (int i = 0; i < num_features; ++i)
        {
            wbuf[i] = 1.0f;
            bbuf[i] = 0.0f;
        }
        mlx_array warr = mlx_array_new_data(wbuf, shape, 1, MLX_FLOAT32);
        mlx_array barr = mlx_array_new_data(bbuf, shape, 1, MLX_FLOAT32);
        free(wbuf);
        free(bbuf);
        h->weight = (mlx_array *)malloc(sizeof(mlx_array));
        h->bias = (mlx_array *)malloc(sizeof(mlx_array));
        if (!h->weight || !h->bias)
        {
            if (h->weight)
            {
                *h->weight = warr;
            }
            else
                mlx_array_free(warr);
            if (h->bias)
            {
                *h->bias = barr;
            }
            else
                mlx_array_free(barr);
            if (h->weight)
                free(h->weight);
            if (h->bias)
                free(h->bias);
            free(h);
            return NULL;
        }
        *h->weight = warr;
        *h->bias = barr;
    }

    return (void *)h;
}

void mlx_nn_instancenorm_free(void *handle)
{
    if (!handle)
        return;
    mlx_nn_instancenorm *h = (mlx_nn_instancenorm *)handle;
    if (h->weight)
    {
        mlx_array_free(*h->weight);
        free(h->weight);
        h->weight = NULL;
    }
    if (h->bias)
    {
        mlx_array_free(*h->bias);
        free(h->bias);
        h->bias = NULL;
    }
    free(h);
}

struct MLXConvBlock
{
    int use_norm;
    void *conv; /* pointer to mlx_array weight (allocated) */
    void *norm; /* underlying instance norm handle */
    MLXLeakyReLU *activation;
    int stride;
    int padding;
    int kernel_size;
    int in_ch;
    int out_ch;
};

MLXConvBlock *mlx_convblock_create(int in_ch, int out_ch, int kernel_size, int padding, int stride, int use_norm)
{
    MLXConvBlock *m = (MLXConvBlock *)malloc(sizeof(MLXConvBlock));
    if (!m)
        return NULL;
    m->use_norm = use_norm;
    m->conv = NULL; /* will store pointer to mlx_array weight */
    m->norm = NULL; /* create instance norm if requested */
    if (use_norm)
    {
        /* Create an mlx-c InstanceNorm module handle if available.
         * Signature assumed: void* mlx_nn_instancenorm_create(int dims, int affine);
         * Adjust if your mlx-c version uses a different name or signature. */
        m->norm = (void *)mlx_nn_instancenorm_create(out_ch, 1);
    }
    m->activation = mlx_leakyrelu_create(0.2f);
    m->stride = stride;
    m->padding = padding;
    m->kernel_size = kernel_size;
    m->in_ch = in_ch;
    m->out_ch = out_ch;

    /* Allocate weight buffer (C_out, KH, KW, C_in) and initialize to zeros */
    size_t wcount = (size_t)out_ch * (size_t)kernel_size * (size_t)kernel_size * (size_t)in_ch;
    float *wbuf = (float *)calloc(wcount, sizeof(float));
    if (wbuf)
    {
        int wshape[4] = {out_ch, kernel_size, kernel_size, in_ch};
        mlx_array w = mlx_array_new_data(wbuf, wshape, 4, MLX_FLOAT32);
        /* copy made by mlx_array_new_data; free host buffer */
        free(wbuf);
        mlx_array *wptr = (mlx_array *)malloc(sizeof(mlx_array));
        if (wptr)
        {
            *wptr = w;
            m->conv = (void *)wptr;
        }
        else
        {
            mlx_array_free(w);
        }
    }
    return m;
}

void mlx_convblock_free(MLXConvBlock *m)
{
    if (!m)
        return;
    /* free conv weight array if allocated */
    if (m->conv)
    {
        mlx_array *wptr = (mlx_array *)m->conv;
        mlx_array_free(*wptr);
        free(wptr);
    }
    /* free instance norm handle if created (signature: mlx_nn_instancenorm_free(void*)) */
    if (m->norm)
    {
        mlx_nn_instancenorm_free(m->norm);
        m->norm = NULL;
    }
    mlx_leakyrelu_free(m->activation);
    free(m);
}

mlx_array_t mlx_convblock_forward(MLXConvBlock *m, mlx_array_t x)
{
    if (!m)
        return x;
    mlx_stream s = mlx_default_cpu_stream_new();

    /* If conv weights are present, run conv; otherwise pass-through */
    mlx_array y = mlx_array_new();
    if (m->conv)
    {
        mlx_array *wptr = (mlx_array *)m->conv;
        if (mlx_conv2d(&y, x, *wptr, m->stride, m->stride, m->padding, m->padding, 1, 1, 1, s) != 0)
        {
            /* conv failed: fallback to input */
            return x;
        }
    }
    else
    {
        y = x;
    }

    /* Instance normalization (NHWC): normalize across H and W axes */
    if (m->use_norm)
    {
        const int axes[] = {1, 2};
        mlx_array mean = mlx_array_new();
        if (mlx_mean_axes(&mean, y, axes, 2, true, s) == 0)
        {
            mlx_array centered = mlx_array_new();
            if (mlx_subtract(&centered, y, mean, s) == 0)
            {
                mlx_array sq = mlx_array_new();
                if (mlx_square(&sq, centered, s) == 0)
                {
                    mlx_array var = mlx_array_new();
                    if (mlx_mean_axes(&var, sq, axes, 2, true, s) == 0)
                    {
                        mlx_array eps = mlx_array_new_float(1e-5f);
                        mlx_array var_eps = mlx_array_new();
                        if (mlx_add(&var_eps, var, eps, s) == 0)
                        {
                            mlx_array std = mlx_array_new();
                            if (mlx_sqrt(&std, var_eps, s) == 0)
                            {
                                mlx_array y_norm = mlx_array_new();
                                if (mlx_divide(&y_norm, centered, std, s) == 0)
                                {
                                    /* replace y with normalized */
                                    mlx_array_free(y);
                                    y = y_norm;
                                }
                                mlx_array_free(std);
                            }
                            mlx_array_free(var_eps);
                        }
                        mlx_array_free(var);
                    }
                    mlx_array_free(sq);
                }
                mlx_array_free(centered);
            }
            mlx_array_free(mean);
        }
    }

    /* Activation */
    mlx_array activated = mlx_leakyrelu_forward(m->activation, y);
    return activated;
}

struct MLXUpsample
{
    int out_h;
    int out_w;
    char mode[16];
    int align_corners;
    void *internal;
};

MLXUpsample *mlx_upsample_create(int out_h, int out_w, const char *mode, int align_corners)
{
    MLXUpsample *m = (MLXUpsample *)malloc(sizeof(MLXUpsample));
    if (!m)
        return NULL;
    m->out_h = out_h;
    m->out_w = out_w;
    strncpy(m->mode, mode ? mode : "linear", sizeof(m->mode) - 1);
    m->mode[sizeof(m->mode) - 1] = '\0';
    m->align_corners = align_corners;
    m->internal = NULL;
    return m;
}

void mlx_upsample_free(MLXUpsample *m)
{
    if (m)
        free(m);
}

mlx_array_t mlx_upsample_forward(MLXUpsample *m, mlx_array_t x)
{
    if (!m)
        return x;
    mlx_stream s = mlx_default_cpu_stream_new();

    size_t ndim = mlx_array_ndim(x);
    if (ndim != 4)
        return x; /* expect NHWC */
    const int *shape = mlx_array_shape(x);
    int b = shape[0];
    int in_h = shape[1];
    int in_w = shape[2];
    int c = shape[3];

    if (in_h == m->out_h && in_w == m->out_w)
        return x;

    if (m->out_h % in_h != 0 || m->out_w % in_w != 0)
    {
        /* Non-integer scale factors not supported by this simple repeat-based upsample */
        return x;
    }

    int scale_h = m->out_h / in_h;
    int scale_w = m->out_w / in_w;

    int tmp_alloc = 0;
    mlx_array tmp = mlx_array_new();
    if (scale_h > 1)
    {
        if (mlx_repeat_axis(&tmp, x, scale_h, 1, s) != 0)
        {
            return x;
        }
        tmp_alloc = 1;
    }
    else
    {
        /* reuse input as temporary holder (no allocation) */
        tmp = x;
        tmp_alloc = 0;
    }

    mlx_array out = mlx_array_new();
    int out_alloc = 0;
    if (scale_w > 1)
    {
        if (mlx_repeat_axis(&out, tmp, scale_w, 2, s) != 0)
        {
            if (tmp_alloc)
                mlx_array_free(tmp);
            return x;
        }
        out_alloc = 1;
    }
    else
    {
        out = tmp;
        out_alloc = tmp_alloc;
    }

    /* If we used the input as the returned array (no alloc), just return it.
     * Otherwise return the newly allocated `out` and leave caller to free it. */
    return out;
}

struct MLXSPADE
{
    int norm_nc;
    int cond_nc;
    int hidden_nc;
    int kernel_size;
    int padding;
    /* weight arrays for convs: pointers to mlx_array allocated with mlx_array_new_data */
    mlx_array *mlp_shared_w; /* shape: (hidden_nc, K, K, cond_nc) */
    mlx_array *mlp_gamma_w;  /* shape: (norm_nc, K, K, hidden_nc) */
    mlx_array *mlp_beta_w;   /* shape: (norm_nc, K, K, hidden_nc) */
    /* cache omitted for simplicity */
};

MLXSPADE *mlx_spade_create(int norm_nc, int cond_nc, int hidden_nc, int kernel_size)
{
    MLXSPADE *m = (MLXSPADE *)malloc(sizeof(MLXSPADE));
    if (!m)
        return NULL;
    m->norm_nc = norm_nc;
    m->cond_nc = cond_nc;
    m->hidden_nc = hidden_nc;
    m->kernel_size = kernel_size;
    m->padding = kernel_size / 2;
    m->mlp_shared_w = NULL;
    m->mlp_gamma_w = NULL;
    m->mlp_beta_w = NULL;

    /* Allocate zero-initialized weights for the three convs */
    size_t shared_count = (size_t)hidden_nc * kernel_size * kernel_size * cond_nc;
    float *shared_buf = (float *)calloc(shared_count, sizeof(float));
    if (shared_buf)
    {
        int sshape[4] = {hidden_nc, kernel_size, kernel_size, cond_nc};
        mlx_array sw = mlx_array_new_data(shared_buf, sshape, 4, MLX_FLOAT32);

        free(shared_buf);
        mlx_array *swp = (mlx_array *)malloc(sizeof(mlx_array));
        if (swp)
        {
            *swp = sw;
            m->mlp_shared_w = swp;
        }
        else
        {
            mlx_array_free(sw);
        }
    }

    size_t gamma_count = (size_t)norm_nc * kernel_size * kernel_size * hidden_nc;
    float *gamma_buf = (float *)calloc(gamma_count, sizeof(float));
    if (gamma_buf)
    {
        int gshape[4] = {norm_nc, kernel_size, kernel_size, hidden_nc};
        mlx_array gw = mlx_array_new_data(gamma_buf, gshape, 4, MLX_FLOAT32);

        free(gamma_buf);
        mlx_array *gwp = (mlx_array *)malloc(sizeof(mlx_array));
        if (gwp)
        {
            *gwp = gw;
            m->mlp_gamma_w = gwp;
        }
        else
        {
            mlx_array_free(gw);
        }
    }

    size_t beta_count = (size_t)norm_nc * kernel_size * kernel_size * hidden_nc;
    float *beta_buf = (float *)calloc(beta_count, sizeof(float));
    if (beta_buf)
    {
        int bshape[4] = {norm_nc, kernel_size, kernel_size, hidden_nc};
        mlx_array bw = mlx_array_new_data(beta_buf, bshape, 4, MLX_FLOAT32);

        free(beta_buf);
        mlx_array *bwp = (mlx_array *)malloc(sizeof(mlx_array));
        if (bwp)
        {
            *bwp = bw;
            m->mlp_beta_w = bwp;
        }
        else
        {
            mlx_array_free(bw);
        }
    }
    return m;
}

void mlx_spade_free(MLXSPADE *m)
{
    if (!m)
        return;
    if (m->mlp_shared_w)
    {
        mlx_array_free(*m->mlp_shared_w);
        free(m->mlp_shared_w);
    }
    if (m->mlp_gamma_w)
    {
        mlx_array_free(*m->mlp_gamma_w);
        free(m->mlp_gamma_w);
    }
    if (m->mlp_beta_w)
    {
        mlx_array_free(*m->mlp_beta_w);
        free(m->mlp_beta_w);
    }
    free(m);
}

mlx_array_t mlx_spade_forward(MLXSPADE *m, mlx_array_t x, mlx_array_t conditioning_input, const char *mode, int align_corners)
{
    if (!m)
        return x;
    mlx_stream s = mlx_default_cpu_stream_new();

    /* Expect NHWC */
    if (mlx_array_ndim(x) != 4 || mlx_array_ndim(conditioning_input) != 4)
        return x;
    const int *xshape = mlx_array_shape(x);
    const int *cshape = mlx_array_shape(conditioning_input);
    int bx = xshape[0], hx = xshape[1], wx = xshape[2], cx = xshape[3];
    int bc = cshape[0], hc = cshape[1], wc = cshape[2], cc = cshape[3];

    /* Upsample conditioning input to x spatial dims if integer scale factor */
    mlx_array cond_up = mlx_array_new();
    int cond_up_alloc = 0;
    if (hc == hx && wc == wx)
    {
        cond_up = conditioning_input;
    }
    else if (hx % hc == 0 && wx % wc == 0)
    {
        int scale_h = hx / hc;
        int scale_w = wx / wc;
        mlx_array tmp = mlx_array_new();
        if (scale_h > 1)
        {
            if (mlx_repeat_axis(&tmp, conditioning_input, scale_h, 1, s) != 0)
            {
                return x;
            }
        }
        else
        {
            tmp = conditioning_input;
        }
        if (scale_w > 1)
        {
            if (mlx_repeat_axis(&cond_up, tmp, scale_w, 2, s) != 0)
            {
                if (tmp.ctx != conditioning_input.ctx)
                    mlx_array_free(tmp);
                return x;
            }
            cond_up_alloc = 1;
            if (tmp.ctx != conditioning_input.ctx)
                mlx_array_free(tmp);
        }
        else
        {
            cond_up = tmp;
        }
    }
    else
    {
        /* non-integer scaling: fallback to using conditioning_input as-is */
        cond_up = conditioning_input;
    }

    /* mlp_shared: conv(cond_up, mlp_shared_w) */
    mlx_array shared = mlx_array_new();
    if (!m->mlp_shared_w)
    {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        return x;
    }
    if (mlx_conv2d(&shared, cond_up, *m->mlp_shared_w, 1, 1, m->padding, m->padding, 1, 1, 1, s) != 0)
    {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        return x;
    }

    /* activation */
    MLXLeakyReLU *tmp_act = mlx_leakyrelu_create(0.2f);
    mlx_array shared_act = mlx_leakyrelu_forward(tmp_act, shared);
    mlx_leakyrelu_free(tmp_act);

    /* gamma and beta */
    mlx_array gamma = mlx_array_new();
    mlx_array beta = mlx_array_new();
    if (!m->mlp_gamma_w || !m->mlp_beta_w)
    {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_array_free(shared);
        mlx_array_free(shared_act);
        return x;
    }
    if (mlx_conv2d(&gamma, shared_act, *m->mlp_gamma_w, 1, 1, m->padding, m->padding, 1, 1, 1, s) != 0)
    {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_array_free(shared);
        mlx_array_free(shared_act);
        return x;
    }
    if (mlx_conv2d(&beta, shared_act, *m->mlp_beta_w, 1, 1, m->padding, m->padding, 1, 1, 1, s) != 0)
    {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_array_free(shared);
        mlx_array_free(shared_act);
        mlx_array_free(gamma);
        return x;
    }

    if (cond_up_alloc)
        mlx_array_free(cond_up);
    mlx_array_free(shared);
    mlx_array_free(shared_act);

    /* instance normalize x across H and W axes */
    const int axes[] = {1, 2};
    mlx_array mean = mlx_array_new();
    if (mlx_mean_axes(&mean, x, axes, 2, true, s) != 0)
    {
        mlx_array_free(gamma);
        mlx_array_free(beta);
        return x;
    }
    mlx_array centered = mlx_array_new();
    if (mlx_subtract(&centered, x, mean, s) != 0)
    {
        mlx_array_free(mean);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        return x;
    }
    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, centered, s) != 0)
    {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        return x;
    }
    mlx_array var = mlx_array_new();
    if (mlx_mean_axes(&var, sq, axes, 2, true, s) != 0)
    {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        return x;
    }
    mlx_array eps = mlx_array_new_float(1e-5f);
    mlx_array var_eps = mlx_array_new();
    if (mlx_add(&var_eps, var, eps, s) != 0)
    {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        return x;
    }
    mlx_array std = mlx_array_new();
    if (mlx_sqrt(&std, var_eps, s) != 0)
    {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        return x;
    }
    mlx_array x_norm = mlx_array_new();
    if (mlx_divide(&x_norm, centered, std, s) != 0)
    {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        return x;
    }

    /* out = x_norm * (1 + gamma) + beta */
    mlx_array one = mlx_array_new_float(1.0f);
    mlx_array one_plus_gamma = mlx_array_new();
    if (mlx_add(&one_plus_gamma, gamma, one, s) != 0)
    {
        /* cleanup */
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(x_norm);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_array_free(one);
        return x;
    }

    mlx_array scaled = mlx_array_new();
    if (mlx_multiply(&scaled, x_norm, one_plus_gamma, s) != 0)
    {
        /* cleanup */
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(x_norm);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_array_free(one);
        mlx_array_free(one_plus_gamma);
        return x;
    }

    mlx_array out = mlx_array_new();
    if (mlx_add(&out, scaled, beta, s) != 0)
    {
        /* cleanup */
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(x_norm);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_array_free(one);
        mlx_array_free(one_plus_gamma);
        mlx_array_free(scaled);
        return x;
    }

    /* cleanup temporaries */
    mlx_array_free(mean);
    mlx_array_free(centered);
    mlx_array_free(sq);
    mlx_array_free(var);
    mlx_array_free(var_eps);
    mlx_array_free(std);
    mlx_array_free(x_norm);
    mlx_array_free(gamma);
    mlx_array_free(beta);
    mlx_array_free(eps);
    mlx_array_free(one);
    mlx_array_free(one_plus_gamma);
    mlx_array_free(scaled);

    return out;
}

/* SPADE mlp weight accessors */
mlx_array *mlx_spade_get_mlp_shared_w(MLXSPADE *m)
{
    if (!m)
        return NULL;
    return m->mlp_shared_w ? m->mlp_shared_w : NULL;
}
mlx_array *mlx_spade_get_mlp_gamma_w(MLXSPADE *m)
{
    if (!m)
        return NULL;
    return m->mlp_gamma_w ? m->mlp_gamma_w : NULL;
}
mlx_array *mlx_spade_get_mlp_beta_w(MLXSPADE *m)
{
    if (!m)
        return NULL;
    return m->mlp_beta_w ? m->mlp_beta_w : NULL;
}

struct MLXSPADEConvBlock
{
    MLXSPADE *spade;
    void *conv;
    MLXLeakyReLU *activation;
    int stride;
    int padding;
    int kernel_size;
    int in_ch;
    int out_ch;
};

int mlx_spadeconv_get_stride(MLXSPADEConvBlock *m)
{
    if (!m)
        return 1;
    return m->stride;
}

int mlx_spade_get_padding(MLXSPADE *m)
{
    if (!m)
        return 0;
    return m->padding;
}

MLXSPADEConvBlock *mlx_spadeconv_create(int in_ch, int out_ch, int cond_ch, int kernel_size, int padding, int stride, int spade_hidden)
{
    MLXSPADEConvBlock *m = (MLXSPADEConvBlock *)malloc(sizeof(MLXSPADEConvBlock));
    if (!m)
        return NULL;
    m->spade = mlx_spade_create(in_ch, cond_ch, spade_hidden, kernel_size);
    m->conv = NULL; /* will store pointer to mlx_array weight */
    m->activation = mlx_leakyrelu_create(0.2f);
    m->stride = stride;
    m->padding = padding;
    m->kernel_size = kernel_size;
    m->in_ch = in_ch;
    m->out_ch = out_ch;

    /* Allocate conv weights (out_ch, KH, KW, in_ch) initialized to zeros */
    size_t wcount = (size_t)out_ch * (size_t)kernel_size * (size_t)kernel_size * (size_t)in_ch;
    float *wbuf = (float *)calloc(wcount, sizeof(float));
    if (wbuf)
    {
        int wshape[4] = {out_ch, kernel_size, kernel_size, in_ch};
        mlx_array w = mlx_array_new_data(wbuf, wshape, 4, MLX_FLOAT32);
        free(wbuf);
        mlx_array *wptr = (mlx_array *)malloc(sizeof(mlx_array));
        if (wptr)
        {
            *wptr = w;
            m->conv = (void *)wptr;
        }
        else
        {
            mlx_array_free(w);
        }
    }
    return m;
}

void mlx_spadeconv_free(MLXSPADEConvBlock *m)
{
    if (!m)
        return;
    if (m->conv)
    {
        mlx_array *wptr = (mlx_array *)m->conv;
        mlx_array_free(*wptr);
        free(wptr);
    }
    mlx_spade_free(m->spade);
    mlx_leakyrelu_free(m->activation);
    free(m);
}

mlx_array_t mlx_spadeconv_forward(MLXSPADEConvBlock *m, mlx_array_t x, mlx_array_t cond)
{
    if (!m)
        return x;
    mlx_stream s = mlx_default_cpu_stream_new();

    /* 1) SPADE modulation */
    mlx_array y = mlx_spade_forward(m->spade, x, cond, "linear", 1);

    /* 2) Activation */
    mlx_array act = mlx_leakyrelu_forward(m->activation, y);
    /* free y if activation returned a different array */
    if (((void *)&y) != ((void *)&act))
    {
        /* compare internal ctx pointers to avoid freeing the returned array */
        if (y.ctx != act.ctx)
            mlx_array_free(y);
    }
    y = act;

    /* 3) Convolution (if weights present) */
    if (m->conv)
    {
        mlx_array out = mlx_array_new();
        mlx_array *wptr = (mlx_array *)m->conv;
        if (mlx_conv2d(&out, y, *wptr, m->stride, m->stride, m->padding, m->padding, 1, 1, 1, s) != 0)
        {
            /* conv failed: cleanup and return input x */
            mlx_array_free(y);
            return x;
        }
        /* free intermediate activation result */
        mlx_array_free(y);
        return out;
    }

    /* no conv: return activated/spade-modulated tensor */
    return y;
}

struct MLXSPADEGenerator
{
    void *init_conv;     /* pointer to mlx_array weight */
    void **spade_blocks; /* array of pointers to MLXSPADEConvBlock */
    int n_blocks;
    void *tail_conv; /* pointer to mlx_array weight */
    MLXLeakyReLU *leaky_relu;
    void *activation; /* tanh handle (unused) */
    int padding_size;
    int kernel_size;
};

MLXSPADEGenerator *mlx_spadegen_create(int num_layer, int kernel_size, int padding_size, int num_features, int min_num_features, int output_channels, int input_channels)
{
    MLXSPADEGenerator *m = (MLXSPADEGenerator *)malloc(sizeof(MLXSPADEGenerator));
    if (!m)
        return NULL;
    m->init_conv = NULL; /* conv weight pointer */
    int body_count = (num_layer - 2) > 0 ? (num_layer - 2) : 0;
    m->n_blocks = body_count;
    m->spade_blocks = NULL;
    m->tail_conv = NULL;
    m->padding_size = padding_size;
    m->kernel_size = kernel_size;
    /* Allocate init conv weights: (num_features, KH, KW, input_channels) */
    size_t init_count = (size_t)num_features * kernel_size * kernel_size * input_channels;
    float *init_buf = (float *)calloc(init_count, sizeof(float));
    if (init_buf)
    {
        int ishape[4] = {num_features, kernel_size, kernel_size, input_channels};
        mlx_array iw = mlx_array_new_data(init_buf, ishape, 4, MLX_FLOAT32);
        free(init_buf);
        mlx_array *iwptr = (mlx_array *)malloc(sizeof(mlx_array));
        if (iwptr)
        {
            *iwptr = iw;
            m->init_conv = (void *)iwptr;
        }
        else
        {
            mlx_array_free(iw);
        }
    }
    if (body_count > 0)
    {
        m->spade_blocks = (void **)calloc(body_count, sizeof(void *));
        int current_features = num_features;
        for (int i = 0; i < body_count; ++i)
        {
            int denom = (1 << (i + 1));
            int block_features = num_features / denom;
            if (block_features < min_num_features)
                block_features = min_num_features;
            int in_ch = current_features;
            int out_ch = block_features;
            m->spade_blocks[i] = (void *)mlx_spadeconv_create(
                in_ch, out_ch, input_channels, kernel_size, padding_size, 1, 64);
            current_features = out_ch;
        }
        /* Allocate tail conv weights (output_channels, KH, KW, current_features) */
        size_t tail_count = (size_t)output_channels * kernel_size * kernel_size * (size_t)current_features;
        float *tail_buf = (float *)calloc(tail_count, sizeof(float));
        if (tail_buf)
        {
            int tshape[4] = {output_channels, kernel_size, kernel_size, current_features};
            mlx_array tw = mlx_array_new_data(tail_buf, tshape, 4, MLX_FLOAT32);
            free(tail_buf);
            mlx_array *twptr = (mlx_array *)malloc(sizeof(mlx_array));
            if (twptr)
            {
                *twptr = tw;
                m->tail_conv = (void *)twptr;
            }
            else
            {
                mlx_array_free(tw);
            }
        }
    }
    m->leaky_relu = mlx_leakyrelu_create(0.2f);
    m->activation = NULL; /* tanh handle if available */
    return m;
}

void mlx_spadegen_free(MLXSPADEGenerator *m)
{
    if (!m)
        return;
    if (m->spade_blocks)
    {
        for (int i = 0; i < m->n_blocks; ++i)
        {
            mlx_spadeconv_free((MLXSPADEConvBlock *)m->spade_blocks[i]);
        }
        free(m->spade_blocks);
    }
    if (m->init_conv)
    {
        mlx_array *iw = (mlx_array *)m->init_conv;
        mlx_array_free(*iw);
        free(iw);
    }
    if (m->tail_conv)
    {
        mlx_array *tw = (mlx_array *)m->tail_conv;
        mlx_array_free(*tw);
        free(tw);
    }
    mlx_leakyrelu_free(m->leaky_relu);
    free(m);
}

mlx_array_t mlx_spadegen_forward(MLXSPADEGenerator *m, mlx_array_t cond)
{
    if (!m)
        return cond;
    mlx_stream s = mlx_default_cpu_stream_new();

    mlx_array x = mlx_array_new();
    if (m->init_conv)
    {
        mlx_array *iw = (mlx_array *)m->init_conv;
        if (mlx_conv2d(&x, cond, *iw, 1, 1, m->padding_size, m->padding_size, 1, 1, 1, s) != 0)
        {
            return cond;
        }
    }
    else
    {
        /* no init conv: treat cond as x */
        x = cond;
    }

    /* leaky relu */
    mlx_array x_act = mlx_leakyrelu_forward(m->leaky_relu, x);
    if (x.ctx != x_act.ctx)
        mlx_array_free(x);
    x = x_act;

    /* body spade blocks */
    if (m->spade_blocks)
    {
        for (int i = 0; i < m->n_blocks; ++i)
        {
            MLXSPADEConvBlock *blk = (MLXSPADEConvBlock *)m->spade_blocks[i];
            if (!blk)
                continue;
            mlx_array nx = mlx_spadeconv_forward(blk, x, cond);
            if (x.ctx != nx.ctx)
                mlx_array_free(x);
            x = nx;
        }
    }

    /* tail conv + tanh */
    if (m->tail_conv)
    {
        mlx_array out = mlx_array_new();
        mlx_array *tw = (mlx_array *)m->tail_conv;
        if (mlx_conv2d(&out, x, *tw, 1, 1, m->padding_size, m->padding_size, 1, 1, 1, s) != 0)
        {
            if (x.ctx != cond.ctx)
                mlx_array_free(x);
            return cond;
        }
        /* tanh */
        mlx_array out_t = mlx_array_new();
        if (mlx_tanh(&out_t, out, s) != 0)
        {
            mlx_array_free(out);
            if (x.ctx != cond.ctx)
                mlx_array_free(x);
            return cond;
        }
        mlx_array_free(out);
        if (x.ctx != cond.ctx)
            mlx_array_free(x);
        return out_t;
    }

    return x;
}

/* Small accessors for SPADE generator internals used by AG tracing. */
int mlx_spadegen_get_n_blocks(MLXSPADEGenerator *m)
{
    if (!m)
        return 0;
    return m->n_blocks;
}

MLXSPADEConvBlock *mlx_spadegen_get_block_at(MLXSPADEGenerator *m, int idx)
{
    if (!m)
        return NULL;
    if (idx < 0 || idx >= m->n_blocks)
        return NULL;
    return (MLXSPADEConvBlock *)m->spade_blocks[idx];
}

mlx_array *mlx_spadegen_get_init_conv(MLXSPADEGenerator *m)
{
    if (!m)
        return NULL;
    return (mlx_array *)m->init_conv;
}

mlx_array *mlx_spadegen_get_tail_conv(MLXSPADEGenerator *m)
{
    if (!m)
        return NULL;
    return (mlx_array *)m->tail_conv;
}

int mlx_spadeconv_get_padding(MLXSPADEConvBlock *m)
{
    if (!m)
        return 0;
    return m->padding;
}

MLXLeakyReLU *mlx_spadeconv_get_activation(MLXSPADEConvBlock *m)
{
    if (!m)
        return NULL;
    return m->activation;
}

MLXSPADE *mlx_spadeconv_get_spade(MLXSPADEConvBlock *m)
{
    if (!m)
        return NULL;
    return m->spade;
}

struct MLXScaleModule
{
    void *head;
    void *body;
    void *tail;
};

MLXScaleModule *mlx_scalemodule_create(void *head, void *body, void *tail)
{
    MLXScaleModule *m = (MLXScaleModule *)malloc(sizeof(MLXScaleModule));
    if (!m)
        return NULL;
    m->head = head;
    m->body = body;
    m->tail = tail;
    return m;
}

void mlx_scalemodule_free(MLXScaleModule *m)
{
    if (m)
        free(m);
}

mlx_array_t mlx_scalemodule_forward(MLXScaleModule *m, mlx_array_t x)
{
    if (!m)
        return x;
    /* Sequence: head(x) -> body(x) -> tail(x)
     *
     * The `head`, `body`, and `tail` fields are opaque `void*` values.
     * To keep this routine generic we expect callers to provide unary
     * forward function pointers cast to `void*` with the signature:
     *
     *   mlx_array_t fn(mlx_array_t input);
     *
     * i.e., pass `(void*)fn` into `mlx_scalemodule_create`. This keeps the
     * C-side dispatcher simple and avoids making assumptions about the
     * concrete module struct layouts. If you instead intend to pass module
     * pointers, replace this dispatcher with a typed call-site implementation.
     */

    typedef mlx_array_t (*scale_fn_t)(mlx_array_t);

    mlx_array_t cur = x;

    if (m->head)
    {
        scale_fn_t fn = (scale_fn_t)m->head;
        mlx_array_t nx = fn(cur);
        if (cur.ctx != nx.ctx)
            mlx_array_free(cur);
        cur = nx;
    }

    if (m->body)
    {
        /* `body` may point to a single function or an array of functions;
         * here we treat it as a single unary function for simplicity. */
        scale_fn_t fn = (scale_fn_t)m->body;
        mlx_array_t nx = fn(cur);
        if (cur.ctx != nx.ctx)
            mlx_array_free(cur);
        cur = nx;
    }

    if (m->tail)
    {
        scale_fn_t fn = (scale_fn_t)m->tail;
        mlx_array_t nx = fn(cur);
        if (cur.ctx != nx.ctx)
            mlx_array_free(cur);
        cur = nx;
    }

    return cur;
}

struct MLXSPADEDiscriminator
{
    void *head;
    void **body;
    int n_body;
    void *tail;
};

MLXSPADEDiscriminator *mlx_spadedisc_create(int num_features, int min_num_features, int num_layer, int kernel_size, int padding_size, int input_channels)
{
    MLXSPADEDiscriminator *m = (MLXSPADEDiscriminator *)malloc(sizeof(MLXSPADEDiscriminator));
    if (!m)
        return NULL;
    m->head = NULL;
    int body_count = (num_layer - 2) > 0 ? (num_layer - 2) : 0;
    m->n_body = body_count;
    m->body = NULL;
    m->tail = NULL;

    /* Allocate head conv weights: (num_features, KH, KW, input_channels) */
    size_t head_count = (size_t)num_features * kernel_size * kernel_size * (size_t)input_channels;
    float *head_buf = (float *)calloc(head_count, sizeof(float));
    if (head_buf)
    {
        int hshape[4] = {num_features, kernel_size, kernel_size, input_channels};
        mlx_array hw = mlx_array_new_data(head_buf, hshape, 4, MLX_FLOAT32);
        free(head_buf);
        mlx_array *hwptr = (mlx_array *)malloc(sizeof(mlx_array));
        if (hwptr)
        {
            *hwptr = hw;
            m->head = (void *)hwptr;
        }
        else
        {
            mlx_array_free(hw);
        }
    }

    if (body_count > 0)
    {
        m->body = (void **)calloc(body_count, sizeof(void *));
        int current_features = num_features;
        for (int i = 0; i < body_count; ++i)
        {
            int denom = (1 << (i + 1));
            int block_features = num_features / denom;
            if (block_features < min_num_features)
                block_features = min_num_features;
            int in_ch = current_features;
            int out_ch = block_features;
            /* Create a SPADE conv block which allocates its own conv weights */
            m->body[i] = (void *)mlx_spadeconv_create(in_ch, out_ch, input_channels, kernel_size, padding_size, 1, 64);
            current_features = out_ch;
        }
        /* Allocate tail conv weights (1, KH, KW, current_features) => single output channel */
        size_t tail_count = (size_t)1 * kernel_size * kernel_size * (size_t)current_features;
        float *tail_buf = (float *)calloc(tail_count, sizeof(float));
        if (tail_buf)
        {
            int tshape[4] = {1, kernel_size, kernel_size, current_features};
            mlx_array tw = mlx_array_new_data(tail_buf, tshape, 4, MLX_FLOAT32);
            free(tail_buf);
            mlx_array *twptr = (mlx_array *)malloc(sizeof(mlx_array));
            if (twptr)
            {
                *twptr = tw;
                m->tail = (void *)twptr;
            }
            else
            {
                mlx_array_free(tw);
            }
        }
    }
    return m;
}

void mlx_spadedisc_free(MLXSPADEDiscriminator *m)
{
    if (!m)
        return;
    if (m->head)
    {
        mlx_array *hw = (mlx_array *)m->head;
        mlx_array_free(*hw);
        free(hw);
    }
    if (m->body)
    {
        for (int i = 0; i < m->n_body; ++i)
        {
            if (m->body[i])
                mlx_spadeconv_free((MLXSPADEConvBlock *)m->body[i]);
        }
        free(m->body);
    }
    if (m->tail)
    {
        mlx_array *tw = (mlx_array *)m->tail;
        mlx_array_free(*tw);
        free(tw);
    }
    free(m);
}

/* Introspection helpers */
mlx_array *mlx_spadeconv_get_conv_weight(MLXSPADEConvBlock *m)
{
    if (!m)
        return NULL;
    return m->conv;
}

mlx_array *mlx_convblock_get_conv_weight(MLXConvBlock *m)
{
    if (!m)
        return NULL;
    return m->conv;
}

mlx_array **mlx_spadegen_get_parameters(MLXSPADEGenerator *m, int *out_count)
{
    if (!m || !out_count)
        return NULL;
    int total = 0;
    if (m->init_conv)
        total++;
    for (int i = 0; i < m->n_blocks; ++i)
    {
        MLXSPADEConvBlock *cb = (MLXSPADEConvBlock *)m->spade_blocks[i];
        if (cb && cb->conv)
            total++;
    }
    if (m->tail_conv)
        total++;
    if (total == 0)
    {
        *out_count = 0;
        return NULL;
    }
    mlx_array **list = (mlx_array **)malloc(sizeof(mlx_array *) * total);
    if (!list)
    {
        *out_count = 0;
        return NULL;
    }
    int idx = 0;
    if (m->init_conv)
        list[idx++] = (mlx_array *)m->init_conv;
    for (int i = 0; i < m->n_blocks; ++i)
    {
        MLXSPADEConvBlock *cb = (MLXSPADEConvBlock *)m->spade_blocks[i];
        if (cb && cb->conv)
            list[idx++] = (mlx_array *)cb->conv;
    }
    if (m->tail_conv)
        list[idx++] = (mlx_array *)m->tail_conv;
    *out_count = idx;
    return list;
}

void mlx_spadegen_free_parameters_list(mlx_array **list)
{
    if (list)
        free(list);
}

mlx_array **mlx_spadedisc_get_parameters(MLXSPADEDiscriminator *m, int *out_count)
{
    if (!m || !out_count)
        return NULL;
    int total = 0;
    if (m->head)
        total++;
    for (int i = 0; i < m->n_body; ++i)
    {
        MLXSPADEConvBlock *cb = (MLXSPADEConvBlock *)m->body[i];
        if (cb && cb->conv)
            total++;
    }
    if (m->tail)
        total++;
    if (total == 0)
    {
        *out_count = 0;
        return NULL;
    }
    mlx_array **list = (mlx_array **)malloc(sizeof(mlx_array *) * total);
    if (!list)
    {
        *out_count = 0;
        return NULL;
    }
    int idx = 0;
    if (m->head)
        list[idx++] = (mlx_array *)m->head;
    for (int i = 0; i < m->n_body; ++i)
    {
        MLXSPADEConvBlock *cb = (MLXSPADEConvBlock *)m->body[i];
        if (cb && cb->conv)
            list[idx++] = (mlx_array *)cb->conv;
    }
    if (m->tail)
        list[idx++] = (mlx_array *)m->tail;
    *out_count = idx;
    return list;
}

void mlx_spadedisc_free_parameters_list(mlx_array **list)
{
    if (list)
        free(list);
}

/* Accessors for discriminator internals */
mlx_array *mlx_spadedisc_get_head_conv(MLXSPADEDiscriminator *m)
{
    if (!m)
        return NULL;
    return (mlx_array *)m->head;
}

int mlx_spadedisc_get_body_count(MLXSPADEDiscriminator *m)
{
    if (!m)
        return 0;
    return m->n_body;
}

MLXSPADEConvBlock *mlx_spadedisc_get_body_at(MLXSPADEDiscriminator *m, int idx)
{
    if (!m)
        return NULL;
    if (idx < 0 || idx >= m->n_body)
        return NULL;
    return (MLXSPADEConvBlock *)m->body[idx];
}

mlx_array *mlx_spadedisc_get_tail_conv(MLXSPADEDiscriminator *m)
{
    if (!m)
        return NULL;
    return (mlx_array *)m->tail;
}

mlx_array_t mlx_spadedisc_forward(MLXSPADEDiscriminator *m, mlx_array_t x)
{
    if (!m)
        return x;
    return x;
}

struct MLXColorQuantization
{
    float temperature;
    /* pure colors stored as an mlx array if available; otherwise keep as a C array */
    float pure_colors[4][3];
};

MLXColorQuantization *mlx_colorquant_create(float temperature)
{
    MLXColorQuantization *m = (MLXColorQuantization *)malloc(sizeof(MLXColorQuantization));
    if (!m)
        return NULL;
    m->temperature = temperature;
    /* default pure colors (match Python implementation) */
    m->pure_colors[0][0] = -1.0f;
    m->pure_colors[0][1] = -1.0f;
    m->pure_colors[0][2] = -1.0f;
    m->pure_colors[1][0] = 1.0f;
    m->pure_colors[1][1] = -1.0f;
    m->pure_colors[1][2] = -1.0f;
    m->pure_colors[2][0] = -1.0f;
    m->pure_colors[2][1] = 1.0f;
    m->pure_colors[2][2] = -1.0f;
    m->pure_colors[3][0] = -1.0f;
    m->pure_colors[3][1] = -1.0f;
    m->pure_colors[3][2] = 1.0f;
    return m;
}

void mlx_colorquant_free(MLXColorQuantization *m)
{
    if (m)
        free(m);
}

mlx_array_t mlx_colorquant_forward(MLXColorQuantization *m, mlx_array_t x, int training)
{
    if (!m)
        return x;

    /* Use default CPU stream */
    mlx_stream s = mlx_default_cpu_stream_new();

    /* Read shape: expect NHWC (b,h,w,c) */
    size_t ndim = mlx_array_ndim(x);
    if (ndim != 4)
    {
        /* unsupported shape: return input as-is */
        return x;
    }
    const int *shape = mlx_array_shape(x);
    int b = shape[0];
    int h = shape[1];
    int w = shape[2];
    int c = shape[3];
    int N = b * h * w;

    /* reshape to (N, c) */
    int flat_shape[2];
    flat_shape[0] = N;
    flat_shape[1] = c;
    mlx_array flat = mlx_array_new();
    if (mlx_reshape(&flat, x, flat_shape, 2, s) != 0)
    {
        return x;
    }

    /* x_norm = sum(x_flat**2, axis=1, keepdims=True) */
    mlx_array x_sq = mlx_array_new();
    if (mlx_square(&x_sq, flat, s) != 0)
    {
        mlx_array_free(flat);
        return x;
    }
    mlx_array x_norm = mlx_array_new();
    if (mlx_sum_axis(&x_norm, x_sq, 1, true, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_sq);
        return x;
    }
    mlx_array_free(x_sq);

    /* pure_colors as an mlx array (K, c) where K=4 */
    int pc_shape[2] = {4, c};
    mlx_array pure = mlx_array_new_data(m->pure_colors, pc_shape, 2, MLX_FLOAT32);

    /* c_norm = sum(pure**2, axis=1, keepdims=True) */
    mlx_array pure_sq = mlx_array_new();
    if (mlx_square(&pure_sq, pure, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        return x;
    }
    mlx_array c_norm = mlx_array_new();
    if (mlx_sum_axis(&c_norm, pure_sq, 1, true, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(pure_sq);
        return x;
    }
    mlx_array_free(pure_sq);

    /* prod = x_flat @ pure.T  (N, K) */
    mlx_array pure_t = mlx_array_new();
    if (mlx_transpose(&pure_t, pure, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        return x;
    }
    mlx_array prod = mlx_array_new();
    if (mlx_matmul(&prod, flat, pure_t, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        return x;
    }

    /* distances = x_norm + c_norm.T - 2 * prod */
    mlx_array c_norm_t = mlx_array_new();
    if (mlx_transpose(&c_norm_t, c_norm, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        return x;
    }

    /* neg2 = scalar -2.0 */
    mlx_array neg2 = mlx_array_new_float(-2.0f);
    mlx_array neg2prod = mlx_array_new();
    if (mlx_multiply(&neg2prod, prod, neg2, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        return x;
    }

    mlx_array tmp = mlx_array_new();
    if (mlx_add(&tmp, x_norm, neg2prod, s) != 0)
    {
        /* cleanup */
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        return x;
    }

    mlx_array distances = mlx_array_new();
    if (mlx_add(&distances, tmp, c_norm_t, s) != 0)
    {
        /* cleanup */
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        return x;
    }

    /* If not training: nearest neighbor assignment
     * Use the MLX helper `mlx_quantize_array` which handles host-side
     * quantization and returns an MLX array, avoiding manual host loops.
     */
    if (!training)
    {
        mlx_array quant = mlx_array_new();
        if (mlx_quantize_array(flat, &quant, pure) != 0)
        {
            /* fallback to existing behavior on error */
            mlx_array_free(quant);
            mlx_array_free(flat);
            mlx_array_free(x_norm);
            mlx_array_free(c_norm);
            mlx_array_free(pure_t);
            mlx_array_free(prod);
            mlx_array_free(c_norm_t);
            mlx_array_free(neg2prod);
            mlx_array_free(tmp);
            mlx_array_free(distances);
            return x;
        }

        int outshape[4] = {b, h, w, c};
        mlx_array out = mlx_array_new();
        if (mlx_reshape(&out, quant, outshape, 4, s) != 0)
        {
            mlx_array_free(quant);
            mlx_array_free(flat);
            mlx_array_free(x_norm);
            mlx_array_free(c_norm);
            mlx_array_free(pure_t);
            mlx_array_free(prod);
            mlx_array_free(c_norm_t);
            mlx_array_free(neg2prod);
            mlx_array_free(tmp);
            mlx_array_free(distances);
            return x;
        }

        /* cleanup temporaries */
        mlx_array_free(quant);
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        mlx_array_free(distances);

        return out;
    }

    /* Training: soft assignment
     * weights = softmax(-distances / temperature, axis=-1)
     * quantized = weights @ pure  -> shape (N, c)
     */
    mlx_array temp_arr = mlx_array_new_float(m->temperature);
    mlx_array div = mlx_array_new();
    if (mlx_divide(&div, distances, temp_arr, s) != 0)
    {
        /* fallback */
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        mlx_array_free(distances);
        return x;
    }

    mlx_array neg = mlx_array_new_float(-1.0f);
    mlx_array scaled = mlx_array_new();
    if (mlx_multiply(&scaled, div, neg, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        mlx_array_free(div);
        mlx_array_free(distances);
        return x;
    }

    mlx_array weights = mlx_array_new();
    if (mlx_softmax_axis(&weights, scaled, 1, false, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        mlx_array_free(div);
        mlx_array_free(scaled);
        mlx_array_free(distances);
        return x;
    }

    /* quant_flat = weights @ pure  (N, c) */
    mlx_array quant_flat = mlx_array_new();
    if (mlx_matmul(&quant_flat, weights, pure, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        mlx_array_free(div);
        mlx_array_free(scaled);
        mlx_array_free(weights);
        mlx_array_free(distances);
        return x;
    }

    int outshape[4] = {b, h, w, c};
    mlx_array out = mlx_array_new();
    if (mlx_reshape(&out, quant_flat, outshape, 4, s) != 0)
    {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        mlx_array_free(div);
        mlx_array_free(scaled);
        mlx_array_free(weights);
        mlx_array_free(quant_flat);
        mlx_array_free(distances);
        return x;
    }

    /* cleanup temporaries */
    mlx_array_free(flat);
    mlx_array_free(x_norm);
    mlx_array_free(c_norm);
    mlx_array_free(pure_t);
    mlx_array_free(prod);
    mlx_array_free(c_norm_t);
    mlx_array_free(neg2prod);
    mlx_array_free(tmp);
    mlx_array_free(div);
    mlx_array_free(scaled);
    mlx_array_free(weights);
    mlx_array_free(quant_flat);
    mlx_array_free(distances);

    return out;
}
