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
#include <limits.h>
#include <mlx/c/mlx.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Ensure common string prototypes are visible */
#include <stddef.h>

/* Project helpers */
#include "../trainning/array_helpers.h"
#include "../trainning/mlx_compat.h"
#include "utils.h"

/**
 * Create a random normal array matching Python's weight initialization:
 * weight = mx.random.normal(shape) * std
 * where std defaults to 0.02
 */
mlx_array mlx_init_conv_weight(const int *shape, int ndim, float std) {
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array result = mlx_array_new();

    /* Generate random normal with mean=0 and scale=std directly
     * mlx_random_normal signature: (res, shape, ndim, dtype, loc, scale, key, stream) */
    if (mlx_random_normal(&result, shape, ndim, MLX_FLOAT32, 0.0f, std,
                          mlx_array_empty, s) != 0) {
        /* Fallback to zeros if random fails */
        mlx_zeros(&result, shape, ndim, MLX_FLOAT32, s);
    }

    mlx_stream_free(s);
    return result;
}

/**
 * Create a zero-initialized bias array matching Python's Conv2d default.
 * bias = mx.zeros((out_channels,))
 */
mlx_array mlx_init_conv_bias(int out_ch) {
    mlx_stream s = mlx_default_gpu_stream_new();
    int shape[1] = {out_ch};
    mlx_array result = mlx_array_new();
    mlx_zeros(&result, shape, 1, MLX_FLOAT32, s);
    mlx_stream_free(s);
    return result;
}

/**
 * Helper: allocate and store an mlx_array on the heap (like conv weights).
 * Returns pointer to allocated mlx_array, or NULL on failure.
 */
mlx_array *mlx_alloc_array_ptr(mlx_array arr) {
    mlx_array *ptr = NULL;
    if (mlx_alloc_pod((void **)&ptr, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(arr);
        return NULL;
    }
    *ptr = arr;
    return ptr;
}

/**
 * Helper: free a heap-allocated mlx_array pointer (frees array + pointer).
 * Accepts void** to work with void* struct fields storing mlx_array pointers.
 */
void mlx_free_array_ptr(void **pptr) {
    if (!pptr || !*pptr) return;
    mlx_array *ap = (mlx_array *)*pptr;
    mlx_array_free(*ap);
    mlx_free_pod(pptr);
}

struct MLXLeakyReLU {
    float negative_slope;
    /* cached scalar arrays to avoid reallocating per-forward call */
    mlx_array slope_scalar; /* holds negative_slope */
    int has_cached_scalars;
};

MLXLeakyReLU *mlx_leakyrelu_create(float negative_slope) {
    MLXLeakyReLU *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXLeakyReLU), 1) != 0)
        return NULL;
    m->negative_slope = negative_slope;
    m->has_cached_scalars = 0;
    /* create scalar cache */
    m->slope_scalar = mlx_array_new_float(negative_slope);
    m->has_cached_scalars = 1;
    return m;
}

void mlx_leakyrelu_free(MLXLeakyReLU *m) {
    if (!m)
        return;
    if (m->has_cached_scalars) {
        mlx_array_free(m->slope_scalar);
    }
    mlx_free_pod((void **)&m);
}

mlx_array_t mlx_leakyrelu_forward(MLXLeakyReLU *m, mlx_array_t x) {
    /* Implement leaky-relu using mlx_maximum for parity with Python MLX.
     * out = max(x, x * negative_slope)
     * Previous implementation used sqrt-based formula which has numerical precision issues.
     */
    if (!m)
        return x;

    mlx_stream s = mlx_default_gpu_stream_new();

    mlx_array scaled = mlx_array_new();
    mlx_array out = mlx_array_new();

    int err = 0;

    /* scaled = x * negative_slope */
    if (mlx_multiply(&scaled, x, m->slope_scalar, s) != 0) {
        err = 1;
        goto cleanup;
    }

    /* out = max(x, scaled) - this is the native LeakyReLU operation */
    if (mlx_maximum(&out, x, scaled, s) != 0) {
        err = 1;
        goto cleanup;
    }

cleanup:
    /* Free temporaries if they were allocated (ctx==0 indicates empty).
     * Do NOT synchronize here — mlx_synchronize inside a value_and_grad
     * closure disrupts MLX's lazy evaluation and breaks gradient tracing.
     * MLX handles reference counting; freeing our handle is safe even if
     * the computation graph still references the array. */
    if (scaled.ctx)
        mlx_array_free(scaled);

    /* Free the stream we created */
    mlx_stream_free(s);

    if (err) {
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
typedef struct mlx_nn_instancenorm_ {
    int num_features;
    int affine;
    mlx_array *weight; /* 1-D array of length num_features (scale), may be NULL */
    mlx_array *bias; /* 1-D array of length num_features (offset), may be NULL */
} mlx_nn_instancenorm;

void *mlx_nn_instancenorm_create(int num_features, int affine) {
    mlx_nn_instancenorm *h = NULL;
    if (mlx_alloc_pod((void **)&h, sizeof(mlx_nn_instancenorm), 1) != 0)
        return NULL;
    h->num_features = num_features;
    h->affine = affine ? 1 : 0;
    h->weight = NULL;
    h->bias = NULL;

    if (h->affine) {
        /* allocate weight initialized to 1.0 and bias initialized to 0.0 */
        int shape[1] = {num_features};
        float *wbuf = NULL;
        float *bbuf = NULL;
        if (num_features > (size_t)INT_MAX) {
            wbuf = (float *)malloc(sizeof(float) * (size_t)num_features);
            bbuf = (float *)malloc(sizeof(float) * (size_t)num_features);
        } else {
            if (mlx_alloc_float_buf(&wbuf, (int)num_features) != 0)
                wbuf = NULL;
            if (mlx_alloc_float_buf(&bbuf, (int)num_features) != 0)
                bbuf = NULL;
        }
        if (!wbuf || !bbuf) {
            if (wbuf) {
                if (num_features > (size_t)INT_MAX)
                    free(wbuf);
                else
                    mlx_free_float_buf(&wbuf, NULL);
            }
            if (bbuf) {
                if (num_features > (size_t)INT_MAX)
                    free(bbuf);
                else
                    mlx_free_float_buf(&bbuf, NULL);
            }
            mlx_free_pod((void **)&h);
            return NULL;
        }
        for (int i = 0; i < num_features; ++i) {
            wbuf[i] = 1.0f;
            bbuf[i] = 0.0f;
        }
        /* FIX 37: Weights initialized to ones(), bias to zeros() — matching
         * Python's _init_instance_norm() in custom_layer.py:
         *   norm.weight = mx.ones(weight.shape)
         *   norm.bias   = mx.zeros(bias.shape)
         *
         * Previously this used random_normal(1.0, 0.02) which consumed RNG
         * values, shifting all subsequent Conv2d weight random values out of
         * alignment with Python.  Python's reset_parameters() never uses
         * random for InstanceNorm — it always uses deterministic ones(). */
        {
            mlx_array warr = mlx_array_new_data(wbuf, shape, 1, MLX_FLOAT32);
            mlx_array barr = mlx_array_new_data(bbuf, shape, 1, MLX_FLOAT32);
            if (num_features > (size_t)INT_MAX) {
                free(wbuf);
                free(bbuf);
            } else {
                mlx_free_float_buf(&wbuf, NULL);
                mlx_free_float_buf(&bbuf, NULL);
            }
            h->weight = NULL;
            h->bias = NULL;
            if (mlx_alloc_pod((void **)&h->weight, sizeof(mlx_array), 1) != 0 ||
                    mlx_alloc_pod((void **)&h->bias, sizeof(mlx_array), 1) != 0) {
                if (h->weight) {
                    *h->weight = warr;
                } else
                    mlx_array_free(warr);
                if (h->bias) {
                    *h->bias = barr;
                } else
                    mlx_array_free(barr);
                if (h->weight)
                    mlx_free_pod((void **)&h->weight);
                if (h->bias)
                    mlx_free_pod((void **)&h->bias);
                mlx_free_pod((void **)&h);
                return NULL;
            }
            *h->weight = warr;
            *h->bias = barr;
        }
    }

    return (void *)h;
}

void mlx_nn_instancenorm_free(void *handle) {
    if (!handle)
        return;
    mlx_nn_instancenorm *h = (mlx_nn_instancenorm *)handle;
    if (h->weight) {
        mlx_array_free(*h->weight);
        mlx_free_pod((void **)&h->weight);
        h->weight = NULL;
    }
    if (h->bias) {
        mlx_array_free(*h->bias);
        mlx_free_pod((void **)&h->bias);
        h->bias = NULL;
    }
    mlx_free_pod((void **)&h);
}

mlx_array *mlx_nn_instancenorm_get_weight(void *handle) {
    if (!handle)
        return NULL;
    mlx_nn_instancenorm *h = (mlx_nn_instancenorm *)handle;
    return h->weight;
}

mlx_array *mlx_nn_instancenorm_get_bias(void *handle) {
    if (!handle)
        return NULL;
    mlx_nn_instancenorm *h = (mlx_nn_instancenorm *)handle;
    return h->bias;
}

struct MLXConvBlock {
    int use_norm;
    void *conv; /* pointer to mlx_array weight (allocated) */
    mlx_array *conv_bias; /* pointer to mlx_array bias (allocated) */
    void *norm; /* underlying instance norm handle */
    MLXLeakyReLU *activation;
    int stride;
    int padding;
    int kernel_size;
    int in_ch;
    int out_ch;
};

MLXConvBlock *mlx_convblock_create(int in_ch, int out_ch, int kernel_size,
                                   int padding, int stride, int use_norm) {
    MLXConvBlock *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXConvBlock), 1) != 0)
        return NULL;
    m->use_norm = use_norm;
    m->conv = NULL; /* will store pointer to mlx_array weight */
    m->conv_bias = NULL;
    m->norm = NULL; /* create instance norm if requested */
    if (use_norm) {
        /* Create an mlx-c InstanceNorm module handle if available.
         * Signature assumed: void* mlx_nn_instancenorm_create(int dims, int
         * affine); Adjust if your mlx-c version uses a different name or signature.
         */
        m->norm = (void *)mlx_nn_instancenorm_create(out_ch, 1);
    }
    m->activation = mlx_leakyrelu_create(0.2f);
    m->stride = stride;
    m->padding = padding;
    m->kernel_size = kernel_size;
    m->in_ch = in_ch;
    m->out_ch = out_ch;

    /* MLX canonical conv-weight layout: (out_ch, KH, KW, in_ch) */
    int wshape[4] = {out_ch, kernel_size, kernel_size, in_ch};
    mlx_array w = mlx_init_conv_weight(wshape, 4, 0.02f);
    mlx_array *wptr = NULL;
    if (mlx_alloc_pod((void **)&wptr, sizeof(mlx_array), 1) == 0) {
        *wptr = w;
        m->conv = (void *)wptr;
    } else {
        mlx_array_free(w);
    }
    /* Allocate conv bias (zero-initialised, shape [out_ch]) */
    m->conv_bias = mlx_alloc_array_ptr(mlx_init_conv_bias(out_ch));
    return m;
}

void mlx_convblock_free(MLXConvBlock *m) {
    if (!m)
        return;
    /* free conv weight array if allocated */
    if (m->conv) {
        mlx_array *wptr = (mlx_array *)m->conv;
        mlx_array_free(*wptr);
        mlx_free_pod((void **)&wptr);
        m->conv = NULL;
    }
    mlx_free_array_ptr((void **)&m->conv_bias);
    /* free instance norm handle if created (signature:
     * mlx_nn_instancenorm_free(void*)) */
    if (m->norm) {
        mlx_nn_instancenorm_free(m->norm);
        m->norm = NULL;
    }
    mlx_leakyrelu_free(m->activation);
    mlx_free_pod((void **)&m);
}

mlx_array_t mlx_convblock_forward(MLXConvBlock *m, mlx_array_t x) {
    if (!m)
        return x;
    mlx_stream s = mlx_default_gpu_stream_new();

    /* If conv weights are present, run conv; otherwise pass-through */
    mlx_array y = mlx_array_new();
    if (m->conv) {
        mlx_array *wptr = (mlx_array *)m->conv;
        if (safe_mlx_conv2d(&y, x, *wptr, m->stride, m->stride, m->padding,
                            m->padding, 1, 1, 1, s) != 0) {
            /* conv failed: fallback to input */
            mlx_stream_free(s);
            return x;
        }
        /* Apply bias: y = y + bias (matching Python nn.Conv2d) */
        if (m->conv_bias) {
            mlx_array biased = mlx_array_new();
            if (mlx_add(&biased, y, *m->conv_bias, s) == 0) {
                mlx_array_free(y);
                y = biased;
            } else {
                mlx_array_free(biased);
            }
        }
    } else {
        y = x;
    }

    /* Instance normalization (NHWC): normalize across H and W axes, then apply affine */
    if (m->use_norm && m->norm) {
        mlx_nn_instancenorm *norm = (mlx_nn_instancenorm *)m->norm;
        const int axes[] = {1, 2};
        mlx_array mean = mlx_array_new();
        if (mlx_mean_axes(&mean, y, axes, 2, true, s) == 0) {
            mlx_array centered = mlx_array_new();
            if (mlx_subtract(&centered, y, mean, s) == 0) {
                mlx_array sq = mlx_array_new();
                if (mlx_square(&sq, centered, s) == 0) {
                    mlx_array var = mlx_array_new();
                    if (mlx_mean_axes(&var, sq, axes, 2, true, s) == 0) {
                        mlx_array eps = mlx_array_new_float(1e-5f);
                        mlx_array var_eps = mlx_array_new();
                        if (mlx_add(&var_eps, var, eps, s) == 0) {
                            mlx_array std = mlx_array_new();
                            if (mlx_sqrt(&std, var_eps, s) == 0) {
                                mlx_array y_norm = mlx_array_new();
                                if (mlx_divide(&y_norm, centered, std, s) == 0) {
                                    /* Apply affine: y = y_norm * weight + bias */
                                    if (norm->affine && norm->weight && norm->bias) {
                                        mlx_array scaled = mlx_array_new();
                                        if (mlx_multiply(&scaled, y_norm, *norm->weight, s) == 0) {
                                            mlx_array affined = mlx_array_new();
                                            if (mlx_add(&affined, scaled, *norm->bias, s) == 0) {
                                                mlx_array_free(y);
                                                mlx_array_free(y_norm);
                                                y = affined;
                                            } else {
                                                mlx_array_free(y);
                                                y = y_norm;
                                            }
                                            mlx_array_free(scaled);
                                        } else {
                                            mlx_array_free(y);
                                            y = y_norm;
                                        }
                                    } else {
                                        /* No affine: just use normalized */
                                        mlx_array_free(y);
                                        y = y_norm;
                                    }
                                }
                                mlx_array_free(std);
                            }
                            mlx_array_free(var_eps);
                        }
                        mlx_array_free(eps);
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
    if (y.ctx != x.ctx && y.ctx != activated.ctx)
        mlx_array_free(y);
    mlx_stream_free(s);
    return activated;
}

struct MLXUpsample {
    int out_h;
    int out_w;
    char mode[16];
    int align_corners;
    void *internal;
};

MLXUpsample *mlx_upsample_create(int out_h, int out_w, const char *mode,
                                 int align_corners) {
    MLXUpsample *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXUpsample), 1) != 0)
        return NULL;
    m->out_h = out_h;
    m->out_w = out_w;
    strncpy(m->mode, mode ? mode : "linear", sizeof(m->mode) - 1);
    m->mode[sizeof(m->mode) - 1] = '\0';
    m->align_corners = align_corners;
    m->internal = NULL;
    return m;
}

void mlx_upsample_free(MLXUpsample *m) {
    if (m)
        mlx_free_pod((void **)&m);
}

/**
 * Bilinear interpolation resize for NHWC tensors.
 * Supports any scale factor including non-integer and downsampling.
 * Matches Python's upsample_linear with align_corners=True.
 */
mlx_array_t mlx_upsample_forward(MLXUpsample *m, mlx_array_t x) {
    if (!m)
        return x;
    mlx_stream s = mlx_default_gpu_stream_new();

    size_t ndim = mlx_array_ndim(x);
    if (ndim != 4) {
        mlx_stream_free(s);
        return x; /* expect NHWC */
    }
    const int *shape = mlx_array_shape(x);
    int b = shape[0];
    int in_h = shape[1];
    int in_w = shape[2];
    int c = shape[3];

    int out_h = m->out_h;
    int out_w = m->out_w;

    if (in_h == out_h && in_w == out_w) {
        mlx_stream_free(s);
        return x;
    }

    /* ALWAYS use bilinear interpolation (align_corners=True) for parity with Python.
     * Previous implementation used mlx_repeat_axis (nearest-neighbor) for integer
     * scale factors, but Python's upsample_linear uses bilinear for ALL cases. */

    /* Generate scaled indices for height dimension */
    /* indices_h[i] = i * (in_h - 1) / (out_h - 1) */
    mlx_array idx_h = mlx_array_new();
    mlx_arange(&idx_h, 0.0, (double)out_h, 1.0, MLX_FLOAT32, s);

    mlx_array scale_h_arr = mlx_array_new();
    double h_scale = (out_h > 1) ? (double)(in_h - 1) / (double)(out_h - 1) : 0.0;
    mlx_array_set_float32(&scale_h_arr, (float)h_scale);

    mlx_array idx_h_scaled = mlx_array_new();
    mlx_multiply(&idx_h_scaled, idx_h, scale_h_arr, s);
    mlx_array_free(idx_h);
    mlx_array_free(scale_h_arr);

    /* Generate scaled indices for width dimension */
    mlx_array idx_w = mlx_array_new();
    mlx_arange(&idx_w, 0.0, (double)out_w, 1.0, MLX_FLOAT32, s);

    mlx_array scale_w_arr = mlx_array_new();
    double w_scale = (out_w > 1) ? (double)(in_w - 1) / (double)(out_w - 1) : 0.0;
    mlx_array_set_float32(&scale_w_arr, (float)w_scale);

    mlx_array idx_w_scaled = mlx_array_new();
    mlx_multiply(&idx_w_scaled, idx_w, scale_w_arr, s);
    mlx_array_free(idx_w);
    mlx_array_free(scale_w_arr);

    /* Clip indices to valid range */
    mlx_array zero_arr = mlx_array_new();
    mlx_array_set_float32(&zero_arr, 0.0f);
    mlx_array max_h_arr = mlx_array_new();
    mlx_array_set_float32(&max_h_arr, (float)(in_h - 1));
    mlx_array max_w_arr = mlx_array_new();
    mlx_array_set_float32(&max_w_arr, (float)(in_w - 1));

    mlx_array idx_h_clipped = mlx_array_new();
    mlx_clip(&idx_h_clipped, idx_h_scaled, zero_arr, max_h_arr, s);
    mlx_array_free(idx_h_scaled);

    mlx_array idx_w_clipped = mlx_array_new();
    mlx_clip(&idx_w_clipped, idx_w_scaled, zero_arr, max_w_arr, s);
    mlx_array_free(idx_w_scaled);
    mlx_array_free(zero_arr);
    mlx_array_free(max_h_arr);
    mlx_array_free(max_w_arr);

    /* Floor and ceil indices */
    mlx_array idx_h_floor = mlx_array_new();
    mlx_floor(&idx_h_floor, idx_h_clipped, s);
    mlx_array idx_h_ceil = mlx_array_new();
    mlx_ceil(&idx_h_ceil, idx_h_clipped, s);

    mlx_array idx_w_floor = mlx_array_new();
    mlx_floor(&idx_w_floor, idx_w_clipped, s);
    mlx_array idx_w_ceil = mlx_array_new();
    mlx_ceil(&idx_w_ceil, idx_w_clipped, s);

    /* Compute weights */
    mlx_array weight_h = mlx_array_new();
    mlx_subtract(&weight_h, idx_h_clipped, idx_h_floor, s);
    mlx_array weight_h_inv = mlx_array_new();
    mlx_array one_arr = mlx_array_new();
    mlx_array_set_float32(&one_arr, 1.0f);
    mlx_subtract(&weight_h_inv, one_arr, weight_h, s);

    mlx_array weight_w = mlx_array_new();
    mlx_subtract(&weight_w, idx_w_clipped, idx_w_floor, s);
    mlx_array weight_w_inv = mlx_array_new();
    mlx_subtract(&weight_w_inv, one_arr, weight_w, s);
    mlx_array_free(one_arr);
    mlx_array_free(idx_h_clipped);
    mlx_array_free(idx_w_clipped);

    /* Convert indices to int32 for gather */
    mlx_array idx_h_floor_i = mlx_array_new();
    mlx_astype(&idx_h_floor_i, idx_h_floor, MLX_INT32, s);
    mlx_array_free(idx_h_floor);
    mlx_array idx_h_ceil_i = mlx_array_new();
    mlx_astype(&idx_h_ceil_i, idx_h_ceil, MLX_INT32, s);
    mlx_array_free(idx_h_ceil);
    mlx_array idx_w_floor_i = mlx_array_new();
    mlx_astype(&idx_w_floor_i, idx_w_floor, MLX_INT32, s);
    mlx_array_free(idx_w_floor);
    mlx_array idx_w_ceil_i = mlx_array_new();
    mlx_astype(&idx_w_ceil_i, idx_w_ceil, MLX_INT32, s);
    mlx_array_free(idx_w_ceil);

    /* Gather samples at 4 corners using take along axis */
    /* x shape: (B, in_h, in_w, C) */
    /* We need to sample at (h_floor, w_floor), (h_floor, w_ceil), (h_ceil, w_floor), (h_ceil, w_ceil) */

    /* First gather along height axis */
    /* Reshape idx_h_floor_i to (1, out_h, 1, 1) for broadcasting */
    int idx_h_shape[4] = {1, out_h, 1, 1};
    mlx_array idx_h_floor_4d = mlx_array_new();
    mlx_reshape(&idx_h_floor_4d, idx_h_floor_i, idx_h_shape, 4, s);
    mlx_array_free(idx_h_floor_i);
    mlx_array idx_h_ceil_4d = mlx_array_new();
    mlx_reshape(&idx_h_ceil_4d, idx_h_ceil_i, idx_h_shape, 4, s);
    mlx_array_free(idx_h_ceil_i);

    /* Broadcast to (B, out_h, in_w, C) */
    int bcast_h_shape[4] = {b, out_h, in_w, c};
    mlx_array idx_h_floor_bcast = mlx_array_new();
    mlx_broadcast_to(&idx_h_floor_bcast, idx_h_floor_4d, bcast_h_shape, 4, s);
    mlx_array_free(idx_h_floor_4d);
    mlx_array idx_h_ceil_bcast = mlx_array_new();
    mlx_broadcast_to(&idx_h_ceil_bcast, idx_h_ceil_4d, bcast_h_shape, 4, s);
    mlx_array_free(idx_h_ceil_4d);

    /* Take along height axis */
    mlx_array x_hfloor = mlx_array_new();
    mlx_take_along_axis(&x_hfloor, x, idx_h_floor_bcast, 1, s);
    mlx_array_free(idx_h_floor_bcast);
    mlx_array x_hceil = mlx_array_new();
    mlx_take_along_axis(&x_hceil, x, idx_h_ceil_bcast, 1, s);
    mlx_array_free(idx_h_ceil_bcast);

    /* Now gather along width axis */
    int idx_w_shape[4] = {1, 1, out_w, 1};
    mlx_array idx_w_floor_4d = mlx_array_new();
    mlx_reshape(&idx_w_floor_4d, idx_w_floor_i, idx_w_shape, 4, s);
    mlx_array_free(idx_w_floor_i);
    mlx_array idx_w_ceil_4d = mlx_array_new();
    mlx_reshape(&idx_w_ceil_4d, idx_w_ceil_i, idx_w_shape, 4, s);
    mlx_array_free(idx_w_ceil_i);

    /* Broadcast to (B, out_h, out_w, C) */
    int bcast_w_shape[4] = {b, out_h, out_w, c};
    mlx_array idx_w_floor_bcast = mlx_array_new();
    mlx_broadcast_to(&idx_w_floor_bcast, idx_w_floor_4d, bcast_w_shape, 4, s);
    mlx_array_free(idx_w_floor_4d);
    mlx_array idx_w_ceil_bcast = mlx_array_new();
    mlx_broadcast_to(&idx_w_ceil_bcast, idx_w_ceil_4d, bcast_w_shape, 4, s);
    mlx_array_free(idx_w_ceil_4d);

    /* Take along width axis for all 4 corners */
    mlx_array x_tl = mlx_array_new(); /* top-left: h_floor, w_floor */
    mlx_take_along_axis(&x_tl, x_hfloor, idx_w_floor_bcast, 2, s);
    mlx_array x_tr = mlx_array_new(); /* top-right: h_floor, w_ceil */
    mlx_take_along_axis(&x_tr, x_hfloor, idx_w_ceil_bcast, 2, s);
    mlx_array_free(x_hfloor);
    mlx_array x_bl = mlx_array_new(); /* bottom-left: h_ceil, w_floor */
    mlx_take_along_axis(&x_bl, x_hceil, idx_w_floor_bcast, 2, s);
    mlx_array x_br = mlx_array_new(); /* bottom-right: h_ceil, w_ceil */
    mlx_take_along_axis(&x_br, x_hceil, idx_w_ceil_bcast, 2, s);
    mlx_array_free(x_hceil);
    mlx_array_free(idx_w_floor_bcast);
    mlx_array_free(idx_w_ceil_bcast);

    /* Reshape weights for broadcasting: (1, out_h, 1, 1) and (1, 1, out_w, 1) */
    int wh_shape[4] = {1, out_h, 1, 1};
    mlx_array wh = mlx_array_new();
    mlx_reshape(&wh, weight_h, wh_shape, 4, s);
    mlx_array_free(weight_h);
    mlx_array wh_inv = mlx_array_new();
    mlx_reshape(&wh_inv, weight_h_inv, wh_shape, 4, s);
    mlx_array_free(weight_h_inv);

    int ww_shape[4] = {1, 1, out_w, 1};
    mlx_array ww = mlx_array_new();
    mlx_reshape(&ww, weight_w, ww_shape, 4, s);
    mlx_array_free(weight_w);
    mlx_array ww_inv = mlx_array_new();
    mlx_reshape(&ww_inv, weight_w_inv, ww_shape, 4, s);
    mlx_array_free(weight_w_inv);

    /* Bilinear interpolation: */
    /* out = (1-wh)*(1-ww)*x_tl + (1-wh)*ww*x_tr + wh*(1-ww)*x_bl + wh*ww*x_br */

    /* Weight for top-left: (1-wh) * (1-ww) */
    mlx_array w_tl = mlx_array_new();
    mlx_multiply(&w_tl, wh_inv, ww_inv, s);

    /* Weight for top-right: (1-wh) * ww */
    mlx_array w_tr = mlx_array_new();
    mlx_multiply(&w_tr, wh_inv, ww, s);

    /* Weight for bottom-left: wh * (1-ww) */
    mlx_array w_bl = mlx_array_new();
    mlx_multiply(&w_bl, wh, ww_inv, s);

    /* Weight for bottom-right: wh * ww */
    mlx_array w_br = mlx_array_new();
    mlx_multiply(&w_br, wh, ww, s);

    mlx_array_free(wh);
    mlx_array_free(wh_inv);
    mlx_array_free(ww);
    mlx_array_free(ww_inv);

    /* Weighted sum */
    mlx_array term_tl = mlx_array_new();
    mlx_multiply(&term_tl, w_tl, x_tl, s);
    mlx_array_free(w_tl);
    mlx_array_free(x_tl);

    mlx_array term_tr = mlx_array_new();
    mlx_multiply(&term_tr, w_tr, x_tr, s);
    mlx_array_free(w_tr);
    mlx_array_free(x_tr);

    mlx_array term_bl = mlx_array_new();
    mlx_multiply(&term_bl, w_bl, x_bl, s);
    mlx_array_free(w_bl);
    mlx_array_free(x_bl);

    mlx_array term_br = mlx_array_new();
    mlx_multiply(&term_br, w_br, x_br, s);
    mlx_array_free(w_br);
    mlx_array_free(x_br);

    mlx_array sum1 = mlx_array_new();
    mlx_add(&sum1, term_tl, term_tr, s);
    mlx_array_free(term_tl);
    mlx_array_free(term_tr);

    mlx_array sum2 = mlx_array_new();
    mlx_add(&sum2, term_bl, term_br, s);
    mlx_array_free(term_bl);
    mlx_array_free(term_br);

    mlx_array out = mlx_array_new();
    mlx_add(&out, sum1, sum2, s);
    mlx_array_free(sum1);
    mlx_array_free(sum2);

    mlx_stream_free(s);
    return out;
}

struct MLXSPADE {
    int norm_nc;
    int cond_nc;
    int hidden_nc;
    int kernel_size;
    int padding;
    /* InstanceNorm with affine=True to match Python */
    void *norm;  /* mlx_nn_instancenorm handle */
    /* weight arrays for convs: pointers to mlx_array allocated with
     * mlx_array_new_data */
    mlx_array *mlp_shared_w; /* shape: (hidden_nc, K, K, cond_nc) */
    mlx_array *mlp_shared_b; /* shape: (hidden_nc,) */
    mlx_array *mlp_gamma_w;  /* shape: (norm_nc, K, K, hidden_nc) */
    mlx_array *mlp_gamma_b;  /* shape: (norm_nc,) */
    mlx_array *mlp_beta_w;   /* shape: (norm_nc, K, K, hidden_nc) */
    mlx_array *mlp_beta_b;   /* shape: (norm_nc,) */
    /* cache omitted for simplicity */
};

MLXSPADE *mlx_spade_create(int norm_nc, int cond_nc, int hidden_nc,
                           int kernel_size) {
    MLXSPADE *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXSPADE), 1) != 0)
        return NULL;
    m->norm_nc = norm_nc;
    m->cond_nc = cond_nc;
    m->hidden_nc = hidden_nc;
    m->kernel_size = kernel_size;
    m->padding = kernel_size / 2;
    m->norm = NULL;
    m->mlp_shared_w = NULL;
    m->mlp_shared_b = NULL;
    m->mlp_gamma_w = NULL;
    m->mlp_gamma_b = NULL;
    m->mlp_beta_w = NULL;
    m->mlp_beta_b = NULL;

    /* Create InstanceNorm with affine=True to match Python */
    m->norm = mlx_nn_instancenorm_create(norm_nc, 1);

    /* Allocate random-initialized weights for the three convs (like Python: random_normal * 0.02) */
    /* canonical: (out_ch=hidden_nc, KH, KW, in_ch=cond_nc) */
    int sshape[4] = {hidden_nc, kernel_size, kernel_size, cond_nc};
    mlx_array sw = mlx_init_conv_weight(sshape, 4, 0.02f);
    mlx_array *swp = NULL;
    if (mlx_alloc_pod((void **)&swp, sizeof(mlx_array), 1) == 0) {
        *swp = sw;
        m->mlp_shared_w = swp;
    } else {
        mlx_array_free(sw);
    }
    m->mlp_shared_b = mlx_alloc_array_ptr(mlx_init_conv_bias(hidden_nc));

    /* canonical: (out_ch=norm_nc, KH, KW, in_ch=hidden_nc) */
    int gshape[4] = {norm_nc, kernel_size, kernel_size, hidden_nc};
    mlx_array gw = mlx_init_conv_weight(gshape, 4, 0.02f);
    mlx_array *gwp = NULL;
    if (mlx_alloc_pod((void **)&gwp, sizeof(mlx_array), 1) == 0) {
        *gwp = gw;
        m->mlp_gamma_w = gwp;
    } else {
        mlx_array_free(gw);
    }
    m->mlp_gamma_b = mlx_alloc_array_ptr(mlx_init_conv_bias(norm_nc));

    /* canonical: (out_ch=norm_nc, KH, KW, in_ch=hidden_nc) */
    int bshape[4] = {norm_nc, kernel_size, kernel_size, hidden_nc};
    mlx_array bw = mlx_init_conv_weight(bshape, 4, 0.02f);
    mlx_array *bwp = NULL;
    if (mlx_alloc_pod((void **)&bwp, sizeof(mlx_array), 1) == 0) {
        *bwp = bw;
        m->mlp_beta_w = bwp;
    } else {
        mlx_array_free(bw);
    }
    m->mlp_beta_b = mlx_alloc_array_ptr(mlx_init_conv_bias(norm_nc));
    return m;
}

void mlx_spade_free(MLXSPADE *m) {
    if (!m)
        return;
    if (m->norm) {
        mlx_nn_instancenorm_free(m->norm);
        m->norm = NULL;
    }
    if (m->mlp_shared_w) {
        mlx_array_free(*m->mlp_shared_w);
        mlx_free_pod((void **)&m->mlp_shared_w);
    }
    mlx_free_array_ptr((void **)&m->mlp_shared_b);
    if (m->mlp_gamma_w) {
        mlx_array_free(*m->mlp_gamma_w);
        mlx_free_pod((void **)&m->mlp_gamma_w);
    }
    mlx_free_array_ptr((void **)&m->mlp_gamma_b);
    if (m->mlp_beta_w) {
        mlx_array_free(*m->mlp_beta_w);
        mlx_free_pod((void **)&m->mlp_beta_w);
    }
    mlx_free_array_ptr((void **)&m->mlp_beta_b);
    mlx_free_pod((void **)&m);
}

mlx_array_t mlx_spade_forward(MLXSPADE *m, mlx_array_t x,
                              mlx_array_t conditioning_input, const char *mode,
                              int align_corners) {
    if (!m)
        return x;
    mlx_stream s = mlx_default_gpu_stream_new();

    /* Expect NHWC */
    if (mlx_array_ndim(x) != 4 || mlx_array_ndim(conditioning_input) != 4) {
        mlx_stream_free(s);
        return x;
    }
    const int *xshape = mlx_array_shape(x);
    const int *cshape = mlx_array_shape(conditioning_input);
    int bx = xshape[0], hx = xshape[1], wx = xshape[2], cx = xshape[3];
    int bc = cshape[0], hc = cshape[1], wc = cshape[2], cc = cshape[3];

    /* ALWAYS use bilinear upsampling for conditioning input (parity with Python).
     * Previous implementation used mlx_repeat_axis (nearest-neighbor) for integer
     * scale factors, but Python's SPADE uses bilinear for ALL cases. */
    mlx_array cond_up = mlx_array_new();
    int cond_up_alloc = 0;
    if (hc == hx && wc == wx) {
        cond_up = conditioning_input;
    } else {
        /* Always use bilinear interpolation to match Python */
        MLXUpsample *u = mlx_upsample_create(hx, wx, "bilinear", 1);
        if (u) {
            cond_up = mlx_upsample_forward(u, conditioning_input);
            cond_up_alloc = (cond_up.ctx != 0);
            mlx_upsample_free(u);
        } else {
            cond_up = conditioning_input;
        }
    }

    /* If cond_up spatial dims still don't match x, explicitly upsample to x */
    if (cond_up.ctx) {
        const int *cshape_up = mlx_array_shape(cond_up);
        if (cshape_up && (cshape_up[1] != hx || cshape_up[2] != wx)) {
            MLXUpsample *u2 = mlx_upsample_create(hx, wx, "bilinear", 1);
            if (u2) {
                mlx_array tmp_up = mlx_upsample_forward(u2, cond_up);
                if (tmp_up.ctx) {
                    /* replace cond_up with resized version */
                    if (cond_up.ctx && cond_up.ctx != conditioning_input.ctx)
                        mlx_array_free(cond_up);
                    cond_up = tmp_up;
                }
                mlx_upsample_free(u2);
            }
        }
    }

    /* mlp_shared: conv(cond_up, mlp_shared_w) */
    mlx_array shared = mlx_array_new();
    if (!m->mlp_shared_w) {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_stream_free(s);
        return x;
    }
    if (safe_mlx_conv2d(&shared, cond_up, *m->mlp_shared_w, 1, 1, m->padding,
                        m->padding, 1, 1, 1, s) != 0) {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_stream_free(s);
        return x;
    }
    /* Apply shared conv bias */
    if (m->mlp_shared_b) {
        mlx_array tmp_b = mlx_array_new();
        if (mlx_add(&tmp_b, shared, *m->mlp_shared_b, s) == 0) {
            mlx_array_free(shared);
            shared = tmp_b;
        } else {
            mlx_array_free(tmp_b);
        }
    }

    /* activation */
    MLXLeakyReLU *tmp_act = mlx_leakyrelu_create(0.2f);
    mlx_array shared_act = mlx_leakyrelu_forward(tmp_act, shared);
    mlx_leakyrelu_free(tmp_act);

    /* gamma and beta */
    mlx_array gamma = mlx_array_new();
    mlx_array beta = mlx_array_new();
    if (!m->mlp_gamma_w || !m->mlp_beta_w) {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_array_free(shared);
        mlx_array_free(shared_act);
        mlx_stream_free(s);
        return x;
    }
    if (safe_mlx_conv2d(&gamma, shared_act, *m->mlp_gamma_w, 1, 1, m->padding,
                        m->padding, 1, 1, 1, s) != 0) {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_array_free(shared);
        mlx_array_free(shared_act);
        mlx_stream_free(s);
        return x;
    }
    /* Apply gamma conv bias */
    if (m->mlp_gamma_b) {
        mlx_array tmp_b = mlx_array_new();
        if (mlx_add(&tmp_b, gamma, *m->mlp_gamma_b, s) == 0) {
            mlx_array_free(gamma);
            gamma = tmp_b;
        } else {
            mlx_array_free(tmp_b);
        }
    }
    if (safe_mlx_conv2d(&beta, shared_act, *m->mlp_beta_w, 1, 1, m->padding,
                        m->padding, 1, 1, 1, s) != 0) {
        if (cond_up_alloc)
            mlx_array_free(cond_up);
        mlx_array_free(shared);
        mlx_array_free(shared_act);
        mlx_array_free(gamma);
        mlx_stream_free(s);
        return x;
    }
    /* Apply beta conv bias */
    if (m->mlp_beta_b) {
        mlx_array tmp_b = mlx_array_new();
        if (mlx_add(&tmp_b, beta, *m->mlp_beta_b, s) == 0) {
            mlx_array_free(beta);
            beta = tmp_b;
        } else {
            mlx_array_free(tmp_b);
        }
    }

    if (cond_up_alloc)
        mlx_array_free(cond_up);
    mlx_array_free(shared);
    mlx_array_free(shared_act);

    /* Instance normalize x across H and W axes, then apply affine (like Python self.norm) */
    mlx_nn_instancenorm *norm = (mlx_nn_instancenorm *)m->norm;
    const int axes[] = {1, 2};
    mlx_array mean = mlx_array_new();
    if (mlx_mean_axes(&mean, x, axes, 2, true, s) != 0) {
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_stream_free(s);
        return x;
    }
    mlx_array centered = mlx_array_new();
    if (mlx_subtract(&centered, x, mean, s) != 0) {
        mlx_array_free(mean);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_stream_free(s);
        return x;
    }
    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, centered, s) != 0) {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_stream_free(s);
        return x;
    }
    mlx_array var = mlx_array_new();
    if (mlx_mean_axes(&var, sq, axes, 2, true, s) != 0) {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_stream_free(s);
        return x;
    }
    mlx_array eps = mlx_array_new_float(1e-5f);
    mlx_array var_eps = mlx_array_new();
    if (mlx_add(&var_eps, var, eps, s) != 0) {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_stream_free(s);
        return x;
    }
    mlx_array std = mlx_array_new();
    if (mlx_sqrt(&std, var_eps, s) != 0) {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_stream_free(s);
        return x;
    }
    mlx_array x_norm = mlx_array_new();
    if (mlx_divide(&x_norm, centered, std, s) != 0) {
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_stream_free(s);
        return x;
    }

    /* Apply affine transformation from InstanceNorm: normalized = x_norm * weight + bias */
    mlx_array normalized = x_norm;
    if (norm && norm->affine && norm->weight && norm->bias) {
        mlx_array affine_scaled = mlx_array_new();
        if (mlx_multiply(&affine_scaled, x_norm, *norm->weight, s) == 0) {
            mlx_array affine_out = mlx_array_new();
            if (mlx_add(&affine_out, affine_scaled, *norm->bias, s) == 0) {
                mlx_array_free(x_norm);
                normalized = affine_out;
            }
            mlx_array_free(affine_scaled);
        }
    }

    /* out = normalized * (1 + gamma) + beta */
    mlx_array one = mlx_array_new_float(1.0f);
    mlx_array one_plus_gamma = mlx_array_new();
    if (mlx_add(&one_plus_gamma, gamma, one, s) != 0) {
        /* cleanup */
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(normalized);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_array_free(one);
        mlx_stream_free(s);
        return x;
    }

    mlx_array scaled = mlx_array_new();
    if (mlx_multiply(&scaled, normalized, one_plus_gamma, s) != 0) {
        /* cleanup */
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(normalized);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_array_free(one);
        mlx_array_free(one_plus_gamma);
        mlx_stream_free(s);
        return x;
    }

    mlx_array out = mlx_array_new();
    if (mlx_add(&out, scaled, beta, s) != 0) {
        /* cleanup */
        mlx_array_free(mean);
        mlx_array_free(centered);
        mlx_array_free(sq);
        mlx_array_free(var);
        mlx_array_free(var_eps);
        mlx_array_free(std);
        mlx_array_free(normalized);
        mlx_array_free(gamma);
        mlx_array_free(beta);
        mlx_array_free(eps);
        mlx_array_free(one);
        mlx_array_free(one_plus_gamma);
        mlx_array_free(scaled);
        mlx_stream_free(s);
        return x;
    }

    /* cleanup temporaries */
    mlx_array_free(mean);
    mlx_array_free(centered);
    mlx_array_free(sq);
    mlx_array_free(var);
    mlx_array_free(var_eps);
    mlx_array_free(std);
    mlx_array_free(normalized);
    mlx_array_free(gamma);
    mlx_array_free(beta);
    mlx_array_free(eps);
    mlx_array_free(one);
    mlx_array_free(one_plus_gamma);
    mlx_array_free(scaled);

    {
        /* debug: print SPADE forward output shape if available */
        int out_ndim = mlx_array_ndim(out);
        if (out_ndim == 4) {
            const int *osh = mlx_array_shape(out);
        }
        mlx_stream_free(s);
        return out;
    }
}

/* SPADE mlp weight accessors */
mlx_array *mlx_spade_get_mlp_shared_w(MLXSPADE *m) {
    if (!m)
        return NULL;
    return m->mlp_shared_w ? m->mlp_shared_w : NULL;
}
mlx_array *mlx_spade_get_mlp_gamma_w(MLXSPADE *m) {
    if (!m)
        return NULL;
    return m->mlp_gamma_w ? m->mlp_gamma_w : NULL;
}
mlx_array *mlx_spade_get_mlp_beta_w(MLXSPADE *m) {
    if (!m)
        return NULL;
    return m->mlp_beta_w ? m->mlp_beta_w : NULL;
}
/* SPADE InstanceNorm accessor for AG tracing */
void *mlx_spade_get_norm(MLXSPADE *m) {
    if (!m)
        return NULL;
    return m->norm;
}

struct MLXSPADEConvBlock {
    MLXSPADE *spade;
    void *conv;
    void *conv_bias; /* pointer to mlx_array bias */
    MLXLeakyReLU *activation;
    int stride;
    int padding;
    int kernel_size;
    int in_ch;
    int out_ch;
};

int mlx_spadeconv_get_stride(MLXSPADEConvBlock *m) {
    if (!m)
        return 1;
    return m->stride;
}

int mlx_spade_get_padding(MLXSPADE *m) {
    if (!m)
        return 0;
    return m->padding;
}

MLXSPADEConvBlock *mlx_spadeconv_create(int in_ch, int out_ch, int cond_ch,
                                        int kernel_size, int padding,
                                        int stride, int spade_hidden) {
    MLXSPADEConvBlock *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXSPADEConvBlock), 1) != 0)
        return NULL;
    m->spade = mlx_spade_create(in_ch, cond_ch, spade_hidden, kernel_size);
    m->conv = NULL; /* will store pointer to mlx_array weight */
    m->conv_bias = NULL;
    m->activation = mlx_leakyrelu_create(0.2f);
    m->stride = stride;
    m->padding = padding;
    m->kernel_size = kernel_size;
    m->in_ch = in_ch;
    m->out_ch = out_ch;

    /* MLX canonical conv-weight layout: (out_ch, KH, KW, in_ch) */
    int wshape[4] = {out_ch, kernel_size, kernel_size, in_ch};
    mlx_array w = mlx_init_conv_weight(wshape, 4, 0.02f);
    mlx_array *wptr = NULL;
    if (mlx_alloc_pod((void **)&wptr, sizeof(mlx_array), 1) == 0) {
        *wptr = w;
        m->conv = (void *)wptr;
    } else {
        mlx_array_free(w);
    }
    /* Allocate conv bias (zero-initialised, shape [out_ch]) to match Python nn.Conv2d */
    m->conv_bias = mlx_alloc_array_ptr(mlx_init_conv_bias(out_ch));
    return m;
}

void mlx_spadeconv_free(MLXSPADEConvBlock *m) {
    if (!m)
        return;
    if (m->conv) {
        mlx_array *wptr = (mlx_array *)m->conv;
        mlx_array_free(*wptr);
        mlx_free_pod((void **)&wptr);
        m->conv = NULL;
    }
    mlx_free_array_ptr(&m->conv_bias);
    mlx_spade_free(m->spade);
    mlx_leakyrelu_free(m->activation);
    mlx_free_pod((void **)&m);
}

mlx_array_t mlx_spadeconv_forward(MLXSPADEConvBlock *m, mlx_array_t x,
                                  mlx_array_t cond) {
    if (!m)
        return x;
    mlx_stream s = mlx_default_gpu_stream_new();

    /* 1) SPADE modulation */
    mlx_array y = mlx_spade_forward(m->spade, x, cond, "linear", 1);

    /* debug: print spadeconv intermediate y shape */
    {
        int y_ndim = mlx_array_ndim(y);
        if (y_ndim == 4) {
            const int *ysh = mlx_array_shape(y);
        }
    }
    /* 2) Activation */
    mlx_array act = mlx_leakyrelu_forward(m->activation, y);
    /* free y if activation returned a different array */
    if (((void *)&y) != ((void *)&act)) {
        /* compare internal ctx pointers to avoid freeing the returned array */
        if (y.ctx != act.ctx)
            mlx_array_free(y);
    }
    y = act;

    /* 3) Convolution (if weights present) */
    if (m->conv) {
        mlx_array out = mlx_array_new();
        mlx_array *wptr = (mlx_array *)m->conv;
        if (safe_mlx_conv2d(&out, y, *wptr, m->stride, m->stride, m->padding,
                            m->padding, 1, 1, 1, s) != 0) {
            /* conv failed: cleanup and return input x */
            mlx_array_free(y);
            mlx_stream_free(s);
            return x;
        }
        /* Apply conv bias */
        if (m->conv_bias) {
            mlx_array *bptr = (mlx_array *)m->conv_bias;
            mlx_array biased = mlx_array_new();
            if (mlx_add(&biased, out, *bptr, s) == 0) {
                mlx_array_free(out);
                out = biased;
            } else {
                mlx_array_free(biased);
            }
        }
        /* debug: print conv output shape */
        {
            int out_ndim = mlx_array_ndim(out);
            if (out_ndim == 4) {
                const int *osh = mlx_array_shape(out);
            }
        }
        /* free intermediate activation result */
        mlx_array_free(y);
        mlx_stream_free(s);
        return out;
    }

    /* no conv: return activated/spade-modulated tensor */
    mlx_stream_free(s);
    return y;
}

struct MLXSPADEGenerator {
    void *init_conv;     /* pointer to mlx_array weight */
    void *init_conv_bias; /* pointer to mlx_array bias */
    void **spade_blocks; /* array of pointers to MLXSPADEConvBlock */
    int n_blocks;
    void *tail_conv; /* pointer to mlx_array weight */
    void *tail_conv_bias; /* pointer to mlx_array bias */
    MLXLeakyReLU *leaky_relu;
    void *activation; /* tanh handle (unused) */
    int padding_size;
    int kernel_size;
};

MLXSPADEGenerator *mlx_spadegen_create(int num_layer, int kernel_size,
                                       int padding_size, int num_features,
                                       int min_num_features,
                                       int output_channels,
                                       int input_channels) {
    MLXSPADEGenerator *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXSPADEGenerator), 1) != 0)
        return NULL;
    m->init_conv = NULL; /* conv weight pointer */
    m->init_conv_bias = NULL;
    int body_count = (num_layer - 2) > 0 ? (num_layer - 2) : 0;
    m->n_blocks = body_count;
    m->spade_blocks = NULL;
    m->tail_conv = NULL;
    m->tail_conv_bias = NULL;
    m->padding_size = padding_size;
    m->kernel_size = kernel_size;
    /* Allocate init conv weights: (num_features, KH, KW, input_channels) */
    /* canonical: (out_ch=num_features, KH, KW, in_ch=input_channels) */
    int ishape[4] = {num_features, kernel_size, kernel_size, input_channels};
    mlx_array iw = mlx_init_conv_weight(ishape, 4, 0.02f);
    mlx_array *iwptr = NULL;
    if (mlx_alloc_pod((void **)&iwptr, sizeof(mlx_array), 1) == 0) {
        *iwptr = iw;
        m->init_conv = (void *)iwptr;
    } else {
        mlx_array_free(iw);
    }
    /* init_conv bias (zero-initialised, shape [num_features]) */
    m->init_conv_bias = mlx_alloc_array_ptr(mlx_init_conv_bias(num_features));
    if (body_count > 0) {
        m->spade_blocks = NULL;
        /* prefer helper allocation but fall back to calloc to match previous
         * behavior if the helper fails */
        if (mlx_alloc_ptr_array((void ***)&m->spade_blocks, body_count) != 0) {
            m->spade_blocks = (void **)calloc(body_count, sizeof(void *));
        }
        int current_features = num_features;
        for (int i = 0; i < body_count; ++i) {
            int denom = (1 << (i + 1));
            int block_features = num_features / denom;
            if (block_features < min_num_features)
                block_features = min_num_features;
            int in_ch = current_features;
            int out_ch = block_features;
            /* Use padding_size (passed from generator options) to match Python */
            m->spade_blocks[i] = (void *)mlx_spadeconv_create(
                                     in_ch, out_ch, input_channels, kernel_size, padding_size, 1, 64);
            current_features = out_ch;
        }
        /* Allocate tail conv weights in MLX canonical layout:
         * (out_ch=output_channels, KH, KW, in_ch=current_features) */
        int tshape[4] = {output_channels, kernel_size, kernel_size,
                         current_features
                        };
        mlx_array tw = mlx_init_conv_weight(tshape, 4, 0.02f);
        mlx_array *twptr = NULL;
        if (mlx_alloc_pod((void **)&twptr, sizeof(mlx_array), 1) == 0) {
            *twptr = tw;
            m->tail_conv = (void *)twptr;
        } else {
            mlx_array_free(tw);
        }
        /* tail_conv bias (zero-initialised, shape [output_channels]) */
        m->tail_conv_bias = mlx_alloc_array_ptr(mlx_init_conv_bias(output_channels));
    }
    m->leaky_relu = mlx_leakyrelu_create(0.2f);
    m->activation = NULL; /* tanh handle if available */
    return m;
}

void mlx_spadegen_free(MLXSPADEGenerator *m) {
    if (!m)
        return;
    if (m->spade_blocks) {
        for (int i = 0; i < m->n_blocks; ++i) {
            mlx_spadeconv_free((MLXSPADEConvBlock *)m->spade_blocks[i]);
        }
        /* free via helper if allocated that way, otherwise free the calloc */
        mlx_free_ptr_array((void ***)&m->spade_blocks, m->n_blocks);
    }
    if (m->init_conv) {
        mlx_array *iw = (mlx_array *)m->init_conv;
        mlx_array_free(*iw);
        mlx_free_pod((void **)&iw);
    }
    mlx_free_array_ptr(&m->init_conv_bias);
    if (m->tail_conv) {
        mlx_array *tw = (mlx_array *)m->tail_conv;
        mlx_array_free(*tw);
        mlx_free_pod((void **)&tw);
    }
    mlx_free_array_ptr(&m->tail_conv_bias);
    mlx_leakyrelu_free(m->leaky_relu);
    mlx_free_pod((void **)&m);
}

mlx_array_t mlx_spadegen_forward(MLXSPADEGenerator *m, mlx_array_t cond) {
    if (!m)
        return cond;
    mlx_stream s = mlx_default_gpu_stream_new();

    if (mlx_array_ndim(cond) == 4) {
        const int *cshape = mlx_array_shape(cond);
        (void)cshape; /* suppress unused warning */
    }

    mlx_array x = mlx_array_new();
    if (m->init_conv) {
        mlx_array *iw = (mlx_array *)m->init_conv;
        if (iw->ctx && mlx_array_ndim(*iw) == 4) {
            const int *iwsh = mlx_array_shape(*iw);
            (void)iwsh; /* suppress unused warning */
        }
        /* Use padding_size (not kernel_size/2) to match Python */
        int pad = m->padding_size;
        if (safe_mlx_conv2d(&x, cond, *iw, 1, 1, pad, pad, 1, 1, 1, s) != 0) {
            mlx_stream_free(s);
            return cond;
        }
        /* Apply init_conv bias */
        if (m->init_conv_bias) {
            mlx_array *bptr = (mlx_array *)m->init_conv_bias;
            mlx_array biased = mlx_array_new();
            if (mlx_add(&biased, x, *bptr, s) == 0) {
                mlx_array_free(x);
                x = biased;
            } else {
                mlx_array_free(biased);
            }
        }
        /* debug: print init conv output shape */
        {
            int x_ndim = mlx_array_ndim(x);
            if (x_ndim == 4) {
                const int *xsh = mlx_array_shape(x);
                (void)xsh; /* suppress unused warning */
            }
        }
    } else {
        /* no init conv: treat cond as x */
        x = cond;
    }

    /* leaky relu */
    mlx_array x_act = mlx_leakyrelu_forward(m->leaky_relu, x);
    if (x.ctx != x_act.ctx)
        mlx_array_free(x);
    x = x_act;
    /* debug: print after leakyrelu */
    {
        int x_ndim = mlx_array_ndim(x);
        if (x_ndim == 4) {
            const int *xsh = mlx_array_shape(x);
        }
    }

    /* body spade blocks */
    if (m->spade_blocks) {
        for (int i = 0; i < m->n_blocks; ++i) {
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
    if (m->tail_conv) {
        mlx_array out = mlx_array_new();
        mlx_array *tw = (mlx_array *)m->tail_conv;
        /* Use padding_size (not kernel_size/2) to match Python */
        int pad = m->padding_size;
        if (safe_mlx_conv2d(&out, x, *tw, 1, 1, pad, pad, 1, 1, 1, s) != 0) {
            if (x.ctx != cond.ctx)
                mlx_array_free(x);
            mlx_stream_free(s);
            return cond;
        }
        /* Apply tail_conv bias */
        if (m->tail_conv_bias) {
            mlx_array *bptr = (mlx_array *)m->tail_conv_bias;
            mlx_array biased = mlx_array_new();
            if (mlx_add(&biased, out, *bptr, s) == 0) {
                mlx_array_free(out);
                out = biased;
            } else {
                mlx_array_free(biased);
            }
        }
        /* tanh */
        mlx_array out_t = mlx_array_new();
        if (mlx_tanh(&out_t, out, s) != 0) {
            mlx_array_free(out);
            if (x.ctx != cond.ctx)
                mlx_array_free(x);
            mlx_stream_free(s);
            return cond;
        }
        mlx_array_free(out);
        if (x.ctx != cond.ctx)
            mlx_array_free(x);
        mlx_stream_free(s);
        return out_t;
    }

    mlx_stream_free(s);
    return x;
}

/* Small accessors for SPADE generator internals used by AG tracing. */
int mlx_spadegen_get_n_blocks(MLXSPADEGenerator *m) {
    if (!m)
        return 0;
    return m->n_blocks;
}

MLXSPADEConvBlock *mlx_spadegen_get_block_at(MLXSPADEGenerator *m, int idx) {
    if (!m)
        return NULL;
    if (idx < 0 || idx >= m->n_blocks)
        return NULL;
    return (MLXSPADEConvBlock *)m->spade_blocks[idx];
}

mlx_array *mlx_spadegen_get_init_conv(MLXSPADEGenerator *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->init_conv;
}

mlx_array *mlx_spadegen_get_tail_conv(MLXSPADEGenerator *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->tail_conv;
}

int mlx_spadeconv_get_padding(MLXSPADEConvBlock *m) {
    if (!m)
        return 0;
    return m->padding;
}

MLXLeakyReLU *mlx_spadeconv_get_activation(MLXSPADEConvBlock *m) {
    if (!m)
        return NULL;
    return m->activation;
}

MLXSPADE *mlx_spadeconv_get_spade(MLXSPADEConvBlock *m) {
    if (!m)
        return NULL;
    return m->spade;
}

struct MLXScaleModule {
    void *head;
    void *body;
    void *tail;
};

MLXScaleModule *mlx_scalemodule_create(void *head, void *body, void *tail) {
    MLXScaleModule *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXScaleModule), 1) != 0)
        return NULL;
    m->head = head;
    m->body = body;
    m->tail = tail;
    return m;
}

void mlx_scalemodule_free(MLXScaleModule *m) {
    if (m)
        mlx_free_pod((void **)&m);
}

mlx_array_t mlx_scalemodule_forward(MLXScaleModule *m, mlx_array_t x) {
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

    if (m->head) {
        scale_fn_t fn = (scale_fn_t)m->head;
        mlx_array_t nx = fn(cur);
        if (cur.ctx != nx.ctx)
            mlx_array_free(cur);
        cur = nx;
    }

    if (m->body) {
        /* `body` may point to a single function or an array of functions;
         * here we treat it as a single unary function for simplicity. */
        scale_fn_t fn = (scale_fn_t)m->body;
        mlx_array_t nx = fn(cur);
        if (cur.ctx != nx.ctx)
            mlx_array_free(cur);
        cur = nx;
    }

    if (m->tail) {
        scale_fn_t fn = (scale_fn_t)m->tail;
        mlx_array_t nx = fn(cur);
        if (cur.ctx != nx.ctx)
            mlx_array_free(cur);
        cur = nx;
    }

    return cur;
}

struct MLXSPADEDiscriminator {
    MLXConvBlock *head;  /* head is a MLXConvBlock, not just weights */
    void **body;         /* array of MLXConvBlock pointers */
    int n_body;
    void *tail;          /* tail conv weight pointer */
    void *tail_bias;     /* tail conv bias pointer */
    int kernel_size;
    int padding_size;
};

MLXSPADEDiscriminator *mlx_spadedisc_create(int num_features,
        int min_num_features, int num_layer,
        int kernel_size, int padding_size,
        int input_channels) {
    MLXSPADEDiscriminator *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXSPADEDiscriminator), 1) != 0)
        return NULL;
    m->head = NULL;
    int body_count = (num_layer - 2) > 0 ? (num_layer - 2) : 0;
    m->n_body = body_count;
    m->body = NULL;
    m->tail = NULL;
    m->tail_bias = NULL;
    m->kernel_size = kernel_size;
    m->padding_size = padding_size;

    /* Head is a MLXConvBlock (conv + norm + leakyrelu) matching Python */
    m->head = mlx_convblock_create(input_channels, num_features, kernel_size,
                                   padding_size, 1, 1 /* use_norm=True */);

    if (body_count > 0) {
        m->body = NULL;
        if (mlx_alloc_ptr_array((void ***)&m->body, body_count) != 0) {
            m->body = (void **)calloc(body_count, sizeof(void *));
        }
        int current_features = num_features;
        for (int i = 0; i < body_count; ++i) {
            /* Python: in_ch = max(num_features // (2**i), min_num_features) */
            int in_ch = num_features >> i;
            if (in_ch < min_num_features) in_ch = min_num_features;
            /* Python: out_ch = max(num_features // (2**(i+1)), min_num_features) */
            int out_ch = num_features >> (i + 1);
            if (out_ch < min_num_features) out_ch = min_num_features;
            /* Create a MLXConvBlock (NOT SPADEConvBlock) to match Python */
            m->body[i] = (void *)mlx_convblock_create(
                             in_ch, out_ch, kernel_size, padding_size, 1, 1 /* use_norm=True */);
            current_features = out_ch;
        }
        /* Allocate tail conv weights: canonical MLX layout (out_ch=1, KH, KW,
         * in_ch=current_features) */
        int tshape[4] = {1, kernel_size, kernel_size, current_features};
        mlx_array tw = mlx_init_conv_weight(tshape, 4, 0.02f);
        mlx_array *twptr = NULL;
        if (mlx_alloc_pod((void **)&twptr, sizeof(mlx_array), 1) == 0) {
            *twptr = tw;
            m->tail = (void *)twptr;
        } else {
            mlx_array_free(tw);
        }
        /* tail bias (zero-initialised, shape [1] since out_ch=1 for disc) */
        m->tail_bias = mlx_alloc_array_ptr(mlx_init_conv_bias(1));
    }
    return m;
}

void mlx_spadedisc_free(MLXSPADEDiscriminator *m) {
    if (!m)
        return;
    if (m->head) {
        mlx_convblock_free(m->head);
        m->head = NULL;
    }
    if (m->body) {
        for (int i = 0; i < m->n_body; ++i) {
            if (m->body[i])
                mlx_convblock_free((MLXConvBlock *)m->body[i]);
        }
        mlx_free_ptr_array((void ***)&m->body, m->n_body);
    }
    if (m->tail) {
        mlx_array *tw = (mlx_array *)m->tail;
        mlx_array_free(*tw);
        mlx_free_pod((void **)&tw);
        m->tail = NULL;
    }
    mlx_free_array_ptr(&m->tail_bias);
    mlx_free_pod((void **)&m);
}

/* Introspection helpers */
mlx_array *mlx_spadeconv_get_conv_weight(MLXSPADEConvBlock *m) {
    if (!m)
        return NULL;
    return m->conv;
}

mlx_array *mlx_convblock_get_conv_weight(MLXConvBlock *m) {
    if (!m)
        return NULL;
    return m->conv;
}

mlx_array *mlx_convblock_get_conv_bias(MLXConvBlock *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->conv_bias;
}

mlx_array *mlx_convblock_get_norm_weight(MLXConvBlock *m) {
    if (!m || !m->norm)
        return NULL;
    return mlx_nn_instancenorm_get_weight(m->norm);
}

mlx_array *mlx_convblock_get_norm_bias(MLXConvBlock *m) {
    if (!m || !m->norm)
        return NULL;
    return mlx_nn_instancenorm_get_bias(m->norm);
}

mlx_array *mlx_spadeconv_get_conv_bias(MLXSPADEConvBlock *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->conv_bias;
}

mlx_array *mlx_spade_get_mlp_shared_b(MLXSPADE *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->mlp_shared_b;
}

mlx_array *mlx_spade_get_mlp_gamma_b(MLXSPADE *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->mlp_gamma_b;
}

mlx_array *mlx_spade_get_mlp_beta_b(MLXSPADE *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->mlp_beta_b;
}

mlx_array *mlx_spadegen_get_init_conv_bias(MLXSPADEGenerator *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->init_conv_bias;
}

mlx_array *mlx_spadegen_get_tail_conv_bias(MLXSPADEGenerator *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->tail_conv_bias;
}

mlx_array *mlx_spadedisc_get_tail_bias(MLXSPADEDiscriminator *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->tail_bias;
}

mlx_array **mlx_spadegen_get_parameters(MLXSPADEGenerator *m, int *out_count) {
    if (!m || !out_count)
        return NULL;

    /* Helper: count all trainable parameters matching Python's nn.Module.parameters() */
#define COUNT_IF(ptr) do { if ((ptr)) total++; } while(0)
#define ADD_IF(ptr) do { if ((ptr)) list[idx++] = (mlx_array *)(ptr); } while(0)

    /* --- Counting pass --- */
    int total = 0;
    COUNT_IF(m->init_conv);
    COUNT_IF(m->init_conv_bias);
    for (int i = 0; i < m->n_blocks; ++i) {
        MLXSPADEConvBlock *cb = (MLXSPADEConvBlock *)m->spade_blocks[i];
        if (!cb) continue;
        MLXSPADE *sp = cb->spade;
        if (sp) {
            COUNT_IF(mlx_nn_instancenorm_get_weight(sp->norm));
            COUNT_IF(mlx_nn_instancenorm_get_bias(sp->norm));
            COUNT_IF(sp->mlp_shared_w);
            COUNT_IF(sp->mlp_shared_b);
            COUNT_IF(sp->mlp_gamma_w);
            COUNT_IF(sp->mlp_gamma_b);
            COUNT_IF(sp->mlp_beta_w);
            COUNT_IF(sp->mlp_beta_b);
        }
        COUNT_IF(cb->conv);
        COUNT_IF(cb->conv_bias);
    }
    COUNT_IF(m->tail_conv);
    COUNT_IF(m->tail_conv_bias);

    if (total == 0) {
        *out_count = 0;
        return NULL;
    }

    mlx_array **list = NULL;
    if (mlx_alloc_pod((void **)&list, sizeof(mlx_array *), total) != 0) {
        *out_count = 0;
        return NULL;
    }

    /* --- Filling pass --- */
    int idx = 0;
    ADD_IF(m->init_conv);
    ADD_IF(m->init_conv_bias);
    for (int i = 0; i < m->n_blocks; ++i) {
        MLXSPADEConvBlock *cb = (MLXSPADEConvBlock *)m->spade_blocks[i];
        if (!cb) continue;
        MLXSPADE *sp = cb->spade;
        if (sp) {
            ADD_IF(mlx_nn_instancenorm_get_weight(sp->norm));
            ADD_IF(mlx_nn_instancenorm_get_bias(sp->norm));
            ADD_IF(sp->mlp_shared_w);
            ADD_IF(sp->mlp_shared_b);
            ADD_IF(sp->mlp_gamma_w);
            ADD_IF(sp->mlp_gamma_b);
            ADD_IF(sp->mlp_beta_w);
            ADD_IF(sp->mlp_beta_b);
        }
        ADD_IF(cb->conv);
        ADD_IF(cb->conv_bias);
    }
    ADD_IF(m->tail_conv);
    ADD_IF(m->tail_conv_bias);

#undef COUNT_IF
#undef ADD_IF

    *out_count = idx;
    return list;
}

void mlx_spadegen_free_parameters_list(mlx_array **list) {
    if (list)
        mlx_free_pod((void **)&list);
}

mlx_array **mlx_spadedisc_get_parameters(MLXSPADEDiscriminator *m,
        int *out_count) {
    if (!m || !out_count)
        return NULL;

#define COUNT_IF(ptr) do { if ((ptr)) total++; } while(0)
#define ADD_IF(ptr) do { if ((ptr)) list[idx++] = (mlx_array *)(ptr); } while(0)

    /* --- Counting pass --- */
    int total = 0;
    /* head ConvBlock: conv weight + bias + norm weight + bias */
    if (m->head) {
        COUNT_IF(m->head->conv);
        COUNT_IF(m->head->conv_bias);
        COUNT_IF(m->head->norm ? mlx_nn_instancenorm_get_weight(m->head->norm) : NULL);
        COUNT_IF(m->head->norm ? mlx_nn_instancenorm_get_bias(m->head->norm) : NULL);
    }
    /* body ConvBlocks */
    for (int i = 0; i < m->n_body; ++i) {
        MLXConvBlock *cb = (MLXConvBlock *)m->body[i];
        if (!cb) continue;
        COUNT_IF(cb->conv);
        COUNT_IF(cb->conv_bias);
        COUNT_IF(cb->norm ? mlx_nn_instancenorm_get_weight(cb->norm) : NULL);
        COUNT_IF(cb->norm ? mlx_nn_instancenorm_get_bias(cb->norm) : NULL);
    }
    /* tail conv weight + bias */
    COUNT_IF(m->tail);
    COUNT_IF(m->tail_bias);

    if (total == 0) {
        *out_count = 0;
        return NULL;
    }

    mlx_array **list = NULL;
    if (mlx_alloc_pod((void **)&list, sizeof(mlx_array *), total) != 0) {
        *out_count = 0;
        return NULL;
    }

    /* --- Filling pass --- */
    int idx = 0;
    if (m->head) {
        ADD_IF(m->head->conv);
        ADD_IF(m->head->conv_bias);
        if (m->head->norm) {
            ADD_IF(mlx_nn_instancenorm_get_weight(m->head->norm));
            ADD_IF(mlx_nn_instancenorm_get_bias(m->head->norm));
        }
    }
    for (int i = 0; i < m->n_body; ++i) {
        MLXConvBlock *cb = (MLXConvBlock *)m->body[i];
        if (!cb) continue;
        ADD_IF(cb->conv);
        ADD_IF(cb->conv_bias);
        if (cb->norm) {
            ADD_IF(mlx_nn_instancenorm_get_weight(cb->norm));
            ADD_IF(mlx_nn_instancenorm_get_bias(cb->norm));
        }
    }
    ADD_IF(m->tail);
    ADD_IF(m->tail_bias);

#undef COUNT_IF
#undef ADD_IF

    *out_count = idx;
    return list;
}

void mlx_spadedisc_free_parameters_list(mlx_array **list) {
    if (list)
        mlx_free_pod((void **)&list);
}

/* Accessors for discriminator internals */
mlx_array *mlx_spadedisc_get_head_conv(MLXSPADEDiscriminator *m) {
    if (!m || !m->head)
        return NULL;
    return (mlx_array *)m->head->conv;
}

MLXConvBlock *mlx_spadedisc_get_head_block(MLXSPADEDiscriminator *m) {
    if (!m)
        return NULL;
    return m->head;
}

int mlx_spadedisc_get_body_count(MLXSPADEDiscriminator *m) {
    if (!m)
        return 0;
    return m->n_body;
}

MLXConvBlock *mlx_spadedisc_get_body_at(MLXSPADEDiscriminator *m,
                                        int idx) {
    if (!m)
        return NULL;
    if (idx < 0 || idx >= m->n_body)
        return NULL;
    return (MLXConvBlock *)m->body[idx];
}

mlx_array *mlx_spadedisc_get_tail_conv(MLXSPADEDiscriminator *m) {
    if (!m)
        return NULL;
    return (mlx_array *)m->tail;
}

mlx_array_t mlx_spadedisc_forward(MLXSPADEDiscriminator *m, mlx_array_t x) {
    if (!m)
        return x;
    mlx_stream s = mlx_default_gpu_stream_new();

    /* Head: MLXConvBlock (conv + norm + leakyrelu) */
    mlx_array cur = mlx_convblock_forward(m->head, x);

    /* Body: sequence of MLXConvBlocks */
    for (int i = 0; i < m->n_body; ++i) {
        MLXConvBlock *blk = (MLXConvBlock *)m->body[i];
        if (!blk) continue;
        mlx_array nx = mlx_convblock_forward(blk, cur);
        if (cur.ctx != x.ctx && cur.ctx != nx.ctx)
            mlx_array_free(cur);
        cur = nx;
    }

    /* Tail: just a conv (no norm, no activation) */
    if (m->tail) {
        mlx_array out = mlx_array_new();
        mlx_array *tw = (mlx_array *)m->tail;
        if (safe_mlx_conv2d(&out, cur, *tw, 1, 1, m->padding_size, m->padding_size,
                            1, 1, 1, s) != 0) {
            if (cur.ctx != x.ctx)
                mlx_array_free(cur);
            mlx_stream_free(s);
            return x;
        }
        /* Apply tail bias */
        if (m->tail_bias) {
            mlx_array *bptr = (mlx_array *)m->tail_bias;
            mlx_array biased = mlx_array_new();
            if (mlx_add(&biased, out, *bptr, s) == 0) {
                mlx_array_free(out);
                out = biased;
            } else {
                mlx_array_free(biased);
            }
        }
        if (cur.ctx != x.ctx)
            mlx_array_free(cur);
        mlx_stream_free(s);
        return out;
    }

    mlx_stream_free(s);
    return cur;
}

struct MLXColorQuantization {
    float temperature;
    /* pure colors stored as an mlx array if available; otherwise keep as a C
     * array */
    float pure_colors[4][3];
};

MLXColorQuantization *mlx_colorquant_create(float temperature) {
    MLXColorQuantization *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXColorQuantization), 1) != 0)
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

void mlx_colorquant_free(MLXColorQuantization *m) {
    if (m)
        mlx_free_pod((void **)&m);
}

mlx_array_t mlx_colorquant_forward(MLXColorQuantization *m, mlx_array_t x,
                                   int training) {
    if (!m)
        return x;

    /* Use default CPU stream */
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array pure = mlx_array_new();
    mlx_array neg2 = mlx_array_new();
    mlx_array temp_arr = mlx_array_new();
    mlx_array neg = mlx_array_new();

#define CLEANUP_RETURN(ret)                                                    \
    do {                                                                       \
        if (pure.ctx)                                                         \
            mlx_array_free(pure);                                             \
        if (neg2.ctx)                                                         \
            mlx_array_free(neg2);                                             \
        if (temp_arr.ctx)                                                     \
            mlx_array_free(temp_arr);                                         \
        if (neg.ctx)                                                          \
            mlx_array_free(neg);                                              \
        mlx_stream_free(s);                                                   \
        return (ret);                                                         \
    } while (0)

    /* Read shape: expect NHWC (b,h,w,c) */
    size_t ndim = mlx_array_ndim(x);
    if (ndim != 4) {
        /* unsupported shape: return input as-is */
        CLEANUP_RETURN(x);
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
    if (mlx_reshape(&flat, x, flat_shape, 2, s) != 0) {
        CLEANUP_RETURN(x);
    }

    /* x_norm = sum(x_flat**2, axis=1, keepdims=True) */
    mlx_array x_sq = mlx_array_new();
    if (mlx_square(&x_sq, flat, s) != 0) {
        mlx_array_free(flat);
        CLEANUP_RETURN(x);
    }
    mlx_array x_norm = mlx_array_new();
    if (mlx_sum_axis(&x_norm, x_sq, 1, true, s) != 0) {
        mlx_array_free(flat);
        mlx_array_free(x_sq);
        CLEANUP_RETURN(x);
    }
    mlx_array_free(x_sq);

    /* pure_colors as an mlx array (K, c) where K=4 */
    int pc_shape[2] = {4, c};
    pure = mlx_array_new_data(m->pure_colors, pc_shape, 2, MLX_FLOAT32);

    /* c_norm = sum(pure**2, axis=1, keepdims=True) */
    mlx_array pure_sq = mlx_array_new();
    if (mlx_square(&pure_sq, pure, s) != 0) {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        CLEANUP_RETURN(x);
    }
    mlx_array c_norm = mlx_array_new();
    if (mlx_sum_axis(&c_norm, pure_sq, 1, true, s) != 0) {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(pure_sq);
        CLEANUP_RETURN(x);
    }
    mlx_array_free(pure_sq);

    /* prod = x_flat @ pure.T  (N, K) */
    mlx_array pure_t = mlx_array_new();
    if (mlx_transpose(&pure_t, pure, s) != 0) {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        CLEANUP_RETURN(x);
    }
    mlx_array prod = mlx_array_new();
    if (mlx_matmul(&prod, flat, pure_t, s) != 0) {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        CLEANUP_RETURN(x);
    }

    /* distances = x_norm + c_norm.T - 2 * prod */
    mlx_array c_norm_t = mlx_array_new();
    if (mlx_transpose(&c_norm_t, c_norm, s) != 0) {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        CLEANUP_RETURN(x);
    }

    /* neg2 = scalar -2.0 */
    neg2 = mlx_array_new_float(-2.0f);
    mlx_array neg2prod = mlx_array_new();
    if (mlx_multiply(&neg2prod, prod, neg2, s) != 0) {
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        CLEANUP_RETURN(x);
    }

    mlx_array tmp = mlx_array_new();
    if (mlx_add(&tmp, x_norm, neg2prod, s) != 0) {
        /* cleanup */
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        CLEANUP_RETURN(x);
    }

    mlx_array distances = mlx_array_new();
    if (mlx_add(&distances, tmp, c_norm_t, s) != 0) {
        /* cleanup */
        mlx_array_free(flat);
        mlx_array_free(x_norm);
        mlx_array_free(c_norm);
        mlx_array_free(pure_t);
        mlx_array_free(prod);
        mlx_array_free(c_norm_t);
        mlx_array_free(neg2prod);
        mlx_array_free(tmp);
        CLEANUP_RETURN(x);
    }

    /* If not training: nearest neighbor assignment (hard quantization)
     * Match Python: indices = mx.argmin(distances, axis=-1); quantized = pure_colors[indices]
     */
    if (!training) {
        /* Try host-side quantization first (faster for small arrays) */
        mlx_array quant = mlx_array_new();
        const float *flat_host = NULL;
        const float *pure_host = NULL;
        int host_quant_success = 0;

        {
            bool ok_flat = false;
            bool ok_pure = false;
            if (_mlx_array_is_available(&ok_flat, flat) == 0 && ok_flat &&
                    _mlx_array_is_available(&ok_pure, pure) == 0 && ok_pure) {
                flat_host = mlx_array_data_float32(flat);
                pure_host = mlx_array_data_float32(pure);
            }
        }
        if (flat_host && pure_host) {
            if (mlx_quantize_array(flat, &quant, pure) == 0) {
                int outshape[4] = {b, h, w, c};
                mlx_array out = mlx_array_new();
                if (mlx_reshape(&out, quant, outshape, 4, s) == 0) {
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
                    CLEANUP_RETURN(out);
                }
                mlx_array_free(out);
            }
            mlx_array_free(quant);
        }

        /* GPU-side hard quantization: indices = argmin(distances, axis=-1)
         * This matches Python: indices = mx.argmin(distances, axis=-1) */
        mlx_array indices = mlx_array_new();
        if (mlx_argmin_axis(&indices, distances, 1, false, s) == 0) {
            /* quantized = pure_colors[indices] via take along axis 0
             * pure has shape (K, c), indices has shape (N,)
             * We need to gather pure[indices[i]] for each pixel i
             */
            mlx_array quantized_flat = mlx_array_new();
            if (mlx_take_axis(&quantized_flat, pure, indices, 0, s) == 0) {
                int outshape[4] = {b, h, w, c};
                mlx_array out = mlx_array_new();
                if (mlx_reshape(&out, quantized_flat, outshape, 4, s) == 0) {
                    /* cleanup temporaries */
                    mlx_array_free(quantized_flat);
                    mlx_array_free(indices);
                    mlx_array_free(flat);
                    mlx_array_free(x_norm);
                    mlx_array_free(c_norm);
                    mlx_array_free(pure_t);
                    mlx_array_free(prod);
                    mlx_array_free(c_norm_t);
                    mlx_array_free(neg2prod);
                    mlx_array_free(tmp);
                    mlx_array_free(distances);
                    CLEANUP_RETURN(out);
                }
                mlx_array_free(out);
            }
            mlx_array_free(quantized_flat);
        }
        mlx_array_free(indices);
        /* If GPU quantization failed, fall through to soft assignment */
    }

    /* Training: soft assignment
     * weights = softmax(-distances / temperature, axis=-1)
     * quantized = weights @ pure  -> shape (N, c)
     */
    temp_arr = mlx_array_new_float(m->temperature);
    mlx_array div = mlx_array_new();
    if (mlx_divide(&div, distances, temp_arr, s) != 0) {
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
        CLEANUP_RETURN(x);
    }

    neg = mlx_array_new_float(-1.0f);
    mlx_array scaled = mlx_array_new();
    if (mlx_multiply(&scaled, div, neg, s) != 0) {
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
        CLEANUP_RETURN(x);
    }

    mlx_array weights = mlx_array_new();
    if (mlx_softmax_axis(&weights, scaled, 1, false, s) != 0) {
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
        CLEANUP_RETURN(x);
    }

    /* quant_flat = weights @ pure  (N, c) */
    mlx_array quant_flat = mlx_array_new();
    if (mlx_matmul(&quant_flat, weights, pure, s) != 0) {
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
        CLEANUP_RETURN(x);
    }

    int outshape[4] = {b, h, w, c};
    mlx_array out = mlx_array_new();
    if (mlx_reshape(&out, quant_flat, outshape, 4, s) != 0) {
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
        CLEANUP_RETURN(x);
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

    CLEANUP_RETURN(out);
}
