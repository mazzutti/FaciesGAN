#include "utils.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* MLX headers for array helpers */
#include "options.h"
#include <errno.h>
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/transforms.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <math.h>

#include "trainning/array_helpers.h"

int mlx_create_dirs(const char *path) {
    if (!path || path[0] == '\0')
        return -1;
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len >= sizeof(tmp))
        return -1;
    strcpy(tmp, path);
    /* Remove trailing slashes */
    while (len > 1 && tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
        --len;
    }

    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0) {
                if (errno != EEXIST)
                    return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0) {
        if (errno != EEXIST)
            return -1;
    }
    return 0;
}

void ensure_dir(const char *path) {
    if (!path || !*path)
        return;

    char tmp[PATH_BUFSZ];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';

    /* Remove trailing slashes (but keep single leading '/' for absolute) */
    size_t len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/') {
        tmp[--len] = '\0';
    }

    /* Iterate prefixes and create directories as needed (mkdir -p behavior) */
    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            struct stat st;
            if (stat(tmp, &st) == -1) {
                if (mkdir(tmp, 0755) == -1 && errno != EEXIST) {
                    fprintf(stderr, "error: mkdir '%s' failed: %s\n", tmp,
                            strerror(errno));
                    *p = '/';
                    return;
                }
            } else if (!S_ISDIR(st.st_mode)) {
                fprintf(stderr,
                        "error: path component '%s' exists and is not a directory\n",
                        tmp);
                *p = '/';
                return;
            }
            *p = '/';
        }
    }

    /* Create the final directory */
    struct stat st;
    if (stat(tmp, &st) == -1) {
        if (mkdir(tmp, 0755) == -1 && errno != EEXIST) {
            fprintf(stderr, "error: mkdir '%s' failed: %s\n", tmp, strerror(errno));
            return;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr, "error: path '%s' exists and is not a directory\n", tmp);
        return;
    }
}

void write_options_json(const TrainningOptions *topt,
                        const int *wells_mask_columns,
                        size_t wells_mask_count) {
    if (!topt || !topt->output_path)
        return;
    char optpath[PATH_BUFSZ];
    join_path(optpath, sizeof(optpath), topt->output_path, OPT_FILE);
    FILE *of = fopen(optpath, "w");
    if (!of)
        return;

    /* Follow the Python writer order and keys closely to maximize parity. */
    fprintf(of, "{\n");

    /* local helper function-like macro */
#define PRINT_DOUBLE(FP)                                                       \
  do {                                                                         \
    double __v = (double)(FP);                                                 \
    if (isfinite(__v) && fabs(__v - round(__v)) < 1e-9)                        \
      fprintf(of, "%.1f", __v);                                                \
    else                                                                       \
      fprintf(of, "%g", __v);                                                  \
  } while (0)

    fprintf(of, "    \"alpha\": %d,\n", topt->alpha);
    fprintf(of, "    \"batch_size\": %d,\n", topt->batch_size);
    fprintf(of, "    \"beta1\": ");
    PRINT_DOUBLE(topt->beta1);
    fprintf(of, ",\n");
    fprintf(of, "    \"crop_size\": %d,\n", topt->crop_size);
    fprintf(of, "    \"discriminator_steps\": %d,\n", topt->discriminator_steps);
    fprintf(of, "    \"num_img_channels\": %d,\n", topt->num_img_channels);
    fprintf(of, "    \"gamma\": ");
    PRINT_DOUBLE(topt->gamma);
    fprintf(of, ",\n");
    fprintf(of, "    \"generator_steps\": %d,\n", topt->generator_steps);
    fprintf(of, "    \"gpu_device\": %d,\n", topt->gpu_device);
    /* Print arrays in multi-line form to match Python's json.dump(indent=4) */
    fprintf(of, "    \"img_color_range\": [\n");
    fprintf(of, "        %d,\n", topt->img_color_min);
    fprintf(of, "        %d\n", topt->img_color_max);
    fprintf(of, "    ],\n");
    fprintf(of, "    \"input_path\": \"%s\",\n",
            topt->input_path ? topt->input_path : "");
    fprintf(of, "    \"kernel_size\": %d,\n", topt->kernel_size);
    fprintf(of, "    \"lambda_grad\": ");
    PRINT_DOUBLE(topt->lambda_grad);
    fprintf(of, ",\n");
    fprintf(of, "    \"lr_d\": ");
    PRINT_DOUBLE(topt->lr_d);
    fprintf(of, ",\n");
    fprintf(of, "    \"lr_decay\": %d,\n", topt->lr_decay);
    fprintf(of, "    \"lr_g\": ");
    PRINT_DOUBLE(topt->lr_g);
    fprintf(of, ",\n");
    if (topt->manual_seed < 0)
        fprintf(of, "    \"manual_seed\": null,\n");
    else
        fprintf(of, "    \"manual_seed\": %d,\n", topt->manual_seed);
    fprintf(of, "    \"max_size\": %d,\n", topt->max_size);
    fprintf(of, "    \"min_num_feature\": %d,\n", topt->min_num_feature);
    fprintf(of, "    \"min_size\": %d,\n", topt->min_size);
    fprintf(of, "    \"noise_amp\": ");
    PRINT_DOUBLE(topt->noise_amp);
    fprintf(of, ",\n");
    fprintf(of, "    \"min_noise_amp\": ");
    PRINT_DOUBLE(topt->min_noise_amp);
    fprintf(of, ",\n");
    fprintf(of, "    \"scale0_noise_amp\": ");
    PRINT_DOUBLE(topt->scale0_noise_amp);
    fprintf(of, ",\n");
    fprintf(of, "    \"well_loss_penalty\": ");
    PRINT_DOUBLE(topt->well_loss_penalty);
    fprintf(of, ",\n");
    fprintf(of, "    \"lambda_diversity\": ");
    PRINT_DOUBLE(topt->lambda_diversity);
    fprintf(of, ",\n");
    fprintf(of, "    \"num_diversity_samples\": %d,\n",
            topt->num_diversity_samples);
    fprintf(of, "    \"num_feature\": %d,\n", topt->num_feature);
    fprintf(of, "    \"num_generated_per_real\": %d,\n",
            topt->num_generated_per_real);
    fprintf(of, "    \"num_iter\": %d,\n", topt->num_iter);
    fprintf(of, "    \"num_layer\": %d,\n", topt->num_layer);
    fprintf(of, "    \"num_real_facies\": %d,\n", topt->num_real_facies);
    fprintf(of, "    \"num_train_pyramids\": %d,\n", topt->num_train_pyramids);
    fprintf(of, "    \"num_parallel_scales\": %d,\n", topt->num_parallel_scales);
    fprintf(of, "    \"noise_channels\": %d,\n", topt->noise_channels);
    fprintf(of, "    \"num_workers\": %d,\n", topt->num_workers);
    fprintf(of, "    \"output_path\": \"%s\",\n",
            topt->output_path ? topt->output_path : "");
    fprintf(of, "    \"padding_size\": %d,\n", topt->padding_size);
    fprintf(of, "    \"regen_npy_gz\": %s,\n", bool_str(topt->regen_npy_gz));
    fprintf(of, "    \"save_interval\": %d,\n", topt->save_interval);
    fprintf(of, "    \"start_scale\": %d,\n", topt->start_scale);
    fprintf(of, "    \"stride\": %d,\n", topt->stride);
    fprintf(of, "    \"stop_scale\": %d,\n", topt->stop_scale);
    fprintf(of, "    \"use_cpu\": %s,\n", bool_str(topt->use_cpu));
    fprintf(of, "    \"use_mlx\": %s,\n", bool_str(topt->use_mlx));
    fprintf(of, "    \"use_wells\": %s,\n", bool_str(topt->use_wells));
    fprintf(of, "    \"use_seismic\": %s,\n", bool_str(topt->use_seismic));
    if (wells_mask_count == 0) {
        fprintf(of, "    \"wells_mask_columns\": [],\n");
    } else {
        fprintf(of, "    \"wells_mask_columns\": [\n");
        for (size_t wi = 0; wi < wells_mask_count; ++wi) {
            fprintf(of, "        %d", wells_mask_columns[wi]);
            if (wi + 1 < wells_mask_count)
                fprintf(of, ",\n");
            else
                fprintf(of, "\n");
        }
        fprintf(of, "    ],\n");
    }
    fprintf(of, "    \"enable_tensorboard\": %s,\n",
            bool_str(topt->enable_tensorboard));
    fprintf(of, "    \"enable_plot_facies\": %s,\n",
            bool_str(topt->enable_plot_facies));
    /* Make compile_backend the final key to match Python's ordering (no aliases).
     */
    fprintf(of, "    \"compile_backend\": %s\n", bool_str(topt->compile_backend));
    /* Match Python json.dump which does not append a trailing newline */
    fprintf(of, "}");
    fclose(of);
}

void format_timestamp(char *buf, size_t bufsz) {
    if (!buf || bufsz == 0)
        return;
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(buf, bufsz, "%Y_%m_%d_%H_%M_%S", &tm);
}

const char *bool_str(int v) {
    return v ? "true" : "false";
}

void join_path(char *dst, size_t dstsz, const char *a, const char *b) {
    if (!dst || dstsz == 0)
        return;
    if (!a)
        a = "";
    if (!b)
        b = "";
    size_t al = strlen(a);
    if (al > 0 && a[al - 1] == '/')
        snprintf(dst, dstsz, "%s%s", a, b);
    else
        snprintf(dst, dstsz, "%s/%s", a, b);
}

void mlx_set_seed(int seed) {
    /* Seed the C library RNG. */
    srand((unsigned int)seed);
    /* Seed POSIX drand48 family as well */
    srand48((long)seed);
}

int mlx_clamp(mlx_array *res, const mlx_array a, float min_val, float max_val,
              const mlx_stream s) {
    if (!res)
        return -1;

    /* Create scalar arrays for min and max */
    mlx_array min_scalar = mlx_array_new_float(min_val);
    mlx_array max_scalar = mlx_array_new_float(max_val);

    int rc = 0;
    mlx_array min_arr = mlx_array_new();
    mlx_array max_arr = mlx_array_new();
    mlx_array tmp = mlx_array_new();

    /* Fill arrays with scalar values matching `a` shape */
    rc = mlx_full_like(&min_arr, a, min_scalar, mlx_array_dtype(a), s);
    if (rc != 0)
        goto cleanup;
    rc = mlx_full_like(&max_arr, a, max_scalar, mlx_array_dtype(a), s);
    if (rc != 0)
        goto cleanup;

    /* tmp = maximum(a, min_arr) */
    rc = mlx_maximum(&tmp, a, min_arr, s);
    if (rc != 0)
        goto cleanup;

    /* res = minimum(tmp, max_arr) */
    rc = mlx_minimum(res, tmp, max_arr, s);

cleanup:
    /* Free temporaries and scalars */
    mlx_array_free(min_scalar);
    mlx_array_free(max_scalar);
    /* mlx_array_free returns int but we ignore errors during cleanup */
    mlx_array_free(min_arr);
    mlx_array_free(max_arr);
    mlx_array_free(tmp);
    return rc;
}

/* Begin moved from utils_extra.c */

int mlx_array_to_float_buffer(const mlx_array a, float **out_buf,
                              size_t *out_elems, int *out_ndim,
                              int **out_shape) {
    if (!out_buf || !out_elems)
        return -1;

    bool ok_avail = false;
    if (_mlx_array_is_available(&ok_avail, a) != 0 || !ok_avail)
        return -1;
    if (mlx_array_dtype(a) != MLX_FLOAT32)
        return -1;
    size_t elems = (size_t)mlx_array_size(a);
    const float *data = mlx_array_data_float32(a);
    if (!data)
        return -1;
    if (elems > INT_MAX)
        return -1;
    float *buf = NULL;
    if (mlx_alloc_float_buf(&buf, (int)elems) != 0)
        return -1;
    memcpy(buf, data, sizeof(float) * elems);
    *out_buf = buf;
    *out_elems = elems;
    if (out_ndim)
        *out_ndim = mlx_array_ndim(a);
    if (out_shape) {
        int ndim = mlx_array_ndim(a);
        int *shape = NULL;
        if (ndim > 0) {
            if (mlx_alloc_int_array(&shape, ndim) != 0) {
                mlx_free_float_buf(&buf, NULL);
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

int mlx_array_from_float_buffer(mlx_array *out, const float *buf,
                                const int *shape, int ndim) {
    if (!out || !buf)
        return -1;
    size_t elems = 1;
    for (int i = 0; i < ndim; ++i)
        elems *= (size_t)shape[i];
    if (elems > INT_MAX)
        return -1;
    float *tmp = NULL;
    if (mlx_alloc_float_buf(&tmp, (int)elems) != 0)
        return -1;
    memcpy(tmp, buf, sizeof(float) * elems);
    int rc = mlx_array_set_data(out, tmp, shape, ndim, MLX_FLOAT32);
    mlx_free_float_buf(&tmp, NULL);
    return rc == 0 ? 0 : -1;
}

void quantize_pixels_float(const float *in_pixels, float *out_pixels,
                           size_t npixels, int c, const float *palette,
                           int ncolors) {
    if (!in_pixels || !out_pixels || !palette)
        return;
    for (size_t i = 0; i < npixels; ++i) {
        const float *px = in_pixels + (size_t)i * c;
        int best = 0;
        float best_dist = FLT_MAX;
        for (int p = 0; p < ncolors; ++p) {
            const float *col = palette + (size_t)p * c;
            float d = 0.0f;
            for (int k = 0; k < c; ++k) {
                float diff = px[k] - col[k];
                d += diff * diff;
            }
            if (d < best_dist) {
                best_dist = d;
                best = p;
            }
        }
        const float *sel = palette + (size_t)best * c;
        for (int k = 0; k < c; ++k)
            out_pixels[i * c + k] = sel[k];
    }
}

void apply_well_mask_cpu(const float *facies, float *out, int h, int w, int c,
                         const unsigned char *mask, const float *well, int wc) {
    if (!facies || !out || !mask || !well)
        return;
    size_t npixels = (size_t)h * (size_t)w;
    for (size_t i = 0; i < npixels * (size_t)c; ++i)
        out[i] = facies[i];

    unsigned char *col_has = NULL;
    if (w > 0) {
        if (mlx_alloc_pod((void **)&col_has, sizeof(unsigned char), w) != 0)
            return;
        memset(col_has, 0, (size_t)w);
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (mask[y * w + x])
                col_has[x] = 1;
        }
    }

    for (int x = 0; x < w; ++x) {
        if (col_has[x]) {
            for (int y = 0; y < h; ++y) {
                for (int ch = 0; ch < c; ++ch) {
                    out[(y * w + x) * c + ch] = 1.0f;
                }
            }
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            if (!mask[idx])
                continue;
            float brightness = 0.0f;
            if (wc == c) {
                for (int ch = 0; ch < c; ++ch)
                    brightness += fabsf(well[idx * wc + ch]);
                brightness = brightness / (float)c;
            } else if (wc == 1) {
                brightness = fabsf(well[idx]);
            }
            if (brightness < 0.3f)
                continue;
            if (wc == c) {
                for (int ch = 0; ch < c; ++ch)
                    out[idx * c + ch] = well[idx * wc + ch];
            } else if (wc == 1) {
                for (int ch = 0; ch < c; ++ch)
                    out[idx * c + ch] = well[idx];
            }
        }
    }

    if (col_has)
        mlx_free_pod((void **)&col_has);
}

int mlx_denorm_array(const mlx_array in, mlx_array *out, int ceiling) {
    if (!out)
        return -1;
    bool ok_avail = false;
    if (_mlx_array_is_available(&ok_avail, in) != 0 || !ok_avail)
        return -1;
    if (mlx_array_dtype(in) != MLX_FLOAT32)
        return -1;
    size_t elems = mlx_array_size(in);
    const float *data = mlx_array_data_float32(in);
    if (!data)
        return -1;
    if (elems > INT_MAX)
        return -1;
    float *buf = NULL;
    if (mlx_alloc_float_buf(&buf, (int)elems) != 0)
        return -1;
    for (size_t i = 0; i < elems; ++i) {
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
    mlx_array out_arr = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
    *out = out_arr;
    mlx_free_float_buf(&buf, NULL);
    return 0;
}

int mlx_quantize_array(const mlx_array in, mlx_array *out,
                       const mlx_array palette) {
    if (!out)
        return -1;
    float *in_buf = NULL;
    size_t in_elems = 0;
    int in_ndim = 0;
    int *in_shape = NULL;
    if (mlx_array_to_float_buffer(in, &in_buf, &in_elems, &in_ndim, &in_shape) !=
            0)
        return -1;
    float *pal_buf = NULL;
    size_t pal_elems = 0;
    int pal_ndim = 0;
    int *pal_shape = NULL;
    if (mlx_array_to_float_buffer(palette, &pal_buf, &pal_elems, &pal_ndim,
                                  &pal_shape) != 0) {
        mlx_free_float_buf(&in_buf, NULL);
        if (in_shape)
            mlx_free_int_array(&in_shape, NULL);
        return -1;
    }
    int c = 1;
    int h = 1, w = 1;
    if (in_ndim == 3) {
        h = in_shape[0];
        w = in_shape[1];
        c = in_shape[2];
    } else if (in_ndim == 2) {
        h = in_shape[0];
        w = in_shape[1];
        c = 1;
    } else {
        mlx_free_float_buf(&in_buf, NULL);
        mlx_free_float_buf(&pal_buf, NULL);
        if (in_shape)
            mlx_free_int_array(&in_shape, NULL);
        if (pal_shape)
            mlx_free_int_array(&pal_shape, NULL);
        return -1;
    }
    size_t npixels = (size_t)h * (size_t)w;
    int ncolors = (int)(pal_elems / (size_t)c);
    if (npixels * (size_t)c > (size_t)INT_MAX) {
        mlx_free_float_buf(&in_buf, NULL);
        mlx_free_float_buf(&pal_buf, NULL);
        if (in_shape)
            mlx_free_int_array(&in_shape, NULL);
        if (pal_shape)
            mlx_free_int_array(&pal_shape, NULL);
        return -1;
    }
    float *out_buf = NULL;
    if (mlx_alloc_float_buf(&out_buf, (int)(npixels * (size_t)c)) != 0) {
        mlx_free_float_buf(&in_buf, NULL);
        mlx_free_float_buf(&pal_buf, NULL);
        if (in_shape)
            mlx_free_int_array(&in_shape, NULL);
        if (pal_shape)
            mlx_free_int_array(&pal_shape, NULL);
        return -1;
    }
    quantize_pixels_float(in_buf, out_buf, npixels, c, pal_buf, ncolors);
    int shape_out[3];
    int ndim_out = 0;
    if (c == 1) {
        shape_out[0] = h;
        shape_out[1] = w;
        ndim_out = 2;
    } else {
        shape_out[0] = h;
        shape_out[1] = w;
        shape_out[2] = c;
        ndim_out = 3;
    }
    mlx_array out_arr =
        mlx_array_new_data(out_buf, shape_out, ndim_out, MLX_FLOAT32);
    *out = out_arr;
    mlx_free_float_buf(&in_buf, NULL);
    mlx_free_float_buf(&pal_buf, NULL);
    if (in_shape)
        mlx_free_int_array(&in_shape, NULL);
    if (pal_shape)
        mlx_free_int_array(&pal_shape, NULL);
    mlx_free_float_buf(&out_buf, NULL);
    return 0;
}

int mlx_apply_well_mask_array(const mlx_array facies, mlx_array *out,
                              const mlx_array mask, const mlx_array well) {
    if (!out)
        return -1;
    float *fac_buf = NULL;
    size_t fac_elems = 0;
    int fac_ndim = 0;
    int *fac_shape = NULL;
    if (mlx_array_to_float_buffer(facies, &fac_buf, &fac_elems, &fac_ndim,
                                  &fac_shape) != 0)
        return -1;
    bool ok_mask = false;
    if (_mlx_array_is_available(&ok_mask, mask) != 0 || !ok_mask) {
        mlx_free_float_buf(&fac_buf, NULL);
        if (fac_shape)
            mlx_free_int_array(&fac_shape, NULL);
        return -1;
    }
    int m_dtype = mlx_array_dtype(mask);
    size_t mask_elems = mlx_array_size(mask);
    unsigned char *mask_buf = NULL;
    int mask_buf_helper = 0;
    if (mask_elems > 0) {
        if (mask_elems > (size_t)INT_MAX) {
            mask_buf = (unsigned char *)malloc(mask_elems);
            mask_buf_helper = 0;
        } else {
            if (mlx_alloc_pod((void **)&mask_buf, sizeof(unsigned char),
                              (int)mask_elems) == 0)
                mask_buf_helper = 1;
            else
                mask_buf = (unsigned char *)malloc(mask_elems);
        }
        if (!mask_buf) {
            mlx_free_float_buf(&fac_buf, NULL);
            if (fac_shape)
                mlx_free_int_array(&fac_shape, NULL);
            return -1;
        }
    }
    if (m_dtype == MLX_BOOL) {
        const bool *mb = mlx_array_data_bool(mask);
        for (size_t i = 0; i < mask_elems; ++i)
            mask_buf[i] = mb[i] ? 1 : 0;
    } else if (m_dtype == MLX_UINT8) {
        const uint8_t *mb = mlx_array_data_uint8(mask);
        for (size_t i = 0; i < mask_elems; ++i)
            mask_buf[i] = mb[i];
    } else {
        mlx_free_float_buf(&fac_buf, NULL);
        if (fac_shape)
            mlx_free_int_array(&fac_shape, NULL);
        if (mask_buf) {
            if (mask_buf_helper)
                mlx_free_pod((void **)&mask_buf);
            else
                free(mask_buf);
        }
        return -1;
    }
    float *well_buf = NULL;
    size_t well_elems = 0;
    int well_ndim = 0;
    int *well_shape = NULL;
    if (mlx_array_to_float_buffer(well, &well_buf, &well_elems, &well_ndim,
                                  &well_shape) != 0) {
        mlx_free_float_buf(&fac_buf, NULL);
        if (fac_shape)
            mlx_free_int_array(&fac_shape, NULL);
        if (mask_buf) {
            if (mask_buf_helper)
                mlx_free_pod((void **)&mask_buf);
            else
                free(mask_buf);
        }
        return -1;
    }
    int h = 1, w = 1, c = 1, wc = 1;
    if (fac_ndim == 3) {
        h = fac_shape[0];
        w = fac_shape[1];
        c = fac_shape[2];
    } else if (fac_ndim == 2) {
        h = fac_shape[0];
        w = fac_shape[1];
        c = 1;
    }
    if (well_ndim == 3) {
        wc = well_shape[2];
    } else if (well_ndim == 2) {
        wc = 1;
    }
    if (fac_elems > INT_MAX) {
        mlx_free_float_buf(&fac_buf, NULL);
        if (fac_shape)
            mlx_free_int_array(&fac_shape, NULL);
        if (mask_buf) {
            if (mask_buf_helper)
                mlx_free_pod((void **)&mask_buf);
            else
                free(mask_buf);
        }
        mlx_free_float_buf(&well_buf, NULL);
        if (well_shape)
            mlx_free_int_array(&well_shape, NULL);
        return -1;
    }
    float *out_buf = NULL;
    if (mlx_alloc_float_buf(&out_buf, (int)fac_elems) != 0) {
        mlx_free_float_buf(&fac_buf, NULL);
        if (fac_shape)
            mlx_free_int_array(&fac_shape, NULL);
        if (mask_buf) {
            if (mask_buf_helper)
                mlx_free_pod((void **)&mask_buf);
            else
                free(mask_buf);
        }
        mlx_free_float_buf(&well_buf, NULL);
        if (well_shape)
            mlx_free_int_array(&well_shape, NULL);
        return -1;
    }
    apply_well_mask_cpu(fac_buf, out_buf, h, w, c, mask_buf, well_buf, wc);
    int ndim_out = fac_ndim;
    const int *shape_in = mlx_array_shape(facies);
    mlx_array out_arr =
        mlx_array_new_data(out_buf, shape_in, ndim_out, MLX_FLOAT32);
    *out = out_arr;
    mlx_free_float_buf(&fac_buf, NULL);
    if (fac_shape)
        mlx_free_int_array(&fac_shape, NULL);
    if (mask_buf) {
        if (mask_buf_helper)
            mlx_free_pod((void **)&mask_buf);
        else
            free(mask_buf);
    }
    mlx_free_float_buf(&well_buf, NULL);
    if (well_shape)
        mlx_free_int_array(&well_shape, NULL);
    mlx_free_float_buf(&out_buf, NULL);
    return 0;
}

/* End moved from utils_extra.c */

/* PNG saving implementation using miniz */
#include "third_party/miniz.h"
#include "third_party/miniz_tdef.h"

int mlx_save_png(const char *path, mlx_array arr) {
    if (!path)
        return -1;

    /* Evaluate array to ensure data is available - use CPU stream for I/O */
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array_eval(arr);
    mlx_synchronize(s);

    int ndim = mlx_array_ndim(arr);
    const int *shape = mlx_array_shape(arr);

    /* Handle (1,H,W,C) or (H,W,C) */
    int h, w, c;
    if (ndim == 4 && shape[0] == 1) {
        h = shape[1];
        w = shape[2];
        c = shape[3];
    } else if (ndim == 3) {
        h = shape[0];
        w = shape[1];
        c = shape[2];
    } else {
        fprintf(stderr, "[mlx_save_png] Unsupported array ndim=%d\n", ndim);
        mlx_stream_free(s);
        return -1;
    }

    /* Get float data */
    float *buf = NULL;
    size_t elems = 0;
    int out_ndim = 0;
    int *out_shape = NULL;
    if (mlx_array_to_float_buffer(arr, &buf, &elems, &out_ndim, &out_shape) != 0) {
        mlx_stream_free(s);
        return -1;
    }

    /* Convert float [0,1] to uint8 [0,255] */
    unsigned char *pixels = (unsigned char *)malloc(h * w * c);
    if (!pixels) {
        mlx_free_float_buf(&buf, NULL);
        if (out_shape)
            mlx_free_int_array(&out_shape, NULL);
        mlx_stream_free(s);
        return -1;
    }

    /* Handle 4D (skip batch dim) vs 3D */
    size_t offset = (ndim == 4) ? (h * w * c) * 0 : 0; /* batch 0 */
    for (int i = 0; i < h * w * c; i++) {
        float val = buf[offset + i];
        /* Clamp to [0,1] and scale to [0,255] */
        if (val < 0.0f)
            val = 0.0f;
        if (val > 1.0f)
            val = 1.0f;
        pixels[i] = (unsigned char)(val * 255.0f + 0.5f);
    }

    mlx_free_float_buf(&buf, NULL);
    if (out_shape)
        mlx_free_int_array(&out_shape, NULL);

    /* Write PNG */
    size_t png_len = 0;
    void *png_data = tdefl_write_image_to_png_file_in_memory(pixels, w, h, c, &png_len);
    free(pixels);

    if (!png_data) {
        fprintf(stderr, "[mlx_save_png] PNG compression failed\n");
        mlx_stream_free(s);
        return -1;
    }

    /* Write to file */
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[mlx_save_png] Cannot open %s for writing\n", path);
        mz_free(png_data);
        mlx_stream_free(s);
        return -1;
    }
    fwrite(png_data, 1, png_len, f);
    fclose(f);
    mz_free(png_data);
    mlx_stream_free(s);
    return 0;
}

int mlx_save_facies_comparison_png(const char *path, mlx_array fake,
                                   mlx_array real) {
    if (!path)
        return -1;

    /* Evaluate arrays in batch - use CPU stream for I/O */
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_vector_array vec = mlx_vector_array_new();
    mlx_vector_array_append_value(vec, fake);
    mlx_vector_array_append_value(vec, real);
    mlx_eval(vec);
    mlx_vector_array_free(vec);
    mlx_synchronize(s);

    /* Get dimensions - both should be same shape */
    int fake_ndim = mlx_array_ndim(fake);
    int real_ndim = mlx_array_ndim(real);
    const int *fake_shape = mlx_array_shape(fake);
    const int *real_shape = mlx_array_shape(real);

    int h, w, c;
    if (fake_ndim == 4 && fake_shape[0] == 1) {
        h = fake_shape[1];
        w = fake_shape[2];
        c = fake_shape[3];
    } else if (fake_ndim == 3) {
        h = fake_shape[0];
        w = fake_shape[1];
        c = fake_shape[2];
    } else {
        fprintf(stderr, "[mlx_save_facies_comparison_png] Unsupported ndim=%d\n",
                fake_ndim);
        mlx_stream_free(s);
        return -1;
    }

    /* Get float data for both */
    float *fake_buf = NULL, *real_buf = NULL;
    size_t fake_elems = 0, real_elems = 0;
    int fake_out_ndim = 0, real_out_ndim = 0;
    int *fake_out_shape = NULL, *real_out_shape = NULL;

    if (mlx_array_to_float_buffer(fake, &fake_buf, &fake_elems, &fake_out_ndim,
                                  &fake_out_shape) != 0) {
        mlx_stream_free(s);
        return -1;
    }
    if (mlx_array_to_float_buffer(real, &real_buf, &real_elems, &real_out_ndim,
                                  &real_out_shape) != 0) {
        mlx_free_float_buf(&fake_buf, NULL);
        if (fake_out_shape)
            mlx_free_int_array(&fake_out_shape, NULL);
        mlx_stream_free(s);
        return -1;
    }

    /* Create side-by-side image: [real | fake] with 2px gap */
    int gap = 2;
    int out_w = w * 2 + gap;
    int out_h = h;
    unsigned char *pixels = (unsigned char *)calloc(out_h * out_w * c, 1);
    if (!pixels) {
        mlx_free_float_buf(&fake_buf, NULL);
        mlx_free_float_buf(&real_buf, NULL);
        if (fake_out_shape)
            mlx_free_int_array(&fake_out_shape, NULL);
        if (real_out_shape)
            mlx_free_int_array(&real_out_shape, NULL);
        mlx_stream_free(s);
        return -1;
    }

    /* Handle batch dimension offset */
    size_t offset = (fake_ndim == 4) ? 0 : 0;

    /* Copy real image to left side */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int ch = 0; ch < c; ch++) {
                float val = real_buf[offset + (y * w + x) * c + ch];
                if (val < 0.0f)
                    val = 0.0f;
                if (val > 1.0f)
                    val = 1.0f;
                pixels[(y * out_w + x) * c + ch] = (unsigned char)(val * 255.0f + 0.5f);
            }
        }
    }

    /* Copy fake image to right side (after gap) */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int ch = 0; ch < c; ch++) {
                float val = fake_buf[offset + (y * w + x) * c + ch];
                if (val < 0.0f)
                    val = 0.0f;
                if (val > 1.0f)
                    val = 1.0f;
                pixels[(y * out_w + (w + gap + x)) * c + ch] =
                    (unsigned char)(val * 255.0f + 0.5f);
            }
        }
    }

    mlx_free_float_buf(&fake_buf, NULL);
    mlx_free_float_buf(&real_buf, NULL);
    if (fake_out_shape)
        mlx_free_int_array(&fake_out_shape, NULL);
    if (real_out_shape)
        mlx_free_int_array(&real_out_shape, NULL);

    /* Write PNG */
    size_t png_len = 0;
    void *png_data =
        tdefl_write_image_to_png_file_in_memory(pixels, out_w, out_h, c, &png_len);
    free(pixels);

    if (!png_data) {
        fprintf(stderr, "[mlx_save_facies_comparison_png] PNG compression failed\n");
        mlx_stream_free(s);
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[mlx_save_facies_comparison_png] Cannot open %s\n", path);
        mz_free(png_data);
        mlx_stream_free(s);
        return -1;
    }
    fwrite(png_data, 1, png_len, f);
    fclose(f);
    mz_free(png_data);
    mlx_stream_free(s);
    return 0;
}

/* Default facies color palette (in [0,1] range) - 4 colors: black, blue, green, red */
static const float DEFAULT_FACIES_PALETTE[4][3] = {
    {0.0f, 0.0f, 0.0f}, /* Black */
    {0.0f, 0.0f, 1.0f}, /* Blue */
    {0.0f, 1.0f, 0.0f}, /* Green */
    {1.0f, 0.0f, 0.0f}, /* Red */
};
static const int DEFAULT_FACIES_PALETTE_SIZE = 4;

/* Simple 5x7 bitmap font for digits and basic letters */
/* Each character is 5 pixels wide, 7 pixels tall */
/* Bit pattern: MSB is leftmost pixel */
static const unsigned char FONT_5X7[128][7] = {
    /* Space (32) */
    [' '] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    /* - (45) */
    ['-'] = {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00},
    /* 0-9 */
    ['0'] = {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
    ['1'] = {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
    ['2'] = {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
    ['3'] = {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},
    ['4'] = {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
    ['5'] = {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
    ['6'] = {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},
    ['7'] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
    ['8'] = {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
    ['9'] = {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},
    /* A-Z (uppercase) */
    ['A'] = {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
    ['B'] = {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},
    ['C'] = {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},
    ['D'] = {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C},
    ['E'] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F},
    ['F'] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},
    ['G'] = {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F},
    ['H'] = {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
    ['I'] = {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
    ['J'] = {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C},
    ['K'] = {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},
    ['L'] = {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},
    ['M'] = {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},
    ['N'] = {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11},
    ['O'] = {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
    ['P'] = {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
    ['Q'] = {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},
    ['R'] = {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},
    ['S'] = {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},
    ['T'] = {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
    ['U'] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
    ['V'] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04},
    ['W'] = {0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11},
    ['X'] = {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
    ['Y'] = {0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04},
    ['Z'] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},
    /* a-z (lowercase - same as uppercase for simplicity) */
    ['a'] = {0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F},
    ['e'] = {0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E},
    ['g'] = {0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E},
    ['i'] = {0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E},
    ['l'] = {0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
    ['n'] = {0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11},
    ['o'] = {0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E},
    ['r'] = {0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10},
    ['s'] = {0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E},
    ['t'] = {0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06},
    ['v'] = {0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04},
    ['w'] = {0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A},
};

/* Draw a single character at (x, y) in pixels buffer */
static void draw_char(unsigned char *pixels, int img_w, int img_h,
                      int x, int y, char ch, int scale,
                      unsigned char r, unsigned char g, unsigned char b) {
    if (ch < 0 || ch > 127) return;
    const unsigned char *glyph = FONT_5X7[(int)ch];
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (glyph[row] & (0x10 >> col)) {
                /* Draw scaled pixel */
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = x + col * scale + sx;
                        int py = y + row * scale + sy;
                        if (px >= 0 && px < img_w && py >= 0 && py < img_h) {
                            int idx = (py * img_w + px) * 3;
                            pixels[idx] = r;
                            pixels[idx + 1] = g;
                            pixels[idx + 2] = b;
                        }
                    }
                }
            }
        }
    }
}

/* Draw a string at (x, y) */
static void draw_string(unsigned char *pixels, int img_w, int img_h,
                        int x, int y, const char *str, int scale,
                        unsigned char r, unsigned char g, unsigned char b) {
    int char_w = 5 * scale + scale; /* char width + 1 pixel spacing */
    for (int i = 0; str[i]; i++) {
        draw_char(pixels, img_w, img_h, x + i * char_w, y, str[i], scale, r, g, b);
    }
}

/* Get string width in pixels */
static int string_width(const char *str, int scale) {
    int len = 0;
    while (str[len]) len++;
    if (len == 0) return 0;
    int char_w = 5 * scale + scale;
    return len * char_w - scale; /* subtract trailing spacing */
}

/* Draw a filled downward-pointing triangle (arrow) at position (x, y)
 * The arrow points down with the tip at (x, y + arrow_size)
 * Top edge is at y with width arrow_size pixels centered on x */
static void draw_arrow_down(unsigned char *pixels, int img_w, int img_h,
                            int x, int y, int arrow_size,
                            unsigned char r, unsigned char g, unsigned char b) {
    for (int row = 0; row < arrow_size; row++) {
        /* Width decreases as we go down: arrow_size at top, 1 at bottom */
        int half_width = (arrow_size - row) / 2;
        int start_x = x - half_width;
        int end_x = x + half_width;
        int py = y + row;
        if (py < 0 || py >= img_h) continue;
        for (int px = start_x; px <= end_x; px++) {
            if (px < 0 || px >= img_w) continue;
            int idx = (py * img_w + px) * 3;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }
}

/* Quantize a single pixel to nearest palette color */
static void quantize_pixel_to_palette(const float *in_rgb, float *out_rgb,
                                      const float (*palette)[3], int num_colors) {
    float best_dist = 1e9f;
    int best_idx = 0;
    for (int c = 0; c < num_colors; c++) {
        float dr = in_rgb[0] - palette[c][0];
        float dg = in_rgb[1] - palette[c][1];
        float db = in_rgb[2] - palette[c][2];
        float dist = dr * dr + dg * dg + db * db;
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = c;
        }
    }
    out_rgb[0] = palette[best_idx][0];
    out_rgb[1] = palette[best_idx][1];
    out_rgb[2] = palette[best_idx][2];
}

/* Resize image using nearest-neighbor interpolation */
static void resize_image_nearest(const float *src, int src_h, int src_w, int channels,
                                 float *dst, int dst_h, int dst_w) {
    for (int y = 0; y < dst_h; y++) {
        int src_y = (int)((float)y * src_h / dst_h);
        if (src_y >= src_h) src_y = src_h - 1;
        for (int x = 0; x < dst_w; x++) {
            int src_x = (int)((float)x * src_w / dst_w);
            if (src_x >= src_w) src_x = src_w - 1;
            for (int ch = 0; ch < channels; ch++) {
                dst[(y * dst_w + x) * channels + ch] = src[(src_y * src_w + src_x) * channels + ch];
            }
        }
    }
}

/* Apply well mask overlay to a cell buffer.
 * - First makes well columns white
 * - Then overlays non-black pixels from well_facies
 * mask_buf: single-sample mask (H x W x C), values > 0.5 indicate well
 * cell_buf: cell to modify (cell_size x cell_size x 3), values in [0,1]
 * well_facies_buf: original facies with well colors (H x W x C)
 * Returns: modifies cell_buf in place
 */
static void apply_well_mask_overlay(float *cell_buf, int cell_size,
                                    const float *mask_buf, int mask_h, int mask_w, int mask_c,
                                    const float *well_facies_buf, int well_h, int well_w, int well_c) {
    /* Find well columns by summing mask vertically */
    int *is_well_col = (int *)calloc(cell_size, sizeof(int));
    if (!is_well_col) return;

    for (int cx = 0; cx < cell_size; cx++) {
        /* Map cell x to mask x */
        int mx = (int)((float)cx * mask_w / cell_size);
        if (mx >= mask_w) mx = mask_w - 1;

        /* Check if any pixel in this column is a well pixel */
        for (int my = 0; my < mask_h; my++) {
            float mask_val = mask_buf[my * mask_w * mask_c + mx * mask_c];
            if (mask_val > 0.5f) {
                is_well_col[cx] = 1;
                break;
            }
        }
    }

    /* Step 1: Set well columns to white */
    for (int cy = 0; cy < cell_size; cy++) {
        for (int cx = 0; cx < cell_size; cx++) {
            if (is_well_col[cx]) {
                int idx = (cy * cell_size + cx) * 3;
                cell_buf[idx] = 1.0f;     /* R = white */
                cell_buf[idx + 1] = 1.0f; /* G = white */
                cell_buf[idx + 2] = 1.0f; /* B = white */
            }
        }
    }

    /* Step 2: Overlay non-black pixels from well_facies */
    for (int cy = 0; cy < cell_size; cy++) {
        for (int cx = 0; cx < cell_size; cx++) {
            if (!is_well_col[cx]) continue;

            /* Map cell coords to mask coords */
            int mx = (int)((float)cx * mask_w / cell_size);
            int my = (int)((float)cy * mask_h / cell_size);
            if (mx >= mask_w) mx = mask_w - 1;
            if (my >= mask_h) my = mask_h - 1;

            /* Check if this pixel is a well pixel */
            float mask_val = mask_buf[my * mask_w * mask_c + mx * mask_c];
            if (mask_val <= 0.5f) continue;

            /* Map cell coords to well_facies coords */
            int wx = (int)((float)cx * well_w / cell_size);
            int wy = (int)((float)cy * well_h / cell_size);
            if (wx >= well_w) wx = well_w - 1;
            if (wy >= well_h) wy = well_h - 1;

            /* Get well facies pixel */
            float r = well_facies_buf[(wy * well_w + wx) * well_c];
            float g = well_facies_buf[(wy * well_w + wx) * well_c + 1];
            float b = well_facies_buf[(wy * well_w + wx) * well_c + 2];

            /* Check if pixel is NOT black (distance from black >= 0.3) */
            float dist = sqrtf(r * r + g * g + b * b);
            if (dist >= 0.3f) {
                /* Overlay the non-black well facies pixel */
                int idx = (cy * cell_size + cx) * 3;
                cell_buf[idx] = r;
                cell_buf[idx + 1] = g;
                cell_buf[idx + 2] = b;
            }
        }
    }

    free(is_well_col);
}

int mlx_save_facies_grid_png(const char *path, mlx_array *fake_samples,
                             int num_fake, mlx_array real, int cell_size,
                             int scale, int epoch) {
    if (!path || !fake_samples || num_fake <= 0)
        return -1;

    /* Evaluate arrays in batch - use CPU stream for I/O */
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_vector_array vec = mlx_vector_array_new();
    mlx_vector_array_append_value(vec, real);
    for (int i = 0; i < num_fake; i++) {
        mlx_vector_array_append_value(vec, fake_samples[i]);
    }
    mlx_eval(vec);
    mlx_vector_array_free(vec);
    mlx_synchronize(s);

    /* Get dimensions from real array */
    int real_ndim = mlx_array_ndim(real);
    const int *real_shape = mlx_array_shape(real);

    int batch_size, h, w, c;
    if (real_ndim == 4) {
        batch_size = real_shape[0];
        h = real_shape[1];
        w = real_shape[2];
        c = real_shape[3];
    } else if (real_ndim == 3) {
        batch_size = 1;
        h = real_shape[0];
        w = real_shape[1];
        c = real_shape[2];
    } else {
        fprintf(stderr, "[mlx_save_facies_grid_png] Unsupported ndim=%d\n",
                real_ndim);
        mlx_stream_free(s);
        return -1;
    }

    /* Grid layout: rows = batch_size, cols = 1 (real) + num_fake (generated) */
    int num_cols = 1 + num_fake;
    int num_rows = batch_size;
    int spacing = 4;
    int margin = 10;
    int title_height = 25;
    int main_title_height = 30;

    int grid_w = margin * 2 + num_cols * cell_size + (num_cols - 1) * spacing;
    int grid_h =
        margin * 2 + main_title_height + num_rows * (cell_size + title_height) +
        (num_rows - 1) * spacing;

    /* Allocate output image (white background) */
    unsigned char *pixels = (unsigned char *)malloc(grid_h * grid_w * 3);
    if (!pixels) {
        mlx_stream_free(s);
        return -1;
    }
    memset(pixels, 255, grid_h * grid_w * 3); /* White background */

    /* Get float buffers */
    float *real_buf = NULL;
    size_t real_elems = 0;
    int real_out_ndim = 0;
    int *real_out_shape = NULL;
    if (mlx_array_to_float_buffer(real, &real_buf, &real_elems, &real_out_ndim,
                                  &real_out_shape) != 0) {
        free(pixels);
        mlx_stream_free(s);
        if (real_out_shape)
            mlx_free_int_array(&real_out_shape, NULL);
        return -1;
    }

    /* Allocate cell buffer for resizing */
    float *cell_buf = (float *)malloc(cell_size * cell_size * c * sizeof(float));
    float *quant_buf = (float *)malloc(cell_size * cell_size * c * sizeof(float));
    if (!cell_buf || !quant_buf) {
        free(pixels);
        mlx_free_float_buf(&real_buf, NULL);
        if (real_out_shape)
            mlx_free_int_array(&real_out_shape, NULL);
        if (cell_buf)
            free(cell_buf);
        if (quant_buf)
            free(quant_buf);
        mlx_stream_free(s);
        return -1;
    }

    /* Helper to draw a cell at (row, col) in the grid */
#define DRAW_CELL(src_buf, src_offset, row, col)                             \
    do {                                                                       \
      /* Resize to cell_size */                                                \
      resize_image_nearest(src_buf + src_offset, h, w, c, cell_buf, cell_size, \
                           cell_size);                                         \
      /* Quantize colors */                                                    \
      for (int py = 0; py < cell_size * cell_size; py++) {                     \
        float in_rgb[3] = {cell_buf[py * c], cell_buf[py * c + 1],             \
                           cell_buf[py * c + 2]};                              \
        float out_rgb[3];                                                      \
        quantize_pixel_to_palette(in_rgb, out_rgb, DEFAULT_FACIES_PALETTE,     \
                                  DEFAULT_FACIES_PALETTE_SIZE);                \
        quant_buf[py * c] = out_rgb[0];                                        \
        quant_buf[py * c + 1] = out_rgb[1];                                    \
        quant_buf[py * c + 2] = out_rgb[2];                                    \
      }                                                                        \
      /* Copy to output pixels */                                              \
      int cell_x = margin + (col) * (cell_size + spacing);                     \
      int cell_y = margin + main_title_height + (row) * (cell_size + title_height + spacing); \
      for (int cy = 0; cy < cell_size; cy++) {                                 \
        for (int cx = 0; cx < cell_size; cx++) {                               \
          int out_idx = ((cell_y + cy) * grid_w + (cell_x + cx)) * 3;          \
          for (int ch = 0; ch < 3; ch++) {                                     \
            float val = quant_buf[(cy * cell_size + cx) * c + ch];             \
            if (val < 0.0f) val = 0.0f;                                        \
            if (val > 1.0f) val = 1.0f;                                        \
            pixels[out_idx + ch] = (unsigned char)(val * 255.0f + 0.5f);       \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    } while (0)

    /* Draw real facies (column 0) */
    for (int row = 0; row < num_rows; row++) {
        size_t offset = (size_t)row * h * w * c;
        DRAW_CELL(real_buf, offset, row, 0);
    }

    /* Draw fake samples (columns 1..num_fake) */
    for (int fi = 0; fi < num_fake; fi++) {
        float *fake_buf = NULL;
        size_t fake_elems = 0;
        int fake_out_ndim = 0;
        int *fake_out_shape = NULL;
        if (mlx_array_to_float_buffer(fake_samples[fi], &fake_buf, &fake_elems,
                                      &fake_out_ndim, &fake_out_shape) == 0) {
            for (int row = 0; row < num_rows; row++) {
                size_t offset = (size_t)row * h * w * c;
                DRAW_CELL(fake_buf, offset, row, fi + 1);
            }
            mlx_free_float_buf(&fake_buf, NULL);
            if (fake_out_shape)
                mlx_free_int_array(&fake_out_shape, NULL);
        } else {
            if (fake_out_shape)
                mlx_free_int_array(&fake_out_shape, NULL);
        }
    }

#undef DRAW_CELL

    /* Clean up */
    free(cell_buf);
    free(quant_buf);
    mlx_free_float_buf(&real_buf, NULL);
    if (real_out_shape)
        mlx_free_int_array(&real_out_shape, NULL);

    /* Write PNG */
    size_t png_len = 0;
    void *png_data =
        tdefl_write_image_to_png_file_in_memory(pixels, grid_w, grid_h, 3, &png_len);
    free(pixels);

    if (!png_data) {
        fprintf(stderr, "[mlx_save_facies_grid_png] PNG compression failed\n");
        mlx_stream_free(s);
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[mlx_save_facies_grid_png] Cannot open %s\n", path);
        mz_free(png_data);
        mlx_stream_free(s);
        return -1;
    }
    fwrite(png_data, 1, png_len, f);
    fclose(f);
    mz_free(png_data);
    mlx_stream_free(s);
    return 0;
}

int mlx_save_facies_grid_png_v2(const char *path, mlx_array *all_fakes,
                                int total_gen, mlx_array real,
                                const int *selected_indices, int num_real,
                                int num_gen_per_real, int cell_size,
                                int scale, int epoch, mlx_array masks) {
    if (!path || !all_fakes || total_gen <= 0 || !selected_indices || num_real <= 0)
        return -1;

    /* Evaluate arrays in batch - use CPU stream for I/O */
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_vector_array vec = mlx_vector_array_new();
    mlx_vector_array_append_value(vec, real);
    for (int i = 0; i < total_gen; i++) {
        mlx_vector_array_append_value(vec, all_fakes[i]);
    }
    mlx_eval(vec);
    mlx_vector_array_free(vec);
    mlx_synchronize(s);

    /* Get dimensions from real array */
    int real_ndim = mlx_array_ndim(real);
    const int *real_shape = mlx_array_shape(real);

    int batch_size, h, w, c;
    if (real_ndim == 4) {
        batch_size = real_shape[0];
        h = real_shape[1];
        w = real_shape[2];
        c = real_shape[3];
    } else if (real_ndim == 3) {
        batch_size = 1;
        h = real_shape[0];
        w = real_shape[1];
        c = real_shape[2];
    } else {
        fprintf(stderr, "[mlx_save_facies_grid_png_v2] Unsupported ndim=%d\n",
                real_ndim);
        mlx_stream_free(s);
        return -1;
    }

    /* Grid layout: rows = num_real, cols = 1 (real) + num_gen_per_real (generated) */
    int num_cols = 1 + num_gen_per_real;
    int num_rows = num_real;
    int spacing = 20;  /* Match Python spacing */
    int margin = 20;   /* Match Python margin */
    int title_height = 20;  /* Height for subplot titles */
    int main_title_height = 30;  /* Height for main title */

    int grid_w = margin * 2 + num_cols * cell_size + (num_cols - 1) * spacing;
    int grid_h =
        margin * 2 + main_title_height + num_rows * (cell_size + title_height + spacing) - spacing;

    /* Allocate output image (white background) */
    unsigned char *pixels = (unsigned char *)malloc(grid_h * grid_w * 3);
    if (!pixels) {
        mlx_stream_free(s);
        return -1;
    }
    memset(pixels, 255, grid_h * grid_w * 3); /* White background */

    /* Draw main title: "Stage X - Real vs Generated Facies" */
    {
        char main_title[128];
        snprintf(main_title, sizeof(main_title), "Stage %d - Well Log, Real vs Generated Facies", scale);
        int font_scale = 2;  /* 2x scale for main title */
        int title_w = string_width(main_title, font_scale);
        int title_x = (grid_w - title_w) / 2;
        draw_string(pixels, grid_w, grid_h, title_x, margin / 2, main_title, font_scale, 0, 0, 0);
    }

    /* Get float buffer for real */
    float *real_buf = NULL;
    size_t real_elems = 0;
    int real_out_ndim = 0;
    int *real_out_shape = NULL;
    if (mlx_array_to_float_buffer(real, &real_buf, &real_elems, &real_out_ndim,
                                  &real_out_shape) != 0) {
        free(pixels);
        mlx_stream_free(s);
        if (real_out_shape)
            mlx_free_int_array(&real_out_shape, NULL);
        return -1;
    }

    /* Allocate cell buffer for resizing */
    float *cell_buf = (float *)malloc(cell_size * cell_size * c * sizeof(float));
    float *quant_buf = (float *)malloc(cell_size * cell_size * c * sizeof(float));
    if (!cell_buf || !quant_buf) {
        free(pixels);
        mlx_free_float_buf(&real_buf, NULL);
        if (real_out_shape)
            mlx_free_int_array(&real_out_shape, NULL);
        if (cell_buf)
            free(cell_buf);
        if (quant_buf)
            free(quant_buf);
        mlx_stream_free(s);
        return -1;
    }

    /* Extract mask buffer if provided */
    float *mask_buf = NULL;
    size_t mask_elems = 0;
    int mask_ndim = 0;
    int *mask_shape = NULL;
    int mask_h = 0, mask_w = 0, mask_c = 0, mask_batch = 0;
    int has_masks = 0;
    if (masks.ctx != NULL) {
        mlx_array_eval(masks);
        mlx_synchronize(s);
        mlx_dtype mask_dtype = mlx_array_dtype(masks);

        /* Convert mask to float32 if needed (masks may be stored as bool) */
        mlx_array mask_float = mlx_array_new();
        if (mask_dtype != MLX_FLOAT32) {
            if (mlx_astype(&mask_float, masks, MLX_FLOAT32, s) != 0) {
                mlx_array_free(mask_float);
            } else {
                mlx_array_eval(mask_float);
                mlx_synchronize(s);
            }
        } else {
            mlx_array_set(&mask_float, masks);
        }

        int float_rc = mlx_array_to_float_buffer(mask_float, &mask_buf, &mask_elems, &mask_ndim, &mask_shape);
        mlx_array_free(mask_float);

        if (float_rc == 0 && mask_buf) {
            has_masks = 1;
            if (mask_ndim == 4) {
                mask_batch = mask_shape[0];
                mask_h = mask_shape[1];
                mask_w = mask_shape[2];
                mask_c = mask_shape[3];
            } else if (mask_ndim == 3) {
                mask_batch = 1;
                mask_h = mask_shape[0];
                mask_w = mask_shape[1];
                mask_c = mask_shape[2];
            }
        } else {
            if (mask_shape)
                mlx_free_int_array(&mask_shape, NULL);
        }
    }

    /* Helper to draw a cell at (row, col) in the grid */
#define DRAW_CELL_V2(src_buf, src_h, src_w, src_c, row, col)                 \
    do {                                                                       \
      /* Resize to cell_size */                                                \
      resize_image_nearest(src_buf, src_h, src_w, src_c, cell_buf, cell_size,  \
                           cell_size);                                         \
      /* Quantize colors */                                                    \
      for (int py = 0; py < cell_size * cell_size; py++) {                     \
        float in_rgb[3] = {cell_buf[py * src_c], cell_buf[py * src_c + 1],     \
                           cell_buf[py * src_c + 2]};                          \
        float out_rgb[3];                                                      \
        quantize_pixel_to_palette(in_rgb, out_rgb, DEFAULT_FACIES_PALETTE,     \
                                  DEFAULT_FACIES_PALETTE_SIZE);                \
        quant_buf[py * src_c] = out_rgb[0];                                    \
        quant_buf[py * src_c + 1] = out_rgb[1];                                \
        quant_buf[py * src_c + 2] = out_rgb[2];                                \
      }                                                                        \
      /* Copy to output pixels */                                              \
      int cell_x = margin + (col) * (cell_size + spacing);                     \
      int cell_y = margin + main_title_height + (row) * (cell_size + title_height + spacing); \
      /* Offset cell_y by title_height so image is below the title */          \
      for (int cy = 0; cy < cell_size; cy++) {                                 \
        for (int cx = 0; cx < cell_size; cx++) {                               \
          int out_idx = ((cell_y + title_height + cy) * grid_w + (cell_x + cx)) * 3; \
          for (int ch = 0; ch < 3; ch++) {                                     \
            float val = quant_buf[(cy * cell_size + cx) * src_c + ch];         \
            if (val < 0.0f) val = 0.0f;                                        \
            if (val > 1.0f) val = 1.0f;                                        \
            pixels[out_idx + ch] = (unsigned char)(val * 255.0f + 0.5f);       \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    } while (0)

    /* Macro to draw cell with mask overlay applied */
#define DRAW_CELL_WITH_MASK(src_buf, src_h, src_w, src_c, row, col,          \
                              mask_ptr, m_h, m_w, m_c, well_ptr, w_h, w_w, w_c)\
    do {                                                                       \
      /* Resize to cell_size */                                                \
      resize_image_nearest(src_buf, src_h, src_w, src_c, cell_buf, cell_size,  \
                           cell_size);                                         \
      /* Quantize colors */                                                    \
      for (int py = 0; py < cell_size * cell_size; py++) {                     \
        float in_rgb[3] = {cell_buf[py * src_c], cell_buf[py * src_c + 1],     \
                           cell_buf[py * src_c + 2]};                          \
        float out_rgb[3];                                                      \
        quantize_pixel_to_palette(in_rgb, out_rgb, DEFAULT_FACIES_PALETTE,     \
                                  DEFAULT_FACIES_PALETTE_SIZE);                \
        quant_buf[py * src_c] = out_rgb[0];                                    \
        quant_buf[py * src_c + 1] = out_rgb[1];                                \
        quant_buf[py * src_c + 2] = out_rgb[2];                                \
      }                                                                        \
      /* Apply well mask overlay if mask is provided */                        \
      if ((mask_ptr) && (well_ptr)) {                                          \
        apply_well_mask_overlay(quant_buf, cell_size,                          \
                                (mask_ptr), (m_h), (m_w), (m_c),               \
                                (well_ptr), (w_h), (w_w), (w_c));              \
      }                                                                        \
      /* Copy to output pixels */                                              \
      int cell_x = margin + (col) * (cell_size + spacing);                     \
      int cell_y = margin + main_title_height + (row) * (cell_size + title_height + spacing); \
      /* Offset cell_y by title_height so image is below the title */          \
      for (int cy = 0; cy < cell_size; cy++) {                                 \
        for (int cx = 0; cx < cell_size; cx++) {                               \
          int out_idx = ((cell_y + title_height + cy) * grid_w + (cell_x + cx)) * 3; \
          for (int ch = 0; ch < 3; ch++) {                                     \
            float val = quant_buf[(cy * cell_size + cx) * src_c + ch];         \
            if (val < 0.0f) val = 0.0f;                                        \
            if (val > 1.0f) val = 1.0f;                                        \
            pixels[out_idx + ch] = (unsigned char)(val * 255.0f + 0.5f);       \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    } while (0)

    /* Draw real facies (column 0) - use selected_indices to pick from batch */
    for (int row = 0; row < num_rows; row++) {
        int idx = selected_indices[row];
        if (idx >= batch_size) idx = 0; /* Safety clamp */
        size_t offset = (size_t)idx * h * w * c;

        /* Get mask for this sample if available */
        float *sample_mask = NULL;
        if (has_masks && mask_buf && idx < mask_batch) {
            sample_mask = mask_buf + (size_t)idx * mask_h * mask_w * mask_c;
        }

        /* Draw with mask overlay (real facies uses itself as well source) */
        DRAW_CELL_WITH_MASK(real_buf + offset, h, w, c, row, 0,
                            sample_mask, mask_h, mask_w, mask_c,
                            real_buf + offset, h, w, c);

        /* Draw row label: "Well N" above the real facies cell */
        char row_label[32];
        snprintf(row_label, sizeof(row_label), "Well %d", row + 1);
        int label_font_scale = 2;
        int label_w = string_width(row_label, label_font_scale);
        int cell_x = margin + 0 * (cell_size + spacing);
        int cell_y = margin + main_title_height + row * (cell_size + title_height + spacing);
        int label_x = cell_x + (cell_size - label_w) / 2;
        int label_y = cell_y;
        draw_string(pixels, grid_w, grid_h, label_x, label_y, row_label, label_font_scale, 0, 0, 0);

        /* Draw well arrow if masks are available */
        if (has_masks && mask_buf) {
            int mask_idx = selected_indices[row];
            if (mask_idx < mask_batch) {
                /* Find center of well columns by summing mask vertically */
                size_t mask_offset = (size_t)mask_idx * mask_h * mask_w * mask_c;
                float *sample_mask = mask_buf + mask_offset;
                float col_sum = 0;
                int well_count = 0;
                for (int mx = 0; mx < mask_w; mx++) {
                    float col_val = 0;
                    for (int my = 0; my < mask_h; my++) {
                        col_val += sample_mask[my * mask_w * mask_c + mx * mask_c];
                    }
                    if (col_val > 0) {
                        col_sum += mx;
                        well_count++;
                    }
                }
                if (well_count > 0) {
                    float center_col = col_sum / well_count;
                    /* Map from mask coords to cell coords */
                    float center_x = (center_col + 0.5f) * ((float)cell_size / (float)mask_w);
                    int arrow_x = cell_x + (int)center_x;
                    int arrow_y = cell_y + title_height - 3;  /* Just above the cell */
                    draw_arrow_down(pixels, grid_w, grid_h, arrow_x, arrow_y, 10, 255, 0, 0);
                }
            }
        }
    }

    /* Draw fake samples: row r gets fakes [r*num_gen_per_real .. (r+1)*num_gen_per_real-1] */
    for (int row = 0; row < num_rows; row++) {
        /* Get the real facies for this row (for well overlay source) */
        int real_idx = selected_indices[row];
        if (real_idx >= batch_size) real_idx = 0;
        float *row_real_buf = real_buf + (size_t)real_idx * h * w * c;

        /* Get mask for this row if available */
        float *row_mask = NULL;
        if (has_masks && mask_buf && real_idx < mask_batch) {
            row_mask = mask_buf + (size_t)real_idx * mask_h * mask_w * mask_c;
        }

        for (int gen = 0; gen < num_gen_per_real; gen++) {
            int fake_idx = row * num_gen_per_real + gen;
            if (fake_idx >= total_gen) continue;

            float *fake_buf = NULL;
            size_t fake_elems = 0;
            int fake_out_ndim = 0;
            int *fake_out_shape = NULL;
            if (mlx_array_to_float_buffer(all_fakes[fake_idx], &fake_buf, &fake_elems,
                                          &fake_out_ndim, &fake_out_shape) == 0) {
                /* ...existing code... */
                mlx_free_float_buf(&fake_buf, NULL);
                if (fake_out_shape)
                    mlx_free_int_array(&fake_out_shape, NULL);
            } else {
                if (fake_out_shape)
                    mlx_free_int_array(&fake_out_shape, NULL);
            }
        }
    }
#undef DRAW_CELL_V2
#undef DRAW_CELL_WITH_MASK

    /* Clean up */
    free(cell_buf);
    free(quant_buf);
    mlx_free_float_buf(&real_buf, NULL);
    if (real_out_shape)
        mlx_free_int_array(&real_out_shape, NULL);
    if (mask_buf)
        mlx_free_float_buf(&mask_buf, NULL);
    if (mask_shape)
        mlx_free_int_array(&mask_shape, NULL);

    /* Write PNG */
    size_t png_len = 0;
    void *png_data =
        tdefl_write_image_to_png_file_in_memory(pixels, grid_w, grid_h, 3, &png_len);
    free(pixels);

    if (!png_data) {
        fprintf(stderr, "[mlx_save_facies_grid_png_v2] PNG compression failed\n");
        mlx_stream_free(s);
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[mlx_save_facies_grid_png_v2] Cannot open %s\n", path);
        mz_free(png_data);
        mlx_stream_free(s);
        return -1;
    }
    fwrite(png_data, 1, png_len, f);
    fclose(f);
    mz_free(png_data);
    mlx_stream_free(s);
    return 0;
}
