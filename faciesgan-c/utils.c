#include "utils.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* MLX headers for array helpers */
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include "options.h"

#include <math.h>

int mlx_create_dirs(const char *path)
{
    if (!path || path[0] == '\0')
        return -1;
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len >= sizeof(tmp))
        return -1;
    strcpy(tmp, path);
    /* Remove trailing slashes */
    while (len > 1 && tmp[len - 1] == '/')
    {
        tmp[len - 1] = '\0';
        --len;
    }

    for (char *p = tmp + 1; *p; ++p)
    {
        if (*p == '/')
        {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0)
            {
                if (errno != EEXIST)
                    return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0)
    {
        if (errno != EEXIST)
            return -1;
    }
    return 0;
}

void ensure_dir(const char *path)
{
    if (!path || !*path)
        return;

    char tmp[PATH_BUFSZ];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';

    /* Remove trailing slashes (but keep single leading '/' for absolute) */
    size_t len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/')
    {
        tmp[--len] = '\0';
    }

    /* Iterate prefixes and create directories as needed (mkdir -p behavior) */
    for (char *p = tmp + 1; *p; ++p)
    {
        if (*p == '/')
        {
            *p = '\0';
            struct stat st;
            if (stat(tmp, &st) == -1)
            {
                if (mkdir(tmp, 0755) == -1 && errno != EEXIST)
                {
                    fprintf(stderr, "error: mkdir '%s' failed: %s\n", tmp, strerror(errno));
                    *p = '/';
                    return;
                }
            }
            else if (!S_ISDIR(st.st_mode))
            {
                fprintf(stderr, "error: path component '%s' exists and is not a directory\n", tmp);
                *p = '/';
                return;
            }
            *p = '/';
        }
    }

    /* Create the final directory */
    struct stat st;
    if (stat(tmp, &st) == -1)
    {
        if (mkdir(tmp, 0755) == -1 && errno != EEXIST)
        {
            fprintf(stderr, "error: mkdir '%s' failed: %s\n", tmp, strerror(errno));
            return;
        }
    }
    else if (!S_ISDIR(st.st_mode))
    {
        fprintf(stderr, "error: path '%s' exists and is not a directory\n", tmp);
        return;
    }
}

void write_options_json(const TrainningOptions *topt,
                        const int *wells_mask_columns,
                        size_t wells_mask_count)
{
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
#define PRINT_DOUBLE(FP)                                    \
    do                                                      \
    {                                                       \
        double __v = (double)(FP);                          \
        if (isfinite(__v) && fabs(__v - round(__v)) < 1e-9) \
            fprintf(of, "%.1f", __v);                       \
        else                                                \
            fprintf(of, "%g", __v);                         \
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
    fprintf(of, "    \"input_path\": \"%s\",\n", topt->input_path ? topt->input_path : "");
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
    fprintf(of, "    \"num_diversity_samples\": %d,\n", topt->num_diversity_samples);
    fprintf(of, "    \"num_feature\": %d,\n", topt->num_feature);
    fprintf(of, "    \"num_generated_per_real\": %d,\n", topt->num_generated_per_real);
    fprintf(of, "    \"num_iter\": %d,\n", topt->num_iter);
    fprintf(of, "    \"num_layer\": %d,\n", topt->num_layer);
    fprintf(of, "    \"num_real_facies\": %d,\n", topt->num_real_facies);
    fprintf(of, "    \"num_train_pyramids\": %d,\n", topt->num_train_pyramids);
    fprintf(of, "    \"num_parallel_scales\": %d,\n", topt->num_parallel_scales);
    fprintf(of, "    \"noise_channels\": %d,\n", topt->noise_channels);
    fprintf(of, "    \"num_workers\": %d,\n", topt->num_workers);
    fprintf(of, "    \"output_path\": \"%s\",\n", topt->output_path ? topt->output_path : "");
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
    if (wells_mask_count == 0)
    {
        fprintf(of, "    \"wells_mask_columns\": [],\n");
    }
    else
    {
        fprintf(of, "    \"wells_mask_columns\": [\n");
        for (size_t wi = 0; wi < wells_mask_count; ++wi)
        {
            fprintf(of, "        %d", wells_mask_columns[wi]);
            if (wi + 1 < wells_mask_count)
                fprintf(of, ",\n");
            else
                fprintf(of, "\n");
        }
        fprintf(of, "    ],\n");
    }
    fprintf(of, "    \"enable_tensorboard\": %s,\n", bool_str(topt->enable_tensorboard));
    fprintf(of, "    \"enable_plot_facies\": %s,\n", bool_str(topt->enable_plot_facies));
    /* Make compile_backend the final key to match Python's ordering (no aliases). */
    fprintf(of, "    \"compile_backend\": %s\n", bool_str(topt->compile_backend));
    /* Match Python json.dump which does not append a trailing newline */
    fprintf(of, "}");
    fclose(of);
}

void format_timestamp(char *buf, size_t bufsz)
{
    if (!buf || bufsz == 0)
        return;
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(buf, bufsz, "%Y_%m_%d_%H_%M_%S", &tm);
}

const char *bool_str(int v)
{
    return v ? "true" : "false";
}

void join_path(char *dst, size_t dstsz, const char *a, const char *b)
{
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

void mlx_set_seed(int seed)
{
    /* Seed the C library RNG. */
    srand((unsigned int)seed);
    /* Seed POSIX drand48 family as well */
    srand48((long)seed);
}

int mlx_clamp(mlx_array *res, const mlx_array a, float min_val, float max_val, const mlx_stream s)
{
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
