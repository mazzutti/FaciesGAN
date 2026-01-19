#include "mlx_dataset.h"
#include "func_cache.h"
#include "io/npz_unzip.h"
#include <mlx/c/io.h>
#include <mlx/c/stream.h>
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/vector.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

int mlx_pyramids_dataset_load(const char *input_path, const char *cache_dir,
                              int desired_num, int stop_scale, int crop_size,
                              int num_img_channels, int use_wells, int use_seismic,
                              int manual_seed,
                              mlx_vector_vector_array *out_facies,
                              mlx_vector_vector_array *out_wells,
                              mlx_vector_vector_array *out_seismic,
                              int *out_num_samples)
{
    if (!input_path || !cache_dir || !out_facies || !out_num_samples)
        return -1;

    char cache_npz[PATH_MAX] = {0};
    int actual_samples = 0;
    if (ensure_function_cache(input_path, cache_dir, desired_num, stop_scale,
                              crop_size, num_img_channels, use_wells, use_seismic,
                              manual_seed, cache_npz, sizeof(cache_npz), &actual_samples) != 0)
    {
        fprintf(stderr, "ensure_function_cache failed for %s\n", input_path);
        return -1;
    }

    int num_samples = actual_samples > 0 ? actual_samples : desired_num;
    mlx_stream s = mlx_default_cpu_stream_new();

    /* initialize output vectors */
    *out_facies = mlx_vector_vector_array_new();
    if (out_wells)
        *out_wells = mlx_vector_vector_array_new();
    if (out_seismic)
        *out_seismic = mlx_vector_vector_array_new();

    for (int si = 0; si < num_samples; ++si)
    {
        mlx_vector_array fac_sample = mlx_vector_array_new();
        mlx_vector_array well_sample = mlx_vector_array_new();
        mlx_vector_array seis_sample = mlx_vector_array_new();

        for (int sc = 0; sc < stop_scale + 1; ++sc)
        {
            char member[64];
            snprintf(member, sizeof(member), "sample_%d/facies_%d.npy", si, sc);
            mlx_io_reader reader;
            if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0)
            {
                mlx_array a = mlx_array_new();
                if (mlx_load_reader(&a, reader, s) != 0)
                {
                    int shape[3] = {crop_size, crop_size, num_img_channels};
                    mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
                }
                mlx_io_reader_free(reader);
                mlx_vector_array_append_value(fac_sample, a);
                mlx_array_free(a);
            }
            else
            {
                int shape[3] = {crop_size, crop_size, num_img_channels};
                mlx_array a = mlx_array_new();
                mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
                mlx_vector_array_append_value(fac_sample, a);
                mlx_array_free(a);
            }

            /* wells/seismic: present in archive if use_wells/use_seismic true */
            if (out_wells)
            {
                snprintf(member, sizeof(member), "sample_%d/wells_%d.npy", si, sc);
                if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0)
                {
                    mlx_array a = mlx_array_new();
                    if (mlx_load_reader(&a, reader, s) != 0)
                    {
                        int shape[3] = {crop_size, crop_size, 1};
                        mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
                    }
                    mlx_io_reader_free(reader);
                    mlx_vector_array_append_value(well_sample, a);
                    mlx_array_free(a);
                }
                else
                {
                    int shape[3] = {crop_size, crop_size, 1};
                    mlx_array a = mlx_array_new();
                    mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
                    mlx_vector_array_append_value(well_sample, a);
                    mlx_array_free(a);
                }
            }

            if (out_seismic)
            {
                snprintf(member, sizeof(member), "sample_%d/seismic_%d.npy", si, sc);
                if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0)
                {
                    mlx_array a = mlx_array_new();
                    if (mlx_load_reader(&a, reader, s) != 0)
                    {
                        int shape[3] = {crop_size, crop_size, 1};
                        mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
                    }
                    mlx_io_reader_free(reader);
                    mlx_vector_array_append_value(seis_sample, a);
                    mlx_array_free(a);
                }
                else
                {
                    int shape[3] = {crop_size, crop_size, 1};
                    mlx_array a = mlx_array_new();
                    mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
                    mlx_vector_array_append_value(seis_sample, a);
                    mlx_array_free(a);
                }
            }
        }

        mlx_vector_vector_array_append_value(*out_facies, fac_sample);
        mlx_vector_array_free(fac_sample);
        if (out_wells)
        {
            mlx_vector_vector_array_append_value(*out_wells, well_sample);
            mlx_vector_array_free(well_sample);
        }
        if (out_seismic)
        {
            mlx_vector_vector_array_append_value(*out_seismic, seis_sample);
            mlx_vector_array_free(seis_sample);
        }
    }

    mlx_stream_free(s);
    *out_num_samples = num_samples;
    return 0;
}
