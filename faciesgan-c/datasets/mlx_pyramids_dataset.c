#include "mlx_pyramids_dataset.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <dirent.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>

#include "func_cache.h"
#include "../io/npz_unzip.h"
#include <mlx/c/array.h>
#include <mlx/c/io.h>
#include <mlx/c/vector.h>
#include <mlx/c/stream.h>
#include <mlx/c/ops.h>

struct MLXPyramidsDataset
{
    mlx_vector_vector_array facies; /* per-sample per-scale mlx_array */
    mlx_vector_vector_array wells;
    mlx_vector_vector_array seismic;
    int n_samples;
};

static void swap_sample(mlx_vector_vector_array *v, int i, int j)
{
    size_t n = mlx_vector_vector_array_size(*v);
    if (i < 0 || j < 0 || (size_t)i >= n || (size_t)j >= n)
        return;
    mlx_vector_array *arr = (mlx_vector_array *)malloc(sizeof(mlx_vector_array) * n);
    if (!arr)
        return;
    for (size_t k = 0; k < n; ++k)
    {
        mlx_vector_array tmp;
        mlx_vector_vector_array_get(&tmp, *v, k);
        arr[k] = tmp;
    }
    mlx_vector_array tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
    mlx_vector_vector_array tmpvec = mlx_vector_vector_array_new_data(arr, n);
    mlx_vector_vector_array_set(v, tmpvec);
    mlx_vector_vector_array_free(tmpvec);
    free(arr);
}

static void shuffle_indices(int *idx, int n, unsigned int seed)
{
    if (seed == 0)
        seed = (unsigned int)time(NULL);
    for (int i = n - 1; i > 0; --i)
    {
        unsigned int r = (unsigned int)((seed * 1664525u + 1013904223u) & 0xffffffffu);
        seed = r;
        int j = r % (i + 1);
        int t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}

int mlx_pyramids_dataset_new(MLXPyramidsDataset **out, const char *input_path, const char *cache_dir,
                             int desired_num, int stop_scale, int crop_size, int num_img_channels,
                             int use_wells, int use_seismic, int manual_seed, int shuffle)
{
    if (!out)
        return -1;
    *out = NULL;
    char npz_path[PATH_MAX] = {0};
    int sample_count = 0;
    int rc = ensure_function_cache(input_path, cache_dir, desired_num, stop_scale, crop_size,
                                   num_img_channels, use_wells, use_seismic, manual_seed,
                                   npz_path, sizeof(npz_path), &sample_count);
    if (rc != 0)
    {
        return rc;
    }

    MLXPyramidsDataset *ds = (MLXPyramidsDataset *)calloc(1, sizeof(*ds));
    if (!ds)
    {
        return -1;
    }

    ds->facies = mlx_vector_vector_array_new();
    ds->wells = mlx_vector_vector_array_new();
    ds->seismic = mlx_vector_vector_array_new();

    int found_samples = 0;
    for (int si = 0; si < sample_count; ++si)
    {
        mlx_vector_array fac_sample = mlx_vector_array_new();
        int scale = 0;
        while (1)
        {
            char member[256];
            snprintf(member, sizeof(member), "facies_%d_scale_%d.npy", si, scale);
            mlx_io_reader r = {0};
            if (npz_extract_member_to_mlx_reader(npz_path, member, &r) != 0)
            {
                /* no more scales */
                if (scale == 0)
                {
                    mlx_vector_array_free(fac_sample);
                    goto sample_end;
                }
                break;
            }
            mlx_stream s = mlx_default_cpu_stream_new();
            mlx_array arr = mlx_array_new();
            if (mlx_load_reader(&arr, r, s) != 0)
            {
                mlx_io_reader_free(r);
                mlx_vector_array_free(fac_sample);
                mlx_pyramids_dataset_free(ds);
                mlx_stream_free(s);
                return -1;
            }
            mlx_io_reader_free(r);
            /* append a copy of arr into the sample vector */
            if (mlx_vector_array_append_value(fac_sample, arr) != 0)
            {
                mlx_array_free(arr);
                mlx_vector_array_free(fac_sample);
                mlx_pyramids_dataset_free(ds);
                mlx_stream_free(s);
                return -1;
            }
            /* the append copies the value, so free local arr */
            mlx_array_free(arr);
            mlx_stream_free(s);
            ++scale;
        }
        /* append sample vector to ds->facies */
        if (mlx_vector_vector_array_append_value(ds->facies, fac_sample) != 0)
        {
            mlx_vector_array_free(fac_sample);
            mlx_pyramids_dataset_free(ds);
            return -1;
        }
        mlx_vector_array_free(fac_sample);
        ++found_samples;
    sample_end:;
    }

    /* wells and seismic optional: for parity keep empty vectors if not present */
    ds->n_samples = found_samples;

    if (shuffle && ds->n_samples > 1)
    {
        mlx_pyramids_dataset_shuffle(ds, (unsigned int)manual_seed);
    }

    *out = ds;
    return 0;
}

void mlx_pyramids_dataset_free(MLXPyramidsDataset *ds)
{
    if (!ds)
        return;
    mlx_vector_vector_array_free(ds->facies);
    mlx_vector_vector_array_free(ds->wells);
    mlx_vector_vector_array_free(ds->seismic);
    free(ds);
}

int mlx_pyramids_dataset_shuffle(MLXPyramidsDataset *ds, unsigned int seed)
{
    if (!ds)
        return -1;
    int n = ds->n_samples;
    if (n <= 1)
        return 0;
    int *idx = (int *)malloc(sizeof(int) * n);
    if (!idx)
        return -1;
    for (int i = 0; i < n; ++i)
        idx[i] = i;
    shuffle_indices(idx, n, seed);

    /* create copies in shuffled order using MLX API */
    mlx_vector_vector_array fac_new = mlx_vector_vector_array_new();
    mlx_vector_vector_array wells_new = mlx_vector_vector_array_new();
    mlx_vector_vector_array seis_new = mlx_vector_vector_array_new();
    for (int i = 0; i < n; ++i)
    {
        mlx_vector_array tmp = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&tmp, ds->facies, idx[i]))
        {
            mlx_vector_array_free(tmp);
            mlx_vector_vector_array_free(fac_new);
            free(idx);
            return -1;
        }
        if (mlx_vector_vector_array_append_value(fac_new, tmp))
        {
            mlx_vector_array_free(tmp);
            mlx_vector_vector_array_free(fac_new);
            free(idx);
            return -1;
        }
        mlx_vector_array_free(tmp);
    }
    /* also reorder wells/seismic if present */
    if (mlx_vector_vector_array_size(ds->wells) > 0)
    {
        for (int i = 0; i < n; ++i)
        {
            mlx_vector_array tmp = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&tmp, ds->wells, idx[i]))
            {
                mlx_vector_array_free(tmp);
                /* continue; best-effort */
            }
            else
            {
                mlx_vector_vector_array_append_value(wells_new, tmp);
                mlx_vector_array_free(tmp);
            }
        }
    }
    if (mlx_vector_vector_array_size(ds->seismic) > 0)
    {
        for (int i = 0; i < n; ++i)
        {
            mlx_vector_array tmp = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&tmp, ds->seismic, idx[i]))
            {
                mlx_vector_array_free(tmp);
            }
            else
            {
                mlx_vector_vector_array_append_value(seis_new, tmp);
                mlx_vector_array_free(tmp);
            }
        }
    }

    mlx_vector_vector_array_free(ds->facies);
    ds->facies = fac_new;
    if (mlx_vector_vector_array_size(wells_new) > 0)
    {
        mlx_vector_vector_array_free(ds->wells);
        ds->wells = wells_new;
    }
    else
    {
        mlx_vector_vector_array_free(wells_new);
    }
    if (mlx_vector_vector_array_size(seis_new) > 0)
    {
        mlx_vector_vector_array_free(ds->seismic);
        ds->seismic = seis_new;
    }
    else
    {
        mlx_vector_vector_array_free(seis_new);
    }

    free(idx);
    return 0;
}

int mlx_pyramids_dataset_clean_cache(const char *cache_dir)
{
    /* remove func_cache_*.npz files in cache_dir */
    if (!cache_dir)
        return -1;
    DIR *d = opendir(cache_dir);
    if (!d)
        return -1;
    struct dirent *entry;
    int rc = 0;
    while ((entry = readdir(d)) != NULL)
    {
        if (strncmp(entry->d_name, "func_cache_", 11) == 0)
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", cache_dir, entry->d_name);
            if (remove(path) != 0)
                rc = -1;
        }
    }
    closedir(d);
    return rc;
}

int mlx_pyramids_dataset_get_scale_stack(MLXPyramidsDataset *ds, int scale, mlx_array *out)
{
    if (!ds || !out)
        return -1;
    int N = ds->n_samples;
    if (N <= 0)
        return -1;

    /* inspect first sample scale to get shape */
    mlx_vector_array sample0;
    if (mlx_vector_vector_array_get(&sample0, ds->facies, 0) != 0)
        return -1;
    if (scale >= (int)mlx_vector_array_size(sample0))
        return -1;
    mlx_array first = mlx_array_new();
    if (mlx_vector_array_get(&first, sample0, scale) != 0)
        return -1;
    const int *shape = mlx_array_shape(first);
    int h = shape[0];
    int w = shape[1];
    int c = shape[2];

    /* Build a vector of arrays for this scale across samples and call mlx_stack */
    mlx_vector_array scale_vec = mlx_vector_array_new();
    for (int i = 0; i < N; ++i)
    {
        mlx_vector_array sample = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample, ds->facies, i))
        {
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        mlx_array elem = mlx_array_new();
        if (mlx_vector_array_get(&elem, sample, scale))
        {
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        if (mlx_vector_array_append_value(scale_vec, elem))
        {
            mlx_array_free(elem);
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        mlx_array_free(elem);
        mlx_vector_array_free(sample);
    }

    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array stacked = mlx_array_new();
    int rc = mlx_stack(&stacked, scale_vec, s);
    mlx_vector_array_free(scale_vec);
    mlx_stream_free(s);
    if (rc != 0)
    {
        mlx_array_free(stacked);
        return -1;
    }
    *out = stacked;
    return 0;
}
