#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dataloader.h"
#include "collate.h"

struct facies_dataset_
{
    mlx_vector_vector_array facies;
    mlx_vector_vector_array wells;
    mlx_vector_vector_array seismic;
    size_t n_samples;
};

struct facies_dataloader_
{
    facies_dataset *ds;
    size_t batch_size;
    size_t *indices;
    size_t idx; // next index
    size_t n_indices;
    bool shuffle;
    bool drop_last;
};

int facies_dataset_new(
    facies_dataset **out,
    const mlx_vector_vector_array facies_pyramids,
    const mlx_vector_vector_array wells_pyramids,
    const mlx_vector_vector_array seismic_pyramids)
{
    if (!out)
        return 1;
    facies_dataset *ds = (facies_dataset *)malloc(sizeof(facies_dataset));
    if (!ds)
        return 1;
    ds->facies = mlx_vector_vector_array_new();
    ds->wells = mlx_vector_vector_array_new();
    ds->seismic = mlx_vector_vector_array_new();
    // copy data references into our vectors
    size_t nf = mlx_vector_vector_array_size(facies_pyramids);
    for (size_t i = 0; i < nf; ++i)
    {
        mlx_vector_array tmp = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&tmp, facies_pyramids, i))
        {
            mlx_vector_vector_array_free(ds->facies);
            free(ds);
            return 1;
        }
        if (mlx_vector_vector_array_append_value(ds->facies, tmp))
        {
            mlx_vector_array_free(tmp);
            mlx_vector_vector_array_free(ds->facies);
            free(ds);
            return 1;
        }
        mlx_vector_array_free(tmp);
    }
    // wells/seismic may be empty
    size_t nw = mlx_vector_vector_array_size(wells_pyramids);
    for (size_t i = 0; i < nw; ++i)
    {
        mlx_vector_array tmp = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&tmp, wells_pyramids, i))
        {
            // cleanup
            mlx_vector_vector_array_free(ds->facies);
            mlx_vector_vector_array_free(ds->wells);
            free(ds);
            return 1;
        }
        if (mlx_vector_vector_array_append_value(ds->wells, tmp))
        {
            mlx_vector_array_free(tmp);
            mlx_vector_vector_array_free(ds->facies);
            mlx_vector_vector_array_free(ds->wells);
            free(ds);
            return 1;
        }
        mlx_vector_array_free(tmp);
    }
    size_t ns = mlx_vector_vector_array_size(seismic_pyramids);
    for (size_t i = 0; i < ns; ++i)
    {
        mlx_vector_array tmp = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&tmp, seismic_pyramids, i))
        {
            mlx_vector_vector_array_free(ds->facies);
            mlx_vector_vector_array_free(ds->wells);
            mlx_vector_vector_array_free(ds->seismic);
            free(ds);
            return 1;
        }
        if (mlx_vector_vector_array_append_value(ds->seismic, tmp))
        {
            mlx_vector_array_free(tmp);
            mlx_vector_vector_array_free(ds->facies);
            mlx_vector_vector_array_free(ds->wells);
            mlx_vector_vector_array_free(ds->seismic);
            free(ds);
            return 1;
        }
        mlx_vector_array_free(tmp);
    }

    ds->n_samples = mlx_vector_vector_array_size(ds->facies);
    *out = ds;
    return 0;
}

int facies_dataset_free(facies_dataset *ds)
{
    if (!ds)
        return 0;
    mlx_vector_vector_array_free(ds->facies);
    mlx_vector_vector_array_free(ds->wells);
    mlx_vector_vector_array_free(ds->seismic);
    free(ds);
    return 0;
}

static void shuffle_indices(size_t *idxs, size_t n, unsigned int seed)
{
    if (!idxs)
        return;
    srand(seed ? seed : (unsigned int)time(NULL));
    for (size_t i = n - 1; i > 0; --i)
    {
        size_t j = (size_t)(rand() % (i + 1));
        size_t t = idxs[i];
        idxs[i] = idxs[j];
        idxs[j] = t;
    }
}

int facies_dataloader_new(
    facies_dataloader **out,
    facies_dataset *ds,
    size_t batch_size,
    bool shuffle,
    bool drop_last,
    unsigned int seed)
{
    if (!out || !ds)
        return 1;
    facies_dataloader *dl = (facies_dataloader *)malloc(sizeof(facies_dataloader));
    if (!dl)
        return 1;
    dl->ds = ds;
    dl->batch_size = batch_size;
    dl->n_indices = ds->n_samples;
    dl->indices = (size_t *)malloc(sizeof(size_t) * dl->n_indices);
    if (!dl->indices)
    {
        free(dl);
        return 1;
    }
    for (size_t i = 0; i < dl->n_indices; ++i)
        dl->indices[i] = i;
    dl->shuffle = shuffle;
    dl->drop_last = drop_last;
    dl->idx = 0;
    if (dl->shuffle)
        shuffle_indices(dl->indices, dl->n_indices, seed);
    *out = dl;
    return 0;
}

int facies_dataloader_reset(facies_dataloader *dl)
{
    if (!dl)
        return 1;
    dl->idx = 0;
    if (dl->shuffle)
        shuffle_indices(dl->indices, dl->n_indices, 0);
    return 0;
}

int facies_dataloader_next(
    facies_dataloader *dl,
    mlx_vector_array *out_facies,
    mlx_vector_array *out_wells,
    mlx_vector_array *out_seismic,
    const mlx_stream s)
{
    if (!dl)
        return 1;
    size_t remaining = dl->n_indices - dl->idx;
    if (remaining == 0)
        return 2; // finished
    if (remaining < dl->batch_size && dl->drop_last)
        return 2;
    size_t cur_batch = dl->batch_size;
    if (remaining < dl->batch_size)
        cur_batch = remaining;

    // prepare temporary vector_vector_array to hold per-sample vector_array
    mlx_vector_vector_array batch_fac = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_wells = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_seismic = mlx_vector_vector_array_new();

    for (size_t i = 0; i < cur_batch; ++i)
    {
        size_t si = dl->indices[dl->idx + i];
        mlx_vector_array sample_fac = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_fac, dl->ds->facies, si))
        {
            // cleanup
            mlx_vector_vector_array_free(batch_fac);
            mlx_vector_vector_array_free(batch_wells);
            mlx_vector_vector_array_free(batch_seismic);
            return 1;
        }
        if (mlx_vector_vector_array_append_value(batch_fac, sample_fac))
        {
            mlx_vector_array_free(sample_fac);
            mlx_vector_vector_array_free(batch_fac);
            mlx_vector_vector_array_free(batch_wells);
            mlx_vector_vector_array_free(batch_seismic);
            return 1;
        }
        mlx_vector_array_free(sample_fac);

        // wells/seismic may be shorter; try to get and append if exists
        if (mlx_vector_vector_array_size(dl->ds->wells) > 0)
        {
            mlx_vector_array sample_w = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_w, dl->ds->wells, si))
            {
                mlx_vector_vector_array_free(batch_fac);
                mlx_vector_vector_array_free(batch_wells);
                mlx_vector_vector_array_free(batch_seismic);
                return 1;
            }
            if (mlx_vector_vector_array_append_value(batch_wells, sample_w))
            {
                mlx_vector_array_free(sample_w);
                mlx_vector_vector_array_free(batch_fac);
                mlx_vector_vector_array_free(batch_wells);
                mlx_vector_vector_array_free(batch_seismic);
                return 1;
            }
            mlx_vector_array_free(sample_w);
        }

        if (mlx_vector_vector_array_size(dl->ds->seismic) > 0)
        {
            mlx_vector_array sample_s = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_s, dl->ds->seismic, si))
            {
                mlx_vector_vector_array_free(batch_fac);
                mlx_vector_vector_array_free(batch_wells);
                mlx_vector_vector_array_free(batch_seismic);
                return 1;
            }
            if (mlx_vector_vector_array_append_value(batch_seismic, sample_s))
            {
                mlx_vector_array_free(sample_s);
                mlx_vector_vector_array_free(batch_fac);
                mlx_vector_vector_array_free(batch_wells);
                mlx_vector_vector_array_free(batch_seismic);
                return 1;
            }
            mlx_vector_array_free(sample_s);
        }
    }

    // Call facies_collate to produce stacked outputs
    int rc = facies_collate(out_facies, out_wells, out_seismic, batch_fac, batch_wells, batch_seismic, s);

    // advance index
    dl->idx += cur_batch;

    // free temporaries
    mlx_vector_vector_array_free(batch_fac);
    mlx_vector_vector_array_free(batch_wells);
    mlx_vector_vector_array_free(batch_seismic);

    return rc == 0 ? 0 : 1;
}

int facies_dataloader_free(facies_dataloader *dl)
{
    if (!dl)
        return 0;
    if (dl->indices)
        free(dl->indices);
    free(dl);
    return 0;
}
