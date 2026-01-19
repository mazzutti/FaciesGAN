#include "prefetcher.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/device.h>
#include <mlx/c/stream.h>
#include "../utils_extra.h"

typedef struct InternalBatch
{
    mlx_array facies;
    mlx_array seismic;
    int valid;
    /* optional per-scale vectors (when pushed via MLX API) */
    mlx_vector_array facies_vec;
    mlx_vector_array wells_vec;
    mlx_vector_array masks_vec;
    mlx_vector_array seismic_vec;
    int is_pyramids;
} InternalBatch;

typedef struct Prefetcher
{
    InternalBatch *buf;
    int capacity;
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    int alive;
    int producer_finished;
    mlx_stream stream;
    mlx_device device;
    int use_device;
} Prefetcher;

PrefetcherHandle prefetcher_create(int max_queue, int device_index)
{
    Prefetcher *p = (Prefetcher *)malloc(sizeof(Prefetcher));
    if (!p)
        return NULL;
    p->capacity = max_queue > 0 ? max_queue : 4;
    p->buf = (InternalBatch *)calloc((size_t)p->capacity, sizeof(InternalBatch));
    p->head = p->tail = p->count = 0;
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->not_empty, NULL);
    pthread_cond_init(&p->not_full, NULL);
    p->alive = 1;
    p->producer_finished = 0;
    p->use_device = 0;
    p->device = mlx_device_new();
    if (device_index >= 0)
    {
        p->device = mlx_device_new_type(MLX_GPU, device_index);
        p->stream = mlx_stream_new_device(p->device);
        p->use_device = 1;
    }
    else
    {
        p->stream = mlx_default_cpu_stream_new();
        p->use_device = 0;
    }
    return (PrefetcherHandle)p;
}

static int create_internal_from_mlx(InternalBatch *ib,
                                    const mlx_array *facies, int n_facies,
                                    const mlx_array *wells, int n_wells,
                                    const mlx_array *masks, int n_masks,
                                    const mlx_array *seismic, int n_seismic,
                                    const mlx_stream stream)
{
    if (!ib)
        return -1;
    ib->valid = 0;
    ib->is_pyramids = 1;
    ib->facies = mlx_array_new();
    ib->seismic = mlx_array_new();
    ib->facies_vec = mlx_vector_array_new();
    ib->wells_vec = mlx_vector_array_new();
    ib->masks_vec = mlx_vector_array_new();
    ib->seismic_vec = mlx_vector_array_new();

    /* copy provided mlx_arrays into internal vectors; if stream is non-CPU,
       attempt to copy arrays onto that stream/device. */
    for (int i = 0; i < n_facies; ++i)
    {
        mlx_array a = facies[i];
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx)
        {
            if (mlx_copy(&a_copy, a, stream) == 0)
            {
                /* use copied */
            }
            else
            {
                mlx_array_free(a_copy);
                a_copy = mlx_array_new();
                mlx_array_set(&a_copy, a);
            }
        }
        else
        {
            mlx_array_set(&a_copy, a);
        }
        mlx_vector_array_append_value(ib->facies_vec, a_copy);
        mlx_array_free(a_copy);
    }
    for (int i = 0; i < n_wells; ++i)
    {
        mlx_array a = wells[i];
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx)
        {
            if (mlx_copy(&a_copy, a, stream) != 0)
                mlx_array_set(&a_copy, a);
        }
        else
            mlx_array_set(&a_copy, a);
        mlx_vector_array_append_value(ib->wells_vec, a_copy);
        mlx_array_free(a_copy);
    }
    for (int i = 0; i < n_masks; ++i)
    {
        mlx_array a = masks[i];
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx)
        {
            if (mlx_copy(&a_copy, a, stream) != 0)
                mlx_array_set(&a_copy, a);
        }
        else
            mlx_array_set(&a_copy, a);
        mlx_vector_array_append_value(ib->masks_vec, a_copy);
        mlx_array_free(a_copy);
    }
    for (int i = 0; i < n_seismic; ++i)
    {
        mlx_array a = seismic[i];
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx)
        {
            if (mlx_copy(&a_copy, a, stream) != 0)
                mlx_array_set(&a_copy, a);
        }
        else
            mlx_array_set(&a_copy, a);
        mlx_vector_array_append_value(ib->seismic_vec, a_copy);
        mlx_array_free(a_copy);
    }

    ib->valid = 1;
    return 0;
}

PrefetcherHandle prefetcher_create_with_stream(int max_queue, mlx_stream stream)
{
    Prefetcher *p = (Prefetcher *)malloc(sizeof(Prefetcher));
    if (!p)
        return NULL;
    p->capacity = max_queue > 0 ? max_queue : 4;
    p->buf = (InternalBatch *)calloc((size_t)p->capacity, sizeof(InternalBatch));
    p->head = p->tail = p->count = 0;
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->not_empty, NULL);
    pthread_cond_init(&p->not_full, NULL);
    p->alive = 1;
    p->producer_finished = 0;
    // adopt provided stream
    p->stream = stream;
    p->use_device = 0;
    p->device = mlx_device_new();
    // try to infer device from stream
    mlx_device dev;
    if (mlx_stream_get_device(&dev, stream) == 0)
    {
        mlx_device_type t;
        if (mlx_device_get_type(&t, dev) == 0)
        {
            if (t == MLX_GPU)
            {
                p->use_device = 1;
                p->device = dev;
            }
            else
            {
                // CPU stream: keep device empty
                mlx_device_free(dev);
            }
        }
    }
    return (PrefetcherHandle)p;
}

static int create_internal_from_host(InternalBatch *ib,
                                     const float *facies, int facies_ndim, const int *facies_shape, int facies_len,
                                     const float *seismic, int seismic_ndim, const int *seismic_shape, int seismic_len,
                                     const mlx_stream stream)
{
    if (!ib)
        return -1;
    ib->valid = 0;
    ib->facies = mlx_array_new();
    ib->seismic = mlx_array_new();
    int rc = 0;
    if (facies && facies_len > 0)
    {
        rc = mlx_array_from_float_buffer(&ib->facies, facies, facies_shape, facies_ndim);
        if (rc != 0)
        {
            mlx_array_free(ib->facies);
            ib->facies = mlx_array_new();
            return -1;
        }
        // If a non-CPU stream/device was requested, copy onto that stream/device
        if (stream.ctx)
        {
            mlx_array dst = mlx_array_new();
            if (mlx_copy(&dst, ib->facies, stream) == 0)
            {
                mlx_array_free(ib->facies);
                ib->facies = dst;
            }
            else
            {
                // copy failed, keep original CPU array
            }
        }
    }
    if (seismic && seismic_len > 0)
    {
        rc = mlx_array_from_float_buffer(&ib->seismic, seismic, seismic_shape, seismic_ndim);
        if (rc != 0)
        {
            mlx_array_free(ib->seismic);
            ib->seismic = mlx_array_new();
            if (mlx_array_ndim(ib->facies) > 0)
                mlx_array_free(ib->facies);
            return -1;
        }
        if (stream.ctx)
        {
            mlx_array dst = mlx_array_new();
            if (mlx_copy(&dst, ib->seismic, stream) == 0)
            {
                mlx_array_free(ib->seismic);
                ib->seismic = dst;
            }
        }
    }
    ib->valid = 1;
    return 0;
}

int prefetcher_push(PrefetcherHandle h,
                    const float *facies, int facies_ndim, const int *facies_shape, int facies_len,
                    const float *seismic, int seismic_ndim, const int *seismic_shape, int seismic_len)
{
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return -1;
    InternalBatch ib;
    if (create_internal_from_host(&ib, facies, facies_ndim, facies_shape, facies_len,
                                  seismic, seismic_ndim, seismic_shape, seismic_len,
                                  p->stream) != 0)
    {
        return -1;
    }

    pthread_mutex_lock(&p->mutex);
    while (p->count == p->capacity && p->alive)
    {
        pthread_cond_wait(&p->not_full, &p->mutex);
    }
    if (!p->alive)
    {
        pthread_mutex_unlock(&p->mutex);
        if (ib.valid)
        {
            if (mlx_array_ndim(ib.facies) > 0)
                mlx_array_free(ib.facies);
            if (mlx_array_ndim(ib.seismic) > 0)
                mlx_array_free(ib.seismic);
        }
        return -1;
    }
    p->buf[p->tail] = ib;
    p->tail = (p->tail + 1) % p->capacity;
    p->count++;
    pthread_cond_signal(&p->not_empty);
    pthread_mutex_unlock(&p->mutex);
    return 0;
}

int prefetcher_push_mlx(PrefetcherHandle h,
                        const mlx_array *facies, int n_facies,
                        const mlx_array *wells, int n_wells,
                        const mlx_array *masks, int n_masks,
                        const mlx_array *seismic, int n_seismic)
{
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return -1;
    InternalBatch ib;
    memset(&ib, 0, sizeof(InternalBatch));
    if (create_internal_from_mlx(&ib, facies, n_facies, wells, n_wells, masks, n_masks, seismic, n_seismic, p->stream) != 0)
    {
        return -1;
    }

    pthread_mutex_lock(&p->mutex);
    while (p->count == p->capacity && p->alive)
    {
        pthread_cond_wait(&p->not_full, &p->mutex);
    }
    if (!p->alive)
    {
        pthread_mutex_unlock(&p->mutex);
        /* free internal vectors */
        if (ib.is_pyramids)
        {
            mlx_vector_array_free(ib.facies_vec);
            mlx_vector_array_free(ib.wells_vec);
            mlx_vector_array_free(ib.masks_vec);
            mlx_vector_array_free(ib.seismic_vec);
        }
        return -1;
    }
    p->buf[p->tail] = ib;
    p->tail = (p->tail + 1) % p->capacity;
    p->count++;
    pthread_cond_signal(&p->not_empty);
    pthread_mutex_unlock(&p->mutex);
    return 0;
}

PrefetchedBatch *prefetcher_pop(PrefetcherHandle h)
{
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return NULL;
    pthread_mutex_lock(&p->mutex);
    while (p->count == 0 && !p->producer_finished)
    {
        pthread_cond_wait(&p->not_empty, &p->mutex);
    }
    if (p->count == 0 && p->producer_finished)
    {
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    InternalBatch ib = p->buf[p->head];
    p->buf[p->head].valid = 0;
    p->head = (p->head + 1) % p->capacity;
    p->count--;
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->mutex);

    PrefetchedBatch *b = (PrefetchedBatch *)malloc(sizeof(PrefetchedBatch));
    if (!b)
        return NULL;
    memset(b, 0, sizeof(PrefetchedBatch));

    if (ib.valid && mlx_array_ndim(ib.facies) > 0)
    {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.facies, &buf, &elems, &ndim, &shape) == 0)
        {
            b->facies = buf;
            b->facies_len = (int)elems;
            b->facies_ndim = ndim;
            for (int i = 0; i < ndim && i < 8; ++i)
                b->facies_shape[i] = shape[i];
            if (shape)
                free(shape);
        }
        mlx_array_free(ib.facies);
    }

    if (ib.valid && mlx_array_ndim(ib.seismic) > 0)
    {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.seismic, &buf, &elems, &ndim, &shape) == 0)
        {
            b->seismic = buf;
            b->seismic_len = (int)elems;
            b->seismic_ndim = ndim;
            for (int i = 0; i < ndim && i < 8; ++i)
                b->seismic_shape[i] = shape[i];
            if (shape)
                free(shape);
        }
        mlx_array_free(ib.seismic);
    }

    return b;
}

/* Pop a prepared pyramids batch and return per-scale MLX arrays. Caller owns the
 * returned `PrefetchedPyramidsBatch` and must call `prefetcher_free_pyramids`.
 */
PrefetchedPyramidsBatch *prefetcher_pop_pyramids(PrefetcherHandle h)
{
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return NULL;
    pthread_mutex_lock(&p->mutex);
    while (p->count == 0 && !p->producer_finished)
    {
        pthread_cond_wait(&p->not_empty, &p->mutex);
    }
    if (p->count == 0 && p->producer_finished)
    {
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    InternalBatch ib = p->buf[p->head];
    p->buf[p->head].valid = 0;
    p->head = (p->head + 1) % p->capacity;
    p->count--;
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->mutex);

    if (!ib.is_pyramids)
    {
        /* Not a pyramids-style batch; cannot return as pyramids */
        return NULL;
    }

    PrefetchedPyramidsBatch *out = (PrefetchedPyramidsBatch *)malloc(sizeof(PrefetchedPyramidsBatch));
    if (!out)
        return NULL;
    memset(out, 0, sizeof(*out));

    int nf = mlx_vector_array_size(ib.facies_vec);
    if (nf > 0)
    {
        out->facies = (mlx_array *)malloc(sizeof(mlx_array) * nf);
        out->n_facies = nf;
        for (int i = 0; i < nf; ++i)
        {
            mlx_array a = mlx_array_new();
            mlx_vector_array_get(&a, ib.facies_vec, i);
            out->facies[i] = a;
        }
    }
    int nw = mlx_vector_array_size(ib.wells_vec);
    if (nw > 0)
    {
        out->wells = (mlx_array *)malloc(sizeof(mlx_array) * nw);
        out->n_wells = nw;
        for (int i = 0; i < nw; ++i)
        {
            mlx_array a = mlx_array_new();
            mlx_vector_array_get(&a, ib.wells_vec, i);
            out->wells[i] = a;
        }
    }
    int nm = mlx_vector_array_size(ib.masks_vec);
    if (nm > 0)
    {
        out->masks = (mlx_array *)malloc(sizeof(mlx_array) * nm);
        out->n_masks = nm;
        for (int i = 0; i < nm; ++i)
        {
            mlx_array a = mlx_array_new();
            mlx_vector_array_get(&a, ib.masks_vec, i);
            out->masks[i] = a;
        }
    }
    int ns = mlx_vector_array_size(ib.seismic_vec);
    if (ns > 0)
    {
        out->seismic = (mlx_array *)malloc(sizeof(mlx_array) * ns);
        out->n_seismic = ns;
        for (int i = 0; i < ns; ++i)
        {
            mlx_array a = mlx_array_new();
            mlx_vector_array_get(&a, ib.seismic_vec, i);
            out->seismic[i] = a;
        }
    }

    /* free internal vectors */
    mlx_vector_array_free(ib.facies_vec);
    mlx_vector_array_free(ib.wells_vec);
    mlx_vector_array_free(ib.masks_vec);
    mlx_vector_array_free(ib.seismic_vec);

    return out;
}

void prefetcher_free_pyramids(PrefetchedPyramidsBatch *b)
{
    if (!b)
        return;
    if (b->facies)
    {
        for (int i = 0; i < b->n_facies; ++i)
            mlx_array_free(b->facies[i]);
        free(b->facies);
    }
    if (b->wells)
    {
        for (int i = 0; i < b->n_wells; ++i)
            mlx_array_free(b->wells[i]);
        free(b->wells);
    }
    if (b->masks)
    {
        for (int i = 0; i < b->n_masks; ++i)
            mlx_array_free(b->masks[i]);
        free(b->masks);
    }
    if (b->seismic)
    {
        for (int i = 0; i < b->n_seismic; ++i)
            mlx_array_free(b->seismic[i]);
        free(b->seismic);
    }
    free(b);
}

PrefetchedBatch *prefetcher_pop_timeout(PrefetcherHandle h, double timeout_s)
{
    if (timeout_s < 0)
    {
        return prefetcher_pop(h);
    }
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return NULL;
    pthread_mutex_lock(&p->mutex);
    if (p->count == 0 && p->producer_finished)
    {
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    if (p->count == 0)
    {
        if (timeout_s == 0.0)
        {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
        struct timespec abs_ts;
        struct timespec now;
        if (clock_gettime(CLOCK_REALTIME, &now) != 0)
        {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
        time_t sec = (time_t)timeout_s;
        long nsec = (long)((timeout_s - (double)sec) * 1e9);
        abs_ts.tv_sec = now.tv_sec + sec;
        abs_ts.tv_nsec = now.tv_nsec + nsec;
        if (abs_ts.tv_nsec >= 1000000000L)
        {
            abs_ts.tv_sec += 1;
            abs_ts.tv_nsec -= 1000000000L;
        }
        int rc = 0;
        while (p->count == 0 && !p->producer_finished && rc != ETIMEDOUT)
        {
            rc = pthread_cond_timedwait(&p->not_empty, &p->mutex, &abs_ts);
        }
        if (rc == ETIMEDOUT && p->count == 0)
        {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
    }
    if (p->count == 0 && p->producer_finished)
    {
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    InternalBatch ib = p->buf[p->head];
    p->buf[p->head].valid = 0;
    p->head = (p->head + 1) % p->capacity;
    p->count--;
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->mutex);

    PrefetchedBatch *b = (PrefetchedBatch *)malloc(sizeof(PrefetchedBatch));
    if (!b)
        return NULL;
    memset(b, 0, sizeof(PrefetchedBatch));

    if (ib.valid && mlx_array_ndim(ib.facies) > 0)
    {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.facies, &buf, &elems, &ndim, &shape) == 0)
        {
            b->facies = buf;
            b->facies_len = (int)elems;
            b->facies_ndim = ndim;
            for (int i = 0; i < ndim && i < 8; ++i)
                b->facies_shape[i] = shape[i];
            if (shape)
                free(shape);
        }
        mlx_array_free(ib.facies);
    }

    if (ib.valid && mlx_array_ndim(ib.seismic) > 0)
    {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.seismic, &buf, &elems, &ndim, &shape) == 0)
        {
            b->seismic = buf;
            b->seismic_len = (int)elems;
            b->seismic_ndim = ndim;
            for (int i = 0; i < ndim && i < 8; ++i)
                b->seismic_shape[i] = shape[i];
            if (shape)
                free(shape);
        }
        mlx_array_free(ib.seismic);
    }

    return b;
}

void prefetcher_mark_finished(PrefetcherHandle h)
{
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return;
    pthread_mutex_lock(&p->mutex);
    p->producer_finished = 1;
    pthread_cond_broadcast(&p->not_empty);
    pthread_mutex_unlock(&p->mutex);
}

void prefetcher_free_batch(PrefetchedBatch *b)
{
    if (!b)
        return;
    if (b->facies)
        free(b->facies);
    if (b->seismic)
        free(b->seismic);
    free(b);
}

void prefetcher_destroy(PrefetcherHandle h)
{
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return;
    pthread_mutex_lock(&p->mutex);
    p->alive = 0;
    p->producer_finished = 1;
    pthread_cond_broadcast(&p->not_empty);
    pthread_cond_broadcast(&p->not_full);
    for (int i = 0; i < p->capacity; ++i)
    {
        if (p->buf[i].valid)
        {
            if (mlx_array_ndim(p->buf[i].facies) > 0)
                mlx_array_free(p->buf[i].facies);
            if (mlx_array_ndim(p->buf[i].seismic) > 0)
                mlx_array_free(p->buf[i].seismic);
        }
    }
    if (p->stream.ctx)
        mlx_stream_free(p->stream);
    if (p->use_device && p->device.ctx)
        mlx_device_free(p->device);
    free(p->buf);
    pthread_mutex_unlock(&p->mutex);
    pthread_mutex_destroy(&p->mutex);
    pthread_cond_destroy(&p->not_empty);
    pthread_cond_destroy(&p->not_full);
    free(p);
}
