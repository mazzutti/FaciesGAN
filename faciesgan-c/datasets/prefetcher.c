#include "prefetcher.h"

#include <errno.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataloader.h"
#include "faciesgan-c/utils.h"
#include <mlx/c/array.h>
#include <mlx/c/device.h>
#include <mlx/c/ops.h>
#include <mlx/c/stream.h>

#include "trainning/array_helpers.h"

typedef struct InternalBatch {
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

typedef struct Prefetcher {
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
    /* scales info for pyramid outputs */
    int *scales;
    int n_scales;
    /* number of active producer threads */
    int producer_count;
    pthread_cond_t producers_zero_cond;
    /* storage for created producer thread handles (join in destroy) */
    pthread_t *producer_threads;
    int producer_threads_capacity;
    int producer_threads_count;
} Prefetcher;

typedef struct PrefetcherIterator {
    Prefetcher *p;
    int closed;
    PrefetchedPyramidsBatch *next_prepared;
    pthread_t thread;
    int preload_in_progress;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    /* number of threads/clients currently using the iterator */
    int ref_count;
} PrefetcherIterator;

PrefetcherHandle prefetcher_create(int max_queue, int device_index,
                                   const int *scales, int n_scales) {
    Prefetcher *p = NULL;
    if (mlx_alloc_pod((void **)&p, sizeof(Prefetcher), 1) != 0)
        return NULL;
    p->capacity = max_queue > 0 ? max_queue : 4;
    p->buf = (InternalBatch *)calloc((size_t)p->capacity, sizeof(InternalBatch));
    p->head = p->tail = p->count = 0;
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->not_empty, NULL);
    pthread_cond_init(&p->not_full, NULL);
    p->producer_count = 0;
    pthread_cond_init(&p->producers_zero_cond, NULL);
    p->producer_threads = NULL;
    p->producer_threads_capacity = 0;
    p->producer_threads_count = 0;
    p->alive = 1;
    p->producer_finished = 0;
    p->use_device = 0;
    p->device = mlx_device_new();
    p->scales = NULL;
    p->n_scales = 0;
    if (n_scales > 0 && scales) {
        if (mlx_alloc_int_array(&p->scales, n_scales) == 0) {
            for (int i = 0; i < n_scales; ++i)
                p->scales[i] = scales[i];
            p->n_scales = n_scales;
        }
    }
    if (device_index >= 0) {
        mlx_device_free(p->device);
        p->device = mlx_device_new_type(MLX_GPU, device_index);
        p->stream = mlx_stream_new_device(p->device);
        p->use_device = 1;
    } else {
        p->stream = mlx_default_gpu_stream_new();
        p->use_device = 0;
    }
    return (PrefetcherHandle)p;
}

static int create_internal_from_mlx(InternalBatch *ib, const mlx_array *facies,
                                    int n_facies, const mlx_array *wells,
                                    int n_wells, const mlx_array *masks,
                                    int n_masks, const mlx_array *seismic,
                                    int n_seismic, const mlx_stream stream) {
    if (!ib)
        return -1;

    /* NOTE: Caller must hold global MLX lock */

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
    for (int i = 0; i < n_facies; ++i) {
        mlx_array a = facies[i];
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx) {
            if (mlx_copy(&a_copy, a, stream) == 0) {
                /* use copied */
            } else {
                mlx_array_free(a_copy);
                a_copy = mlx_array_new();
                mlx_array_set(&a_copy, a);
            }
        } else {
            mlx_array_set(&a_copy, a);
        }
        /* ensure vector stores an independent buffer to avoid aliasing */
        {
            mlx_array to_append = mlx_array_new();
            if (stream.ctx) {
                if (mlx_copy(&to_append, a_copy, stream) == 0) {
                    mlx_vector_array_append_value(ib->facies_vec, to_append);
                    mlx_array_free(to_append);
                    if (stream.ctx)
                        mlx_synchronize(stream);
                    mlx_array_free(a_copy);
                } else {
                    mlx_array_free(to_append);
                    mlx_vector_array_append_value(ib->facies_vec, a_copy);
                    mlx_array_free(a_copy);
                }
            } else {
                /* CPU stream: `a_copy` is already CPU-resident; append it directly */
                mlx_array_free(to_append);
                mlx_vector_array_append_value(ib->facies_vec, a_copy);
                mlx_array_free(a_copy);
            }
        }
    }
    for (int i = 0; i < n_wells; ++i) {
        mlx_array a = wells[i];
        /* Debug: check input well array (NO eval - not thread-safe) */
        int input_ndim = (int)mlx_array_ndim(a);
        (void)input_ndim;
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx) {
            if (mlx_copy(&a_copy, a, stream) != 0) {
                mlx_array_set(&a_copy, a);
            }
        } else {
            mlx_array_set(&a_copy, a);
        }
        /* ensure vector stores independent buffer */
        {
            mlx_array to_append = mlx_array_new();
            if (stream.ctx) {
                if (mlx_copy(&to_append, a_copy, stream) == 0) {
                    mlx_vector_array_append_value(ib->wells_vec, to_append);
                    mlx_array_free(to_append);
                    if (stream.ctx)
                        mlx_synchronize(stream);
                    mlx_array_free(a_copy);
                } else {
                    mlx_array_free(to_append);
                    mlx_vector_array_append_value(ib->wells_vec, a_copy);
                    mlx_array_free(a_copy);
                }
            } else {
                mlx_array_free(to_append);
                mlx_vector_array_append_value(ib->wells_vec, a_copy);
                mlx_array_free(a_copy);
            }
        }
    }

    /* NOTE: Mask computation has been moved to the main thread (in the training
     * loop) to avoid thread-safety issues with MLX. The producer thread only
     * passes through the wells data, and masks are computed after prefetcher_pop.
     * See mlx_trainer_api.c for the mask computation logic.
     */

    /* If explicit masks were provided, store them */
    for (int i = 0; i < n_masks; ++i) {
        mlx_array a = masks[i];
        /* Skip empty mask arrays - don't append them to the vector */
        if (!a.ctx) {
            continue;
        }
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx) {
            if (mlx_copy(&a_copy, a, stream) != 0)
                mlx_array_set(&a_copy, a);
        } else
            mlx_array_set(&a_copy, a);
        /* ensure vector stores independent buffer */
        {
            mlx_array to_append = mlx_array_new();
            if (stream.ctx) {
                if (mlx_copy(&to_append, a_copy, stream) == 0) {
                    mlx_vector_array_append_value(ib->masks_vec, to_append);
                    mlx_array_free(to_append);
                    if (stream.ctx)
                        mlx_synchronize(stream);
                    mlx_array_free(a_copy);
                } else {
                    mlx_array_free(to_append);
                    mlx_vector_array_append_value(ib->masks_vec, a_copy);
                    mlx_array_free(a_copy);
                }
            } else {
                mlx_array_free(to_append);
                mlx_vector_array_append_value(ib->masks_vec, a_copy);
                mlx_array_free(a_copy);
            }
        }
    }
    for (int i = 0; i < n_seismic; ++i) {
        mlx_array a = seismic[i];
        mlx_array a_copy = mlx_array_new();
        if (stream.ctx) {
            if (mlx_copy(&a_copy, a, stream) != 0)
                mlx_array_set(&a_copy, a);
        } else
            mlx_array_set(&a_copy, a);
        /* ensure vector stores independent buffer */
        {
            mlx_array to_append = mlx_array_new();
            if (stream.ctx) {
                if (mlx_copy(&to_append, a_copy, stream) == 0) {
                    mlx_vector_array_append_value(ib->seismic_vec, to_append);
                    mlx_array_free(to_append);
                    if (stream.ctx)
                        mlx_synchronize(stream);
                    mlx_array_free(a_copy);
                } else {
                    mlx_array_free(to_append);
                    mlx_vector_array_append_value(ib->seismic_vec, a_copy);
                    mlx_array_free(a_copy);
                }
            } else {
                mlx_array_free(to_append);
                mlx_vector_array_append_value(ib->seismic_vec, a_copy);
                mlx_array_free(a_copy);
            }
        }
    }

    ib->valid = 1;

    /* NOTE: Caller releases lock */

    return 0;
}

PrefetcherHandle prefetcher_create_with_stream(int max_queue, mlx_stream stream,
        const int *scales,
        int n_scales) {
    Prefetcher *p = NULL;
    if (mlx_alloc_pod((void **)&p, sizeof(Prefetcher), 1) != 0)
        return NULL;
    p->capacity = max_queue > 0 ? max_queue : 4;
    p->buf = (InternalBatch *)calloc((size_t)p->capacity, sizeof(InternalBatch));
    p->head = p->tail = p->count = 0;
    pthread_mutex_init(&p->mutex, NULL);
    pthread_cond_init(&p->not_empty, NULL);
    pthread_cond_init(&p->not_full, NULL);
    p->alive = 1;
    p->producer_finished = 0;
    p->producer_threads = NULL;
    p->producer_threads_capacity = 0;
    p->producer_threads_count = 0;
    // adopt provided stream
    p->stream = stream;
    p->use_device = 0;
    p->device = mlx_device_new();
    p->scales = NULL;
    p->n_scales = 0;
    if (n_scales > 0 && scales) {
        if (mlx_alloc_int_array(&p->scales, n_scales) == 0) {
            for (int i = 0; i < n_scales; ++i)
                p->scales[i] = scales[i];
            p->n_scales = n_scales;
        }
    }

    // try to infer device from stream
    mlx_device dev = mlx_device_new();
    if (mlx_stream_get_device(&dev, stream) == 0) {
        mlx_device_type t;
        if (mlx_device_get_type(&t, dev) == 0) {
            if (t == MLX_GPU) {
                p->use_device = 1;
                mlx_device_free(p->device);
                p->device = dev;
            } else {
                // CPU stream: keep device empty
                mlx_device_free(dev);
            }
        }
    }
    return (PrefetcherHandle)p;
}

/* Background producer that reads from a facies_dataloader and pushes into
 * the provided prefetcher handle. Ownership of the provided `stream` is
 * consumed by the thread and it will be freed when the producer finishes.
 */
typedef struct PrefetcherDLProducerArgs {
    struct MLXDataloader *dl;
    PrefetcherHandle ph;
    mlx_stream s;
} PrefetcherDLProducerArgs;

static void *prefetcher_dataloader_producer(void *v) {
    PrefetcherDLProducerArgs *a = (PrefetcherDLProducerArgs *)v;
    struct MLXDataloader *dl = a->dl;
    PrefetcherHandle ph = a->ph;
    mlx_stream s = a->s;
    Prefetcher *pref = (Prefetcher *)ph;

    /* Loop infinitely (like PyTorch's cycle) until the prefetcher is stopped.
     * When the dataloader exhausts its samples (rc == 2), reset and continue
     * to provide data for multiple training iterations/epochs. */
    while (1) {
        /* Check if the prefetcher has been signaled to stop */
        pthread_mutex_lock(&pref->mutex);
        int alive = pref->alive;
        pthread_mutex_unlock(&pref->mutex);
        if (!alive)
            break;

        mlx_vector_array facs = mlx_vector_array_new();
        mlx_vector_array wells_out = mlx_vector_array_new();
        mlx_vector_array seis_out = mlx_vector_array_new();

        /* Acquire global MLX lock for ALL MLX operations in this iteration */
        mlx_global_lock();
        int rc = facies_dataloader_next(dl, &facs, &wells_out, &seis_out, s);

        if (rc == 2) {
            /* End of epoch: reset dataloader and continue cycling */
            mlx_vector_array_free(facs);
            mlx_vector_array_free(wells_out);
            mlx_vector_array_free(seis_out);
            mlx_global_unlock();
            facies_dataloader_reset(dl);
            continue;
        } else if (rc != 0) {
            mlx_vector_array_free(facs);
            mlx_vector_array_free(wells_out);
            mlx_vector_array_free(seis_out);
            mlx_global_unlock();
            break;
        }

        int nsc = (int)mlx_vector_array_size(facs);
        mlx_array *fac_arr = NULL;
        mlx_array *well_arr = NULL;
        mlx_array *sei_arr = NULL;
        if (nsc > 0) {
            if (mlx_alloc_mlx_array_raw(&fac_arr, nsc) == 0) {
                for (int i = 0; i < nsc; ++i) {
                    mlx_array tmp = mlx_array_new();
                    if (mlx_vector_array_get(&tmp, facs, i) != 0) {
                        mlx_array_free(tmp);
                        tmp = mlx_array_new();
                    }
                    fac_arr[i] = tmp;
                }
            } else {
                fac_arr = NULL;
            }
        }
        int nw = (int)mlx_vector_array_size(wells_out);
        if (nw > 0) {
            if (mlx_alloc_mlx_array_raw(&well_arr, nw) == 0) {
                for (int i = 0; i < nw; ++i) {
                    mlx_array tmp = mlx_array_new();
                    int get_rc = mlx_vector_array_get(&tmp, wells_out, i);
                    if (get_rc != 0) {
                        mlx_array_free(tmp);
                        tmp = mlx_array_new();
                    } else {
                        /* NO eval here - not thread-safe */
                        int tmp_ndim = (int)mlx_array_ndim(tmp);
                    }
                    well_arr[i] = tmp;
                }
            } else {
                well_arr = NULL;
            }
        }
        int ns = (int)mlx_vector_array_size(seis_out);
        if (ns > 0) {
            if (mlx_alloc_mlx_array_raw(&sei_arr, ns) == 0) {
                for (int i = 0; i < ns; ++i) {
                    mlx_array tmp = mlx_array_new();
                    if (mlx_vector_array_get(&tmp, seis_out, i) != 0) {
                        mlx_array_free(tmp);
                        tmp = mlx_array_new();
                    }
                    sei_arr[i] = tmp;
                }
            } else {
                sei_arr = NULL;
            }
        }

        /* Release lock before pushing to queue (may block) */
        mlx_global_unlock();

        /* Note: masks are computed from wells inside create_internal_from_mlx,
         * so we pass NULL for masks here. This ensures consistent mask computation
         * and avoids issues with empty placeholder arrays. */

        /* push into prefetcher (handles its own MLX lock for create_internal_from_mlx) */
        prefetcher_push_mlx(ph, fac_arr, nsc, well_arr, nw, NULL, 0, sei_arr, ns);

        /* Free element handles (obtained from mlx_vector_array_get, which creates
         * copies of the C++ shared_ptr) and container memory. The elements have
         * already been copied into the prefetcher's internal batch by
         * create_internal_from_mlx, so we can safely free these handles. */
        if (fac_arr) {
            for (int i = 0; i < nsc; ++i)
                mlx_array_free(fac_arr[i]);
            mlx_free_mlx_array_raw(&fac_arr, nsc);
        }
        if (well_arr) {
            for (int i = 0; i < nw; ++i)
                mlx_array_free(well_arr[i]);
            mlx_free_mlx_array_raw(&well_arr, nw);
        }
        if (sei_arr) {
            for (int i = 0; i < ns; ++i)
                mlx_array_free(sei_arr[i]);
            mlx_free_mlx_array_raw(&sei_arr, ns);
        }

        mlx_vector_array_free(facs);
        mlx_vector_array_free(wells_out);
        mlx_vector_array_free(seis_out);
    }

    prefetcher_mark_finished(ph);
    if (a->s.ctx)
        mlx_stream_free(a->s);
    Prefetcher *p = (Prefetcher *)ph;
    pthread_mutex_lock(&p->mutex);
    p->producer_count--;
    pthread_cond_signal(&p->producers_zero_cond);
    pthread_mutex_unlock(&p->mutex);
    mlx_free_pod((void **)&a);
    return NULL;
}

int prefetcher_start_from_dataloader(PrefetcherHandle ph,
                                     struct MLXDataloader *dl,
                                     mlx_stream stream) {
    if (!ph || !dl)
        return -1;
    PrefetcherDLProducerArgs *args = NULL;
    if (mlx_alloc_pod((void **)&args, sizeof(PrefetcherDLProducerArgs), 1) != 0)
        return -1;
    args->dl = dl;
    args->ph = ph;
    args->s = stream;
    pthread_t t;
    Prefetcher *p = (Prefetcher *)ph;
    pthread_mutex_lock(&p->mutex);
    if (!p->alive) {
        pthread_mutex_unlock(&p->mutex);
        mlx_free_pod((void **)&args);
        return -1;
    }
    p->producer_count++;
    pthread_mutex_unlock(&p->mutex);
    if (pthread_create(&t, NULL, prefetcher_dataloader_producer, args) != 0) {
        mlx_free_pod((void **)&args);
        pthread_mutex_lock(&p->mutex);
        p->producer_count--;
        pthread_mutex_unlock(&p->mutex);
        return -1;
    }
    /* store thread handle so we can join it on destroy */
    pthread_mutex_lock(&p->mutex);
    if (p->producer_threads_count == p->producer_threads_capacity) {
        int newcap = p->producer_threads_capacity == 0
                     ? 4
                     : p->producer_threads_capacity * 2;
        pthread_t *nptr =
            (pthread_t *)realloc(p->producer_threads, sizeof(pthread_t) * newcap);
        if (nptr) {
            p->producer_threads = nptr;
            p->producer_threads_capacity = newcap;
        }
    }
    if (p->producer_threads_count < p->producer_threads_capacity) {
        p->producer_threads[p->producer_threads_count++] = t;
    }
    pthread_mutex_unlock(&p->mutex);
    return 0;
}

static int create_internal_from_host(InternalBatch *ib, const float *facies,
                                     int facies_ndim, const int *facies_shape,
                                     int facies_len, const float *seismic,
                                     int seismic_ndim, const int *seismic_shape,
                                     int seismic_len, const mlx_stream stream) {
    if (!ib)
        return -1;
    ib->valid = 0;
    ib->facies = mlx_array_new();
    ib->seismic = mlx_array_new();
    int rc = 0;
    if (facies && facies_len > 0) {
        rc = mlx_array_from_float_buffer(&ib->facies, facies, facies_shape,
                                         facies_ndim);
        if (rc != 0) {
            mlx_array_free(ib->facies);
            ib->facies = mlx_array_new();
            return -1;
        }
        // If a non-CPU stream/device was requested, copy onto that stream/device
        if (stream.ctx) {
            mlx_array dst = mlx_array_new();
            if (mlx_copy(&dst, ib->facies, stream) == 0) {
                mlx_array_free(ib->facies);
                ib->facies = dst;
            } else {
                // copy failed, keep original CPU array
            }
        }
    }
    if (seismic && seismic_len > 0) {
        rc = mlx_array_from_float_buffer(&ib->seismic, seismic, seismic_shape,
                                         seismic_ndim);
        if (rc != 0) {
            mlx_array_free(ib->seismic);
            ib->seismic = mlx_array_new();
            if (mlx_array_ndim(ib->facies) > 0)
                mlx_array_free(ib->facies);
            return -1;
        }
        if (stream.ctx) {
            mlx_array dst = mlx_array_new();
            if (mlx_copy(&dst, ib->seismic, stream) == 0) {
                mlx_array_free(ib->seismic);
                ib->seismic = dst;
            }
        }
    }
    ib->valid = 1;
    return 0;
}

int prefetcher_push(PrefetcherHandle h, const float *facies, int facies_ndim,
                    const int *facies_shape, int facies_len,
                    const float *seismic, int seismic_ndim,
                    const int *seismic_shape, int seismic_len) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return -1;
    InternalBatch ib;
    if (create_internal_from_host(&ib, facies, facies_ndim, facies_shape,
                                  facies_len, seismic, seismic_ndim,
                                  seismic_shape, seismic_len, p->stream) != 0) {
        return -1;
    }

    pthread_mutex_lock(&p->mutex);
    while (p->count == p->capacity && p->alive) {
        pthread_cond_wait(&p->not_full, &p->mutex);
    }
    if (!p->alive) {
        pthread_mutex_unlock(&p->mutex);
        if (ib.valid) {
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

int prefetcher_push_mlx(PrefetcherHandle h, const mlx_array *facies,
                        int n_facies, const mlx_array *wells, int n_wells,
                        const mlx_array *masks, int n_masks,
                        const mlx_array *seismic, int n_seismic) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return -1;
    InternalBatch ib;
    memset(&ib, 0, sizeof(InternalBatch));

    /* MLX operations - acquire lock */
    mlx_global_lock();
    if (create_internal_from_mlx(&ib, facies, n_facies, wells, n_wells, masks,
                                 n_masks, seismic, n_seismic, p->stream) != 0) {
        mlx_global_unlock();
        return -1;
    }
    mlx_global_unlock();
    /* MLX operations done - lock released */

    /* Queue operations - may block waiting for space, must NOT hold MLX lock */
    pthread_mutex_lock(&p->mutex);
    while (p->count == p->capacity && p->alive) {
        pthread_cond_wait(&p->not_full, &p->mutex);
    }
    if (!p->alive) {
        pthread_mutex_unlock(&p->mutex);
        /* During shutdown: free the InternalBatch we just created.
           create_internal_from_mlx copies elements into its own vectors,
           so these handles are independent and must be freed. */
        if (ib.is_pyramids) {
            if (ib.facies_vec.ctx)
                mlx_vector_array_free(ib.facies_vec);
            if (ib.wells_vec.ctx)
                mlx_vector_array_free(ib.wells_vec);
            if (ib.masks_vec.ctx)
                mlx_vector_array_free(ib.masks_vec);
            if (ib.seismic_vec.ctx)
                mlx_vector_array_free(ib.seismic_vec);
        }
        if (ib.facies.ctx)
            mlx_array_free(ib.facies);
        if (ib.seismic.ctx)
            mlx_array_free(ib.seismic);
        return -1;
    }
    p->buf[p->tail] = ib;
    p->tail = (p->tail + 1) % p->capacity;
    p->count++;
    pthread_cond_signal(&p->not_empty);
    pthread_mutex_unlock(&p->mutex);
    return 0;
}

PrefetchedBatch *prefetcher_pop(PrefetcherHandle h) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return NULL;
    pthread_mutex_lock(&p->mutex);
    while (p->count == 0 && !p->producer_finished) {
        pthread_cond_wait(&p->not_empty, &p->mutex);
    }
    if (p->count == 0 && p->producer_finished) {
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    InternalBatch ib = p->buf[p->head];
    p->buf[p->head].valid = 0;
    p->head = (p->head + 1) % p->capacity;
    p->count--;
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->mutex);

    PrefetchedBatch *b = NULL;
    if (mlx_alloc_pod((void **)&b, sizeof(PrefetchedBatch), 1) != 0)
        return NULL;

    if (ib.valid && mlx_array_ndim(ib.facies) > 0) {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.facies, &buf, &elems, &ndim, &shape) ==
                0) {
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

    if (ib.valid && mlx_array_ndim(ib.seismic) > 0) {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.seismic, &buf, &elems, &ndim, &shape) ==
                0) {
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

/* Pop a prepared pyramids batch and return per-scale MLX arrays. Caller owns
 * the returned `PrefetchedPyramidsBatch` and must call
 * `prefetcher_free_pyramids`.
 */
PrefetchedPyramidsBatch *prefetcher_pop_pyramids(PrefetcherHandle h) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p) {
        return NULL;
    }
    pthread_mutex_lock(&p->mutex);
    while (p->count == 0 && !p->producer_finished) {
        pthread_cond_wait(&p->not_empty, &p->mutex);
    }
    if (p->count == 0 && p->producer_finished) {
        /* Give producers a short grace period to fill the buffer in case
           `producer_finished` was set concurrently with pending pushes. Use a
           timed wait (5s) on the `not_empty` condvar; if nothing arrives we
           fall back to returning NULL. */
        struct timespec abs_ts, now;
        if (clock_gettime(CLOCK_REALTIME, &now) == 0) {
            abs_ts.tv_sec = now.tv_sec + 5; /* 5 second grace */
            abs_ts.tv_nsec = now.tv_nsec;
            int rc = 0;
            while (p->count == 0 && rc != ETIMEDOUT) {
                rc = pthread_cond_timedwait(&p->not_empty, &p->mutex, &abs_ts);
            }
        }
        if (p->count == 0) {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
    }
    InternalBatch ib = p->buf[p->head];
    p->buf[p->head].valid = 0;
    p->head = (p->head + 1) % p->capacity;
    p->count--;
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->mutex);

    if (!ib.is_pyramids) {
        return NULL;
    }

    PrefetchedPyramidsBatch *out = NULL;
    if (mlx_alloc_pod((void **)&out, sizeof(PrefetchedPyramidsBatch), 1) != 0)
        return NULL;
    memset(out, 0, sizeof(*out));

    int n_scales = p->n_scales;
    out->n_scales = n_scales;
    if (n_scales > 0 && p->scales) {
        if (mlx_alloc_int_array(&out->scales, n_scales) == 0) {
            for (int i = 0; i < n_scales; ++i)
                out->scales[i] = p->scales[i];
        }
    }


    /* allocate arrays aligned to n_scales and fill entries from internal vectors
       where available; otherwise leave empty mlx_array objects */
    mlx_global_lock(); /* protect all MLX operations from concurrent access */
    int nf = mlx_vector_array_size(ib.facies_vec);
    if (mlx_alloc_mlx_array_raw(&out->facies, n_scales) != 0)
        out->facies = NULL;
    for (int i = 0; i < n_scales; ++i) {
        if (i < nf) {
            /* get element, then set it into output array */
            mlx_array tmp = mlx_array_new();
            mlx_vector_array_get(&tmp, ib.facies_vec, i);
            /* Use array_set instead of copy to avoid potential stream issues */
            out->facies[i] = mlx_array_new();
            mlx_array_set(&out->facies[i], tmp);
            mlx_array_free(tmp);
        } else {
            out->facies[i] = mlx_array_new();
        }
    }

    /* Create pointer arrays that reference the allocated mlx_array values
     * so callers that expect `mlx_array **` can use them directly without
     * performing per-batch pointer loops. Use generic pointer-array helpers
     * so we don't attempt to free pointees twice. */
    if (out->facies) {
        if (mlx_alloc_ptr_array((void ***)&out->facies_ptrs, n_scales) == 0) {
            for (int i = 0; i < n_scales; ++i)
                out->facies_ptrs[i] = &out->facies[i];
        } else {
            out->facies_ptrs = NULL;
        }
    } else {
        out->facies_ptrs = NULL;
    }

    int nw = mlx_vector_array_size(ib.wells_vec);
    if (nw > 0) {
        if (mlx_alloc_mlx_array_raw(&out->wells, n_scales) != 0)
            out->wells = NULL;
        for (int i = 0; i < n_scales; ++i) {
            if (i < nw) {
                mlx_array tmp = mlx_array_new();
                mlx_vector_array_get(&tmp, ib.wells_vec, i);
                out->wells[i] = mlx_array_new();
                mlx_array_set(&out->wells[i], tmp);
                mlx_array_free(tmp);
            } else {
                out->wells[i] = mlx_array_new();
            }
        }
    } else {
        out->wells = NULL; /* match Python: empty dict semantics */
    }
    if (out->wells) {
        if (mlx_alloc_ptr_array((void ***)&out->wells_ptrs, n_scales) == 0) {
            for (int i = 0; i < n_scales; ++i)
                out->wells_ptrs[i] = &out->wells[i];
        } else {
            out->wells_ptrs = NULL;
        }
    } else {
        out->wells_ptrs = NULL;
    }

    int nm = mlx_vector_array_size(ib.masks_vec);
    if (nm > 0) {
        if (mlx_alloc_mlx_array_raw(&out->masks, n_scales) != 0)
            out->masks = NULL;
        for (int i = 0; i < n_scales; ++i) {
            if (i < nm) {
                mlx_array tmp = mlx_array_new();
                mlx_vector_array_get(&tmp, ib.masks_vec, i);
                out->masks[i] = mlx_array_new();
                mlx_array_set(&out->masks[i], tmp);
                mlx_array_free(tmp);
            } else {
                out->masks[i] = mlx_array_new();
            }
        }
    } else {
        out->masks = NULL; /* match Python: empty dict semantics */
    }
    if (out->masks) {
        if (mlx_alloc_ptr_array((void ***)&out->masks_ptrs, n_scales) == 0) {
            for (int i = 0; i < n_scales; ++i)
                out->masks_ptrs[i] = &out->masks[i];
        } else {
            out->masks_ptrs = NULL;
        }
    } else {
        out->masks_ptrs = NULL;
    }

    int ns = mlx_vector_array_size(ib.seismic_vec);
    if (ns > 0) {
        if (mlx_alloc_mlx_array_raw(&out->seismic, n_scales) != 0)
            out->seismic = NULL;
        for (int i = 0; i < n_scales; ++i) {
            if (i < ns) {
                mlx_array tmp = mlx_array_new();
                mlx_vector_array_get(&tmp, ib.seismic_vec, i);
                out->seismic[i] = mlx_array_new();
                mlx_array_set(&out->seismic[i], tmp);
                mlx_array_free(tmp);
            } else {
                out->seismic[i] = mlx_array_new();
            }
        }
    } else {
        out->seismic = NULL; /* match Python: empty dict semantics */
    }
    if (out->seismic) {
        if (mlx_alloc_ptr_array((void ***)&out->seismic_ptrs, n_scales) == 0) {
            for (int i = 0; i < n_scales; ++i)
                out->seismic_ptrs[i] = &out->seismic[i];
        } else {
            out->seismic_ptrs = NULL;
        }
    } else {
        out->seismic_ptrs = NULL;
    }

    /* free internal vectors */
    mlx_vector_array_free(ib.facies_vec);
    mlx_vector_array_free(ib.wells_vec);
    mlx_vector_array_free(ib.masks_vec);
    mlx_vector_array_free(ib.seismic_vec);
    mlx_global_unlock();

    return out;
}

void prefetcher_free_pyramids(PrefetchedPyramidsBatch *b) {
    if (!b)
        return;
    int n = b->n_scales;
    if (b->facies) {
        for (int i = 0; i < n; ++i)
            mlx_array_free(b->facies[i]);
        mlx_free_mlx_array_raw(&b->facies, n);
    }
    if (b->facies_ptrs) {
        mlx_free_ptr_array((void ***)&b->facies_ptrs, n);
    }
    if (b->wells) {
        for (int i = 0; i < n; ++i)
            mlx_array_free(b->wells[i]);
        mlx_free_mlx_array_raw(&b->wells, n);
    }
    if (b->wells_ptrs) {
        mlx_free_ptr_array((void ***)&b->wells_ptrs, n);
    }
    if (b->masks) {
        for (int i = 0; i < n; ++i)
            mlx_array_free(b->masks[i]);
        mlx_free_mlx_array_raw(&b->masks, n);
    }
    if (b->masks_ptrs) {
        mlx_free_ptr_array((void ***)&b->masks_ptrs, n);
    }
    if (b->seismic) {
        for (int i = 0; i < n; ++i)
            mlx_array_free(b->seismic[i]);
        mlx_free_mlx_array_raw(&b->seismic, n);
    }
    if (b->seismic_ptrs) {
        mlx_free_ptr_array((void ***)&b->seismic_ptrs, n);
    }
    if (b->scales)
        mlx_free_int_array(&b->scales, &n);
    mlx_free_pod((void **)&b);
}

PrefetchedBatch *prefetcher_pop_timeout(PrefetcherHandle h, double timeout_s) {
    if (timeout_s < 0) {
        return prefetcher_pop(h);
    }
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return NULL;
    pthread_mutex_lock(&p->mutex);
    if (p->count == 0 && p->producer_finished) {
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    if (p->count == 0) {
        if (timeout_s == 0.0) {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
        struct timespec abs_ts;
        struct timespec now;
        if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
        time_t sec = (time_t)timeout_s;
        long nsec = (long)((timeout_s - (double)sec) * 1e9);
        abs_ts.tv_sec = now.tv_sec + sec;
        abs_ts.tv_nsec = now.tv_nsec + nsec;
        if (abs_ts.tv_nsec >= 1000000000L) {
            abs_ts.tv_sec += 1;
            abs_ts.tv_nsec -= 1000000000L;
        }
        int rc = 0;
        while (p->count == 0 && !p->producer_finished && rc != ETIMEDOUT) {
            rc = pthread_cond_timedwait(&p->not_empty, &p->mutex, &abs_ts);
        }
        if (rc == ETIMEDOUT && p->count == 0) {
            pthread_mutex_unlock(&p->mutex);
            return NULL;
        }
    }
    if (p->count == 0 && p->producer_finished) {
        pthread_mutex_unlock(&p->mutex);
        return NULL;
    }
    InternalBatch ib = p->buf[p->head];
    p->buf[p->head].valid = 0;
    p->head = (p->head + 1) % p->capacity;
    p->count--;
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->mutex);

    PrefetchedBatch *b = NULL;
    if (mlx_alloc_pod((void **)&b, sizeof(PrefetchedBatch), 1) != 0)
        return NULL;

    if (ib.valid && mlx_array_ndim(ib.facies) > 0) {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.facies, &buf, &elems, &ndim, &shape) ==
                0) {
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

    if (ib.valid && mlx_array_ndim(ib.seismic) > 0) {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(ib.seismic, &buf, &elems, &ndim, &shape) ==
                0) {
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

void prefetcher_mark_finished(PrefetcherHandle h) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return;
    pthread_mutex_lock(&p->mutex);
    p->producer_finished = 1;
    pthread_cond_broadcast(&p->not_empty);
    pthread_mutex_unlock(&p->mutex);
}

int prefetcher_stop(PrefetcherHandle h) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return -1;
    /* signal shutdown and capture thread list to join outside mutex */
    pthread_mutex_lock(&p->mutex);
    p->alive = 0;
    p->producer_finished = 1;
    pthread_cond_broadcast(&p->not_empty);
    pthread_cond_broadcast(&p->not_full);
    int n_producers = p->producer_threads_count;
    pthread_t *threads_copy = NULL;
    if (n_producers > 0) {
        if (mlx_alloc_pod((void **)&threads_copy, sizeof(pthread_t), n_producers) == 0) {
            for (int i = 0; i < n_producers; ++i)
                threads_copy[i] = p->producer_threads[i];
        } else {
            threads_copy = NULL;
            n_producers = 0;
        }
        /* clear stored handles so destroy won't attempt duplicate joins */
        free(p->producer_threads);
        p->producer_threads = NULL;
        p->producer_threads_capacity = 0;
        p->producer_threads_count = 0;
    }
    pthread_mutex_unlock(&p->mutex);

    /* join producers we observed */
    for (int i = 0; i < n_producers; ++i) {
        pthread_join(threads_copy[i], NULL);
    }
    if (threads_copy)
        mlx_free_pod((void **)&threads_copy);

    /* wait for producer_count to reach zero (producers signal on exit) */
    pthread_mutex_lock(&p->mutex);
    while (p->producer_count > 0) {
        pthread_cond_wait(&p->producers_zero_cond, &p->mutex);
    }
    pthread_mutex_unlock(&p->mutex);
    return 0;
}

void prefetcher_free_batch(PrefetchedBatch *b) {
    if (!b)
        return;
    if (b->facies)
        free(b->facies);
    if (b->seismic)
        free(b->seismic);
    free(b);
}

void prefetcher_destroy(PrefetcherHandle h) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return;
    /* signal shutdown and capture thread list to join outside mutex */
    pthread_mutex_lock(&p->mutex);
    p->alive = 0;
    p->producer_finished = 1;
    pthread_cond_broadcast(&p->not_empty);
    pthread_cond_broadcast(&p->not_full);
    int n_producers = p->producer_threads_count;
    pthread_t *threads_copy = NULL;
    if (n_producers > 0) {
        if (mlx_alloc_pod((void **)&threads_copy, sizeof(pthread_t), n_producers) ==
                0) {
            for (int i = 0; i < n_producers; ++i)
                threads_copy[i] = p->producer_threads[i];
        } else {
            threads_copy = NULL;
            n_producers = 0;
        }
    }
    pthread_mutex_unlock(&p->mutex);

    /* join all producer threads (they will decrement producer_count on exit) */
    for (int i = 0; i < n_producers; ++i) {
        pthread_join(threads_copy[i], NULL);
    }
    if (threads_copy)
        mlx_free_pod((void **)&threads_copy);
    /* safe to clean buffers now: producers have exited */
    for (int i = 0; i < p->capacity; ++i) {
        if (p->buf[i].valid) {
            if (p->buf[i].facies.ctx)
                mlx_array_free(p->buf[i].facies);
            if (p->buf[i].seismic.ctx)
                mlx_array_free(p->buf[i].seismic);
            if (p->buf[i].is_pyramids) {
                if (p->buf[i].facies_vec.ctx)
                    mlx_vector_array_free(p->buf[i].facies_vec);
                if (p->buf[i].wells_vec.ctx)
                    mlx_vector_array_free(p->buf[i].wells_vec);
                if (p->buf[i].masks_vec.ctx)
                    mlx_vector_array_free(p->buf[i].masks_vec);
                if (p->buf[i].seismic_vec.ctx)
                    mlx_vector_array_free(p->buf[i].seismic_vec);
            }
        }
    }
    if (p->stream.ctx)
        mlx_stream_free(p->stream);
    if (p->use_device && p->device.ctx)
        mlx_device_free(p->device);
    if (p->scales)
        mlx_free_int_array(&p->scales, &p->n_scales);
    if (p->producer_threads)
        free(p->producer_threads);
    free(p->buf);
    pthread_cond_destroy(&p->producers_zero_cond);
    pthread_mutex_destroy(&p->mutex);
    pthread_cond_destroy(&p->not_empty);
    pthread_cond_destroy(&p->not_full);
    mlx_free_pod((void **)&p);
}

PrefetcherIteratorHandle prefetcher_iterator_create(PrefetcherHandle h) {
    if (!h)
        return NULL;
    Prefetcher *p = (Prefetcher *)h;
    PrefetcherIterator *it = NULL;
    if (mlx_alloc_pod((void **)&it, sizeof(PrefetcherIterator), 1) != 0)
        return NULL;
    it->p = p;
    it->closed = 0;
    it->next_prepared = NULL;
    it->preload_in_progress = 0;
    it->ref_count = 0;
    pthread_mutex_init(&it->mutex, NULL);
    pthread_cond_init(&it->cond, NULL);
    return (PrefetcherIteratorHandle)it;
}

PrefetchedPyramidsBatch *
prefetcher_iterator_next(PrefetcherIteratorHandle it_h) {
    if (!it_h)
        return NULL;
    PrefetcherIterator *it = (PrefetcherIterator *)it_h;
    pthread_mutex_lock(&it->mutex);
    if (it->closed || !it->p) {
        pthread_mutex_unlock(&it->mutex);
        return NULL;
    }
    /* If preload is in progress, wait for it to complete */
    while (it->preload_in_progress) {
        pthread_cond_wait(&it->cond, &it->mutex);
    }
    /* mark usage */
    it->ref_count++;
    /* If we have a prepared batch from a previous preload, return it and
       start a new preload in background. */
    if (it->next_prepared) {
        PrefetchedPyramidsBatch *res = it->next_prepared;
        it->next_prepared = NULL;
        /* don't decrement ref_count until after we start preload to avoid
            destroy freeing `it` between unlocking and preload start */
        pthread_mutex_unlock(&it->mutex);
        /* start background preload for the subsequent batch */
        prefetcher_iterator_preload((PrefetcherIteratorHandle)it);
        /* now decrement usage and signal if zero */
        pthread_mutex_lock(&it->mutex);
        it->ref_count--;
        if (it->ref_count == 0)
            pthread_cond_signal(&it->cond);
        if (it->ref_count == 0)
            ;
        pthread_mutex_unlock(&it->mutex);
        return res;
    }
    /* No prepared batch available: blockingly pop next batch. Strong-sync: if
       the popped batch is NULL (non-pyramids or transient empty), retry until
       we obtain a pyramids batch. Use a short sleep between retries to avoid
       busy-waiting. */
    pthread_mutex_unlock(&it->mutex);
    PrefetchedPyramidsBatch *b = NULL;
    while (b == NULL) {
        b = prefetcher_pop_pyramids((PrefetcherHandle)it->p);
        if (b)
            break;
        /* If producers have finished and the buffer is empty, abort retrying
           and return NULL to the caller (no more data). Check under the
           prefetcher mutex to observe a consistent state. */
        Prefetcher *p = (Prefetcher *)it->p;
        int finished_and_empty = 0;
        pthread_mutex_lock(&p->mutex);
        if (p->producer_finished && p->count == 0)
            finished_and_empty = 1;
        pthread_mutex_unlock(&p->mutex);
        if (finished_and_empty) {
            break;
        }
        /* short sleep (100ms) before retrying */
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 100 * 1000000L;
        nanosleep(&ts, NULL);
    }
    /* log when retry loop exits (helps diagnose NULL/ retry behavior) */
    /* trigger background preload for next batch while still considered in-use */
    if (b)
        prefetcher_iterator_preload((PrefetcherIteratorHandle)it);
    /* decrement usage and signal if zero */
    pthread_mutex_lock(&it->mutex);
    it->ref_count--;
    if (it->ref_count == 0)
        pthread_cond_signal(&it->cond);
    if (it->ref_count == 0)
        ;
    pthread_mutex_unlock(&it->mutex);
    return b;
}

void prefetcher_iterator_destroy(PrefetcherIteratorHandle it_h) {
    if (!it_h)
        return;
    PrefetcherIterator *it = (PrefetcherIterator *)it_h;
    pthread_mutex_lock(&it->mutex);
    it->closed = 1;
    /* wait for any preload in progress */
    while (it->preload_in_progress || it->ref_count > 0) {
        pthread_cond_wait(&it->cond, &it->mutex);
    }

    /* free any prepared batch */
    if (it->next_prepared) {
        prefetcher_free_pyramids(it->next_prepared);
        it->next_prepared = NULL;
    }
    pthread_mutex_unlock(&it->mutex);
    pthread_mutex_destroy(&it->mutex);
    pthread_cond_destroy(&it->cond);
    mlx_free_pod((void **)&it);
}

static void *iterator_preload_thread(void *arg) {
    PrefetcherIterator *it = (PrefetcherIterator *)arg;
    PrefetchedPyramidsBatch *b = prefetcher_pop_pyramids((PrefetcherHandle)it->p);
    pthread_mutex_lock(&it->mutex);
    it->next_prepared = b;
    it->preload_in_progress = 0;
    pthread_cond_signal(&it->cond);
    pthread_cond_signal(&it->cond);
    pthread_mutex_unlock(&it->mutex);
    return NULL;
}

int prefetcher_iterator_preload(PrefetcherIteratorHandle it_h) {
    if (!it_h)
        return -1;
    PrefetcherIterator *it = (PrefetcherIterator *)it_h;
    pthread_mutex_lock(&it->mutex);
    if (it->preload_in_progress) {
        pthread_mutex_unlock(&it->mutex);
        return 0;
    }
    it->preload_in_progress = 1;
    pthread_mutex_unlock(&it->mutex);

    pthread_t t;
    if (pthread_create(&t, NULL, iterator_preload_thread, it) != 0) {
        pthread_mutex_lock(&it->mutex);
        it->preload_in_progress = 0;
        pthread_mutex_unlock(&it->mutex);
        return -1;
    }
    pthread_detach(t);
    return 0;
}

int prefetcher_set_stream(PrefetcherHandle h, mlx_stream stream) {
    Prefetcher *p = (Prefetcher *)h;
    if (!p)
        return -1;
    /* free old stream if present */
    if (p->stream.ctx)
        mlx_stream_free(p->stream);
    p->stream = stream;
    p->use_device = 0;
    if (p->device.ctx)
        mlx_device_free(p->device);
    p->device = mlx_device_new();
    mlx_device dev = mlx_device_new();
    if (mlx_stream_get_device(&dev, stream) == 0) {
        mlx_device_type t;
        if (mlx_device_get_type(&t, dev) == 0) {
            if (t == MLX_GPU) {
                p->use_device = 1;
                mlx_device_free(p->device);
                p->device = dev;
            } else {
                mlx_device_free(dev);
            }
        }
    }
    return 0;
}
