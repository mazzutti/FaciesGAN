#include "prefetcher.h"

#include <errno.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataloader.h"
#include "faciesgan-c/utils.h"
#include <mlx/c/array.h>
#include <mlx/c/device.h>
#include <mlx/c/ops.h>
#include <mlx/c/stream.h>

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
} Prefetcher;

typedef struct PrefetcherIterator {
  Prefetcher *p;
  int closed;
  PrefetchedPyramidsBatch *next_prepared;
  pthread_t thread;
  int preload_in_progress;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} PrefetcherIterator;

PrefetcherHandle prefetcher_create(int max_queue, int device_index,
                                   const int *scales, int n_scales) {
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
  p->scales = NULL;
  p->n_scales = 0;
  if (n_scales > 0 && scales) {
    p->scales = (int *)malloc(sizeof(int) * n_scales);
    if (p->scales) {
      for (int i = 0; i < n_scales; ++i)
        p->scales[i] = scales[i];
      p->n_scales = n_scales;
    }
  }
  if (device_index >= 0) {
    p->device = mlx_device_new_type(MLX_GPU, device_index);
    p->stream = mlx_stream_new_device(p->device);
    p->use_device = 1;
  } else {
    p->stream = mlx_default_cpu_stream_new();
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
          if (stream.ctx)
            mlx_synchronize(stream);
          mlx_array_free(a_copy);
        } else {
          mlx_array_free(to_append);
          mlx_vector_array_append_value(ib->facies_vec, a_copy);
        }
      } else {
        /* CPU stream: `a_copy` is already CPU-resident; append it directly */
        mlx_array_free(to_append);
        mlx_vector_array_append_value(ib->facies_vec, a_copy);
      }
    }
  }
  for (int i = 0; i < n_wells; ++i) {
    mlx_array a = wells[i];
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
          mlx_vector_array_append_value(ib->wells_vec, to_append);
          if (stream.ctx)
            mlx_synchronize(stream);
          mlx_array_free(a_copy);
        } else {
          mlx_array_free(to_append);
          mlx_vector_array_append_value(ib->wells_vec, a_copy);
        }
      } else {
        mlx_array_free(to_append);
        mlx_vector_array_append_value(ib->wells_vec, a_copy);
      }
    }
  }

  /* If wells were provided but no explicit masks, compute masks like the
   * Python prefetcher does using MLX ops on the provided stream (vectorized):
   * masks = greater(sum(abs(wells), axis=channels, keepdims=true), 0)
   */
  if (n_wells > 0 && n_masks == 0) {
    int nw = mlx_vector_array_size(ib->wells_vec);
    for (int wi = 0; wi < nw; ++wi) {
      /* obtain the vector element (tmp_well). Attempt to make a deep copy
         onto the target stream; if copy succeeds we will free the copy
         after use. If copy fails, `well` will alias the vector element and
         must NOT be freed here (to avoid double-free). */
      mlx_array tmp_well = mlx_array_new();
      mlx_vector_array_get(&tmp_well, ib->wells_vec, wi);
      mlx_array well = mlx_array_new();
      int well_copied = 0;
      if (stream.ctx) {
        if (mlx_copy(&well, tmp_well, stream) == 0) {
          well_copied = 1; /* we own `well` and should free it later */
        } else {
          /* fallback: use the vector element directly; do not free it */
          mlx_array_free(well);
          well = tmp_well;
          well_copied = 0;
        }
      } else {
        /* CPU stream: operate directly on the vector element */
        mlx_array_free(well);
        well = tmp_well;
        well_copied = 0;
      }

      /* abs(well) */
      mlx_array abs_arr = mlx_array_new();
      if (mlx_abs(&abs_arr, well, stream) != 0) {
        mlx_array_free(abs_arr);
        mlx_array_free(well);
        continue;
      }

      /* sum over channel axis (axis=3 to match Python shape [B,H,W,C])
         but defensively handle arrays with fewer dims. */
      int abs_ndim = (int)mlx_array_ndim(abs_arr);
      if (abs_ndim <= 0) {
        mlx_array_free(abs_arr);
        mlx_array_free(well);
        continue;
      }
      int axis = 3;
      if (axis >= abs_ndim)
        axis = abs_ndim - 1;
      mlx_array sum_arr = mlx_array_new();
      if (mlx_sum_axis(&sum_arr, abs_arr, axis, true, stream) != 0) {
        mlx_array_free(abs_arr);
        mlx_array_free(sum_arr);
        mlx_array_free(well);
        continue;
      }
      mlx_array_free(abs_arr);

      /* zeros like sum_arr */
      mlx_array zero = mlx_array_new();
      if (mlx_zeros_like(&zero, sum_arr, stream) != 0) {
        mlx_array_free(sum_arr);
        mlx_array_free(zero);
        mlx_array_free(well);
        continue;
      }

      /* mask = greater(sum_arr, zero) */
      mlx_array mask_arr = mlx_array_new();
      if (mlx_greater(&mask_arr, sum_arr, zero, stream) == 0) {
        mlx_array to_append = mlx_array_new();
        if (stream.ctx) {
          if (mlx_copy(&to_append, mask_arr, stream) == 0) {
            mlx_vector_array_append_value(ib->masks_vec, to_append);
            if (stream.ctx)
              mlx_synchronize(stream);
            mlx_array_free(mask_arr);
          } else {
            mlx_array_free(to_append);
            mlx_vector_array_append_value(ib->masks_vec, mask_arr);
          }
        } else {
          mlx_array_free(to_append);
          mlx_vector_array_append_value(ib->masks_vec, mask_arr);
        }
      }
      mlx_array_free(zero);
      mlx_array_free(sum_arr);
      if (well_copied) {
        /* only free the deep copy we created */
        mlx_array_free(well);
      }
    }
  }
  for (int i = 0; i < n_masks; ++i) {
    mlx_array a = masks[i];
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
          if (stream.ctx)
            mlx_synchronize(stream);
          mlx_array_free(a_copy);
        } else {
          mlx_array_free(to_append);
          mlx_vector_array_append_value(ib->masks_vec, a_copy);
        }
      } else {
        mlx_array_free(to_append);
        mlx_vector_array_append_value(ib->masks_vec, a_copy);
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
          if (stream.ctx)
            mlx_synchronize(stream);
          mlx_array_free(a_copy);
        } else {
          mlx_array_free(to_append);
          mlx_vector_array_append_value(ib->seismic_vec, a_copy);
        }
      } else {
        mlx_array_free(to_append);
        mlx_vector_array_append_value(ib->seismic_vec, a_copy);
      }
    }
  }

  ib->valid = 1;
  return 0;
}

PrefetcherHandle prefetcher_create_with_stream(int max_queue, mlx_stream stream,
                                               const int *scales,
                                               int n_scales) {
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
  p->scales = NULL;
  p->n_scales = 0;
  if (n_scales > 0 && scales) {
    p->scales = (int *)malloc(sizeof(int) * n_scales);
    if (p->scales) {
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

  while (1) {
    mlx_vector_array facs = mlx_vector_array_new();
    mlx_vector_array wells_out = mlx_vector_array_new();
    mlx_vector_array seis_out = mlx_vector_array_new();
    int rc = facies_dataloader_next(dl, &facs, &wells_out, &seis_out, s);
    if (rc == 2) {
      mlx_vector_array_free(facs);
      mlx_vector_array_free(wells_out);
      mlx_vector_array_free(seis_out);
      break;
    } else if (rc != 0) {
      mlx_vector_array_free(facs);
      mlx_vector_array_free(wells_out);
      mlx_vector_array_free(seis_out);
      break;
    }

    int nsc = (int)mlx_vector_array_size(facs);
    mlx_array *fac_arr = NULL;
    mlx_array *well_arr = NULL;
    mlx_array *sei_arr = NULL;
    if (nsc > 0) {
      fac_arr = (mlx_array *)malloc(sizeof(mlx_array) * nsc);
      for (int i = 0; i < nsc; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, facs, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        fac_arr[i] = tmp;
      }
    }
    int nw = (int)mlx_vector_array_size(wells_out);
    if (nw > 0) {
      well_arr = (mlx_array *)malloc(sizeof(mlx_array) * nw);
      for (int i = 0; i < nw; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, wells_out, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        well_arr[i] = tmp;
      }
    }
    int ns = (int)mlx_vector_array_size(seis_out);
    if (ns > 0) {
      sei_arr = (mlx_array *)malloc(sizeof(mlx_array) * ns);
      for (int i = 0; i < ns; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, seis_out, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        sei_arr[i] = tmp;
      }
    }

    /* push into prefetcher (copies into internal buffers/stream) */
    prefetcher_push_mlx(ph, fac_arr, nsc, well_arr, nw, NULL, 0, sei_arr, ns);

    /* Free only container memory; elements may have been moved into
       prefetcher internal vectors. */
    if (fac_arr)
      free(fac_arr);
    if (well_arr)
      free(well_arr);
    if (sei_arr)
      free(sei_arr);

    mlx_vector_array_free(facs);
    mlx_vector_array_free(wells_out);
    mlx_vector_array_free(seis_out);
  }

  prefetcher_mark_finished(ph);
  if (a->s.ctx)
    mlx_stream_free(a->s);
  free(a);
  return NULL;
}

int prefetcher_start_from_dataloader(PrefetcherHandle ph,
                                     struct MLXDataloader *dl,
                                     mlx_stream stream) {
  if (!ph || !dl)
    return -1;
  PrefetcherDLProducerArgs *args =
      (PrefetcherDLProducerArgs *)malloc(sizeof(PrefetcherDLProducerArgs));
  if (!args)
    return -1;
  args->dl = dl;
  args->ph = ph;
  args->s = stream;
  pthread_t t;
  if (pthread_create(&t, NULL, prefetcher_dataloader_producer, args) != 0) {
    free(args);
    return -1;
  }
  pthread_detach(t);
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
  if (create_internal_from_mlx(&ib, facies, n_facies, wells, n_wells, masks,
                               n_masks, seismic, n_seismic, p->stream) != 0) {
    return -1;
  }

  /* debug prints removed */

  pthread_mutex_lock(&p->mutex);
  while (p->count == p->capacity && p->alive) {
    pthread_cond_wait(&p->not_full, &p->mutex);
  }
  if (!p->alive) {
    pthread_mutex_unlock(&p->mutex);
    /* During shutdown: avoid freeing internal vector elements here.
       Some elements may alias buffers owned elsewhere and freeing them
       here can cause double-free. Let process exit reclaim memory or
       handle thorough cleanup in a controlled shutdown path. */
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

  PrefetchedBatch *b = (PrefetchedBatch *)malloc(sizeof(PrefetchedBatch));
  if (!b)
    return NULL;
  memset(b, 0, sizeof(PrefetchedBatch));

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

  if (!ib.is_pyramids) {
    /* Not a pyramids-style batch; cannot return as pyramids */
    return NULL;
  }

  PrefetchedPyramidsBatch *out =
      (PrefetchedPyramidsBatch *)malloc(sizeof(PrefetchedPyramidsBatch));
  if (!out)
    return NULL;
  memset(out, 0, sizeof(*out));

  int n_scales = p->n_scales;
  out->n_scales = n_scales;
  if (n_scales > 0 && p->scales) {
    out->scales = (int *)malloc(sizeof(int) * n_scales);
    for (int i = 0; i < n_scales; ++i)
      out->scales[i] = p->scales[i];
  }

  /* allocate arrays aligned to n_scales and fill entries from internal vectors
     where available; otherwise leave empty mlx_array objects */
  int nf = mlx_vector_array_size(ib.facies_vec);
  out->facies = (mlx_array *)malloc(sizeof(mlx_array) * n_scales);
  for (int i = 0; i < n_scales; ++i) {
    if (i < nf) {
      /* get element, then deep-copy it into an independent mlx_array to
         avoid shared-buffer double-free when freeing the internal vector */
      mlx_array tmp = mlx_array_new();
      mlx_vector_array_get(&tmp, ib.facies_vec, i);
      mlx_array dst = mlx_array_new();
      if (mlx_copy(&dst, tmp, p->stream) == 0) {
        mlx_array_free(tmp);
        out->facies[i] = dst;
      } else {
        /* fallback: use the obtained tmp if copy failed */
        out->facies[i] = tmp;
      }
    } else {
      out->facies[i] = mlx_array_new();
    }
  }

  int nw = mlx_vector_array_size(ib.wells_vec);
  if (nw > 0) {
    out->wells = (mlx_array *)malloc(sizeof(mlx_array) * n_scales);
    for (int i = 0; i < n_scales; ++i) {
      if (i < nw) {
        mlx_array tmp = mlx_array_new();
        mlx_vector_array_get(&tmp, ib.wells_vec, i);
        mlx_array dst = mlx_array_new();
        if (mlx_copy(&dst, tmp, p->stream) == 0) {
          mlx_array_free(tmp);
          out->wells[i] = dst;
        } else {
          out->wells[i] = tmp;
        }
      } else {
        out->wells[i] = mlx_array_new();
      }
    }
  } else {
    out->wells = NULL; /* match Python: empty dict semantics */
  }

  int nm = mlx_vector_array_size(ib.masks_vec);
  if (nm > 0) {
    out->masks = (mlx_array *)malloc(sizeof(mlx_array) * n_scales);
    for (int i = 0; i < n_scales; ++i) {
      if (i < nm) {
        mlx_array tmp = mlx_array_new();
        mlx_vector_array_get(&tmp, ib.masks_vec, i);
        mlx_array dst = mlx_array_new();
        if (mlx_copy(&dst, tmp, p->stream) == 0) {
          mlx_array_free(tmp);
          out->masks[i] = dst;
        } else {
          out->masks[i] = tmp;
        }
      } else {
        out->masks[i] = mlx_array_new();
      }
    }
  } else {
    out->masks = NULL; /* match Python: empty dict semantics */
  }

  int ns = mlx_vector_array_size(ib.seismic_vec);
  if (ns > 0) {
    out->seismic = (mlx_array *)malloc(sizeof(mlx_array) * n_scales);
    for (int i = 0; i < n_scales; ++i) {
      if (i < ns) {
        mlx_array tmp = mlx_array_new();
        mlx_vector_array_get(&tmp, ib.seismic_vec, i);
        mlx_array dst = mlx_array_new();
        if (mlx_copy(&dst, tmp, p->stream) == 0) {
          mlx_array_free(tmp);
          out->seismic[i] = dst;
        } else {
          out->seismic[i] = tmp;
        }
      } else {
        out->seismic[i] = mlx_array_new();
      }
    }
  } else {
    out->seismic = NULL; /* match Python: empty dict semantics */
  }

  /* free internal vectors */
  mlx_vector_array_free(ib.facies_vec);
  mlx_vector_array_free(ib.wells_vec);
  mlx_vector_array_free(ib.masks_vec);
  mlx_vector_array_free(ib.seismic_vec);

  return out;
}

void prefetcher_free_pyramids(PrefetchedPyramidsBatch *b) {
  if (!b)
    return;
  int n = b->n_scales;
  if (b->facies) {
    for (int i = 0; i < n; ++i)
      mlx_array_free(b->facies[i]);
    free(b->facies);
  }
  if (b->wells) {
    for (int i = 0; i < n; ++i)
      mlx_array_free(b->wells[i]);
    free(b->wells);
  }
  if (b->masks) {
    for (int i = 0; i < n; ++i)
      mlx_array_free(b->masks[i]);
    free(b->masks);
  }
  if (b->seismic) {
    for (int i = 0; i < n; ++i)
      mlx_array_free(b->seismic[i]);
    free(b->seismic);
  }
  if (b->scales)
    free(b->scales);
  free(b);
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

  PrefetchedBatch *b = (PrefetchedBatch *)malloc(sizeof(PrefetchedBatch));
  if (!b)
    return NULL;
  memset(b, 0, sizeof(PrefetchedBatch));

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
  pthread_mutex_lock(&p->mutex);
  p->alive = 0;
  p->producer_finished = 1;
  pthread_cond_broadcast(&p->not_empty);
  pthread_cond_broadcast(&p->not_full);
  for (int i = 0; i < p->capacity; ++i) {
    if (p->buf[i].valid) {
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
  if (p->scales)
    free(p->scales);
  free(p->buf);
  pthread_mutex_unlock(&p->mutex);
  pthread_mutex_destroy(&p->mutex);
  pthread_cond_destroy(&p->not_empty);
  pthread_cond_destroy(&p->not_full);
  free(p);
}

PrefetcherIteratorHandle prefetcher_iterator_create(PrefetcherHandle h) {
  if (!h)
    return NULL;
  Prefetcher *p = (Prefetcher *)h;
  PrefetcherIterator *it =
      (PrefetcherIterator *)malloc(sizeof(PrefetcherIterator));
  if (!it) {
    return NULL;
  }
  it->p = p;
  it->closed = 0;
  it->next_prepared = NULL;
  it->preload_in_progress = 0;
  pthread_mutex_init(&it->mutex, NULL);
  pthread_cond_init(&it->cond, NULL);
  return (PrefetcherIteratorHandle)it;
}

PrefetchedPyramidsBatch *
prefetcher_iterator_next(PrefetcherIteratorHandle it_h) {
  if (!it_h)
    return NULL;
  PrefetcherIterator *it = (PrefetcherIterator *)it_h;
  if (it->closed || !it->p)
    return NULL;

  pthread_mutex_lock(&it->mutex);
  /* If we have a prepared batch from a previous preload, return it and
     start a new preload in background. */
  if (it->next_prepared) {
    PrefetchedPyramidsBatch *res = it->next_prepared;
    it->next_prepared = NULL;
    pthread_mutex_unlock(&it->mutex);
    /* start background preload for the subsequent batch (after unlocking) */
    prefetcher_iterator_preload((PrefetcherIteratorHandle)it);
    return res;
  }
  /* No prepared batch available: blockingly pop next batch */
  pthread_mutex_unlock(&it->mutex);
  PrefetchedPyramidsBatch *b = prefetcher_pop_pyramids((PrefetcherHandle)it->p);
  /* trigger background preload for next batch */
  prefetcher_iterator_preload((PrefetcherIteratorHandle)it);
  return b;
}

void prefetcher_iterator_destroy(PrefetcherIteratorHandle it_h) {
  if (!it_h)
    return;
  PrefetcherIterator *it = (PrefetcherIterator *)it_h;
  pthread_mutex_lock(&it->mutex);
  it->closed = 1;
  /* wait for any preload in progress */
  while (it->preload_in_progress) {
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
  free(it);
}

static void *iterator_preload_thread(void *arg) {
  PrefetcherIterator *it = (PrefetcherIterator *)arg;
  PrefetchedPyramidsBatch *b = prefetcher_pop_pyramids((PrefetcherHandle)it->p);
  pthread_mutex_lock(&it->mutex);
  it->next_prepared = b;
  it->preload_in_progress = 0;
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
    return 0; /* already in progress */
  }
  it->preload_in_progress = 1;
  pthread_t th;
  int rc = pthread_create(&th, NULL, iterator_preload_thread, it);
  if (rc == 0) {
    pthread_detach(th);
    pthread_mutex_unlock(&it->mutex);
    return 0;
  }
  it->preload_in_progress = 0;
  pthread_mutex_unlock(&it->mutex);
  return -1;
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
  p->device = mlx_device_new();
  mlx_device dev = mlx_device_new();
  if (mlx_stream_get_device(&dev, stream) == 0) {
    mlx_device_type t;
    if (mlx_device_get_type(&t, dev) == 0) {
      if (t == MLX_GPU) {
        p->use_device = 1;
        p->device = dev;
      } else {
        mlx_device_free(dev);
      }
    }
  }
  return 0;
}
