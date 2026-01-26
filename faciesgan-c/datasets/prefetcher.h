// Minimal ctypes-friendly prefetcher API
#ifndef FACIESGAN_PREFETCHER_H
#define FACIESGAN_PREFETCHER_H

#include "dataloader.h"
#include <mlx/c/device.h>
#include <mlx/c/stream.h>
#include <mlx/c/vector.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PrefetchedBatch {
    float *facies;
    int facies_ndim;
    int facies_shape[8];
    int facies_len;

    float *wells;
    int wells_ndim;
    int wells_shape[8];
    int wells_len;

    float *seismic;
    int seismic_ndim;
    int seismic_shape[8];
    int seismic_len;
} PrefetchedBatch;

/* Prepared pyramids batch (per-scale MLX arrays). To match the Python
 * `MLXDataPrefetcher` API this exposes per-scale arrays aligned to the
 * trainer-provided `scales` list. Each array has length `n_scales`; when an
 * entry is absent the corresponding mlx_array will be empty (ndim == 0).
 */
typedef struct PrefetchedPyramidsBatch {
    int n_scales;
    int *scales; /* heap-allocated array of length n_scales */

    mlx_array *facies;  /* array length n_scales */
    mlx_array *wells;   /* array length n_scales */
    mlx_array *masks;   /* array length n_scales */
    mlx_array *seismic; /* array length n_scales */
    /* Convenience pointer arrays referencing the above `mlx_array` values.
     * These are allocated with `mlx_alloc_ptr_array` and point into the
     * corresponding `*_vals` arrays (i.e. elements are addresses of
     * `facies[i]`). They are optional and may be NULL. The caller should
     * treat them as non-owning pointers to the `*_vals` storage; the
     * underlying `mlx_array` values are freed by `prefetcher_free_pyramids`. */
    mlx_array **facies_ptrs;
    mlx_array **wells_ptrs;
    mlx_array **masks_ptrs;
    mlx_array **seismic_ptrs;
} PrefetchedPyramidsBatch;

typedef void *PrefetcherHandle;

// Opaque iterator handle providing Python-like iterator semantics in C.
typedef void *PrefetcherIteratorHandle;

// Create an iterator for the given prefetcher. The iterator does not take
// ownership of the prefetcher; destroying the prefetcher while the iterator
// is active leads to undefined behavior.
PrefetcherIteratorHandle prefetcher_iterator_create(PrefetcherHandle h);

// Retrieve the next pyramids batch from the iterator. Returns a newly
// allocated `PrefetchedPyramidsBatch*` which the caller must free using
// `prefetcher_free_pyramids`. Returns NULL when the prefetcher is finished
// and no more batches are available or on error.
PrefetchedPyramidsBatch *prefetcher_iterator_next(PrefetcherIteratorHandle it);

// Destroy the iterator and free its resources. Does not destroy the
// underlying prefetcher.
void prefetcher_iterator_destroy(PrefetcherIteratorHandle it);

// Start asynchronous preload of the next batch on the iterator. This will
// spawn a background thread that calls `prefetcher_pop_pyramids` and stores
// the prepared batch inside the iterator for the subsequent `next` call.
// Calling this when a preload is already in progress is a no-op.
int prefetcher_iterator_preload(PrefetcherIteratorHandle it);

// Change the stream used by the prefetcher. The provided `stream` is copied
// into the prefetcher and used for subsequent copies and operations.
// Returns 0 on success.
int prefetcher_set_stream(PrefetcherHandle h, mlx_stream stream);

// Create a prefetcher with a ring buffer of capacity `max_queue` and the
// explicit list of `scales` (length `n_scales`) that will be used to
// align per-scale arrays returned by `prefetcher_pop_pyramids`.
// If `device_index` >= 0, prefetcher will copy arrays onto that device's
// stream. Use -1 for CPU (default).
PrefetcherHandle prefetcher_create(int max_queue, int device_index,
                                   const int *scales, int n_scales);

// Create a prefetcher using an explicit MLX stream. The provided `stream`
// is copied into the prefetcher and used for enqueueing copies.
PrefetcherHandle prefetcher_create_with_stream(int max_queue, mlx_stream stream,
        const int *scales, int n_scales);

/* Push already-constructed MLX arrays representing per-scale pyramids.
 * Each pointer array must point to `n_*` mlx_array values. Arrays are copied
 * into the prefetcher and optionally copied onto the prefetcher's stream.
 */
int prefetcher_push_mlx(PrefetcherHandle h, const mlx_array *facies,
                        int n_facies, const mlx_array *wells, int n_wells,
                        const mlx_array *masks, int n_masks,
                        const mlx_array *seismic, int n_seismic);

/* Pop a prepared pyramids batch (per-scale MLX arrays). Caller must free
 * the batch with `prefetcher_free_pyramids`.
 */
PrefetchedPyramidsBatch *prefetcher_pop_pyramids(PrefetcherHandle h);

/* Free a pyramids batch returned by `prefetcher_pop_pyramids`.
 */
void prefetcher_free_pyramids(PrefetchedPyramidsBatch *b);

// Push a batch into the prefetcher (copies buffers). Returns 0 on success.
int prefetcher_push(PrefetcherHandle h, const float *facies, int facies_ndim,
                    const int *facies_shape, int facies_len,
                    const float *seismic, int seismic_ndim,
                    const int *seismic_shape, int seismic_len);

// Pop a prepared batch; caller must call prefetcher_free_batch on the result.
// Returns NULL if the prefetcher is destroyed.
PrefetchedBatch *prefetcher_pop(PrefetcherHandle h);

// Pop with timeout in seconds (double). If `timeout_s` < 0, block forever.
// Returns NULL on timeout or if the prefetcher is destroyed/finished.
PrefetchedBatch *prefetcher_pop_timeout(PrefetcherHandle h, double timeout_s);

// Free a batch returned by prefetcher_pop.
void prefetcher_free_batch(PrefetchedBatch *b);

// Destroy the prefetcher and free all resources.
void prefetcher_destroy(PrefetcherHandle h);

// Mark that the producer is finished (no more pushes). This causes
// `prefetcher_pop` to return NULL once the internal queue is empty. Does not
// free the prefetcher.
void prefetcher_mark_finished(PrefetcherHandle h);

/* Stop background producers and wait for them to exit. This does not free
 * the prefetcher itself; call `prefetcher_destroy` after `prefetcher_stop`
 * to free resources. Returns 0 on success. */
int prefetcher_stop(PrefetcherHandle h);
/* Start a background producer thread that reads batches from a
 * `MLXDataloader` and pushes them into the given `PrefetcherHandle`.
 * The function spawns a detached thread and returns 0 on success. The
 * provided `stream` is used when calling `facies_dataloader_next` and is
 * freed by the background thread when finished. Returns non-zero on
 * failure.
 */
int prefetcher_start_from_dataloader(PrefetcherHandle ph,
                                     struct MLXDataloader *dl,
                                     mlx_stream stream);

#ifdef __cplusplus
}
#endif

#endif // FACIESGAN_PREFETCHER_H
