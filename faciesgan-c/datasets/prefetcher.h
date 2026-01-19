// Minimal ctypes-friendly prefetcher API
#ifndef FACIESGAN_PREFETCHER_H
#define FACIESGAN_PREFETCHER_H

#include <stdint.h>
#include <mlx/c/stream.h>
#include <mlx/c/device.h>
#include <mlx/c/vector.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct PrefetchedBatch
    {
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

    /* Prepared pyramids batch (per-scale MLX arrays) */
    typedef struct PrefetchedPyramidsBatch
    {
        mlx_array *facies; /* pointer to array of mlx_array, length n_facies */
        int n_facies;

        mlx_array *wells;
        int n_wells;

        mlx_array *masks;
        int n_masks;

        mlx_array *seismic;
        int n_seismic;
    } PrefetchedPyramidsBatch;

    typedef void *PrefetcherHandle;

    // Create a prefetcher with a ring buffer of capacity `max_queue`.
    // Create a prefetcher with a ring buffer of capacity `max_queue`.
    // If `device_index` >= 0, prefetcher will copy arrays onto that device's stream.
    // Use -1 for CPU (default).
    PrefetcherHandle prefetcher_create(int max_queue, int device_index);

    // Create a prefetcher using an explicit MLX stream. The provided `stream`
    // is copied into the prefetcher and used for enqueueing copies.
    PrefetcherHandle prefetcher_create_with_stream(int max_queue, mlx_stream stream);

    /* Push already-constructed MLX arrays representing per-scale pyramids.
     * Each pointer array must point to `n_*` mlx_array values. Arrays are copied
     * into the prefetcher and optionally copied onto the prefetcher's stream.
     */
    int prefetcher_push_mlx(PrefetcherHandle h,
                            const mlx_array *facies, int n_facies,
                            const mlx_array *wells, int n_wells,
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
    int prefetcher_push(PrefetcherHandle h,
                        const float *facies, int facies_ndim, const int *facies_shape, int facies_len,
                        const float *seismic, int seismic_ndim, const int *seismic_shape, int seismic_len);

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

    // Mark that the producer is finished (no more pushes). This causes `prefetcher_pop`
    // to return NULL once the internal queue is empty. Does not free the prefetcher.
    void prefetcher_mark_finished(PrefetcherHandle h);

#ifdef __cplusplus
}
#endif

#endif // FACIESGAN_PREFETCHER_H
