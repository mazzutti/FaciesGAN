#ifndef FACIES_DATALOADER_H
#define FACIES_DATALOADER_H

#include "mlx/c/stream.h"
#include "mlx/c/vector.h"
#include "mlx_dataset.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MLXDataloader;

typedef int (*facies_collate_fn)(mlx_vector_array *out_facies,
                                 mlx_vector_array *out_wells,
                                 mlx_vector_array *out_seismic,
                                 const mlx_vector_vector_array facies_samples,
                                 const mlx_vector_vector_array wells_samples,
                                 const mlx_vector_vector_array seismic_samples,
                                 const mlx_stream s, void *user_ctx);

// Sampler callback: produce next index. Return 0=ok, 2=finished, 1=error.
typedef int (*facies_sampler_next_fn)(void *ctx, size_t *out_index);
// Batch sampler callback: produce up to max_count indices into out_indices;
// returns 0=ok, 2=finished, 1=error.
typedef int (*facies_batch_sampler_next_fn)(void *ctx, size_t *out_indices,
                                            int max_count, int *out_count);
// Optional reset callbacks for samplers
typedef int (*facies_sampler_reset_fn)(void *ctx);
// Worker init function: called in worker process with worker id
typedef int (*facies_worker_init_fn)(int worker_id, void *ctx);
// Create dataset from precomputed pyramids (vectors of per-sample vector_array)
/* Use `MLXPyramidsDataset` directly for dataset APIs. */
int facies_dataset_new(MLXPyramidsDataset **out,
                       const mlx_vector_vector_array facies_pyramids,
                       const mlx_vector_vector_array wells_pyramids,
                       const mlx_vector_vector_array seismic_pyramids);

// Free dataset
int facies_dataset_free(MLXPyramidsDataset *ds);

// Create dataloader
int facies_dataloader_new(struct MLXDataloader **out, MLXPyramidsDataset *ds,
                          size_t batch_size, bool shuffle, bool drop_last,
                          unsigned int seed);

// Extended constructor supporting multi-worker prefetching and options.
int facies_dataloader_new_ex(
    struct MLXDataloader **out, MLXPyramidsDataset *ds, size_t batch_size,
    bool shuffle, bool drop_last, unsigned int seed, int num_workers,
    int prefetch_factor, bool persistent_workers, int timeout_ms,
    facies_collate_fn collate, void *collate_ctx, bool pin_memory,
    facies_sampler_next_fn sampler_next, void *sampler_ctx,
    facies_sampler_reset_fn sampler_reset,
    facies_batch_sampler_next_fn batch_sampler_next, void *batch_sampler_ctx,
    facies_sampler_reset_fn batch_sampler_reset,
    facies_worker_init_fn worker_init, void *worker_init_ctx,
    unsigned int worker_init_ctx_len, const char *worker_init_lib,
    const char *worker_init_sym);

// Reset iterator to start
int facies_dataloader_reset(struct MLXDataloader *dl);

// Get next batch: returns 0 on success, 2 when iteration finished, 1 on error.
int facies_dataloader_next(struct MLXDataloader *dl,
                           mlx_vector_array *out_facies,
                           mlx_vector_array *out_wells,
                           mlx_vector_array *out_seismic, const mlx_stream s);

// Free dataloader
int facies_dataloader_free(struct MLXDataloader *dl);

#ifdef __cplusplus
}
#endif

#endif
