#ifndef FACIES_DATALOADER_H
#define FACIES_DATALOADER_H

#include <stddef.h>
#include <stdbool.h>
#include "mlx/c/vector.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct facies_dataset_ facies_dataset;
    typedef struct facies_dataloader_ facies_dataloader;

    // Create dataset from precomputed pyramids (vectors of per-sample vector_array)
    int facies_dataset_new(
        facies_dataset **out,
        const mlx_vector_vector_array facies_pyramids,
        const mlx_vector_vector_array wells_pyramids,
        const mlx_vector_vector_array seismic_pyramids);

    // Free dataset
    int facies_dataset_free(facies_dataset *ds);

    // Create dataloader
    int facies_dataloader_new(
        facies_dataloader **out,
        facies_dataset *ds,
        size_t batch_size,
        bool shuffle,
        bool drop_last,
        unsigned int seed);

    // Reset iterator to start
    int facies_dataloader_reset(facies_dataloader *dl);

    // Get next batch: returns 0 on success, 2 when iteration finished, 1 on error.
    int facies_dataloader_next(
        facies_dataloader *dl,
        mlx_vector_array *out_facies,
        mlx_vector_array *out_wells,
        mlx_vector_array *out_seismic,
        const mlx_stream s);

    // Free dataloader
    int facies_dataloader_free(facies_dataloader *dl);

#ifdef __cplusplus
}
#endif

#endif
