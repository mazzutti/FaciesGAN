/* Zero-copy MLX-native collate helper
 * Exposes a simple API to transpose and stack per-sample pyramids
 * into per-scale stacked arrays (facies/wells/seismic).
 */
#ifndef FACIES_COLLATE_H
#define FACIES_COLLATE_H

#include <stddef.h>
#include "mlx/c/vector.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int facies_collate(
        mlx_vector_array *out_facies,
        mlx_vector_array *out_wells,
        mlx_vector_array *out_seismic,
        const mlx_vector_vector_array facies_samples,
        const mlx_vector_vector_array wells_samples,
        const mlx_vector_vector_array seismic_samples,
        const mlx_stream s);

#ifdef __cplusplus
}
#endif

#endif
