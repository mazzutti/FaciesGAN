#include <stdlib.h>
#include "mlx/c/vector.h"
#include "mlx/c/ops.h"
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include "collate.h"

static int process_optional(mlx_vector_array *out_vec, const mlx_vector_vector_array src, const mlx_stream s)
{
    size_t nsrc = mlx_vector_vector_array_size(src);
    if (nsrc == 0)
        return 0;
    // check first sample has scales
    mlx_vector_array s0 = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&s0, src, 0))
        return 1;
    size_t sscales = mlx_vector_array_size(s0);
    mlx_vector_array_free(s0);
    for (size_t si = 0; si < sscales; ++si)
    {
        mlx_vector_array scale_vec = mlx_vector_array_new();
        for (size_t bi = 0; bi < nsrc; ++bi)
        {
            mlx_vector_array sample_v = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_v, src, bi))
            {
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_v);
                return 1;
            }
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, sample_v, si))
            {
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_v);
                return 1;
            }
            if (mlx_vector_array_append_value(scale_vec, elem))
            {
                mlx_array_free(elem);
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_v);
                return 1;
            }
            mlx_array_free(elem);
            mlx_vector_array_free(sample_v);
        }
        mlx_array stacked = mlx_array_new();
        if (mlx_stack(&stacked, scale_vec, s))
        {
            mlx_vector_array_free(scale_vec);
            mlx_array_free(stacked);
            return 1;
        }
        if (out_vec)
        {
            if (mlx_vector_array_append_value(*out_vec, stacked))
            {
                mlx_array_free(stacked);
                mlx_vector_array_free(scale_vec);
                return 1;
            }
        }
        mlx_array_free(stacked);
        mlx_vector_array_free(scale_vec);
    }
    return 0;
}

int facies_collate(
    mlx_vector_array *out_facies,
    mlx_vector_array *out_wells,
    mlx_vector_array *out_seismic,
    const mlx_vector_vector_array facies_samples,
    const mlx_vector_vector_array wells_samples,
    const mlx_vector_vector_array seismic_samples,
    const mlx_stream s)
{
    // Initialize outputs to empty vectors
    if (out_facies)
        *out_facies = mlx_vector_array_new();
    if (out_wells)
        *out_wells = mlx_vector_array_new();
    if (out_seismic)
        *out_seismic = mlx_vector_array_new();

    size_t batch = mlx_vector_vector_array_size(facies_samples);
    if (batch == 0)
        return 0;

    // Determine number of scales from first sample
    mlx_vector_array first_sample = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&first_sample, facies_samples, 0))
        return 1;
    size_t scales = mlx_vector_array_size(first_sample);
    mlx_vector_array_free(first_sample);

    for (size_t si = 0; si < scales; ++si)
    {
        // Build vector of arrays for this scale across batch
        mlx_vector_array scale_vec = mlx_vector_array_new();
        for (size_t bi = 0; bi < batch; ++bi)
        {
            mlx_vector_array sample_fac = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_fac, facies_samples, bi))
            {
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_fac);
                return 1;
            }
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, sample_fac, si))
            {
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_fac);
                return 1;
            }
            if (mlx_vector_array_append_value(scale_vec, elem))
            {
                mlx_array_free(elem);
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_fac);
                return 1;
            }
            // appended (copied) into scale_vec; free temporary
            mlx_array_free(elem);
            mlx_vector_array_free(sample_fac);
        }

        // Stack along new batch axis
        mlx_array stacked = mlx_array_new();
        if (mlx_stack(&stacked, scale_vec, s))
        {
            mlx_vector_array_free(scale_vec);
            mlx_array_free(stacked);
            return 1;
        }

        // Append stacked result to output facies vector
        if (out_facies)
        {
            if (mlx_vector_array_append_value(*out_facies, stacked))
            {
                mlx_array_free(stacked);
                mlx_vector_array_free(scale_vec);
                return 1;
            }
        }
        // Free local temporaries
        mlx_array_free(stacked);
        mlx_vector_array_free(scale_vec);
    }

    // Process wells and seismic
    if (process_optional(out_wells, wells_samples, s))
        return 1;
    if (process_optional(out_seismic, seismic_samples, s))
        return 1;

    return 0;
}
