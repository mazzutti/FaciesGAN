#include "collate.h"
#include "faciesgan-c/utils.h"
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"
#include "trainning/array_helpers.h"
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

static int process_optional(mlx_vector_array *out_vec,
                            const mlx_vector_vector_array src,
                            const mlx_stream s) {
    size_t nsrc = mlx_vector_vector_array_size(src);
    if (nsrc == 0)
        return 0;
    // check first sample has scales
    mlx_vector_array s0 = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&s0, src, 0))
        return 1;
    size_t sscales = mlx_vector_array_size(s0);
    mlx_vector_array_free(s0);
    for (size_t si = 0; si < sscales; ++si) {
        mlx_vector_array scale_vec = mlx_vector_array_new();
        for (size_t bi = 0; bi < nsrc; ++bi) {
            mlx_vector_array sample_v = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_v, src, bi)) {
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_v);
                return 1;
            }
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, sample_v, si)) {
                mlx_array_free(elem);
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_v);
                return 1;
            }
            if (mlx_vector_array_append_value(scale_vec, elem)) {
                mlx_array_free(elem);
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_v);
                return 1;
            }
            mlx_array_free(elem);
            mlx_vector_array_free(sample_v);
        }
        mlx_array stacked = mlx_array_new();
        size_t vec_n = mlx_vector_array_size(scale_vec);
        if (vec_n == 0) {
            mlx_vector_array_free(scale_vec);
            mlx_array_free(stacked);
            return 1;
        }
        if (mlx_stack(&stacked, scale_vec, s)) {
            mlx_vector_array_free(scale_vec);
            mlx_array_free(stacked);
            return 1;
        }

        /* Diagnostic check: ensure `stacked` is available and sane before
         * appending into the output vector. This helps catch invalid arrays
         * produced by the stacking operation (observed as null internal
         * pointers when accessed later).
         * NOTE: MLX arrays are lazy - we need to evaluate before checking ndim. */
        {
            bool ok = false;
            /* Evaluate the array to ensure it's materialized before checking */
            mlx_array_eval(stacked);
            _mlx_array_is_available(&ok, stacked);
            int ndim = (int)mlx_array_ndim(stacked);
            if (!ok || ndim <= 0) {
                mlx_array_free(stacked);
                /* Substitute a scalar zero so downstream code receives a valid
                 * mlx_array instance instead of crashing on null internals. */
                stacked = mlx_array_new_float(0.0f);
            }
        }
        if (out_vec) {
            if (mlx_vector_array_append_value(*out_vec, stacked)) {
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

int facies_collate(mlx_vector_array *out_facies, mlx_vector_array *out_wells,
                   mlx_vector_array *out_seismic,
                   const mlx_vector_vector_array facies_samples,
                   const mlx_vector_vector_array wells_samples,
                   const mlx_vector_vector_array seismic_samples,
                   const mlx_stream s) {
    // Initialize outputs to empty vectors (free any prior handle first)
    if (out_facies) {
        if ((*out_facies).ctx)
            mlx_vector_array_free(*out_facies);
        *out_facies = mlx_vector_array_new();
    }
    if (out_wells) {
        if ((*out_wells).ctx)
            mlx_vector_array_free(*out_wells);
        *out_wells = mlx_vector_array_new();
    }
    if (out_seismic) {
        if ((*out_seismic).ctx)
            mlx_vector_array_free(*out_seismic);
        *out_seismic = mlx_vector_array_new();
    }

    size_t batch = mlx_vector_vector_array_size(facies_samples);
    if (batch == 0)
        return 0;

    // Determine number of scales from first sample
    mlx_vector_array first_sample = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&first_sample, facies_samples, 0))
        return 1;
    size_t scales = mlx_vector_array_size(first_sample);
    mlx_vector_array_free(first_sample);

    for (size_t si = 0; si < scales; ++si) {
        // Build vector of arrays for this scale across batch
        mlx_vector_array scale_vec = mlx_vector_array_new();
        for (size_t bi = 0; bi < batch; ++bi) {
            mlx_vector_array sample_fac = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_fac, facies_samples, bi)) {
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_fac);
                return 1;
            }
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, sample_fac, si)) {
                mlx_array_free(elem);
                mlx_vector_array_free(scale_vec);
                mlx_vector_array_free(sample_fac);
                return 1;
            }
            if (mlx_vector_array_append_value(scale_vec, elem)) {
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
        size_t vec_n = mlx_vector_array_size(scale_vec);
        if (vec_n == 0) {
            mlx_vector_array_free(scale_vec);
            mlx_array_free(stacked);
            return 1;
        }
        if (mlx_stack(&stacked, scale_vec, s)) {
            mlx_vector_array_free(scale_vec);
            mlx_array_free(stacked);
            return 1;
        }

        // Append stacked result to output facies vector
        if (out_facies) {
            /* Make a deep copy of `stacked` onto host so it does not depend on
             * the lifetime of `scale_vec` or internal non-owning buffers from
             * `mlx::core::stack`. This prevents use-after-free when callers
             * later access the appended arrays. */
            float *buf = NULL;
            size_t elems = 0;
            int ndim = 0;
            int *shape = NULL;
            int copy_ok = 0;
            if (mlx_array_to_float_buffer(stacked, &buf, &elems, &ndim, &shape) ==
                    0 &&
                    buf && elems > 0 && ndim > 0) {
                /* create new MLX array that owns its data (mlx_array_new_data
                 * copies the provided buffer) */
                mlx_array safe = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
                if (mlx_vector_array_append_value(*out_facies, safe)) {
                    mlx_array_free(safe);
                    if (elems > (size_t)INT_MAX)
                        free(buf);
                    else
                        mlx_free_float_buf(&buf, NULL);
                    mlx_free_int_array(&shape, &ndim);
                    mlx_array_free(stacked);
                    mlx_vector_array_free(scale_vec);
                    return 1;
                }
                /* appended; free temporary resources */
                mlx_array_free(safe);
                free(buf);
                free(shape);
                copy_ok = 1;
            }
            if (!copy_ok) {
                if (mlx_vector_array_append_value(*out_facies, stacked)) {
                    mlx_array_free(stacked);
                    mlx_vector_array_free(scale_vec);
                    return 1;
                }
            }
        }
        // Free local temporaries
        mlx_array_free(stacked);
        mlx_vector_array_free(scale_vec);
    }

    // Process wells and seismic
    if (process_optional(out_wells, wells_samples, s)) {
        if (out_facies) mlx_vector_array_free(*out_facies);
        if (out_wells) mlx_vector_array_free(*out_wells);
        if (out_seismic) mlx_vector_array_free(*out_seismic);
        return 1;
    }
    if (process_optional(out_seismic, seismic_samples, s)) {
        if (out_facies) mlx_vector_array_free(*out_facies);
        if (out_wells) mlx_vector_array_free(*out_wells);
        if (out_seismic) mlx_vector_array_free(*out_seismic);
        return 1;
    }

    return 0;
}
