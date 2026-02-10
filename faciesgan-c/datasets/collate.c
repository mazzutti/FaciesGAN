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

static int process_scales_vectorized(mlx_vector_array *out_vec,
                                     const mlx_vector_vector_array samples,
                                     const mlx_stream s,
                                     int deep_copy);

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

        /* Optional diagnostic check: ensure `stacked` is available and sane
         * before appending into the output vector. Disabled by default for
         * performance; enable by setting FACIESGAN_COLLATE_VALIDATE=1. */
        if (getenv("FACIESGAN_COLLATE_VALIDATE")) {
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

static int append_stacked_out(mlx_vector_array *out_vec,
                              mlx_array stacked,
                              int deep_copy) {
    if (!out_vec)
        return 0;
    if (!deep_copy) {
        if (mlx_vector_array_append_value(*out_vec, stacked))
            return 1;
        return 0;
    }
    /* Make a deep copy of `stacked` onto host so it does not depend on
     * the lifetime of internal non-owning buffers from mlx::core::stack. */
    float *buf = NULL;
    size_t elems = 0;
    int ndim = 0;
    int *shape = NULL;
    int copy_ok = 0;
    if (mlx_array_to_float_buffer(stacked, &buf, &elems, &ndim, &shape) == 0 &&
            buf && elems > 0 && ndim > 0) {
        mlx_array safe = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
        if (mlx_vector_array_append_value(*out_vec, safe)) {
            mlx_array_free(safe);
            if (elems > (size_t)INT_MAX)
                free(buf);
            else
                mlx_free_float_buf(&buf, NULL);
            mlx_free_int_array(&shape, &ndim);
            return 1;
        }
        mlx_array_free(safe);
        free(buf);
        free(shape);
        copy_ok = 1;
    }
    if (!copy_ok) {
        if (mlx_vector_array_append_value(*out_vec, stacked))
            return 1;
    }
    return 0;
}

static int process_scales_vectorized(mlx_vector_array *out_vec,
                                     const mlx_vector_vector_array samples,
                                     const mlx_stream s,
                                     int deep_copy) {
    if (!out_vec)
        return 0;
    size_t batch = mlx_vector_vector_array_size(samples);
    if (batch == 0)
        return 0;
    mlx_vector_array first_sample = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&first_sample, samples, 0))
        return 1;
    size_t scales = mlx_vector_array_size(first_sample);
    mlx_vector_array_free(first_sample);

    mlx_vector_array batch_vec = mlx_vector_array_new();
    for (size_t bi = 0; bi < batch; ++bi) {
        mlx_vector_array sample_vec = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_vec, samples, bi)) {
            mlx_vector_array_free(batch_vec);
            mlx_vector_array_free(sample_vec);
            return 1;
        }
        int sample_n = (int)mlx_vector_array_size(sample_vec);
        if (sample_n > 0) {
            mlx_array first = mlx_array_new();
            if (mlx_vector_array_get(&first, sample_vec, 0) != 0) {
                mlx_array_free(first);
                mlx_vector_array_free(sample_vec);
                mlx_vector_array_free(batch_vec);
                return 1;
            }
            int first_ndim = (int)mlx_array_ndim(first);
            const int *first_sh = mlx_array_shape(first);
            for (int si = 1; si < sample_n; ++si) {
                mlx_array cur = mlx_array_new();
                if (mlx_vector_array_get(&cur, sample_vec, si) != 0) {
                    mlx_array_free(cur);
                    mlx_array_free(first);
                    mlx_vector_array_free(sample_vec);
                    mlx_vector_array_free(batch_vec);
                    return 1;
                }
                int cur_ndim = (int)mlx_array_ndim(cur);
                const int *cur_sh = mlx_array_shape(cur);
                int mismatch = (cur_ndim != first_ndim);
                if (!mismatch && first_sh && cur_sh) {
                    for (int di = 0; di < cur_ndim; ++di) {
                        if (cur_sh[di] != first_sh[di]) {
                            mismatch = 1;
                            break;
                        }
                    }
                }
                mlx_array_free(cur);
                if (mismatch) {
                    mlx_array_free(first);
                    mlx_vector_array_free(sample_vec);
                    mlx_vector_array_free(batch_vec);
                    return 2; /* incompatible shapes; let caller fallback */
                }
            }
            mlx_array_free(first);
        }
        mlx_array sample_stacked = mlx_array_new();
        if (mlx_stack(&sample_stacked, sample_vec, s)) {
            mlx_array_free(sample_stacked);
            mlx_vector_array_free(sample_vec);
            mlx_vector_array_free(batch_vec);
            return 2; /* stack failure; let caller fallback */
        }
        if (mlx_vector_array_append_value(batch_vec, sample_stacked)) {
            mlx_array_free(sample_stacked);
            mlx_vector_array_free(sample_vec);
            mlx_vector_array_free(batch_vec);
            return 1;
        }
        mlx_array_free(sample_stacked);
        mlx_vector_array_free(sample_vec);
    }

    mlx_array batch_stacked = mlx_array_new();
    if (mlx_stack(&batch_stacked, batch_vec, s)) {
        mlx_vector_array_free(batch_vec);
        mlx_array_free(batch_stacked);
        return 2; /* stack failure; let caller fallback */
    }
    mlx_vector_array_free(batch_vec);

    int ndim = (int)mlx_array_ndim(batch_stacked);
    const int *bsh = mlx_array_shape(batch_stacked);
    if (ndim < 2 || !bsh || ndim > 8) {
        mlx_array_free(batch_stacked);
        return 1;
    }

    int start[8];
    int stop[8];
    int strides_buf[8];
    for (int i = 0; i < ndim; ++i) {
        start[i] = 0;
        stop[i] = bsh[i];
        strides_buf[i] = 1;
    }

    int out_ndim = ndim - 1;
    int out_shape[8];

    for (size_t si = 0; si < scales; ++si) {
        start[1] = (int)si;
        stop[1] = (int)si + 1;
        mlx_array slice = mlx_array_new();
        if (mlx_slice(&slice, batch_stacked, start, ndim, stop, ndim, strides_buf,
                      ndim, s) != 0) {
            mlx_array_free(slice);
            mlx_array_free(batch_stacked);
            return 1;
        }
        int oi = 0;
        for (int di = 0; di < ndim; ++di) {
            if (di == 1)
                continue;
            out_shape[oi++] = bsh[di];
        }
        mlx_array reshaped = mlx_array_new();
        if (mlx_reshape(&reshaped, slice, out_shape, (size_t)out_ndim, s) != 0) {
            mlx_array_free(slice);
            mlx_array_free(reshaped);
            mlx_array_free(batch_stacked);
            return 1;
        }
        mlx_array_free(slice);
        if (append_stacked_out(out_vec, reshaped, deep_copy)) {
            mlx_array_free(reshaped);
            mlx_array_free(batch_stacked);
            return 1;
        }
        mlx_array_free(reshaped);
    }
    mlx_array_free(batch_stacked);
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

    int deep_copy = getenv("FACIESGAN_COLLATE_NO_COPY") ? 0 : 1;
    int use_facies_fallback = 1;
    if (getenv("FACIESGAN_COLLATE_VECTORIZE")) {
        int vrc = process_scales_vectorized(out_facies, facies_samples, s, deep_copy);
        if (vrc == 0) {
            use_facies_fallback = 0;
        } else {
            if (out_facies) {
                mlx_vector_array_free(*out_facies);
                *out_facies = mlx_vector_array_new();
            }
        }
    }
    if (use_facies_fallback) {
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
                if (deep_copy &&
                        mlx_array_to_float_buffer(stacked, &buf, &elems, &ndim, &shape) ==
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
                if (!deep_copy || !copy_ok) {
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
    }

process_optional_inputs:
    // Process wells and seismic
    if (getenv("FACIESGAN_COLLATE_VECTORIZE")) {
        int wrc = process_scales_vectorized(out_wells, wells_samples, s, 0);
        if (wrc != 0) {
            if (out_wells) {
                mlx_vector_array_free(*out_wells);
                *out_wells = mlx_vector_array_new();
            }
            if (process_optional(out_wells, wells_samples, s)) {
                if (out_facies) mlx_vector_array_free(*out_facies);
                if (out_wells) mlx_vector_array_free(*out_wells);
                if (out_seismic) mlx_vector_array_free(*out_seismic);
                return 1;
            }
        }
    } else if (process_optional(out_wells, wells_samples, s)) {
        if (out_facies) mlx_vector_array_free(*out_facies);
        if (out_wells) mlx_vector_array_free(*out_wells);
        if (out_seismic) mlx_vector_array_free(*out_seismic);
        return 1;
    }
    if (getenv("FACIESGAN_COLLATE_VECTORIZE")) {
        int src = process_scales_vectorized(out_seismic, seismic_samples, s, 0);
        if (src != 0) {
            if (out_seismic) {
                mlx_vector_array_free(*out_seismic);
                *out_seismic = mlx_vector_array_new();
            }
            if (process_optional(out_seismic, seismic_samples, s)) {
                if (out_facies) mlx_vector_array_free(*out_facies);
                if (out_wells) mlx_vector_array_free(*out_wells);
                if (out_seismic) mlx_vector_array_free(*out_seismic);
                return 1;
            }
        }
    } else if (process_optional(out_seismic, seismic_samples, s)) {
        if (out_facies) mlx_vector_array_free(*out_facies);
        if (out_wells) mlx_vector_array_free(*out_wells);
        if (out_seismic) mlx_vector_array_free(*out_seismic);
        return 1;
    }

    return 0;
}
