#include "train_utils.h"
#include "array_helpers.h"
#include "models/custom_layer.h"
#include "models/facies_gan.h"
#include <execinfo.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int mlx_compute_rec_input(int scale, const int *indexes, int n_indexes,
                          mlx_array **facies_pyramid, mlx_array **out) {
    if (!facies_pyramid || !out)
        return -1;
    if (scale < 0)
        return -1;

    /* create a default CPU stream for MLX ops and ensure it's freed */
    mlx_stream s = mlx_gpu_stream();

    /* scale 0 -> zeros_like(real) */
    if (scale == 0) {
        if (!facies_pyramid[0]) {
            return -1;
        }
        mlx_array z = mlx_array_new();
        if (mlx_zeros_like(&z, *facies_pyramid[0], s) != 0) {
            return -1;
        }
        mlx_array *res = NULL;
        if (mlx_alloc_pod((void **)&res, sizeof(mlx_array), 1) != 0) {
            mlx_array_free(z);
            return -1;
        }
        *res = z;
        *out = res;
        return 0;
    }

    /* gather previous-scale entries by indexes using mlx_take_axis
     * which preserves shape better than mlx_gather_single */
    if (!facies_pyramid[scale - 1] || !facies_pyramid[scale])
        return -1;
    if (!indexes || n_indexes <= 0)
        return -1;

    int idx_shape[1] = {n_indexes};
    mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
    mlx_array sel = mlx_array_new();
    /* Use mlx_take_axis to select indices along batch axis (axis=0) */
    if (mlx_take_axis(&sel, *facies_pyramid[scale - 1], idx, 0, s) != 0) {
        mlx_array_free(idx);
        return -1;
    }
    mlx_array_free(idx);

    /* upsample to match target spatial dims */
    const int *shape = mlx_array_shape(*facies_pyramid[scale]);
    if (!shape) {
        mlx_array_free(sel);
        return -1;
    }
    int target_h = shape[1];
    int target_w = shape[2];

    MLXUpsample *u = mlx_upsample_create(target_h, target_w, "linear", 1);
    if (!u) {
        mlx_array_free(sel);
        return -1;
    }
    mlx_array up = mlx_upsample_forward(u, sel);
    mlx_upsample_free(u);

    /* Free the intermediate selection array - mlx_upsample_forward creates
       a new output array, so `sel` is no longer needed. */
    mlx_array_free(sel);

    mlx_array *res = NULL;
    if (mlx_alloc_pod((void **)&res, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(up);
        return -1;
    }
    *res = up;
    *out = res;
    return 0;
}

int mlx_init_rec_noise_and_amp(MLXFaciesGAN *m, int scale, const int *indexes,
                               int n_indexes, const mlx_array *real,
                               mlx_array **wells_pyramid,
                               mlx_array **seismic_pyramid,
                               float scale0_noise_amp, float noise_amp_factor,
                               float min_noise_amp) {
    if (!m || !real)
        return -1;
    if (scale < 0)
        return -1;

    /* Acquire global MLX lock for all MLX operations in this function */
    mlx_global_lock();

    /* Mirror Python: if len(self.model.rec_noise) > scale: return
     * i.e. if we already stored rec noise (and therefore amp) for this scale,
     * nothing to do. */
    if (mlx_faciesgan_get_n_rec_noise(m) > scale) {
        mlx_global_unlock();
        return 0;
    }

    /* === Generate and store reconstruction noise for this scale ===
     * Mirrors Python:
     *   scale 0: z_rec = generate_noise((*real.shape[1:3], noise_channels))
     *            → ALL channels are pure random noise (no actual conditioning)
     *   scale>0: z_rec = generate_noise(reduced_shape) + concat(wells, seismic)
     *            → noise channels + actual conditioning data
     * Python pads with zero_padding in both cases. */
    {
        mlx_array **rec_gen = NULL;
        int n_rec_gen = 0;
        /* FIX 33: At scale 0, Python generates z_rec as ALL random noise with
         * noise_channels total channels — it does NOT concatenate actual
         * conditioning data. Passing NULL wells/seismic to get_pyramid_noise
         * produces gen_input_channels of pure noise, matching Python. */
        mlx_array **rec_wells = (scale == 0) ? NULL : wells_pyramid;
        mlx_array **rec_seis  = (scale == 0) ? NULL : seismic_pyramid;
        if (mlx_faciesgan_get_pyramid_noise(m, scale, indexes, n_indexes,
                                            &rec_gen, &n_rec_gen,
                                            rec_wells, rec_seis,
                                            0) == 0 && n_rec_gen > scale) {
            /* Store noise for THIS scale (rec_gen[scale]) */
            mlx_faciesgan_store_rec_noise(m, scale, rec_gen[scale]);
            /* Free the generated noise arrays */
            for (int ri = 0; ri < n_rec_gen; ++ri) {
                if (rec_gen[ri]) {
                    mlx_array_free(*rec_gen[ri]);
                    mlx_free_pod((void **)&rec_gen[ri]);
                }
            }
            mlx_free_pod((void **)&rec_gen);
        }
    }

    /* get pyramid noises for amp computation.
     * Python scale 0: get_pyramid_noise(scale, indexes) - NO wells/seismic
     * Python scale>0: get_pyramid_noise(scale, indexes, wells, seismic) */
    mlx_array **noises = NULL;
    int n_noises = 0;
    if (scale == 0) {
        /* Scale 0: no conditioning in noise for amp computation */
        if (mlx_faciesgan_get_pyramid_noise(m, scale, indexes, n_indexes, &noises,
                                            &n_noises, NULL, NULL, 0) != 0) {
            mlx_global_unlock();
            return -1;
        }
    } else {
        /* Scale > 0: include conditioning */
        if (mlx_faciesgan_get_pyramid_noise(m, scale, indexes, n_indexes, &noises,
                                            &n_noises, wells_pyramid, seismic_pyramid,
                                            0) != 0) {
            mlx_global_unlock();
            return -1;
        }
    }

    /* obtain amplitude list for generation.
     * Python scale 0: [1.0] * (scale + 1) = [1.0]
     * Python scale>0: self.model.noise_amps + [1.0] */
    float *use_amps = NULL;
    int use_n = 0;
    if (scale == 0) {
        /* Scale 0: all amps are 1.0 */
        use_n = 1;
        if (mlx_alloc_float_buf(&use_amps, use_n) != 0) {
            mlx_free_mlx_array_ptrs(&noises, n_noises);
            mlx_global_unlock();
            return -1;
        }
        use_amps[0] = 1.0f;
    } else {
        /* Scale > 0: existing amps + [1.0] for new scale */
        if (mlx_faciesgan_get_noise_amplitude(m, scale, &use_amps, &use_n) != 0) {
            /* fallback to ones */
            use_n = scale + 1;
            if (mlx_alloc_float_buf(&use_amps, use_n) != 0) {
                mlx_free_mlx_array_ptrs(&noises, n_noises);
                mlx_global_unlock();
                return -1;
            }
            for (int i = 0; i < use_n; ++i)
                use_amps[i] = 1.0f;
        }
    }

    /* generate fake for this scale (numeric forward) */
    /* Convert returned array-of-pointers into a contiguous array of mlx_array
       values required by mlx_faciesgan_generate_fake.
       Transfer ownership from noises[i] → zvals[i] without copying the
       underlying MLX array (which would allocate a new stream + graph node).
       After the transfer, null out noises[i] so the later
       mlx_free_mlx_array_ptrs skips them and avoids a double-free. */
    mlx_array *zvals = NULL;
    if (n_noises > 0) {
        if (mlx_alloc_mlx_array_vals(&zvals, n_noises) != 0) {
            mlx_free_mlx_array_ptrs(&noises, n_noises);
            if (use_amps)
                mlx_free_float_buf(&use_amps, &use_n);
            mlx_global_unlock();
            return -1;
        }
        for (int i = 0; i < n_noises; ++i) {
            if (noises[i] && (*noises[i]).ctx) {
                mlx_array_free(zvals[i]);
                zvals[i] = *noises[i];  /* take ownership of the handle */
                /* Null out the source so mlx_free_mlx_array_ptrs won't
                 * free the same underlying array a second time. We only
                 * free the wrapper struct here (not the MLX array). */
                free(noises[i]);
                noises[i] = NULL;
            }
        }
        /* Ensure none of the zvals are empty */
        for (int i = 0; i < n_noises; ++i) {
            if (mlx_array_ndim(zvals[i]) == 0) {
                mlx_stream _s = mlx_gpu_stream();
                int shape0[4] = {1, 32, 32, 1};
                mlx_array tmp = mlx_array_new();
                if (mlx_zeros(&tmp, shape0, 4, MLX_FLOAT32, _s) == 0) {
                    mlx_array_free(zvals[i]);
                    zvals[i] = tmp;
                } else {
                    mlx_array_free(tmp);
                }
            }
        }
    }

    /* FIX: Python uses in_noise=None (default) and start_scale=0 (default)
     * for BOTH scale 0 and scale > 0. This runs the FULL progressive chain
     * from scale 0 to stop_scale, which is critical for correct RMSE at
     * scale > 0.
     *
     * Previously C used: in_noise=zeros(real_shape), start_scale=scale,
     * stop_scale=scale which only ran the current scale with zero input. */
    mlx_array in_noise = mlx_array_new();  /* empty = no in_noise (Python None) */

    mlx_array_t fake = mlx_faciesgan_generate_fake(m, zvals, n_noises, use_amps,
                       use_n, in_noise, 0 /* start_scale=0 */, scale);

    /* free noises (we free underlying arrays and the pointer container) */
    mlx_free_mlx_array_ptrs(&noises, n_noises);
    if (zvals)
        mlx_free_mlx_array_vals(&zvals, n_noises);
    if (use_amps)
        mlx_free_float_buf(&use_amps, &use_n);

    /* compute rmse = sqrt(mean((fake - real)^2))
     * Python: rmse = mx.sqrt(nn.losses.mse_loss(fake, real))
     * No padding of real - shapes should match since we now use the full
     * progressive chain (start_scale=0). */
    mlx_stream s = mlx_gpu_stream();

    /* Ensure fake and real shapes match (H/W) before subtract to avoid
     * broadcast errors when parallel scales are active. */
    const int *fake_shape = mlx_array_shape(fake);
    const int *real_shape = mlx_array_shape(*real);
    if (fake_shape && real_shape) {
        int fake_ndim = mlx_array_ndim(fake);
        int real_ndim = mlx_array_ndim(*real);
        if (fake_ndim == 4 && real_ndim == 4) {
            if (fake_shape[0] != real_shape[0] || fake_shape[3] != real_shape[3]) {
                mlx_array_free(fake);
                mlx_array_free(in_noise);
                mlx_global_unlock();
                return -1;
            }
            if (fake_shape[1] != real_shape[1] || fake_shape[2] != real_shape[2]) {
                MLXUpsample *u = mlx_upsample_create(real_shape[1], real_shape[2], "linear", 1);
                if (!u) {
                    mlx_array_free(fake);
                    mlx_array_free(in_noise);
                    mlx_global_unlock();
                    return -1;
                }
                mlx_array resized = mlx_upsample_forward(u, fake);
                mlx_upsample_free(u);
                mlx_array_free(fake);
                fake = resized;
            }
        }
    }

    mlx_array diff = mlx_array_new();
    if (mlx_subtract(&diff, fake, *real, s) != 0) {
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        mlx_global_unlock();
        return -1;
    }
    mlx_array sq = mlx_array_new();
    int sq_rc = mlx_square(&sq, diff, s);
    if (sq_rc != 0) {
        mlx_array_free(diff);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        mlx_global_unlock();
        return -1;
    }
    mlx_array_free(diff);

    mlx_array mean = mlx_array_new();
    if (mlx_mean(&mean, sq, false, s) != 0) {
        mlx_array_free(sq);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        mlx_global_unlock();
        return -1;
    }
    mlx_array_free(sq);

    mlx_array root = mlx_array_new();
    if (mlx_sqrt(&root, mean, s) != 0) {
        mlx_array_free(mean);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        mlx_global_unlock();
        return -1;
    }
    mlx_array_free(mean);

    /* FIX: Force evaluation before reading the scalar value.
     * MLX uses lazy evaluation - without eval, the computation graph
     * may not be materialized and mlx_array_data_float32 returns NULL. */
    mlx_array_eval(root);

    double rmse = 0.0;
    {
        const float *pdata = mlx_array_data_float32(root);
        if (pdata)
            rmse = (double)pdata[0];
    }

    /* Compute new amp matching Python exactly:
     * Scale 0: amp = scale0_noise_amp * rmse (NO min clamping)
     * Scale>0: amp = max(noise_amp * rmse, min_noise_amp) */
    double amp = 0.0;
    if (scale == 0) {
        amp = (double)scale0_noise_amp * rmse;
    } else {
        amp = (double)noise_amp_factor * rmse;
        if (amp < (double)min_noise_amp)
            amp = (double)min_noise_amp;
    }

    /* Build new amp array (scale+1) by copying existing amps if present */
    float *new_amps = NULL;
    if (mlx_alloc_float_buf(&new_amps, scale + 1) != 0) {
        mlx_array_free(root);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        mlx_global_unlock();
        return -1;
    }
    /* try to read existing amps to preserve previous values */
    float *amps = NULL;
    int n_amps = 0;
    int prev_n_amps = 0;  /* remember count before free zeroes it */
    if (mlx_faciesgan_get_noise_amps(m, &amps, &n_amps) == 0 && amps &&
            n_amps > 0) {
        prev_n_amps = n_amps;
        for (int i = 0; i < scale + 1; ++i)
            new_amps[i] = (i < n_amps) ? amps[i] : 1.0f;
        mlx_free_float_buf(&amps, &n_amps);
    } else {
        for (int i = 0; i < scale + 1; ++i)
            new_amps[i] = 1.0f;
    }

    /* FIX: For scale > 0 when amp already exists, AVERAGE with existing value
     * matching Python: self.model.noise_amps[scale] = (amp + existing) / 2 */
    if (scale > 0 && scale < prev_n_amps && prev_n_amps > 0) {
        new_amps[scale] = (float)((amp + (double)new_amps[scale]) / 2.0);
    } else {
        new_amps[scale] = (float)amp;
    }

    /* set on model */
    int res = mlx_faciesgan_set_noise_amps(m, new_amps, scale + 1);

    mlx_free_float_buf(&new_amps, NULL);
    mlx_array_free(root);
    mlx_array_free(fake);
    mlx_array_free(in_noise);

    mlx_global_unlock();
    return res;
}
