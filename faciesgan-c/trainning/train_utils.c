#include "train_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include "models/custom_layer.h"
#include "models/facies_gan.h"
#include <math.h>

int mlx_compute_rec_input(int scale, const int *indexes, int n_indexes, mlx_array **facies_pyramid, mlx_array **out)
{
    if (!facies_pyramid || !out)
        return -1;
    if (scale < 0)
        return -1;

    /* create a default CPU stream for MLX ops and ensure it's freed */
    mlx_stream s = mlx_default_cpu_stream_new();

    /* scale 0 -> zeros_like(real) */
    if (scale == 0)
    {
        if (!facies_pyramid[0])
        {
            mlx_stream_free(s);
            return -1;
        }
        mlx_array z = mlx_array_new();
        if (mlx_zeros_like(&z, *facies_pyramid[0], s) != 0)
        {
            mlx_stream_free(s);
            return -1;
        }
        mlx_array *res = (mlx_array *)malloc(sizeof(mlx_array));
        if (!res)
        {
            mlx_array_free(z);
            mlx_stream_free(s);
            return -1;
        }
        *res = z;
        *out = res;
        mlx_stream_free(s);
        return 0;
    }

    /* gather previous-scale entries by indexes */
    if (!facies_pyramid[scale - 1] || !facies_pyramid[scale])
        return -1;
    if (!indexes || n_indexes <= 0)
        return -1;

    int idx_shape[1] = {n_indexes};
    mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
    mlx_array sel = mlx_array_new();
    if (mlx_gather_single(&sel, *facies_pyramid[scale - 1], idx, 0, NULL, 0, s) != 0)
    {
        mlx_array_free(idx);
        mlx_stream_free(s);
        return -1;
    }
    mlx_array_free(idx);

    /* upsample to match target spatial dims */
    const int *shape = mlx_array_shape(*facies_pyramid[scale]);
    if (!shape)
    {
        mlx_array_free(sel);
        return -1;
    }
    int target_h = shape[1];
    int target_w = shape[2];

    MLXUpsample *u = mlx_upsample_create(target_h, target_w, "linear", 1);
    if (!u)
    {
        mlx_array_free(sel);
        mlx_stream_free(s);
        return -1;
    }
    mlx_array up = mlx_upsample_forward(u, sel);
    mlx_upsample_free(u);

    /* free intermediate selection if different from result */
    if (sel.ctx)
        mlx_array_free(sel);

    mlx_array *res = (mlx_array *)malloc(sizeof(mlx_array));
    if (!res)
    {
        mlx_array_free(up);
        mlx_stream_free(s);
        return -1;
    }
    *res = up;
    *out = res;
    mlx_stream_free(s);
    return 0;
}

int mlx_init_rec_noise_and_amp(MLXFaciesGAN *m, int scale, const int *indexes, int n_indexes, const mlx_array *real, mlx_array **wells_pyramid, mlx_array **seismic_pyramid)
{
    if (!m || !real)
        return -1;
    if (scale < 0)
        return -1;

    /* if noise amps already set up to this scale, nothing to do */
    float *amps = NULL;
    int n_amps = 0;
    if (mlx_faciesgan_get_noise_amps(m, &amps, &n_amps) == 0)
    {
        if (amps)
        {
            if (n_amps > scale)
            {
                free(amps);
                return 0;
            }
            free(amps);
        }
    }

    /* Defaults (match Python defaults in options.c) */
    const double default_scale0_noise_amp = 1.0;
    const double default_noise_amp = 0.1;
    const double default_min_noise_amp = 0.1;

    /* get pyramid noises */
    mlx_array **noises = NULL;
    int n_noises = 0;
    if (mlx_faciesgan_get_pyramid_noise(m, scale, indexes, n_indexes, &noises, &n_noises, wells_pyramid, seismic_pyramid, 0) != 0)
        return -1;

    /* obtain amplitude list for generation (will be freed) */
    float *use_amps = NULL;
    int use_n = 0;
    if (mlx_faciesgan_get_noise_amplitude(m, scale, &use_amps, &use_n) != 0)
    {
        /* fallback to ones */
        use_amps = (float *)malloc(sizeof(float) * (scale + 1));
        if (!use_amps)
        {
            for (int i = 0; i < n_noises; ++i)
            {
                if (noises[i])
                {
                    mlx_array_free(*noises[i]);
                    free(noises[i]);
                }
            }
            free(noises);
            return -1;
        }
        for (int i = 0; i < scale + 1; ++i)
            use_amps[i] = 1.0f;
        use_n = scale + 1;
    }

    /* generate fake for this scale (numeric forward) */
    mlx_array in_noise = mlx_array_new();
    mlx_array_t fake = mlx_faciesgan_generate_fake(m, (const mlx_array *)noises, n_noises, use_amps, use_n, in_noise, scale, scale);

    /* free noises */
    for (int i = 0; i < n_noises; ++i)
    {
        if (noises[i])
        {
            mlx_array_free(*noises[i]);
            free(noises[i]);
        }
    }
    free(noises);
    if (use_amps)
        free(use_amps);

    /* compute rmse = sqrt(mean((fake - real)^2)) */
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array diff = mlx_array_new();
    if (mlx_subtract(&diff, fake, *real, s) != 0)
    {
        mlx_stream_free(s);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        return -1;
    }
    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, diff, s) != 0)
    {
        mlx_array_free(diff);
        mlx_stream_free(s);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        return -1;
    }
    mlx_array_free(diff);

    mlx_array mean = mlx_array_new();
    if (mlx_mean(&mean, sq, false, s) != 0)
    {
        mlx_array_free(sq);
        mlx_stream_free(s);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        return -1;
    }
    mlx_array_free(sq);

    mlx_array root = mlx_array_new();
    if (mlx_sqrt(&root, mean, s) != 0)
    {
        mlx_array_free(mean);
        mlx_stream_free(s);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        return -1;
    }
    mlx_array_free(mean);

    /* extract scalar value */
    mlx_array_eval(root);
    const float *pdata = mlx_array_data_float32(root);
    double rmse = 0.0;
    if (pdata)
        rmse = (double)pdata[0];

    /* compute new amp */
    double amp = default_noise_amp * rmse;
    if (scale == 0)
        amp = default_scale0_noise_amp * rmse;
    if (amp < default_min_noise_amp)
        amp = default_min_noise_amp;

    /* Build new amp array (scale+1) by copying existing amps if present */
    float *new_amps = (float *)malloc(sizeof(float) * (scale + 1));
    if (!new_amps)
    {
        mlx_array_free(root);
        mlx_stream_free(s);
        mlx_array_free(fake);
        mlx_array_free(in_noise);
        return -1;
    }
    /* try to read existing amps to preserve previous values */
    if (mlx_faciesgan_get_noise_amps(m, &amps, &n_amps) == 0 && amps && n_amps > 0)
    {
        for (int i = 0; i < scale + 1; ++i)
            new_amps[i] = (i < n_amps) ? amps[i] : 1.0f;
        free(amps);
    }
    else
    {
        for (int i = 0; i < scale + 1; ++i)
            new_amps[i] = 1.0f;
    }
    new_amps[scale] = (float)amp;

    /* set on model */
    int res = mlx_faciesgan_set_noise_amps(m, new_amps, scale + 1);

    free(new_amps);
    mlx_array_free(root);
    mlx_stream_free(s);
    mlx_array_free(fake);
    mlx_array_free(in_noise);
    return res;
}
