#include "facies_gan.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
/* Standard string/memory functions provided by <string.h> */
#include "custom_layer.h"
#include "trainning/array_helpers.h"
#include "trainning/scalar_pool.h"
#include "trainning/mlx_compat.h"
#include <limits.h>
#include <mlx/c/io.h>
#include <mlx/c/map.h>
#include <mlx/c/transforms.h>
#include <trainning/train_step.h>

struct MLXFaciesGAN {
    int num_layer;
    int kernel_size;
    int padding_size;
    int num_img_channels;
    int num_feature;
    int min_num_feature;
    int discriminator_steps;
    int generator_steps;
    /* channel semantics matching Python base class */
    int gen_input_channels;
    int base_channel;
    int num_diversity_samples;
    MLXGenerator *generator;
    MLXDiscriminator *discriminator;
    /* optional shapes & noise amp bookkeeping (set via set_shapes /
     * set_noise_amps) */
    int *shapes;
    int n_shapes;
    float *noise_amps;
    int n_noise_amps;
    /* optional wells storage (loaded from M_FILE) */
    mlx_array **wells;
    int n_wells;
    /* Stored reconstruction noise (one per scale, generated once and reused).
     * Mirrors Python's self.rec_noise list. */
    mlx_array **rec_noise;
    int n_rec_noise;
};

/* Global runtime flag: use create-graph for GP by default. */
static int g_use_create_graph_gp = 1;

void mlx_faciesgan_set_use_create_graph_gp(int use) {
    g_use_create_graph_gp = use ? 1 : 0;
}
int mlx_faciesgan_get_use_create_graph_gp(void) {
    return g_use_create_graph_gp;
}

MLXFaciesGAN *mlx_faciesgan_create(int num_layer, int kernel_size,
                                   int padding_size, int num_img_channels,
                                   int num_feature, int min_num_feature,
                                   int discriminator_steps,
                                   int generator_steps) {
    MLXFaciesGAN *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(MLXFaciesGAN), 1) != 0)
        return NULL;
    memset(m, 0, sizeof(*m));
    m->num_layer = num_layer;
    m->kernel_size = kernel_size;
    m->padding_size = padding_size;
    m->num_img_channels = num_img_channels;
    /* default channel semantics: match Python defaults where
        base_channel == num_img_channels and gen_input_channels
        default to num_img_channels unless caller overrides later */
    m->gen_input_channels = num_img_channels;
    m->base_channel = num_img_channels;
    m->num_diversity_samples = 1;
    m->num_feature = num_feature;
    m->min_num_feature = min_num_feature;
    m->discriminator_steps = discriminator_steps;
    m->generator_steps = generator_steps;
    m->generator = NULL;
    m->discriminator = NULL;
    m->shapes = NULL;
    m->n_shapes = 0;
    m->noise_amps = NULL;
    m->n_noise_amps = 0;
    m->rec_noise = NULL;
    m->n_rec_noise = 0;
    return m;
}

int mlx_faciesgan_set_gen_input_channels(MLXFaciesGAN *m, int channels) {
    if (!m || channels < 1)
        return -1;
    m->gen_input_channels = channels;
    return 0;
}

int mlx_faciesgan_get_gen_input_channels(MLXFaciesGAN *m) {
    if (!m)
        return 3;
    return m->gen_input_channels;
}

int mlx_faciesgan_set_num_diversity_samples(MLXFaciesGAN *m, int n) {
    if (!m || n < 1)
        return -1;
    m->num_diversity_samples = n;
    return 0;
}

int mlx_faciesgan_get_num_diversity_samples(MLXFaciesGAN *m) {
    if (!m)
        return 1;
    return m->num_diversity_samples > 0 ? m->num_diversity_samples : 1;
}

int mlx_faciesgan_compute_diversity_loss(MLXFaciesGAN *m,
        mlx_array **fake_samples,
        int n_samples, float lambda_diversity,
        mlx_array **out_loss) {

    if (!out_loss)
        return -1;
    if (lambda_diversity <= 0.0f || !fake_samples || n_samples < 2) {
        *out_loss = NULL;
        if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0)
            return -1;
        **out_loss = mlx_array_new_float(0.0f);
        return 0;
    }

    mlx_stream s = mlx_gpu_stream();

    /* Validate first sample exists */
    if (!fake_samples[0]) return -1;
    size_t total_elems = (size_t)mlx_array_size(*fake_samples[0]);
    if (total_elems == 0) {
        *out_loss = NULL;
        if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0)
            return -1;
        **out_loss = mlx_array_new_float(0.0f);
        return 0;
    }

    /* --- Vectorized diversity loss via broadcasting ---
     * 1. Flatten each sample to [M] and stack → [n, M]
     * 2. Expand dims:  a=[n,1,M]  b=[1,n,M]
     * 3. diff = a - b  → [n,n,M]  (broadcast)
     * 4. sq = diff²    → [n,n,M]
     * 5. mean_sq = mean(sq, axis=2) → [n,n]
     * 6. vals = exp(-10 * mean_sq)  → [n,n]
     * 7. loss = lambda * (sum(vals) - n) / (n*(n-1))
     *    (diagonal entries are exp(0)=1, matrix is symmetric)
     */
    int rc = 0;

    /* 1. Flatten and stack */
    int flat_shape[1] = {(int)total_elems};
    mlx_vector_array sample_vec = mlx_vector_array_new();
    for (int i = 0; i < n_samples; ++i) {
        if (!fake_samples[i]) {
            mlx_vector_array_free(sample_vec);
            return -1;
        }
        mlx_array flat = mlx_array_new();
        if (mlx_reshape(&flat, *fake_samples[i], flat_shape, 1, s) != 0) {
            mlx_array_free(flat);
            mlx_vector_array_free(sample_vec);
            return -1;
        }
        mlx_vector_array_append_value(sample_vec, flat);
        mlx_array_free(flat);
    }
    mlx_array stacked = mlx_array_new();  /* [n, M] */
    rc = mlx_stack_axis(&stacked, sample_vec, 0, s);
    mlx_vector_array_free(sample_vec);
    if (rc != 0) {
        mlx_array_free(stacked);
        return -1;
    }

    /* 2. expand_dims: a=[n,1,M]  b=[1,n,M] */
    mlx_array a = mlx_array_new();
    mlx_array b = mlx_array_new();
    if (mlx_expand_dims(&a, stacked, 1, s) != 0 ||
            mlx_expand_dims(&b, stacked, 0, s) != 0) {
        mlx_array_free(stacked);
        mlx_array_free(a);
        mlx_array_free(b);
        return -1;
    }
    mlx_array_free(stacked);

    /* 3-4. diff = a - b, sq = diff² */
    mlx_array diff = mlx_array_new();
    if (mlx_subtract(&diff, a, b, s) != 0) {
        mlx_array_free(a);
        mlx_array_free(b);
        mlx_array_free(diff);
        return -1;
    }
    mlx_array_free(a);
    mlx_array_free(b);

    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, diff, s) != 0) {
        mlx_array_free(diff);
        mlx_array_free(sq);
        return -1;
    }
    mlx_array_free(diff);

    /* 5. mean over spatial axis (last axis, index 2) */
    mlx_array mean_sq = mlx_array_new();
    if (mlx_mean_axis(&mean_sq, sq, 2, false, s) != 0) {
        mlx_array_free(sq);
        mlx_array_free(mean_sq);
        return -1;
    }
    mlx_array_free(sq);

    /* 6. vals = exp(-10 * mean_sq) */
    mlx_array scaled = mlx_array_new();
    if (mlx_multiply(&scaled, mean_sq, mlx_scalar_neg_ten(), s) != 0) {
        mlx_array_free(mean_sq);
        mlx_array_free(scaled);
        return -1;
    }
    mlx_array_free(mean_sq);

    mlx_array vals = mlx_array_new();
    if (mlx_exp(&vals, scaled, s) != 0) {
        mlx_array_free(scaled);
        mlx_array_free(vals);
        return -1;
    }
    mlx_array_free(scaled);

    /* 7. sum_all, subtract diagonal (n ones), divide by n*(n-1) pairs×2 */
    mlx_array sum_all = mlx_array_new();
    if (mlx_sum(&sum_all, vals, false, s) != 0) {
        mlx_array_free(vals);
        mlx_array_free(sum_all);
        return -1;
    }
    mlx_array_free(vals);

    mlx_array n_arr = mlx_array_new_float((float)n_samples);
    mlx_array sum_off = mlx_array_new();
    if (mlx_subtract(&sum_off, sum_all, n_arr, s) != 0) {
        mlx_array_free(sum_all);
        mlx_array_free(n_arr);
        mlx_array_free(sum_off);
        return -1;
    }
    mlx_array_free(sum_all);
    mlx_array_free(n_arr);

    /* Divide by n*(n-1) and multiply by lambda */
    float scale_factor = lambda_diversity / (float)(n_samples * (n_samples - 1));
    mlx_array scale_arr = mlx_array_new_float(scale_factor);
    mlx_array outv = mlx_array_new();
    if (mlx_multiply(&outv, sum_off, scale_arr, s) != 0) {
        mlx_array_free(sum_off);
        mlx_array_free(scale_arr);
        mlx_array_free(outv);
        return -1;
    }
    mlx_array_free(sum_off);
    mlx_array_free(scale_arr);

    *out_loss = NULL;
    if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(outv);
        return -1;
    }
    **out_loss = outv;
    return 0;
}

int mlx_faciesgan_compute_masked_loss(MLXFaciesGAN *m, const mlx_array *fake,
                                      const mlx_array *real,
                                      const mlx_array *well,
                                      const mlx_array *mask,
                                      float well_loss_penalty,
                                      mlx_array **out_loss) {

    if (!out_loss)
        return -1;
    if (!well || !mask || !fake || !real) {
        *out_loss = NULL;
        if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0)
            return -1;
        **out_loss = mlx_array_new_float(0.0f);
        return 0;
    }

    mlx_stream s = mlx_gpu_stream();

    mlx_array fmasked = mlx_array_new();
    mlx_array rmasked = mlx_array_new();
    if (mlx_multiply(&fmasked, *fake, *mask, s) != 0 ||
            mlx_multiply(&rmasked, *real, *mask, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        return -1;
    }

    mlx_array diff = mlx_array_new();
    if (mlx_subtract(&diff, fmasked, rmasked, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        return -1;
    }

    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, diff, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_array_free(sq);
        return -1;
    }

    int ndim = (int)mlx_array_ndim(diff);
    int axes[8];
    for (int a = 0; a < ndim && a < 8; ++a)
        axes[a] = a;

    mlx_array mean = mlx_array_new();
    if (mlx_mean_axes(&mean, sq, axes, ndim, true, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        return -1;
    }

    mlx_array lambda_arr = mlx_array_new_float(well_loss_penalty);
    mlx_array outv = mlx_array_new();
    if (mlx_multiply(&outv, mean, lambda_arr, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        mlx_array_free(lambda_arr);
        mlx_array_free(outv);
        return -1;
    }

    *out_loss = NULL;
    if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(outv);
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        mlx_array_free(lambda_arr);
        return -1;
    }
    **out_loss = outv;
    /* Cleanup intermediates on success path */
    mlx_array_free(fmasked);
    mlx_array_free(rmasked);
    mlx_array_free(diff);
    mlx_array_free(sq);
    mlx_array_free(mean);
    mlx_array_free(lambda_arr);
    return 0;
}

void mlx_faciesgan_free(MLXFaciesGAN *m) {
    if (!m)
        return;
    if (m->generator) {
        mlx_generator_free(m->generator);
        m->generator = NULL;
    }
    if (m->discriminator) {
        mlx_discriminator_free(m->discriminator);
        m->discriminator = NULL;
    }
    if (m->shapes)
        mlx_free_int_array(&m->shapes, &m->n_shapes);
    if (m->noise_amps)
        mlx_free_float_buf(&m->noise_amps, NULL);
    if (m->wells) {
        for (int i = 0; i < m->n_wells; ++i) {
            if (m->wells[i]) {
                mlx_array_free(*m->wells[i]);
                mlx_free_pod((void **)&m->wells[i]);
            }
        }
        mlx_free_pod((void **)&m->wells);
        m->wells = NULL;
        m->n_wells = 0;
    }
    if (m->rec_noise) {
        for (int i = 0; i < m->n_rec_noise; ++i) {
            if (m->rec_noise[i]) {
                mlx_array_free(*m->rec_noise[i]);
                mlx_free_pod((void **)&m->rec_noise[i]);
            }
        }
        mlx_free_pod((void **)&m->rec_noise);
        m->rec_noise = NULL;
        m->n_rec_noise = 0;
    }
    mlx_free_pod((void **)&m);
}

MLXDiscriminator *mlx_faciesgan_build_discriminator(MLXFaciesGAN *m) {
    if (!m)
        return NULL;
    if (m->discriminator)
        return m->discriminator;
    m->discriminator = mlx_discriminator_create(
                           m->num_layer, m->kernel_size, m->padding_size, m->num_img_channels);
    if (!m->discriminator)
        return NULL;
    /* create initial scale */
    if (mlx_discriminator_create_scale(m->discriminator, m->num_feature,
                                       m->min_num_feature) != 0) {
        /* best-effort: if scale creation fails, still return discriminator */
    }
    return m->discriminator;
}
MLXGenerator *mlx_faciesgan_build_generator(MLXFaciesGAN *m) {
    if (!m)
        return NULL;
    if (m->generator)
        return m->generator;
    m->generator =
        mlx_generator_create(m->num_layer, m->kernel_size, m->padding_size,
                             m->gen_input_channels, m->num_img_channels);
    if (!m->generator)
        return NULL;
    /* Create a default scale 0 so parameters exist */
    if (mlx_generator_create_scale(m->generator, 0, m->num_feature,
                                   m->min_num_feature) != 0) {
        /* if failed, free generator and return NULL */
        mlx_generator_free(m->generator);
        m->generator = NULL;
        return NULL;
    }
    return m->generator;
}

int mlx_faciesgan_create_generator_scale(MLXFaciesGAN *m, int scale,
        int num_features,
        int min_num_features) {
    if (!m)
        return -1;
    if (!m->generator) {
        m->generator =
            mlx_generator_create(m->num_layer, m->kernel_size, m->padding_size,
                                 m->gen_input_channels, m->num_img_channels);
        if (!m->generator)
            return -1;
    }
    return mlx_generator_create_scale(m->generator, scale, num_features,
                                      min_num_features);
}

int mlx_faciesgan_create_discriminator_scale(MLXFaciesGAN *m,
        int num_features,
        int min_num_features) {
    if (!m)
        return -1;
    /* Ensure discriminator exists */
    MLXDiscriminator *d = mlx_faciesgan_build_discriminator(m);
    if (!d)
        return -1;
    return mlx_discriminator_create_scale(d, num_features, min_num_features);
}

mlx_array_t mlx_faciesgan_generate_fake(MLXFaciesGAN *m,
                                        const mlx_array *z_list, int z_count,
                                        const float *amp, int amp_count,
                                        mlx_array_t in_noise, int start_scale,
                                        int stop_scale) {
    if (!m)
        return in_noise;
    /* If caller provided explicit amp array, prefer it. Otherwise use
       stored noise amps if available; else fall back to implicit 1.0s. */
    const float *use_amp = amp;
    int use_amp_count = amp_count;
    float *tmp_amps = NULL;
    int tmp_amps_helper = 0;
    if ((!use_amp || use_amp_count <= 0) && m->noise_amps &&
            m->n_noise_amps > 0) {
        /* Determine target scale to size the amp list. Use stop_scale if provided,
           otherwise infer from z_count. */
        int target = stop_scale;
        if (target < 0)
            target = (z_count > 0) ? (z_count - 1) : 0;
        int n = target + 1;
        if (n <= m->n_noise_amps) {
            use_amp = m->noise_amps;
            use_amp_count = m->n_noise_amps;
        } else {
            /* allocate temporary array sized to n and fill from available amps or 1.0
             */
            tmp_amps = NULL;
            if (n > (size_t)INT_MAX) {
                tmp_amps = (float *)malloc(sizeof(float) * n);
                tmp_amps_helper = 0;
            } else {
                if (mlx_alloc_float_buf(&tmp_amps, (int)n) != 0) {
                    tmp_amps = NULL;
                    tmp_amps_helper = 0;
                } else {
                    tmp_amps_helper = 1;
                }
            }
            if (!tmp_amps) {
                use_amp = NULL;
                use_amp_count = 0;
            } else {
                for (int i = 0; i < n; ++i)
                    tmp_amps[i] = (i < m->n_noise_amps) ? m->noise_amps[i] : 1.0f;
                use_amp = tmp_amps;
                use_amp_count = n;
            }
        }
    }

    /* Call existing MLX generator forward which implements the progressive
       synthesis behavior consistent with the Python implementation. */
    mlx_array_t out =
        mlx_generator_forward(m->generator, z_list, z_count, use_amp,
                              use_amp_count, in_noise, start_scale, stop_scale);

    if (tmp_amps) {
        if (tmp_amps_helper)
            mlx_free_float_buf(&tmp_amps, NULL);
        else
            free(tmp_amps);
    }
    return out;
}


int mlx_faciesgan_set_shapes(MLXFaciesGAN *m, const int *shapes, int n_scales) {
    if (!m)
        return -1;
    if (m->shapes) {
        free(m->shapes);
        m->shapes = NULL;
        m->n_shapes = 0;
    }
    if (!shapes || n_scales <= 0)
        return 0;
    /* assume each scale shape is 4 ints (N,H,W,C) */
    int total = n_scales * 4;
    m->shapes = (int *)malloc(sizeof(int) * total);
    if (!m->shapes)
        return -1;
    memcpy(m->shapes, shapes, sizeof(int) * total);
    m->n_shapes = n_scales;
    return 0;
}
int mlx_faciesgan_set_noise_amps(MLXFaciesGAN *m, const float *amps, int n) {
    if (!m)
        return -1;
    if (m->noise_amps) {
        mlx_free_float_buf(&m->noise_amps, NULL);
        m->noise_amps = NULL;
        m->n_noise_amps = 0;
    }
    if (!amps || n <= 0)
        return 0;
    if (mlx_alloc_float_buf(&m->noise_amps, n) != 0)
        return -1;
    memcpy(m->noise_amps, amps, sizeof(float) * n);
    m->n_noise_amps = n;
    return 0;
}

int mlx_faciesgan_get_shapes_flat(MLXFaciesGAN *m, int **out_shapes,
                                  int *out_n_scales) {
    if (!m || !out_shapes || !out_n_scales)
        return -1;
    if (!m->shapes || m->n_shapes == 0) {
        *out_shapes = NULL;
        *out_n_scales = 0;
        return 0;
    }
    int total = m->n_shapes * 4;
    int *copy = NULL;
    if (mlx_alloc_pod((void **)&copy, sizeof(int), total) != 0)
        return -1;
    memcpy(copy, m->shapes, sizeof(int) * total);
    *out_shapes = copy;
    *out_n_scales = m->n_shapes;
    return 0;
}

int mlx_faciesgan_get_noise_amps(MLXFaciesGAN *m, float **out_amps,
                                 int *out_n) {
    if (!m || !out_amps || !out_n)
        return -1;
    if (!m->noise_amps || m->n_noise_amps == 0) {
        *out_amps = NULL;
        *out_n = 0;
        return 0;
    }
    float *copy = NULL;
    if (mlx_alloc_float_buf(&copy, m->n_noise_amps) != 0)
        return -1;
    memcpy(copy, m->noise_amps, sizeof(float) * m->n_noise_amps);
    *out_amps = copy;
    *out_n = m->n_noise_amps;
    return 0;
}

int mlx_faciesgan_get_noise_amplitude(MLXFaciesGAN *m, int scale,
                                      float **out_amps, int *out_n) {
    if (!m || !out_amps || !out_n)
        return -1;
    if (!m->noise_amps || m->n_noise_amps == 0) {
        /* return list of 1.0s of length scale+1 */
        int n = scale + 1;
        float *arr = NULL;
        if (n > (size_t)INT_MAX) {
            arr = (float *)malloc(sizeof(float) * n);
            if (!arr)
                return -1;
            for (int i = 0; i < n; ++i)
                arr[i] = 1.0f;
            *out_amps = arr;
        } else {
            if (mlx_alloc_float_buf(&arr, n) != 0)
                return -1;
            for (int i = 0; i < n; ++i)
                arr[i] = 1.0f;
            *out_amps = arr;
        }
        *out_n = n;
        return 0;
    }
    int n = scale + 1;
    float *arr = NULL;
    if (n > (size_t)INT_MAX) {
        arr = (float *)malloc(sizeof(float) * n);
        if (!arr)
            return -1;
        for (int i = 0; i < n; ++i)
            arr[i] = (i < m->n_noise_amps) ? m->noise_amps[i] : 1.0f;
        *out_amps = arr;
    } else {
        if (mlx_alloc_float_buf(&arr, n) != 0)
            return -1;
        for (int i = 0; i < n; ++i)
            arr[i] = (i < m->n_noise_amps) ? m->noise_amps[i] : 1.0f;
        *out_amps = arr;
    }
    *out_n = n;
    return 0;
}

int mlx_faciesgan_get_pyramid_noise(MLXFaciesGAN *m, int scale,
                                    const int *indexes, int n_indexes,
                                    mlx_array ***out_noises, int *out_n,
                                    mlx_array **wells_pyramid,
                                    mlx_array **seismic_pyramid, int rec) {
    if (!m || !out_noises || !out_n)
        return -1;
    if (scale < 0)
        return -1;

    /* If rec=1 and stored reconstruction noise exists, return copies of stored
     * noise.  This mirrors Python's get_pyramid_noise(rec=True) which calls
     * get_rec_noise(scale) returning self.rec_noise[:scale+1]. */
    if (rec && m->rec_noise && m->n_rec_noise > 0) {
        int n = scale + 1;
        mlx_array **arr = NULL;
        if (mlx_alloc_pod((void **)&arr, sizeof(mlx_array *), n) != 0)
            return -1;
        for (int i = 0; i < n; ++i)
            arr[i] = NULL;

        mlx_stream s = mlx_gpu_stream();
        for (int i = 0; i < n; ++i) {
            mlx_array *a = NULL;
            if (mlx_alloc_pod((void **)&a, sizeof(mlx_array), 1) != 0) {
                for (int j = 0; j < n; ++j) {
                    if (arr[j]) {
                        mlx_array_free(*arr[j]);
                        mlx_free_pod((void **)&arr[j]);
                    }
                }
                mlx_free_pod((void **)&arr);
                return -1;
            }
            *a = mlx_array_new();
            if (i < m->n_rec_noise && m->rec_noise[i]) {
                /* Return a copy of the stored rec noise (like Python's mx.array(tensor)) */
                mlx_copy(a, *m->rec_noise[i], s);
            } else {
                /* Fallback: generate zeros if rec noise not available for this scale.
                 * Rec noise is stored AFTER padding, so spatial dims should include
                 * padding. Channels should be gen_input_channels (noise + conditioning). */
                int shape[4] = {1, 32, 32, m->gen_input_channels};
                if (m->shapes && m->n_shapes > i) {
                    const int *s0 = &m->shapes[i * 4];
                    shape[0] = s0[0];
                    shape[1] = s0[1] + 2 * m->padding_size;
                    shape[2] = s0[2] + 2 * m->padding_size;
                    shape[3] = m->gen_input_channels;
                }
                mlx_zeros(a, shape, 4, MLX_FLOAT32, s);
            }
            arr[i] = a;
        }
        *out_noises = arr;
        *out_n = n;
        return 0;
    }

    int n = scale + 1;
    mlx_array **arr = NULL;
    if (mlx_alloc_pod((void **)&arr, sizeof(mlx_array *), n) != 0)
        return -1;
    for (int i = 0; i < n; ++i)
        arr[i] = NULL;

    /* Use shapes set via `set_shapes` when available; otherwise fallback
       to a reasonable default shape (1,32,32,num_img_channels). */
    for (int i = 0; i < n; ++i) {
        int batch = 1;
        int h = 32;
        int w = 32;
        /* Determine whether conditioning (wells/seismic) is provided for this scale
         */
        int cond_present = ((wells_pyramid && wells_pyramid[i]) ||
                            (seismic_pyramid && seismic_pyramid[i]))
                           ? 1
                           : 0;
        int c = 0;

        /* FIX: batch = len(indexes) in Python — always use n_indexes when
         * available, regardless of conditioning presence. */
        if (n_indexes > 0)
            batch = n_indexes;

        /* Derive spatial dims from stored shapes (Python: self.shapes[scale]).
         * Fall back to conditioning shape only if shapes not available. */
        if (m->shapes && m->n_shapes > i) {
            const int *s0 = &m->shapes[i * 4];
            if (batch <= 1) batch = s0[0];
            h = s0[1];
            w = s0[2];
            /* If conditioning is present, use base_channel (noise-only channels),
               otherwise use gen_input_channels to match Python semantics. */
            c = cond_present ? m->base_channel : m->gen_input_channels;
        } else if (cond_present && wells_pyramid && wells_pyramid[i]) {
            const int *cond_shape = mlx_array_shape(*wells_pyramid[i]);
            if (cond_shape) {
                h = cond_shape[1];
                w = cond_shape[2];
            }
            c = m->base_channel;
        } else if (cond_present && seismic_pyramid && seismic_pyramid[i]) {
            const int *cond_shape = mlx_array_shape(*seismic_pyramid[i]);
            if (cond_shape) {
                h = cond_shape[1];
                w = cond_shape[2];
            }
            c = m->base_channel;
        } else {
            /* Fallback defaults: base_channel vs gen_input_channels */
            c = cond_present ? m->base_channel : m->gen_input_channels;
        }
        mlx_stream s = mlx_gpu_stream();
        mlx_array *a = NULL;
        if (mlx_alloc_pod((void **)&a, sizeof(mlx_array), 1) != 0) {
            for (int j = 0; j < n; ++j) {
                if (arr[j]) {
                    mlx_array_free(*arr[j]);
                    mlx_free_pod((void **)&arr[j]);
                }
            }
            mlx_free_pod((void **)&arr);
            return -1;
        }
        /* Initialize the mlx_array object so its C++ internals are constructed
           before any operator= or other methods are invoked. This avoids
           UB when assigning into freshly-allocated memory. */
        *a = mlx_array_new();

        /* NOTE: When conditioning is present, we create noise WITHOUT padding first.
           After concatenating with conditioning data, we apply padding.
           When conditioning is NOT present, we include padding directly in noise creation.

           With conditioning: noise_channels = 3, cond_channels = 3, total = 6 after concat
           Without conditioning: total_channels = gen_input_channels (typically 3)
        */
        /* FIX: Use m->base_channel for noise-only channels (Python: self.base_channel
         * which equals noise_channels, typically 3). Never hardcode. */
        int noise_only_c = m->base_channel;

        /* FIX: BOTH conditioning and no-conditioning paths create noise at
         * (batch, H, W, C) WITHOUT padding. The common padding code at the
         * end adds zero-padding once, matching Python's generate_padding().
         *
         * Previously the no-conditioning path pre-padded with random values
         * then padded AGAIN → double padding + random in border region. */
        if (cond_present) {
            /* Conditioning path: noise at base_channel channels */
            int noise_shape[4] = {batch, h, w, noise_only_c};
            if (mlx_random_normal(a, noise_shape, 4, MLX_FLOAT32, 0.0f, 1.0f, mlx_array_empty,
                                  s) != 0) {
                if (mlx_zeros(a, noise_shape, 4, MLX_FLOAT32, s) != 0) {
                    mlx_free_pod((void **)&a);
                    for (int j = 0; j < n; ++j) {
                        if (arr[j]) {
                            mlx_array_free(*arr[j]);
                            mlx_free_pod((void **)&arr[j]);
                        }
                    }
                    mlx_free_mlx_array_ptrs(&arr, n);
                    return -1;
                }
            }
        } else {
            /* No-conditioning path: noise at gen_input_channels,
             * unpadded spatial dims (padding applied at end) */
            int shape[4] = {batch, h, w, c};
            if (mlx_random_normal(a, shape, 4, MLX_FLOAT32, 0.0f, 1.0f, mlx_array_empty,
                                  s) != 0) {
                if (mlx_zeros(a, shape, 4, MLX_FLOAT32, s) != 0) {
                    mlx_free_pod((void **)&a);
                    for (int j = 0; j < n; ++j) {
                        if (arr[j]) {
                            mlx_array_free(*arr[j]);
                            mlx_free_pod((void **)&arr[j]);
                        }
                    }
                    mlx_free_mlx_array_ptrs(&arr, n);
                    return -1;
                }
            }
        }

        /* If conditioning provided, slice by `indexes` and concat with noise along
           channel axis. Support both wells and seismic concatenation (noise, wells,
           seismic). */
        if (indexes && n_indexes > 0 && wells_pyramid && wells_pyramid[i] &&
                seismic_pyramid && seismic_pyramid[i]) {
            int idx_shape[1] = {n_indexes};
            mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
            mlx_array wells_sel = mlx_array_new();
            mlx_array seismic_sel = mlx_array_new();
            /* Use mlx_take_axis for proper indexing (equivalent to Python's array[indices]) */
            if (mlx_take_axis(&wells_sel, *wells_pyramid[i], idx, 0, s) == 0 &&
                    mlx_take_axis(&seismic_sel, *seismic_pyramid[i], idx, 0, s) == 0) {
                mlx_vector_array vec = mlx_vector_array_new_data(
                (const mlx_array[]) {
                    *a, wells_sel, seismic_sel
                }, 3);
                mlx_array znew = mlx_array_new();
                if (mlx_concatenate_axis(&znew, vec, 3, s) == 0) {
                    mlx_array_free(*a);
                    *a = znew;
                } else {
                    mlx_array_free(znew);
                }
                mlx_vector_array_free(vec);
                mlx_array_free(wells_sel);
                mlx_array_free(seismic_sel);
            } else {
                mlx_array_free(wells_sel);
                mlx_array_free(seismic_sel);
            }
            mlx_array_free(idx);
        } else if (indexes && n_indexes > 0 && wells_pyramid && wells_pyramid[i]) {
            /* build indices array */
            int idx_shape[1] = {n_indexes};
            mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
            mlx_array cond_sel = mlx_array_new();
            /* Use mlx_take_axis for proper indexing (equivalent to Python's well[indexes]) */
            if (mlx_take_axis(&cond_sel, *wells_pyramid[i], idx, 0, s) == 0) {
                int cond_sel_ndim = (int)mlx_array_ndim(cond_sel);
                /* concat noise and cond_sel */
                mlx_vector_array vec =
                mlx_vector_array_new_data((const mlx_array[]) {
                    *a, cond_sel
                }, 2);
                mlx_array znew = mlx_array_new();
                if (mlx_concatenate_axis(&znew, vec, 3, s) == 0) {
                    /* replace a with concatenated */
                    mlx_array_free(*a);
                    *a = znew;
                } else {
                    mlx_array_free(znew);
                }
                mlx_vector_array_free(vec);
                mlx_array_free(cond_sel);
            } else {
                /* failed to take: ignore and continue with base noise */
                mlx_array_free(cond_sel);
            }
            mlx_array_free(idx);
        } else if (indexes && n_indexes > 0 && seismic_pyramid &&
                   seismic_pyramid[i]) {
            int idx_shape[1] = {n_indexes};
            mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
            mlx_array cond_sel = mlx_array_new();
            /* Use mlx_take_axis for proper indexing (equivalent to Python's seismic[indexes]) */
            if (mlx_take_axis(&cond_sel, *seismic_pyramid[i], idx, 0, s) == 0) {
                mlx_vector_array vec =
                mlx_vector_array_new_data((const mlx_array[]) {
                    *a, cond_sel
                }, 2);
                mlx_array znew = mlx_array_new();
                if (mlx_concatenate_axis(&znew, vec, 3, s) == 0) {
                    mlx_array_free(*a);
                    *a = znew;
                } else {
                    mlx_array_free(znew);
                }
                mlx_vector_array_free(vec);
                mlx_array_free(cond_sel);
            } else {
                mlx_array_free(cond_sel);
            }
            mlx_array_free(idx);
        }

        /* Apply padding consistent with Python generate_padding */
        int p = m->num_layer * (m->kernel_size / 2);
        int axes[2] = {1, 2};
        int low_pad[2] = {p, p};
        int high_pad[2] = {p, p};
        mlx_array pad_val = mlx_array_new_float(0.0f);
        mlx_array padded = mlx_array_new();
        if (mlx_pad(&padded, *a, axes, 2, low_pad, 2, high_pad, 2, pad_val,
                    "constant", s) == 0) {
            mlx_array_free(*a);
            *a = padded;
        } else {
            mlx_array_free(padded);
        }
        mlx_array_free(pad_val);

        arr[i] = a;
    }

    *out_noises = arr;
    *out_n = n;
    return 0;
}

/* Store a reconstruction noise tensor for the given scale.
 * Mirrors Python: self.model.rec_noise.append(z_rec)
 * The array is deep-copied so the caller retains ownership of `noise`. */
int mlx_faciesgan_store_rec_noise(MLXFaciesGAN *m, int scale,
                                  const mlx_array *noise) {
    if (!m || !noise || scale < 0)
        return -1;
    /* Ensure capacity for scale+1 entries */
    int need = scale + 1;
    if (need > m->n_rec_noise) {
        mlx_array **tmp = NULL;
        if (mlx_alloc_pod((void **)&tmp, sizeof(mlx_array *), need) != 0)
            return -1;
        for (int i = 0; i < need; ++i)
            tmp[i] = NULL;
        /* Copy existing pointers */
        if (m->rec_noise) {
            for (int i = 0; i < m->n_rec_noise; ++i)
                tmp[i] = m->rec_noise[i];
            mlx_free_pod((void **)&m->rec_noise);
        }
        m->rec_noise = tmp;
        m->n_rec_noise = need;
    }
    /* Free previous entry if overwriting */
    if (m->rec_noise[scale]) {
        mlx_array_free(*m->rec_noise[scale]);
        mlx_free_pod((void **)&m->rec_noise[scale]);
    }
    /* Deep copy the noise array */
    mlx_array *a = NULL;
    if (mlx_alloc_pod((void **)&a, sizeof(mlx_array), 1) != 0)
        return -1;
    *a = mlx_array_new();
    mlx_stream s = mlx_gpu_stream();
    if (mlx_copy(a, *noise, s) != 0) {
        mlx_array_free(*a);
        mlx_free_pod((void **)&a);
        return -1;
    }
    /* Evaluate to materialise the copy */
    mlx_array_eval(*a);
    m->rec_noise[scale] = a;
    return 0;
}

int mlx_faciesgan_get_n_rec_noise(MLXFaciesGAN *m) {
    return m ? m->n_rec_noise : 0;
}

/* ---------------------- Checkpoint I/O helpers ------------------------- */
int mlx_faciesgan_save_generator_state(MLXFaciesGAN *m, const char *scale_path,
                                       int scale) {
    if (!m || !scale_path)
        return -1;
    if (!m->generator)
        return -1;

    int n = 0;
    mlx_array **plist = mlx_generator_get_parameters(m->generator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_generator_free_parameters_list(plist);
        return 0; /* nothing to save */
    }

    /* build file path: scale_path/generator.npz */
    char file[1024];
    snprintf(file, sizeof(file), "%s/generator.npz", scale_path);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();

    for (int i = 0; i < n; ++i) {
        char key[64];
        snprintf(key, sizeof(key), "param_%06d", i);
        if (plist[i]) {
            /* insert a copy/alias of the array value */
            mlx_map_string_to_array_insert(params, key, *plist[i]);
        }
    }

    int rc = mlx_save_safetensors(file, params, meta);

    mlx_map_string_to_array_free(params);
    mlx_map_string_to_string_free(meta);
    mlx_generator_free_parameters_list(plist);
    return rc == 0 ? 0 : -1;
}

int mlx_faciesgan_save_discriminator_state(MLXFaciesGAN *m,
        const char *scale_path, int scale) {
    if (!m || !scale_path)
        return -1;
    if (!m->discriminator)
        return -1;

    int n = 0;
    mlx_array **plist = mlx_discriminator_get_parameters(m->discriminator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_discriminator_free_parameters_list(plist);
        return 0;
    }

    char file[1024];
    snprintf(file, sizeof(file), "%s/discriminator.npz", scale_path);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();

    for (int i = 0; i < n; ++i) {
        char key[64];
        snprintf(key, sizeof(key), "param_%06d", i);
        if (plist[i])
            mlx_map_string_to_array_insert(params, key, *plist[i]);
    }

    int rc = mlx_save_safetensors(file, params, meta);
    mlx_map_string_to_array_free(params);
    mlx_map_string_to_string_free(meta);
    mlx_discriminator_free_parameters_list(plist);
    return rc == 0 ? 0 : -1;
}

int mlx_faciesgan_load_generator_state(MLXFaciesGAN *m, const char *scale_path,
                                       int scale) {
    if (!m || !scale_path || !m->generator)
        return -1;

    char file[1024];
    snprintf(file, sizeof(file), "%s/generator.npz", scale_path);

    mlx_stream s = mlx_gpu_stream();
    mlx_map_string_to_array map;
    mlx_map_string_to_string meta;
    if (mlx_load_safetensors(&map, &meta, file, s) != 0) {
        return -1;
    }

    int n = 0;
    mlx_array **plist = mlx_generator_get_parameters(m->generator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_generator_free_parameters_list(plist);
        mlx_map_string_to_array_free(map);
        mlx_map_string_to_string_free(meta);
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        char key[64];
        snprintf(key, sizeof(key), "param_%06d", i);
        mlx_array tmp = mlx_array_new();
        if (mlx_map_string_to_array_get(&tmp, map, key) == 0) {
            /* copy loaded array into parameter storage */
            mlx_array_set(plist[i], tmp);
        }
    }

    mlx_generator_free_parameters_list(plist);
    mlx_map_string_to_array_free(map);
    mlx_map_string_to_string_free(meta);
    return 0;
}

int mlx_faciesgan_load_discriminator_state(MLXFaciesGAN *m,
        const char *scale_path, int scale) {
    if (!m || !scale_path || !m->discriminator)
        return -1;

    char file[1024];
    snprintf(file, sizeof(file), "%s/discriminator.npz", scale_path);

    mlx_stream s = mlx_gpu_stream();
    mlx_map_string_to_array map;
    mlx_map_string_to_string meta;
    if (mlx_load_safetensors(&map, &meta, file, s) != 0) {
        return -1;
    }

    int n = 0;
    mlx_array **plist = mlx_discriminator_get_parameters(m->discriminator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_discriminator_free_parameters_list(plist);
        mlx_map_string_to_array_free(map);
        mlx_map_string_to_string_free(meta);
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        char key[64];
        snprintf(key, sizeof(key), "param_%06d", i);
        mlx_array tmp = mlx_array_new();
        if (mlx_map_string_to_array_get(&tmp, map, key) == 0) {
            mlx_array_set(plist[i], tmp);
        }
    }

    mlx_discriminator_free_parameters_list(plist);
    mlx_map_string_to_array_free(map);
    mlx_map_string_to_string_free(meta);
    return 0;
}

int mlx_faciesgan_save_shape(MLXFaciesGAN *m, const char *scale_path,
                             int scale) {
    if (!m || !scale_path)
        return -1;
    if (!m->shapes || scale < 0 || scale >= m->n_shapes)
        return -1;

    const int *s0 = &m->shapes[scale * 4];
    int shape_vals[4] = {s0[0], s0[1], s0[2], s0[3]};

    mlx_array arr = mlx_array_new();
    int dims[1] = {4};
    if (mlx_array_set_data(&arr, shape_vals, dims, 1, MLX_INT32) != 0)
        return -1;

    char file[1024];
    snprintf(file, sizeof(file), "%s/shape.npz", scale_path);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    mlx_map_string_to_array_insert(params, "shape", arr);
    int rc = mlx_save_safetensors(file, params, meta);
    mlx_map_string_to_array_free(params);
    mlx_map_string_to_string_free(meta);
    return rc == 0 ? 0 : -1;
}

int mlx_faciesgan_load_shape(MLXFaciesGAN *m, const char *scale_path,
                             int scale) {
    if (!m || !scale_path)
        return -1;
    char file[1024];
    snprintf(file, sizeof(file), "%s/shape.npz", scale_path);

    mlx_stream s = mlx_gpu_stream();
    mlx_map_string_to_array map;
    mlx_map_string_to_string meta;
    if (mlx_load_safetensors(&map, &meta, file, s) != 0) {
        return -1;
    }

    /* try to read key "shape" first, else iterate to first element */
    mlx_array arr = mlx_array_new();
    if (mlx_map_string_to_array_get(&arr, map, "shape") != 0) {
        /* fallback: iterate first element */
        const char *k = NULL;
        mlx_array itval = mlx_array_new();
        mlx_map_string_to_array_iterator it =
            mlx_map_string_to_array_iterator_new(map);
        if (mlx_map_string_to_array_iterator_next(&k, &itval, it) == 0 && k) {
            arr = itval;
        }
        mlx_map_string_to_array_iterator_free(it);
    }

    /* read int32 contents */
    size_t elems = mlx_array_size(arr);
    if (elems >= 3) {
        const int32_t *data = mlx_array_data_int32(arr);
        if (data) {
            /* append shape tuple (take up to 4 ints) */
            int tocopy = elems >= 4 ? 4 : (int)elems;
            int *newflat =
                (int *)realloc(m->shapes, sizeof(int) * (m->n_shapes + 1) * 4);
            if (!newflat) {
                mlx_map_string_to_array_free(map);
                mlx_map_string_to_string_free(meta);
                return -1;
            }
            m->shapes = newflat;
            int base = m->n_shapes * 4;
            for (int i = 0; i < 4; ++i)
                m->shapes[base + i] = (i < tocopy) ? (int)data[i] : 0;
            m->n_shapes += 1;
        }
    }

    mlx_map_string_to_array_free(map);
    mlx_map_string_to_string_free(meta);
    return 0;
}

int mlx_faciesgan_load_amp(MLXFaciesGAN *m, const char *scale_path) {
    if (!m || !scale_path)
        return -1;
    char ampfile[1024];
    snprintf(ampfile, sizeof(ampfile), "%s/amp.txt", scale_path);
    FILE *f = fopen(ampfile, "r");
    if (!f)
        return -1;
    double v = 1.0;
    if (fscanf(f, "%lf", &v) == 1) {
        float *na =
            (float *)realloc(m->noise_amps, sizeof(float) * (m->n_noise_amps + 1));
        if (!na) {
            fclose(f);
            return -1;
        }
        m->noise_amps = na;
        m->noise_amps[m->n_noise_amps++] = (float)v;
    }
    fclose(f);
    return 0;
}

int mlx_faciesgan_load_wells(MLXFaciesGAN *m, const char *scale_path) {
    /* No-op placeholder: wells handling is managed at a higher level in Python.
       Keep this function for API parity; return success if nothing to do. */

    /* scale_path unused in the C implementation (managed at Python level) */
    return 0;
}


int mlx_faciesgan_compute_recovery_loss(
    MLXFaciesGAN *m, const int *indexes, int n_indexes, int scale,
    const mlx_array rec_in, const mlx_array real, mlx_array **wells_pyramid,
    mlx_array **seismic_pyramid, float alpha, mlx_array **out_loss) {
    /* indexes are unused for this code path */
    /* unused parameters for numeric recovery loss helper */
    if (!m || !out_loss)
        return -1;
    if (alpha == 0.0f) {
        if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0)
            return -1;
        **out_loss = mlx_array_new_float(0.0f);
        return 0;
    }

    /* Compute simple numeric MSE between rec_in and real, then scale by alpha. */
    mlx_stream s = mlx_gpu_stream();
    mlx_array diff = mlx_array_new();
    if (mlx_subtract(&diff, rec_in, real, s) != 0) {
        return -1;
    }
    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, diff, s) != 0) {
        mlx_array_free(diff);
        return -1;
    }
    int ndim = (int)mlx_array_ndim(diff);
    int axes[8];
    for (int a = 0; a < ndim && a < 8; ++a)
        axes[a] = a;
    mlx_array mean = mlx_array_new();
    if (mlx_mean_axes(&mean, sq, axes, ndim, true, s) != 0) {
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        return -1;
    }

    mlx_array alpha_arr = mlx_array_new_float(alpha);
    mlx_array outv = mlx_array_new();
    if (mlx_multiply(&outv, mean, alpha_arr, s) != 0) {
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        mlx_array_free(alpha_arr);
        mlx_array_free(outv);
        return -1;
    }

    if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(outv);
        return -1;
    }
    **out_loss = outv;

    mlx_array_free(diff);
    mlx_array_free(sq);
    mlx_array_free(mean);
    mlx_array_free(alpha_arr);
    return 0;
}


/* Evaluate all model parameters (parity with Python mx.eval(self.model.state)).
 * This forces lazy computation graphs to materialize and releases intermediate
 * arrays, preventing memory accumulation during training.
 * Uses batched mlx_eval() to match Python's mx.eval([generator.state, discriminator.state]).
 */
int mlx_faciesgan_eval_all_parameters(MLXFaciesGAN *m) {
    if (!m)
        return -1;

    /* Collect all parameters into a single vector for batched eval */
    mlx_vector_array param_vec = mlx_vector_array_new();

    /* Collect all generator parameters */
    if (m->generator) {
        int gen_n = 0;
        mlx_array **gen_params = mlx_generator_get_parameters(m->generator, &gen_n);
        if (gen_params && gen_n > 0) {
            for (int i = 0; i < gen_n; ++i) {
                if (gen_params[i] && gen_params[i]->ctx) {
                    mlx_vector_array_append_value(param_vec, *gen_params[i]);
                }
            }
            mlx_generator_free_parameters_list(gen_params);
        }
    }

    /* Collect all discriminator parameters */
    if (m->discriminator) {
        int disc_n = 0;
        mlx_array **disc_params = mlx_discriminator_get_parameters(m->discriminator, &disc_n);
        if (disc_params && disc_n > 0) {
            for (int i = 0; i < disc_n; ++i) {
                if (disc_params[i] && disc_params[i]->ctx) {
                    mlx_vector_array_append_value(param_vec, *disc_params[i]);
                }
            }
            mlx_discriminator_free_parameters_list(disc_params);
        }
    }

    /* Batch eval all parameters at once */
    mlx_eval(param_vec);
    mlx_vector_array_free(param_vec);

    return 0;
}

/* Append all model parameters (gen + disc) to an external vector_array
 * for deferred batch evaluation.  This is the non-eval counterpart of
 * mlx_faciesgan_eval_all_parameters, used when the caller builds a
 * combined eval vector (model params + optimizer state + metrics). */
int mlx_faciesgan_append_params_to_vec(MLXFaciesGAN *m, mlx_vector_array vec) {
    if (!m || !vec.ctx)
        return -1;

    if (m->generator) {
        int gen_n = 0;
        mlx_array **gen_params = mlx_generator_get_parameters(m->generator, &gen_n);
        if (gen_params && gen_n > 0) {
            for (int i = 0; i < gen_n; ++i) {
                if (gen_params[i] && gen_params[i]->ctx)
                    mlx_vector_array_append_value(vec, *gen_params[i]);
            }
            mlx_generator_free_parameters_list(gen_params);
        }
    }

    if (m->discriminator) {
        int disc_n = 0;
        mlx_array **disc_params = mlx_discriminator_get_parameters(m->discriminator, &disc_n);
        if (disc_params && disc_n > 0) {
            for (int i = 0; i < disc_n; ++i) {
                if (disc_params[i] && disc_params[i]->ctx)
                    mlx_vector_array_append_value(vec, *disc_params[i]);
            }
            mlx_discriminator_free_parameters_list(disc_params);
        }
    }

    return 0;
}
