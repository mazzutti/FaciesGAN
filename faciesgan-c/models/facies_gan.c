#include "facies_gan.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
/* Standard string/memory functions provided by <string.h> */
#include "custom_layer.h"
#include "trainning/array_helpers.h"
#include <limits.h>
#include <mlx/c/io.h>
#include <mlx/c/map.h>
#include <trainning/autodiff.h>
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
    return m;
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

    /* Use CPU stream for numeric ops */
    mlx_stream s = mlx_default_cpu_stream_new();

    /* Determine element count from first sample */
    mlx_array *first = fake_samples[0];
    if (!first) {
        mlx_stream_free(s);
        return -1;
    }
    mlx_array_eval(*first);
    size_t total_elems = (size_t)mlx_array_size(*first);
    if (total_elems == 0) {
        mlx_stream_free(s);
        *out_loss = NULL;
        if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0)
            return -1;
        **out_loss = mlx_array_new_float(0.0f);
        return 0;
    }

    /* Accumulate sum of exp(-10 * mean_sq) over pairs */
    mlx_array acc = mlx_array_new_float(0.0f);
    int pairs = 0;
    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            if (!fake_samples[i] || !fake_samples[j])
                continue;
            mlx_array diff = mlx_array_new();
            if (mlx_subtract(&diff, *fake_samples[i], *fake_samples[j], s) != 0) {
                mlx_array_free(diff);
                continue;
            }
            mlx_array sq = mlx_array_new();
            if (mlx_square(&sq, diff, s) != 0) {
                mlx_array_free(diff);
                mlx_array_free(sq);
                continue;
            }
            /* mean over all axes */
            int ndim = (int)mlx_array_ndim(diff);
            int *axes = NULL;
            if (ndim > 0) {
                if (mlx_alloc_int_array(&axes, ndim) != 0) {
                    mlx_array_free(diff);
                    mlx_array_free(sq);
                    continue;
                }
                for (int a = 0; a < ndim; ++a)
                    axes[a] = a;
            }
            mlx_array mean = mlx_array_new();
            if (mlx_mean_axes(&mean, sq, axes, ndim, true, s) != 0) {
                mlx_free_int_array(&axes, &ndim);
                mlx_array_free(diff);
                mlx_array_free(sq);
                mlx_array_free(mean);
                continue;
            }
            mlx_free_int_array(&axes, &ndim);

            /* approximate exp(-10 * mean_sq) with 1 / (1 + 10 * mean_sq) */
            mlx_array coef = mlx_array_new_float(10.0f);
            mlx_array prod = mlx_array_new();
            if (mlx_multiply(&prod, mean, coef, s) != 0) {
                mlx_array_free(diff);
                mlx_array_free(sq);
                mlx_array_free(mean);
                mlx_array_free(coef);
                mlx_array_free(prod);
                continue;
            }
            mlx_array one = mlx_array_new_float(1.0f);
            mlx_array den = mlx_array_new();
            if (mlx_add(&den, one, prod, s) != 0) {
                mlx_array_free(diff);
                mlx_array_free(sq);
                mlx_array_free(mean);
                mlx_array_free(coef);
                mlx_array_free(prod);
                mlx_array_free(one);
                mlx_array_free(den);
                continue;
            }
            mlx_array val = mlx_array_new();
            if (mlx_divide(&val, one, den, s) != 0) {
                mlx_array_free(diff);
                mlx_array_free(sq);
                mlx_array_free(mean);
                mlx_array_free(coef);
                mlx_array_free(prod);
                mlx_array_free(one);
                mlx_array_free(den);
                mlx_array_free(val);
                continue;
            }

            /* accumulate into acc */
            mlx_array tmp = mlx_array_new();
            if (mlx_add(&tmp, acc, val, s) == 0) {
                mlx_array_free(acc);
                acc = tmp;
            } else {
                mlx_array_free(tmp);
            }

            mlx_array_free(coef);
            mlx_array_free(prod);
            mlx_array_free(one);
            mlx_array_free(den);
            mlx_array_free(val);

            mlx_array_free(diff);
            mlx_array_free(sq);
            mlx_array_free(mean);
            pairs++;
        }
    }

    if (pairs == 0) {
        mlx_array_free(acc);
        mlx_stream_free(s);
        *out_loss = NULL;
        if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0)
            return -1;
        **out_loss = mlx_array_new_float(0.0f);
        return 0;
    }

    /* divide acc by number of pairs, multiply by lambda_diversity */
    mlx_array denom = mlx_array_new_float((float)pairs);
    mlx_array mean_acc = mlx_array_new();
    int rc = mlx_divide(&mean_acc, acc, denom, s);
    if (rc != 0) {
        mlx_array_free(acc);
        mlx_array_free(denom);
        mlx_stream_free(s);
        return -1;
    }
    mlx_array_free(acc);
    mlx_array_free(denom);

    mlx_array lambda_arr = mlx_array_new_float(lambda_diversity);
    mlx_array outv = mlx_array_new();
    if (mlx_multiply(&outv, mean_acc, lambda_arr, s) != 0) {
        mlx_array_free(mean_acc);
        mlx_array_free(lambda_arr);
        mlx_stream_free(s);
        return -1;
    }
    mlx_array_free(mean_acc);
    mlx_array_free(lambda_arr);

    *out_loss = NULL;
    if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(outv);
        mlx_stream_free(s);
        return -1;
    }
    **out_loss = outv;
    mlx_stream_free(s);
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

    mlx_stream s = mlx_default_cpu_stream_new();

    mlx_array fmasked = mlx_array_new();
    mlx_array rmasked = mlx_array_new();
    if (mlx_multiply(&fmasked, *fake, *mask, s) != 0 ||
            mlx_multiply(&rmasked, *real, *mask, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_stream_free(s);
        return -1;
    }

    mlx_array diff = mlx_array_new();
    if (mlx_subtract(&diff, fmasked, rmasked, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_stream_free(s);
        return -1;
    }

    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, diff, s) != 0) {
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_stream_free(s);
        return -1;
    }

    int ndim = (int)mlx_array_ndim(diff);
    int *axes = NULL;
    if (ndim > 0) {
        if (mlx_alloc_int_array(&axes, ndim) != 0) {
            mlx_array_free(fmasked);
            mlx_array_free(rmasked);
            mlx_array_free(diff);
            mlx_array_free(sq);
            mlx_stream_free(s);
            return -1;
        }
        for (int a = 0; a < ndim; ++a)
            axes[a] = a;
    }

    mlx_array mean = mlx_array_new();
    if (mlx_mean_axes(&mean, sq, axes, ndim, true, s) != 0) {
        mlx_free_int_array(&axes, &ndim);
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        mlx_stream_free(s);
        return -1;
    }
    mlx_free_int_array(&axes, &ndim);

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
        mlx_stream_free(s);
        return -1;
    }

    *out_loss = NULL;
    if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(outv);
        mlx_stream_free(s);
        mlx_array_free(fmasked);
        mlx_array_free(rmasked);
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        mlx_array_free(lambda_arr);
        mlx_stream_free(s);
        return -1;
    }
    **out_loss = outv;
    return 0;
}

void mlx_faciesgan_free(MLXFaciesGAN *m) {
    if (!m)
        return;
    if (m->shapes)
        mlx_free_int_array(&m->shapes, &m->n_shapes);
    if (m->noise_amps)
        mlx_free_float_buf(&m->noise_amps, NULL);
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

AGValue *mlx_faciesgan_generator_forward_ag(MLXFaciesGAN *m, AGValue **z_list,
        int z_count, const float *amp,
        int amp_count, AGValue *in_noise,
        int start_scale, int stop_scale) {
    if (!m)
        return NULL;
    int n_gens = mlx_generator_get_n_gens(m->generator);
    if (n_gens == 0)
        return NULL;
    if (start_scale < 0)
        start_scale = 0;
    if (stop_scale < 0)
        stop_scale = n_gens - 1;
    if (stop_scale >= n_gens)
        stop_scale = n_gens - 1;

    /* Initialize out_facie: prefer provided `in_noise`, else use first z as base.
     */
    AGValue *out_facie = NULL;
    if (in_noise)
        out_facie = in_noise;
    else if (z_count > 0 && z_list)
        out_facie = z_list[0];
    if (!out_facie)
        return NULL;

    /* Progressive multi-scale loop: use AG ops to mirror numeric forward. */
    int zero_padding = m->num_layer * (m->kernel_size / 2);
    int full_zero_padding = 2 * zero_padding;
    for (int index = start_scale; index <= stop_scale; ++index) {
        /* If no corresponding z provided, stop progressing. */
        if (!z_list || index >= z_count)
            break;

        mlx_array *zarr = ag_value_array(z_list[index]);
        if (!zarr)
            break;
        const int *zshape = mlx_array_shape(*zarr);
        if (mlx_array_ndim(*zarr) != 4)
            break;

        int target_h = zshape[1] - full_zero_padding;
        int target_w = zshape[2] - full_zero_padding;

        AGValue *ups = ag_upsample(out_facie, target_h, target_w, "linear", 1);
        ag_register_temp_value(ups);

        AGValue *z_in = z_list[index];

        /* Optionally scale noise channel by amp */
        if (amp && amp_count > index && z_in) {
            AGValue *scale = ag_scalar_float(amp[index]);
            ag_register_temp_value(scale);
            AGValue *scaled = ag_mul(z_in, scale);
            ag_register_temp_value(scaled);
            z_in = scaled;
        }

        /* Pad z_in to match upsampled spatial dims and add */
        int axes[2] = {1, 2};
        int low_pad[2] = {zero_padding, zero_padding};
        int high_pad[2] = {zero_padding, zero_padding};
        AGValue *padded =
            ag_pad(ups, axes, 2, low_pad, 2, high_pad, 2, 0.0f, "constant");
        ag_register_temp_value(padded);
        if (z_in) {
            AGValue *zadd = ag_add(z_in, padded);
            ag_register_temp_value(zadd);
            z_in = zadd;
        }

        AGValue *cur = z_in ? z_in : ups;

        /* Head conv */
        MLXConvBlock *head = mlx_scale_get_head(m->generator, index);
        if (head) {
            mlx_array *wh = mlx_convblock_get_conv_weight(head);
            if (wh) {
                AGValue *w_ag = ag_value_from_array(wh, 1);
                ag_register_temp_value(w_ag);
                cur = ag_conv2d(cur, w_ag, 1, 1, 1, 1, 1, 1, 1);
                cur = ag_leaky_relu(cur, 0.2f);
                ag_register_temp_value(cur);
            }
        }

        /* Body convs */
        int body_n = mlx_scale_get_body_count(m->generator, index);
        for (int b = 0; b < body_n; ++b) {
            MLXConvBlock *cb = mlx_scale_get_body_at(m->generator, index, b);
            if (!cb)
                continue;
            mlx_array *wb = mlx_convblock_get_conv_weight(cb);
            if (!wb)
                continue;
            AGValue *wbag = ag_value_from_array(wb, 1);
            ag_register_temp_value(wbag);
            cur = ag_conv2d(cur, wbag, 1, 1, 1, 1, 1, 1, 1);
            cur = ag_leaky_relu(cur, 0.2f);
            ag_register_temp_value(cur);
        }

        /* Tail conv */
        if (mlx_scale_has_tail_conv(m->generator, index)) {
            mlx_array *wt = mlx_scale_get_tail_conv(m->generator, index);
            if (wt) {
                AGValue *wtag = ag_value_from_array(wt, 1);
                ag_register_temp_value(wtag);
                cur = ag_conv2d(cur, wtag, 1, 1, 1, 1, 1, 1, 1);
                ag_register_temp_value(cur);
            }
        }

        AGValue *new_out = ag_add(cur, ups);
        ag_register_temp_value(new_out);
        out_facie = new_out;
    }

    return out_facie;
}

AGValue *mlx_faciesgan_discriminator_forward_ag(MLXFaciesGAN *m, AGValue *input,
        int scale) {
    if (!m || !input)
        return NULL;
    void *disc_ptr = mlx_discriminator_get_disc_ptr(m->discriminator, scale);
    if (!disc_ptr)
        return NULL;
    MLXSPADEDiscriminator *disc = (MLXSPADEDiscriminator *)disc_ptr;

    AGValue *x = input;

    /* Head conv */
    mlx_array *wh = mlx_spadedisc_get_head_conv(disc);
    if (wh) {
        AGValue *w_ag = ag_value_from_array(wh, 1);
        ag_register_temp_value(w_ag);
        x = ag_conv2d(x, w_ag, 1, 1, 1, 1, 1, 1, 1);
        x = ag_leaky_relu(x, 0.2f);
        ag_register_temp_value(x);
    }

    /* Body convs */
    int body_n = mlx_spadedisc_get_body_count(disc);
    for (int b = 0; b < body_n; ++b) {
        MLXSPADEConvBlock *cb = mlx_spadedisc_get_body_at(disc, b);
        if (!cb)
            continue;
        mlx_array *wb = mlx_spadeconv_get_conv_weight(cb);
        if (!wb)
            continue;
        AGValue *wbag = ag_value_from_array(wb, 1);
        ag_register_temp_value(wbag);
        x = ag_conv2d(x, wbag, 1, 1, 1, 1, 1, 1, 1);
        x = ag_leaky_relu(x, 0.2f);
        ag_register_temp_value(x);
    }

    /* Tail conv -> scalar output */
    mlx_array *wt = mlx_spadedisc_get_tail_conv(disc);
    if (wt) {
        AGValue *wtag = ag_value_from_array(wt, 1);
        ag_register_temp_value(wtag);
        x = ag_conv2d(x, wtag, 1, 1, 0, 0, 1, 1, 1);
        ag_register_temp_value(x);
    }

    /* Reduce spatial/channel dims to a scalar (sum over axes). */
    AGValue *s = ag_sum_axis(x, 1, 0);
    ag_register_temp_value(s);
    s = ag_sum_axis(s, 1, 0);
    ag_register_temp_value(s);
    s = ag_sum_axis(s, 1, 0);
    ag_register_temp_value(s);
    return s;
}

AGValue *mlx_faciesgan_compute_gradient_penalty_ag(MLXFaciesGAN *m,
        AGValue *real, AGValue *fake,
        int scale,
        float lambda_grad) {
    if (!m || !real || !fake)
        return NULL;

    /* Create random alpha with shape (batch,1,1,1) using MLX random_uniform,
     * then wrap as AGValue (no grad). For debug / CPU-only smoke runs we allow
     * skipping MLX array eval via the FACIESGAN_SKIP_ARRAY_EVAL env var and
     * fall back to a fixed scalar alpha. */
    /* Sample random alpha with shape (batch,1,1,1) to match Python behavior. */
    AGValue *alpha = NULL;
    {
        mlx_array *rarr = ag_value_array(real);
        if (rarr) {
            const int *rsh = mlx_array_shape(*rarr);
            if (rsh) {
                int batch = rsh[0];
                int shape[4] = {batch, 1, 1, 1};
                mlx_array a = mlx_array_new();
                mlx_array low = mlx_array_new_float(0.0f);
                mlx_array high = mlx_array_new_float(1.0f);
                mlx_stream s2 = mlx_default_cpu_stream_new();
                if (mlx_random_uniform(&a, low, high, shape, 4, MLX_FLOAT32,
                                       mlx_array_empty, s2) == 0) {
                    alpha = ag_value_from_new_array(&a, 0);
                    ag_register_temp_value(alpha);
                } else {
                    mlx_array_free(a);
                }
                mlx_array_free(low);
                mlx_array_free(high);
                mlx_stream_free(s2);
            }
        }
        if (!alpha) {
            /* fallback to scalar 0.5 if sampling failed */
            alpha = ag_scalar_float(0.5f);
            ag_register_temp_value(alpha);
        }
    }

    /* interpolates = alpha * real + (1-alpha) * fake */
    AGValue *one = ag_scalar_float(1.0f);
    ag_register_temp_value(one);
    /* Quick guard: if `real` and `fake` have incompatible spatial shapes,
     * skip gradient penalty for robustness during smoke/debug runs. */
    {
        mlx_array *rchk = ag_value_array(real);
        mlx_array *fchk = ag_value_array(fake);
        if (rchk && fchk) {
            const int *rsh = mlx_array_shape(*rchk);
            const int *fsh = mlx_array_shape(*fchk);
            if (rsh && fsh && (rsh[1] != fsh[1] || rsh[2] != fsh[2])) {
                AGValue *zero = ag_scalar_float(0.0f);
                ag_register_temp_value(zero);
                return zero;
            }
        }
    }
    /* Ensure `real` and `fake` have compatible spatial shapes before mixing.
     * If they differ, pad `real` centrally to match `fake`'s spatial dims. */
    {
        mlx_array *rarr2 = ag_value_array(real);
        mlx_array *farr = ag_value_array(fake);
        if (rarr2 && farr) {
            const int *rsh = mlx_array_shape(*rarr2);
            const int *fsh = mlx_array_shape(*farr);
            if (rsh && fsh && (rsh[1] != fsh[1] || rsh[2] != fsh[2])) {
                int dh = fsh[1] - rsh[1];
                int dw = fsh[2] - rsh[2];
                int low_h = dh > 0 ? dh / 2 : 0;
                int high_h = dh > 0 ? dh - low_h : 0;
                int low_w = dw > 0 ? dw / 2 : 0;
                int high_w = dw > 0 ? dw - low_w : 0;
                int axes_pad[2] = {1, 2};
                int low_pad[2] = {low_h, low_w};
                int high_pad[2] = {high_h, high_w};
                mlx_array pad_zero = mlx_array_new_float(0.0f);
                mlx_array padded = mlx_array_new();
                mlx_stream s2 = mlx_default_cpu_stream_new();
                if (mlx_pad(&padded, *rarr2, axes_pad, 2, low_pad, 2, high_pad, 2,
                            pad_zero, "constant", s2) == 0) {
                    /* wrap padded real as new AGValue and use it instead of original */
                    AGValue *real_p = ag_value_from_new_array(&padded, 0);
                    ag_register_temp_value(real_p);
                    real = real_p;
                } else {
                    mlx_array_free(padded);
                }
                mlx_array_free(pad_zero);
                mlx_stream_free(s2);
            }
        }
    }
    AGValue *inv_alpha = ag_sub(one, alpha);
    ag_register_temp_value(inv_alpha);
    AGValue *term1 = ag_mul(alpha, real);
    ag_register_temp_value(term1);
    AGValue *term2 = ag_mul(inv_alpha, fake);
    ag_register_temp_value(term2);
    AGValue *interpolates = ag_add(term1, term2);
    ag_register_temp_value(interpolates);

    /* Discriminator output for interpolates */
    AGValue *out = mlx_faciesgan_discriminator_forward_ag(m, interpolates, scale);
    if (!out)
        return NULL;
    ag_register_temp_value(out);

    /* Reduce to scalar by summing all axes */
    mlx_array *out_arr = ag_value_array(out);
    if (!out_arr)
        return NULL;
    int out_ndim = (int)mlx_array_ndim(*out_arr);
    AGValue *sval = out;
    for (int ax = 0; ax < out_ndim; ++ax)
        sval = ag_sum_axis(sval, 0, 0);
    ag_register_temp_value(sval);

    /* Build create-graph backward to obtain symbolic gradients of `interpolates`
     */
    if (ag_backward_create_graph(sval) != 0)
        return NULL;

    AGValue *grad_interpolates = ag_value_get_grad_ag(interpolates);
    if (!grad_interpolates)
        return NULL;
    ag_register_temp_value(grad_interpolates);

    /* grad_norm = sqrt(sum(square(grad_interpolates), axis=-1) + 1e-12) */
    AGValue *g_sq = ag_square(grad_interpolates);
    ag_register_temp_value(g_sq);
    mlx_array *garr = ag_value_array(g_sq);
    if (!garr)
        return NULL;
    int g_ndim = (int)mlx_array_ndim(*garr);
    int last_axis = g_ndim - 1;
    AGValue *g_sum = ag_sum_axis(g_sq, last_axis, 0);
    ag_register_temp_value(g_sum);
    AGValue *eps = ag_scalar_float(1e-12f);
    ag_register_temp_value(eps);
    AGValue *g_sum_eps = ag_add(g_sum, eps);
    ag_register_temp_value(g_sum_eps);
    AGValue *g_norm = ag_sqrt(g_sum_eps);
    ag_register_temp_value(g_norm);

    /* penalty = mean(square(g_norm - 1.0)) * lambda_grad */
    AGValue *onef = ag_scalar_float(1.0f);
    ag_register_temp_value(onef);
    AGValue *diff = ag_sub(g_norm, onef);
    ag_register_temp_value(diff);
    AGValue *sq = ag_square(diff);
    ag_register_temp_value(sq);

    /* reduce sq to scalar by summing all axes then divide by number of elements
     */
    mlx_array *sq_arr = ag_value_array(sq);
    if (!sq_arr)
        return NULL;
    int sq_ndim = (int)mlx_array_ndim(*sq_arr);
    AGValue *sum_sq = sq;
    for (int ax = 0; ax < sq_ndim; ++ax)
        sum_sq = ag_sum_axis(sum_sq, 0, 0);
    ag_register_temp_value(sum_sq);

    /* denom = number of elements in sq array (as float) */
    size_t elems = mlx_array_size(*sq_arr);
    AGValue *den = ag_scalar_float((float)elems);
    ag_register_temp_value(den);
    AGValue *mean = ag_divide(sum_sq, den);
    ag_register_temp_value(mean);

    AGValue *lam = ag_scalar_float(lambda_grad);
    ag_register_temp_value(lam);
    AGValue *pen = ag_mul(mean, lam);
    ag_register_temp_value(pen);

    return pen;
}

AGValue *mlx_faciesgan_compute_diversity_loss_ag(MLXFaciesGAN *m,
        AGValue **fake_samples,
        int n_samples,
        float lambda_diversity) {

    if (!fake_samples || n_samples < 2 || lambda_diversity <= 0.0f)
        return ag_scalar_float(0.0f);

    AGValue *acc = NULL;
    int pairs = 0;
    AGValue *one = ag_scalar_float(1.0f);
    ag_register_temp_value(one);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            if (!fake_samples[i] || !fake_samples[j])
                continue;
            AGValue *diff = ag_sub(fake_samples[i], fake_samples[j]);
            ag_register_temp_value(diff);
            AGValue *sq = ag_square(diff);
            ag_register_temp_value(sq);

            /* reduce to scalar sum */
            mlx_array *arr = ag_value_array(sq);
            if (!arr)
                continue;
            int ndim = (int)mlx_array_ndim(*arr);
            AGValue *s = sq;
            for (int ax = 0; ax < ndim; ++ax)
                s = ag_sum_axis(s, 0, 0);
            ag_register_temp_value(s);

            /* mean = s / elems */
            size_t elems = mlx_array_size(*arr);
            AGValue *den = ag_scalar_float((float)elems);
            ag_register_temp_value(den);
            AGValue *mean_sq = ag_divide(s, den);
            ag_register_temp_value(mean_sq);

            /* approximate exp(-10 * mean_sq) with 1 / (1 + 10 * mean_sq) */
            AGValue *coef = ag_scalar_float(10.0f);
            ag_register_temp_value(coef);
            AGValue *prod = ag_mul(mean_sq, coef);
            ag_register_temp_value(prod);
            AGValue *den2 = ag_add(one, prod);
            ag_register_temp_value(den2);
            AGValue *val = ag_divide(one, den2);
            ag_register_temp_value(val);

            if (!acc)
                acc = val;
            else {
                acc = ag_add(acc, val);
                ag_register_temp_value(acc);
            }
            pairs++;
        }
    }

    if (!acc || pairs == 0)
        return ag_scalar_float(0.0f);

    AGValue *pairs_v = ag_scalar_float((float)pairs);
    ag_register_temp_value(pairs_v);
    AGValue *mean = ag_divide(acc, pairs_v);
    ag_register_temp_value(mean);
    AGValue *lam = ag_scalar_float(lambda_diversity);
    ag_register_temp_value(lam);
    AGValue *out = ag_mul(mean, lam);
    ag_register_temp_value(out);
    return out;
}

AGValue *mlx_faciesgan_compute_masked_loss_ag(MLXFaciesGAN *m, AGValue *fake,
        AGValue *real, AGValue *well,
        AGValue *mask,
        float well_loss_penalty) {

    if (!fake || !real)
        return ag_scalar_float(0.0f);

    AGValue *diff = ag_sub(fake, real);
    ag_register_temp_value(diff);
    if (mask) {
        diff = ag_mul(diff, mask);
        ag_register_temp_value(diff);
    }
    AGValue *sq = ag_square(diff);
    ag_register_temp_value(sq);

    mlx_array *arr = ag_value_array(sq);
    if (!arr)
        return ag_scalar_float(0.0f);
    int ndim = (int)mlx_array_ndim(*arr);
    AGValue *s = sq;
    for (int ax = 0; ax < ndim; ++ax)
        s = ag_sum_axis(s, 0, 0);
    ag_register_temp_value(s);

    size_t elems = mlx_array_size(*arr);
    AGValue *den = ag_scalar_float((float)elems);
    ag_register_temp_value(den);
    AGValue *mean = ag_divide(s, den);
    ag_register_temp_value(mean);

    AGValue *lam = ag_scalar_float(well_loss_penalty);
    ag_register_temp_value(lam);
    AGValue *out = ag_mul(mean, lam);
    ag_register_temp_value(out);
    return out;
}

AGValue *mlx_faciesgan_discriminator_forward_ag_with_params(
    MLXFaciesGAN *m, AGValue *input, int scale, AGValue ***out_params,
    int *out_n_params) {
    if (!m || !input) {
        if (out_params)
            *out_params = NULL;
        if (out_n_params)
            *out_n_params = 0;
        return NULL;
    }
    /* Build AG forward (so returned value participates in autodiff). */
    AGValue *out_ag = mlx_faciesgan_discriminator_forward_ag(m, input, scale);

    /* Collect discriminator parameters and wrap as AGValue (requires_grad=1). */
    int n = 0;
    mlx_array **plist = mlx_discriminator_get_parameters(m->discriminator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_discriminator_free_parameters_list(plist);
        if (out_params)
            *out_params = NULL;
        if (out_n_params)
            *out_n_params = 0;
        return out_ag;
    }
    AGValue **params = NULL;
    if (mlx_alloc_ptr_array((void ***)&params, n) != 0) {
        mlx_discriminator_free_parameters_list(plist);
        if (out_params)
            *out_params = NULL;
        if (out_n_params)
            *out_n_params = 0;
        return out_ag;
    }
    for (int i = 0; i < n; ++i) {
        params[i] = ag_value_from_array(plist[i], 1);
        ag_register_temp_value(params[i]);
    }
    mlx_discriminator_free_parameters_list(plist);

    if (out_params)
        *out_params = params;
    else
        mlx_free_ptr_array((void ***)&params, n);
    if (out_n_params)
        *out_n_params = n;
    return out_ag;
}

AGValue *mlx_faciesgan_generator_forward_ag_with_params(
    MLXFaciesGAN *m, AGValue **z_list, int z_count, const float *amp,
    int amp_count, AGValue *in_noise, int start_scale, int stop_scale,
    AGValue ***out_params, int *out_n_params) {
    if (!m) {
        if (out_params)
            *out_params = NULL;
        if (out_n_params)
            *out_n_params = 0;
        return NULL;
    }

    /* Prepare z input arrays (non-owning copies of AGValue arrays). */
    mlx_array *zvals = NULL;
    if (z_count > 0 && z_list) {
        if (mlx_alloc_mlx_array_vals(&zvals, z_count) != 0) {
            if (out_params)
                *out_params = NULL;
            if (out_n_params)
                *out_n_params = 0;
            return NULL;
        }
        for (int i = 0; i < z_count; ++i) {
            mlx_array *a = ag_value_array(z_list[i]);
            if (a)
                zvals[i] = *a;
            else
                zvals[i] = mlx_array_new();
        }
    }

    /* Prepare input noise array (if provided) */
    mlx_array in_arr = mlx_array_new();
    if (in_noise) {
        mlx_array *tmp = ag_value_array(in_noise);
        if (tmp)
            in_arr = *tmp;
    }

    /* Call numeric generator forward and wrap the result as an AGValue. */
    mlx_array_t numeric_out =
        mlx_generator_forward(m->generator, (const mlx_array *)zvals, z_count,
                              amp, amp_count, in_arr, start_scale, stop_scale);

    if (zvals)
        mlx_free_mlx_array_vals(&zvals, z_count);

    AGValue *out_ag = ag_value_from_new_array(&numeric_out, 0);
    ag_register_temp_value(out_ag);

    /* Collect generator parameter AG wrappers (requires_grad=1). */
    int n = 0;
    mlx_array **plist = mlx_generator_get_parameters(m->generator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_generator_free_parameters_list(plist);
        if (out_params)
            *out_params = NULL;
        if (out_n_params)
            *out_n_params = 0;
        return out_ag;
    }

    AGValue **params = NULL;
    if (mlx_alloc_ptr_array((void ***)&params, n) != 0) {
        mlx_generator_free_parameters_list(plist);
        if (out_params)
            *out_params = NULL;
        if (out_n_params)
            *out_n_params = 0;
        return out_ag;
    }
    for (int i = 0; i < n; ++i) {
        params[i] = ag_value_from_array(plist[i], 1);
        ag_register_temp_value(params[i]);
    }
    mlx_generator_free_parameters_list(plist);

    if (out_params)
        *out_params = params;
    else
        mlx_free_ptr_array((void ***)&params, n);
    if (out_n_params)
        *out_n_params = n;
    return out_ag;
}

int mlx_faciesgan_optimize_generator_from_ag(MLXFaciesGAN *m, MLXOptimizer *opt,
        AGValue **params, int n) {
    if (!m || !opt)
        return -1;
    if (!params || n <= 0)
        return -1;
    mlx_array **grads = NULL;
    if (ag_collect_grads(params, n, &grads) != 0)
        return -1;
    int res = mlx_faciesgan_apply_sgd_to_generator(m, opt, grads, n);
    for (int i = 0; i < n; ++i) {
        if (grads[i]) {
            mlx_array_free(*grads[i]);
            mlx_free_pod((void **)&grads[i]);
        }
    }
    mlx_free_mlx_array_ptrs(&grads, n);
    return res;
}
int mlx_faciesgan_optimize_discriminator_from_ag(MLXFaciesGAN *m,
        MLXOptimizer *opt,
        AGValue **params, int n) {
    if (!m || !opt)
        return -1;
    if (!params || n <= 0)
        return -1;
    mlx_array **grads = NULL;
    if (ag_collect_grads(params, n, &grads) != 0)
        return -1;
    int res = mlx_faciesgan_apply_sgd_to_discriminator(m, opt, grads, n);
    for (int i = 0; i < n; ++i) {
        if (grads[i]) {
            mlx_array_free(*grads[i]);
            mlx_free_pod((void **)&grads[i]);
        }
    }
    mlx_free_mlx_array_ptrs(&grads, n);
    return res;
}
int mlx_faciesgan_train_step_from_ag(MLXFaciesGAN *m, MLXOptimizer *opt_g,
                                     AGValue **gen_params, int gen_n,
                                     MLXOptimizer *opt_d, AGValue **disc_params,
                                     int disc_n) {
    if (!m)
        return -1;
    int rg = 0, rd = 0;
    if (opt_g)
        rg = mlx_faciesgan_optimize_generator_from_ag(m, opt_g, gen_params, gen_n);
    if (opt_d)
        rd = mlx_faciesgan_optimize_discriminator_from_ag(m, opt_d, disc_params,
             disc_n);
    if ((opt_g && rg != 0) || (opt_d && rd != 0))
        return -1;
    return 0;
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
    /* unused parameters in this helper; no-op */
    if (!m || !out_noises || !out_n)
        return -1;
    if (scale < 0)
        return -1;

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
        int cond_present = (indexes && n_indexes > 0 &&
                            ((wells_pyramid && wells_pyramid[i]) ||
                             (seismic_pyramid && seismic_pyramid[i])))
                           ? 1
                           : 0;
        int c = 0;
        if (m->shapes && m->n_shapes > i) {
            const int *s0 = &m->shapes[i * 4];
            batch = s0[0];
            h = s0[1];
            w = s0[2];
            /* If conditioning is present, use stored base_channel (shapes' channel)
               otherwise use generator input channels to match Python semantics. */
            c = cond_present ? s0[3] : m->gen_input_channels;
        } else {
            /* Fallback defaults: base_channel vs gen_input_channels */
            c = cond_present ? m->base_channel : m->gen_input_channels;
        }
        mlx_stream s = mlx_default_cpu_stream_new();
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
        int shape[4] = {batch, h, w, c};
        /* base noise */
        if (mlx_random_normal(a, shape, 4, MLX_FLOAT32, 0.0f, 1.0f, mlx_array_empty,
                              s) != 0) {
            /* fallback: create zeros */
            if (mlx_zeros(a, shape, 4, MLX_FLOAT32, s) != 0) {
                mlx_free_pod((void **)&a);
                for (int j = 0; j < n; ++j) {
                    if (arr[j]) {
                        mlx_array_free(*arr[j]);
                        mlx_free_pod((void **)&arr[j]);
                    }
                }
                mlx_free_mlx_array_ptrs(&arr, n);
                mlx_stream_free(s);
                return -1;
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
            int wells_ndim = (int)mlx_array_ndim(*wells_pyramid[i]);
            int seismic_ndim = (int)mlx_array_ndim(*seismic_pyramid[i]);
            /* Build slice sizes for wells and seismic (axes after axis=0) */
            int *wells_slice = NULL;
            int wells_slice_num = 0;
            if (wells_ndim > 1) {
                wells_slice_num = wells_ndim - 1;
                if ((size_t)wells_slice_num > (size_t)INT_MAX) {
                    wells_slice = (int *)malloc(sizeof(int) * (size_t)wells_slice_num);
                } else {
                    if (mlx_alloc_int_array(&wells_slice, wells_slice_num) != 0)
                        wells_slice = NULL;
                }
                const int *w_sh = mlx_array_shape(*wells_pyramid[i]);
                for (int si = 0; si < wells_slice_num; ++si)
                    wells_slice[si] = w_sh[si + 1];
            }
            int *seis_slice = NULL;
            int seis_slice_num = 0;
            if (seismic_ndim > 1) {
                seis_slice_num = seismic_ndim - 1;
                if ((size_t)seis_slice_num > (size_t)INT_MAX) {
                    seis_slice = (int *)malloc(sizeof(int) * (size_t)seis_slice_num);
                } else {
                    if (mlx_alloc_int_array(&seis_slice, seis_slice_num) != 0)
                        seis_slice = NULL;
                }
                const int *z_sh = mlx_array_shape(*seismic_pyramid[i]);
                for (int si = 0; si < seis_slice_num; ++si)
                    seis_slice[si] = z_sh[si + 1];
            }
            if (wells_ndim > 0 && seismic_ndim > 0 &&
                    mlx_gather_single(&wells_sel, *wells_pyramid[i], idx, 0, wells_slice,
                                      wells_slice_num, s) == 0 &&
                    mlx_gather_single(&seismic_sel, *seismic_pyramid[i], idx, 0,
                                      seis_slice, seis_slice_num, s) == 0) {
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
            if (wells_slice)
                mlx_free_int_array(&wells_slice, &wells_slice_num);
            if (seis_slice)
                mlx_free_int_array(&seis_slice, &seis_slice_num);
            mlx_array_free(idx);
        } else if (indexes && n_indexes > 0 && wells_pyramid && wells_pyramid[i]) {
            /* build indices array */
            int idx_shape[1] = {n_indexes};
            mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
            mlx_array cond_sel = mlx_array_new();
            int cond_ndim = (int)mlx_array_ndim(*wells_pyramid[i]);
            int *slice_sizes = NULL;
            int slice_num = 0;
            if (cond_ndim > 0) {
                /* pass full shape (all dimensions) to mlx_gather_single */
                slice_num = cond_ndim;
                if ((size_t)slice_num > (size_t)INT_MAX) {
                    slice_sizes = (int *)malloc(sizeof(int) * (size_t)slice_num);
                } else {
                    if (mlx_alloc_int_array(&slice_sizes, slice_num) != 0)
                        slice_sizes = NULL;
                }
                const int *sh = mlx_array_shape(*wells_pyramid[i]);
                for (int si = 0; si < slice_num; ++si)
                    slice_sizes[si] = sh[si];
            }
            if (cond_ndim > 0 &&
                    mlx_gather_single(&cond_sel, *wells_pyramid[i], idx, 0, slice_sizes,
                                      slice_num, s) == 0) {
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
                /* failed to gather: ignore and continue with base noise */
                mlx_array_free(cond_sel);
            }
            mlx_array_free(idx);
            if (slice_sizes)
                mlx_free_int_array(&slice_sizes, &slice_num);
        } else if (indexes && n_indexes > 0 && seismic_pyramid &&
                   seismic_pyramid[i]) {
            int idx_shape[1] = {n_indexes};
            mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
            mlx_array cond_sel = mlx_array_new();
            int cond_ndim = (int)mlx_array_ndim(*seismic_pyramid[i]);
            int *slice_sizes2 = NULL;
            int slice_num2 = 0;
            if (cond_ndim > 0) {
                slice_num2 = cond_ndim;
                if ((size_t)slice_num2 > (size_t)INT_MAX) {
                    slice_sizes2 = (int *)malloc(sizeof(int) * (size_t)slice_num2);
                } else {
                    if (mlx_alloc_int_array(&slice_sizes2, slice_num2) != 0)
                        slice_sizes2 = NULL;
                }
                const int *sh2 = mlx_array_shape(*seismic_pyramid[i]);
                for (int si = 0; si < slice_num2; ++si)
                    slice_sizes2[si] = sh2[si];
            }
            if (cond_ndim > 0 &&
                    mlx_gather_single(&cond_sel, *seismic_pyramid[i], idx, 0,
                                      slice_sizes2, slice_num2, s) == 0) {
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
            if (slice_sizes2)
                mlx_free_int_array(&slice_sizes2, &slice_num2);
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

        mlx_stream_free(s);
        arr[i] = a;
    }

    *out_noises = arr;
    *out_n = n;
    return 0;
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

    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_map_string_to_array map;
    mlx_map_string_to_string meta;
    if (mlx_load_safetensors(&map, &meta, file, s) != 0) {
        mlx_stream_free(s);
        return -1;
    }

    int n = 0;
    mlx_array **plist = mlx_generator_get_parameters(m->generator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_generator_free_parameters_list(plist);
        mlx_map_string_to_array_free(map);
        mlx_map_string_to_string_free(meta);
        mlx_stream_free(s);
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
    mlx_stream_free(s);
    return 0;
}

int mlx_faciesgan_load_discriminator_state(MLXFaciesGAN *m,
        const char *scale_path, int scale) {
    if (!m || !scale_path || !m->discriminator)
        return -1;

    char file[1024];
    snprintf(file, sizeof(file), "%s/discriminator.npz", scale_path);

    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_map_string_to_array map;
    mlx_map_string_to_string meta;
    if (mlx_load_safetensors(&map, &meta, file, s) != 0) {
        mlx_stream_free(s);
        return -1;
    }

    int n = 0;
    mlx_array **plist = mlx_discriminator_get_parameters(m->discriminator, &n);
    if (!plist || n == 0) {
        if (plist)
            mlx_discriminator_free_parameters_list(plist);
        mlx_map_string_to_array_free(map);
        mlx_map_string_to_string_free(meta);
        mlx_stream_free(s);
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
    mlx_stream_free(s);
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

    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_map_string_to_array map;
    mlx_map_string_to_string meta;
    if (mlx_load_safetensors(&map, &meta, file, s) != 0) {
        mlx_stream_free(s);
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
                mlx_stream_free(s);
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
    mlx_stream_free(s);
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

int mlx_faciesgan_optimize_discriminator_scales(
    MLXFaciesGAN *m, const int *indexes, int n_indexes,
    MLXOptimizer **optimizers_by_scale, mlx_array **facies_pyramid,
    mlx_array **wells_pyramid, mlx_array **seismic_pyramid,
    const int *active_scales, int n_active_scales) {
    if (!m)
        return -1;
    /* indexes/n_indexes are unused in this C shim */
    if (!optimizers_by_scale || !facies_pyramid)
        return -1;

    for (int i = 0; i < n_active_scales; ++i) {
        int scale = active_scales[i];
        MLXOptimizer *opt = optimizers_by_scale[scale];
        if (!opt)
            continue;

        /* Collect discriminator parameters and wrap as AGValue (requires_grad=1)
           so ag_collect_grads can later extract gradients (even if zero).
           This is a placeholder: full AG-forward wiring should create a real
           computation graph before calling `ag_backward` to populate grads. */
        int n = 0;
        mlx_array **plist = mlx_discriminator_get_parameters(m->discriminator, &n);
        if (!plist || n == 0) {
            if (plist)
                mlx_discriminator_free_parameters_list(plist);
            continue;
        }
        AGValue **params = NULL;
        if (mlx_alloc_ptr_array((void ***)&params, n) != 0) {
            mlx_discriminator_free_parameters_list(plist);
            continue;
        }
        for (int p = 0; p < n; ++p) {
            params[p] = ag_value_from_array(plist[p], 1);
            ag_register_temp_value(params[p]);
        }
        mlx_discriminator_free_parameters_list(plist);

        /* Placeholder backward: create zero scalar and backward to produce zero
         * grads. */
        /* Build real adversarial loss using AG ops.
           - D(real) -> target 1
           - D(fake) -> target 0
           Generator is invoked with `wells_pyramid[scale]` as a simple noise input.
        */
        AGValue *real_in = NULL;
        if (facies_pyramid && facies_pyramid[scale]) {
            real_in = ag_value_from_array(facies_pyramid[scale], 0);
            ag_register_temp_value(real_in);
        }
        AGValue *noise_in = NULL;
        mlx_array **gen_noises = NULL;
        int gen_n_noises = 0;
        int _transferred_noise_idx = -1;
        if (mlx_faciesgan_get_pyramid_noise(
                    m, scale, indexes, n_indexes, &gen_noises, &gen_n_noises,
                    wells_pyramid, seismic_pyramid, 0) == 0) {
            if (gen_n_noises > scale && gen_noises[scale]) {
                /* transfer ownership of the mlx_array into the AGValue wrapper */
                noise_in = ag_value_from_new_array(gen_noises[scale], 0);
                ag_register_temp_value(noise_in);
                _transferred_noise_idx = scale;
            }
            for (int ni = 0; ni < gen_n_noises; ++ni) {
                if (gen_noises[ni]) {
                    if (ni == _transferred_noise_idx) {
                        mlx_free_pod((void **)&gen_noises[ni]);
                    } else {
                        mlx_array_free(*gen_noises[ni]);
                        mlx_free_pod((void **)&gen_noises[ni]);
                    }
                }
            }
            mlx_free_ptr_array((void ***)&gen_noises, gen_n_noises);
        } else if (wells_pyramid && wells_pyramid[scale]) {
            noise_in = ag_value_from_array(wells_pyramid[scale], 0);
            ag_register_temp_value(noise_in);
        }

        AGValue *fake = mlx_faciesgan_generator_forward_ag(m, NULL, 0, NULL, 0,
                        noise_in, scale, scale);
        if (fake)
            ag_register_temp_value(fake);

        AGValue *d_real = NULL;
        AGValue *d_fake = NULL;
        if (real_in) {
            d_real = mlx_faciesgan_discriminator_forward_ag(m, real_in, scale);
            if (d_real)
                ag_register_temp_value(d_real);
        }
        if (fake) {
            d_fake = mlx_faciesgan_discriminator_forward_ag(m, fake, scale);
            if (d_fake)
                ag_register_temp_value(d_fake);
        }

        /* WGAN-GP discriminator loss: E[D(fake)] - E[D(real)] + lambda * GP */
        AGValue *loss = NULL;
        if (d_real && d_fake) {
            /* adv = D(fake) - D(real) */
            AGValue *adv = ag_sub(d_fake, d_real);
            ag_register_temp_value(adv);
            loss = adv;
            ag_register_temp_value(loss);
        } else if (d_fake) {
            loss = d_fake;
            ag_register_temp_value(loss);
        } else if (d_real) {
            /* encourage D(real) to be larger -> negative term */
            AGValue *neg = ag_mul(d_real, ag_scalar_float(-1.0f));
            ag_register_temp_value(neg);
            loss = neg;
            ag_register_temp_value(loss);
        }

        /* Gradient penalty: sample interpolation between real and fake, compute
         * ||grad_x D(x_hat)||_2 and penalize (||.||-1)^2 */
        float gp_lambda = 10.0f;
        if (real_in && fake) {
            /* eps = 0.5 (broadcastable) for simplicity; production should randomize
             * per-sample */
            AGValue *eps = ag_scalar_float(0.5f);
            ag_register_temp_value(eps);
            AGValue *one = ag_scalar_float(1.0f);
            ag_register_temp_value(one);
            AGValue *one_minus = ag_sub(one, eps);
            ag_register_temp_value(one_minus);
            AGValue *part1 = ag_mul(eps, real_in);
            ag_register_temp_value(part1);
            AGValue *part2 = ag_mul(one_minus, fake);
            ag_register_temp_value(part2);
            AGValue *interp = ag_add(part1, part2);
            ag_register_temp_value(interp);

            AGValue *d_interp =
                mlx_faciesgan_discriminator_forward_ag(m, interp, scale);
            if (d_interp)
                ag_register_temp_value(d_interp);

            /* Build create_graph grads: grad of d_interp wrt interp as
             * AGValue->grad_ag */
            if (d_interp) {
                if (mlx_faciesgan_get_use_create_graph_gp()) {
                    ag_backward_create_graph(d_interp);
                    /* interp->grad_ag now holds gradient AGValue */
                    AGValue *g = ag_value_get_grad_ag(interp);
                    if (g) {
                        AGValue *g2 = ag_square(g);
                        ag_register_temp_value(g2);
                        /* sum over H,W,channel axes. Determine shape dims via underlying
                         * array */
                        mlx_array *gar = ag_value_array(g);
                        if (gar) {
                            int ndim = mlx_array_ndim(*gar);
                            AGValue *s = g2;
                            /* reduce axes 3,2,1 if present (NHWC) */
                            for (int ax = ndim - 1; ax >= 1; --ax) {
                                s = ag_sum_axis(s, ax, 1);
                                ag_register_temp_value(s);
                            }
                            AGValue *std = ag_sqrt(s);
                            ag_register_temp_value(std);
                            /* reduce to scalar across batch */
                            AGValue *norm = std;
                            for (int ax = 0; ax < 1; ++ax) { /* reduce batch to scalar */
                                norm = ag_sum_axis(norm, 0, 0);
                                ag_register_temp_value(norm);
                            }
                            AGValue *sub = ag_sub(norm, ag_scalar_float(1.0f));
                            ag_register_temp_value(sub);
                            AGValue *sq = ag_square(sub);
                            ag_register_temp_value(sq);
                            AGValue *gp = ag_mul(sq, ag_scalar_float(gp_lambda));
                            ag_register_temp_value(gp);
                            loss = loss ? ag_add(loss, gp) : gp;
                            ag_register_temp_value(loss);
                        }
                    }
                } else {
                    /* Numeric fallback: compute numeric grads for d_interp, form GP
                     * scalar, add as constant. */
                    ag_backward(d_interp);
                    mlx_array *garr = ag_value_get_grad(interp);
                    if (garr) {
                        mlx_array_eval(*garr);
                        const float *gdata = mlx_array_data_float32(*garr);
                        size_t gcount = (size_t)mlx_array_size(*garr);
                        double sumsq = 0.0;
                        for (size_t ii = 0; ii < gcount; ++ii)
                            sumsq += (double)gdata[ii] * gdata[ii];
                        double norm = sqrt(sumsq);
                        double gp_num = (norm - 1.0) * (norm - 1.0) * (double)gp_lambda;
                        AGValue *gp_const = ag_scalar_float((float)gp_num);
                        ag_register_temp_value(gp_const);
                        loss = loss ? ag_add(loss, gp_const) : gp_const;
                        ag_register_temp_value(loss);
                    }
                    /* clear numeric grads produced by this temporary backward so later
                     * ag_backward(loss) is clean */
                    ag_zero_grad_all();
                }
            }
        }

        /* L2 regularization (weight decay) over discriminator params */
        float l2_lambda = 1e-4f;
        AGValue *l2_sum = NULL;
        for (int pi = 0; pi < n; ++pi) {
            AGValue *p = params[pi];
            if (!p)
                continue;
            AGValue *psq = ag_square(p);
            ag_register_temp_value(psq);
            /* reduce to scalar */
            mlx_array *arr = ag_value_array(psq);
            int ndim = (int)mlx_array_ndim(*arr);
            AGValue *s = psq;
            for (int ax = 0; ax < ndim; ++ax)
                s = ag_sum_axis(s, 0, 0);
            ag_register_temp_value(s);
            if (!l2_sum)
                l2_sum = s;
            else {
                l2_sum = ag_add(l2_sum, s);
                ag_register_temp_value(l2_sum);
            }
        }
        if (l2_sum) {
            AGValue *lam = ag_scalar_float(l2_lambda);
            ag_register_temp_value(lam);
            AGValue *reg = ag_mul(l2_sum, lam);
            ag_register_temp_value(reg);
            loss = loss ? ag_add(loss, reg) : reg;
            ag_register_temp_value(loss);
        }

        if (loss) {
            ag_backward(loss);
            /* Apply collected grads */
            mlx_faciesgan_optimize_discriminator_from_ag(m, opt, params, n);
            /* reset/autodiff cleanup */
            ag_reset_tape();
        }

        /* cleanup param wrappers */
        for (int p = 0; p < n; ++p) {
            ag_value_free(params[p]);
        }
        mlx_free_ptr_array((void ***)&params, n);
    }

    return 0;
}

int mlx_faciesgan_optimize_generator_scales(
    MLXFaciesGAN *m, const int *indexes, int n_indexes,
    MLXOptimizer **optimizers_by_scale, mlx_array **facies_pyramid,
    mlx_array **rec_in_pyramid, mlx_array **wells_pyramid,
    mlx_array **masks_pyramid, mlx_array **seismic_pyramid,
    const int *active_scales, int n_active_scales) {
    if (!m)
        return -1;
    /* indexes/n_indexes not used by this helper shim */
    if (!optimizers_by_scale)
        return -1;

    for (int i = 0; i < n_active_scales; ++i) {
        int scale = active_scales[i];
        MLXOptimizer *opt = optimizers_by_scale[scale];
        if (!opt)
            continue;

        /* Collect generator parameters and wrap as AGValue (requires_grad=1).
           Placeholder backward with zero loss as above. */
        int n = 0;
        mlx_array **plist = mlx_generator_get_parameters(m->generator, &n);
        if (!plist || n == 0) {
            if (plist)
                mlx_generator_free_parameters_list(plist);
            continue;
        }
        AGValue **params = NULL;
        if (mlx_alloc_ptr_array((void ***)&params, n) != 0) {
            mlx_generator_free_parameters_list(plist);
            continue;
        }
        for (int p = 0; p < n; ++p) {
            params[p] = ag_value_from_array(plist[p], 1);
            ag_register_temp_value(params[p]);
        }
        mlx_generator_free_parameters_list(plist);

        /* Build simple generator adversarial loss: want D(fake) -> 1 */
        AGValue *noise_in = NULL;
        mlx_array **gen_noises = NULL;
        int gen_n_noises = 0;
        int _transferred_noise_idx = -1;
        if (mlx_faciesgan_get_pyramid_noise(
                    m, scale, indexes, n_indexes, &gen_noises, &gen_n_noises,
                    wells_pyramid, seismic_pyramid, 0) == 0) {
            if (gen_n_noises > scale && gen_noises[scale]) {
                noise_in = ag_value_from_new_array(gen_noises[scale], 0);
                ag_register_temp_value(noise_in);
                _transferred_noise_idx = scale;
            }
            for (int ni = 0; ni < gen_n_noises; ++ni) {
                if (gen_noises[ni]) {
                    if (ni == _transferred_noise_idx) {
                        mlx_free_pod((void **)&gen_noises[ni]);
                    } else {
                        mlx_array_free(*gen_noises[ni]);
                        mlx_free_pod((void **)&gen_noises[ni]);
                    }
                }
            }
            mlx_free_ptr_array((void ***)&gen_noises, gen_n_noises);
        } else if (wells_pyramid && wells_pyramid[scale]) {
            noise_in = ag_value_from_array(wells_pyramid[scale], 0);
            ag_register_temp_value(noise_in);
        }
        AGValue *fake = mlx_faciesgan_generator_forward_ag(m, NULL, 0, NULL, 0,
                        noise_in, scale, scale);
        if (fake)
            ag_register_temp_value(fake);
        AGValue *d_fake = NULL;
        if (fake) {
            d_fake = mlx_faciesgan_discriminator_forward_ag(m, fake, scale);
            if (d_fake)
                ag_register_temp_value(d_fake);
        }
        AGValue *loss = NULL;
        if (d_fake) {
            AGValue *one = ag_scalar_float(1.0f);
            ag_register_temp_value(one);
            AGValue *err = ag_sub(d_fake, one);
            ag_register_temp_value(err);
            loss = ag_square(err);
            ag_register_temp_value(loss);
        }

        /* Reconstruction loss (if rec_in_pyramid provided). Optionally apply mask.
         */
        if (rec_in_pyramid && rec_in_pyramid[scale]) {
            AGValue *rec = ag_value_from_array(rec_in_pyramid[scale], 0);
            ag_register_temp_value(rec);
            AGValue *diff = NULL;
            if (fake)
                diff = ag_sub(fake, rec);
            else
                diff = ag_sub(ag_value_from_array(rec_in_pyramid[scale], 0), rec);
            ag_register_temp_value(diff);
            if (masks_pyramid && masks_pyramid[scale]) {
                AGValue *mask = ag_value_from_array(masks_pyramid[scale], 0);
                ag_register_temp_value(mask);
                diff = ag_mul(diff, mask);
                ag_register_temp_value(diff);
            }
            AGValue *sq = ag_square(diff);
            ag_register_temp_value(sq);
            /* reduce to scalar */
            mlx_array *arr = ag_value_array(sq);
            int ndim = (int)mlx_array_ndim(*arr);
            AGValue *s = sq;
            for (int ax = 0; ax < ndim; ++ax)
                s = ag_sum_axis(s, 0, 0);
            ag_register_temp_value(s);
            /* weight reconstruction loss */
            AGValue *w = ag_scalar_float(10.0f);
            ag_register_temp_value(w);
            AGValue *rec_loss = ag_mul(s, w);
            ag_register_temp_value(rec_loss);
            loss = loss ? ag_add(loss, rec_loss) : rec_loss;
            ag_register_temp_value(loss);
        }

        /* Numeric diagnostics: compute masked and recovery losses (non-AG) if
           conditioning provided. These are numeric helpers and won't affect
           autodiff; print values for visibility. */
        if (fake && masks_pyramid && masks_pyramid[scale] && rec_in_pyramid &&
                rec_in_pyramid[scale]) {
            mlx_array *num_loss = NULL;
            if (mlx_faciesgan_compute_masked_loss(
                        m, ag_value_array(fake), rec_in_pyramid[scale],
                        (wells_pyramid ? wells_pyramid[scale] : NULL),
                        masks_pyramid[scale], 10.0f, &num_loss) == 0 &&
                    num_loss) {
                mlx_array arr = *num_loss;
                mlx_array_eval(arr);
                float v = 0.0f;
                mlx_array_item_float32(&v, arr);
                /* debug: masked-loss printing removed */
                mlx_array_free(arr);
                mlx_free_pod((void **)&num_loss);
            }
        }

        if (rec_in_pyramid && rec_in_pyramid[scale] && facies_pyramid &&
                facies_pyramid[scale]) {
            mlx_array *rec_loss_num = NULL;
            if (mlx_faciesgan_compute_recovery_loss(
                        m, NULL, 0, scale, *rec_in_pyramid[scale], *facies_pyramid[scale],
                        wells_pyramid, seismic_pyramid, 1.0f, &rec_loss_num) == 0 &&
                    rec_loss_num) {
                mlx_array arr = *rec_loss_num;
                mlx_array_eval(arr);
                float v = 0.0f;
                mlx_array_item_float32(&v, arr);
                /* debug: recovery-loss printing removed */
                mlx_array_free(arr);
                mlx_free_pod((void **)&rec_loss_num);
            }
        }

        if (loss) {
            ag_backward(loss);
            mlx_faciesgan_optimize_generator_from_ag(m, opt, params, n);
            ag_reset_tape();
        }

        for (int p = 0; p < n; ++p) {
            ag_value_free(params[p]);
        }
        mlx_free_ptr_array((void ***)&params, n);
    }

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
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array diff = mlx_array_new();
    if (mlx_subtract(&diff, rec_in, real, s) != 0) {
        mlx_stream_free(s);
        return -1;
    }
    mlx_array sq = mlx_array_new();
    if (mlx_square(&sq, diff, s) != 0) {
        mlx_array_free(diff);
        mlx_stream_free(s);
        return -1;
    }
    int ndim = (int)mlx_array_ndim(diff);
    int *axes = NULL;
    if (ndim > 0) {
        if (mlx_alloc_int_array(&axes, ndim) != 0) {
            mlx_array_free(diff);
            mlx_array_free(sq);
            mlx_stream_free(s);
            return -1;
        }
        for (int a = 0; a < ndim; ++a)
            axes[a] = a;
    }
    mlx_array mean = mlx_array_new();
    if (mlx_mean_axes(&mean, sq, axes, ndim, true, s) != 0) {
        mlx_free_int_array(&axes, &ndim);
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        mlx_stream_free(s);
        return -1;
    }
    mlx_free_int_array(&axes, &ndim);

    mlx_array alpha_arr = mlx_array_new_float(alpha);
    mlx_array outv = mlx_array_new();
    if (mlx_multiply(&outv, mean, alpha_arr, s) != 0) {
        mlx_array_free(diff);
        mlx_array_free(sq);
        mlx_array_free(mean);
        mlx_array_free(alpha_arr);
        mlx_array_free(outv);
        mlx_stream_free(s);
        return -1;
    }

    if (mlx_alloc_pod((void **)out_loss, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(outv);
        mlx_stream_free(s);
        return -1;
    }
    **out_loss = outv;

    mlx_array_free(diff);
    mlx_array_free(sq);
    mlx_array_free(mean);
    mlx_array_free(alpha_arr);
    mlx_stream_free(s);
    return 0;
}

AGValue *mlx_faciesgan_compute_recovery_loss_ag(
    MLXFaciesGAN *m, const int *indexes, int n_indexes, int scale,
    AGValue *rec_in, AGValue *real, mlx_array **wells_pyramid,
    mlx_array **seismic_pyramid, float alpha) {

    /* unused parameters for AG variant of recovery loss */
    if (!rec_in || !real || alpha == 0.0f)
        return ag_scalar_float(0.0f);

    /* debug prints removed */

    /* If `rec_in` has an extra singleton leading dimension (e.g. shape
       (1,1,H,W,C)) but `real` is (1,H,W,C), squeeze that dimension so
       broadcasting works. */
    {
        mlx_array *rarr = ag_value_array(rec_in);
        mlx_array *rearr = ag_value_array(real);
        if (rarr && rearr) {
            int r_nd = (int)mlx_array_ndim(*rarr);
            const int *r_sh = mlx_array_shape(*rarr);
            int s_nd = (int)mlx_array_ndim(*rearr);
            /* handle the pattern discovered in logs: rec_in ndim==5 and
               a singleton second axis */
            if (r_nd == 5 && r_sh && r_sh[1] == 1 && s_nd == 4) {
                mlx_stream _s = mlx_default_cpu_stream_new();
                mlx_array tmp = mlx_array_new();
                int new_shape[4] = {r_sh[0], r_sh[2], r_sh[3], r_sh[4]};
                if (mlx_reshape(&tmp, *rarr, new_shape, 4, _s) == 0) {
                    AGValue *squeezed = ag_value_from_new_array(&tmp, 0);
                    ag_register_temp_value(squeezed);
                    rec_in = squeezed;
                } else {
                    /* reshape failed: free tmp if needed */
                    mlx_array_free(tmp);
                }
                mlx_stream_free(_s);
            }
            /* After optional squeeze, ensure spatial dims match `real` by
               upsampling/downsampling via AG op so gradients flow. */
            mlx_array *rarr2 = ag_value_array(rec_in);
            if (rarr2) {
                const int *rsh2 = mlx_array_shape(*rarr2);
                const int *sh_real = mlx_array_shape(*rearr);
                if (rsh2 && sh_real) {
                    int rh = rsh2[1];
                    int rw = rsh2[2];
                    int th = sh_real[1];
                    int tw = sh_real[2];
                    if (rh != th || rw != tw) {
                        /* Avoid attempting to downsample via ag_upsample (unsupported).
                         * Instead, upsample the smaller of the two operands so that
                         * shapes match. This preserves gradients and avoids calling
                         * mlx_upsample_forward with a smaller target than the source. */
                        if (rh < th || rw < tw) {
                            AGValue *ups = ag_upsample(rec_in, th, tw, "linear", 1);
                            if (ups) {
                                ag_register_temp_value(ups);
                                rec_in = ups;
                            }
                        } else {
                            AGValue *ups_real = ag_upsample(real, rh, rw, "linear", 1);
                            if (ups_real) {
                                ag_register_temp_value(ups_real);
                                real = ups_real;
                            }
                        }
                    }
                }
            }
        }
    }

    AGValue *diff = ag_sub(rec_in, real);
    ag_register_temp_value(diff);
    AGValue *sq = ag_square(diff);
    ag_register_temp_value(sq);

    mlx_array *arr = ag_value_array(sq);
    if (!arr)
        return ag_scalar_float(0.0f);
    int ndim = (int)mlx_array_ndim(*arr);
    AGValue *s = sq;
    for (int ax = 0; ax < ndim; ++ax)
        s = ag_sum_axis(s, 0, 0);
    ag_register_temp_value(s);

    size_t elems = mlx_array_size(*arr);
    AGValue *den = ag_scalar_float((float)elems);
    ag_register_temp_value(den);
    AGValue *mean = ag_divide(s, den);
    ag_register_temp_value(mean);

    AGValue *a = ag_scalar_float(alpha);
    ag_register_temp_value(a);
    AGValue *out = ag_mul(mean, a);
    ag_register_temp_value(out);
    return out;
}
