#include "train_step.h"
#include "array_helpers.h"
#include "autodiff.h"
#include "base_manager.h"
#include "discriminator.h"
#include "generator.h"
#include "optimizer.h"
#include "train_utils.h"
#include <stdlib.h>

/* Wrapper for mlx_array_eval. Always evaluates to force computation and
   release resources. This is critical for memory management in MLX - without
   eval, the computation graph keeps growing and Metal resources accumulate.
*/
static int safe_mlx_array_eval(mlx_array a) {
    return mlx_array_eval(a);
}

int mlx_faciesgan_apply_sgd_to_generator(MLXFaciesGAN *m, MLXOptimizer *opt,
        mlx_array **grads, int n) {
    if (!m || !opt)
        return -1;
    MLXGenerator *g = mlx_faciesgan_build_generator(m);
    if (!g)
        return -1;
    int param_n = 0;
    mlx_array **params = mlx_generator_get_parameters(g, &param_n);
    if (!params || param_n == 0) {
        if (params)
            mlx_generator_free_parameters_list(params);
        return -1;
    }
    if (n != param_n) {
        mlx_generator_free_parameters_list(params);
        return -1;
    }
    int r = mlx_adam_step(opt, params, grads, param_n);
    mlx_generator_free_parameters_list(params);
    return r;
}

int mlx_faciesgan_apply_sgd_to_discriminator(MLXFaciesGAN *m, MLXOptimizer *opt,
        mlx_array **grads, int n) {
    if (!m || !opt)
        return -1;
    MLXDiscriminator *d = mlx_faciesgan_build_discriminator(m);
    if (!d)
        return -1;
    int param_n = 0;
    mlx_array **params = mlx_discriminator_get_parameters(d, &param_n);
    if (!params || param_n == 0) {
        if (params)
            mlx_discriminator_free_parameters_list(params);
        return -1;
    }
    if (n != param_n) {
        mlx_discriminator_free_parameters_list(params);
        return -1;
    }
    int r = mlx_adam_step(opt, params, grads, param_n);
    mlx_discriminator_free_parameters_list(params);
    return r;
}

int mlx_faciesgan_train_step(MLXFaciesGAN *m, MLXOptimizer *opt_g,
                             mlx_array **gen_grads, int gen_n,
                             MLXOptimizer *opt_d, mlx_array **disc_grads,
                             int disc_n) {
    if (!m)
        return -1;
    int rg = 0, rd = 0;
    if (opt_g)
        rg = mlx_faciesgan_apply_sgd_to_generator(m, opt_g, gen_grads, gen_n);
    if (opt_d)
        rd = mlx_faciesgan_apply_sgd_to_discriminator(m, opt_d, disc_grads, disc_n);
    if ((opt_g && rg != 0) || (opt_d && rd != 0))
        return -1;
    return 0;
}

int mlx_base_apply_sgd_to_generator(MLXBaseManager *mgr, MLXOptimizer *opt,
                                    mlx_array **grads, int n) {
    if (!mgr)
        return -1;
    MLXFaciesGAN *fg = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
    if (!fg)
        return -1;
    return mlx_faciesgan_apply_sgd_to_generator(fg, opt, grads, n);
}

int mlx_base_apply_sgd_to_discriminator(MLXBaseManager *mgr, MLXOptimizer *opt,
                                        mlx_array **grads, int n) {
    if (!mgr)
        return -1;
    MLXFaciesGAN *fg = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
    if (!fg)
        return -1;
    return mlx_faciesgan_apply_sgd_to_discriminator(fg, opt, grads, n);
}

int mlx_base_train_step(MLXBaseManager *mgr, MLXOptimizer *opt_g,
                        mlx_array **gen_grads, int gen_n, MLXOptimizer *opt_d,
                        mlx_array **disc_grads, int disc_n) {
    if (!mgr)
        return -1;
    MLXFaciesGAN *fg = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
    if (!fg)
        return -1;
    return mlx_faciesgan_train_step(fg, opt_g, gen_grads, gen_n, opt_d,
                                    disc_grads, disc_n);
}

int mlx_base_apply_sgd_to_generator_from_ag(MLXBaseManager *mgr,
        MLXOptimizer *opt, AGValue **params,
        int n) {
    if (!mgr || !opt)
        return -1;
    if (!params || n <= 0)
        return -1;
    mlx_array **grads = NULL;
    if (ag_collect_grads(params, n, &grads) != 0)
        return -1;
    int res = mlx_base_apply_sgd_to_generator(mgr, opt, grads, n);
    /* free collected grads */
    for (int i = 0; i < n; ++i) {
        if (grads[i]) {
            mlx_array_free(*grads[i]);
            mlx_free_pod((void **)&grads[i]);
        }
    }
    mlx_free_ptr_array((void ***)&grads, n);
    return res;
}

int mlx_base_apply_sgd_to_discriminator_from_ag(MLXBaseManager *mgr,
        MLXOptimizer *opt,
        AGValue **params, int n) {
    if (!mgr || !opt)
        return -1;
    if (!params || n <= 0)
        return -1;
    mlx_array **grads = NULL;
    if (ag_collect_grads(params, n, &grads) != 0)
        return -1;
    int res = mlx_base_apply_sgd_to_discriminator(mgr, opt, grads, n);
    for (int i = 0; i < n; ++i) {
        if (grads[i]) {
            mlx_array_free(*grads[i]);
            mlx_free_pod((void **)&grads[i]);
        }
    }
    mlx_free_ptr_array((void ***)&grads, n);
    return res;
}

int mlx_base_train_step_from_ag(MLXBaseManager *mgr, MLXOptimizer *opt_g,
                                AGValue **gen_params, int gen_n,
                                MLXOptimizer *opt_d, AGValue **disc_params,
                                int disc_n) {
    if (!mgr)
        return -1;
    int rg = 0, rd = 0;
    if (opt_g)
        rg = mlx_base_apply_sgd_to_generator_from_ag(mgr, opt_g, gen_params, gen_n);
    if (opt_d)
        rd = mlx_base_apply_sgd_to_discriminator_from_ag(mgr, opt_d, disc_params,
             disc_n);
    if ((opt_g && rg != 0) || (opt_d && rd != 0))
        return -1;
    return 0;
}

int mlx_faciesgan_collect_metrics_and_grads(
    MLXFaciesGAN *m, const int *indexes, int n_indexes,
    const int *active_scales, int n_active_scales, mlx_array **facies_pyramid,
    mlx_array **rec_in_pyramid, mlx_array **wells_pyramid,
    mlx_array **masks_pyramid, mlx_array **seismic_pyramid,
    float lambda_diversity, float well_loss_penalty, float alpha,
    float lambda_grad, MLXResults **out_results) {
    if (!m || !out_results)
        return -1;

    MLXResults *res = NULL;
    if (mlx_alloc_pod((void **)&res, sizeof(MLXResults), 1) != 0)
        return -1;
    res->n_scales = n_active_scales;
    res->scales =
        (MLXScaleResults *)calloc(n_active_scales, sizeof(MLXScaleResults));
    if (!res->scales) {
        mlx_free_pod((void **)&res);
        return -1;
    }

    for (int si = 0; si < n_active_scales; ++si) {
        int scale = active_scales[si];
        MLXScaleResults *sr = &res->scales[si];
        sr->scale = scale;
        sr->gen_grads = NULL;
        sr->gen_n = 0;
        sr->disc_grads = NULL;
        sr->disc_n = 0;
        sr->metrics.fake = NULL;
        sr->metrics.well = NULL;
        sr->metrics.div = NULL;
        sr->metrics.rec = NULL;
        sr->metrics.total = NULL;

        if (!facies_pyramid || !facies_pyramid[scale])
            continue;

        /* Prepare AG parameters for generator */
        int gen_param_n = 0;
        mlx_array **gen_params_list = NULL;
        /* get parameters from generator instance */
        MLXGenerator *g = mlx_faciesgan_build_generator(m);
        if (!g)
            continue;
        gen_params_list = mlx_generator_get_parameters(g, &gen_param_n);
        if (!gen_params_list || gen_param_n == 0) {
            if (gen_params_list)
                mlx_generator_free_parameters_list(gen_params_list);
            continue;
        }

        AGValue **gen_params_ag = NULL;
        if (mlx_alloc_ptr_array((void ***)&gen_params_ag, gen_param_n) != 0) {
            mlx_generator_free_parameters_list(gen_params_list);
            continue;
        }
        for (int p = 0; p < gen_param_n; ++p) {
            gen_params_ag[p] = ag_value_from_array(gen_params_list[p], 1);
        }
        mlx_generator_free_parameters_list(gen_params_list);

        /* Build fake via AG using current noise pyramid (all scales up to current)
         * matching Python semantics: generator(get_pyramid_noise(scale,...), ...) */
        mlx_array **gen_noises = NULL;
        int gen_n_noises = 0;
        if (mlx_faciesgan_get_pyramid_noise(
                    m, scale, indexes, n_indexes, &gen_noises, &gen_n_noises,
                    wells_pyramid, seismic_pyramid, 0) != 0) {
            /* Failed to get noise pyramid */
            for (int p = 0; p < gen_param_n; ++p) {
                if (gen_params_ag[p])
                    ag_value_free(gen_params_ag[p]);
            }
            mlx_free_ptr_array((void ***)&gen_params_ag, gen_param_n);
            continue;
        }

        /* Convert noise arrays to AGValues for differentiation */
        AGValue **z_list = NULL;
        int z_count = gen_n_noises;
        if (mlx_alloc_ptr_array((void ***)&z_list, z_count) != 0) {
            for (int ni = 0; ni < gen_n_noises; ++ni) {
                if (gen_noises[ni]) {
                    mlx_array_free(*gen_noises[ni]);
                    mlx_free_pod((void **)&gen_noises[ni]);
                }
            }
            mlx_free_ptr_array((void ***)&gen_noises, gen_n_noises);
            for (int p = 0; p < gen_param_n; ++p) {
                if (gen_params_ag[p])
                    ag_value_free(gen_params_ag[p]);
            }
            mlx_free_ptr_array((void ***)&gen_params_ag, gen_param_n);
            continue;
        }
        for (int ni = 0; ni < z_count; ++ni) {
            if (gen_noises[ni]) {
                /* ag_value_from_new_array makes a copy; must free original */
                z_list[ni] = ag_value_from_new_array(gen_noises[ni], 0);
                ag_register_temp_value(z_list[ni]);
                /* Free the original mlx_array after copy was made */
                mlx_array_free(*gen_noises[ni]);
                mlx_free_pod((void **)&gen_noises[ni]);
            } else {
                z_list[ni] = NULL;
            }
        }
        /* Free the container array */
        mlx_free_ptr_array((void ***)&gen_noises, gen_n_noises);

        /* Build amplitude array: use noise_amps[0..scale-1] + 1.0 for current scale */
        float *amp = NULL;
        int amp_count = scale + 1;
        if (mlx_alloc_float_buf(&amp, amp_count) == 0) {
            float *noise_amps = NULL;
            int n_amps = 0;
            mlx_faciesgan_get_noise_amps(m, &noise_amps, &n_amps);
            for (int ai = 0; ai < amp_count; ++ai) {
                if (ai < n_amps && noise_amps)
                    amp[ai] = noise_amps[ai];
                else
                    amp[ai] = 1.0f;  /* default amplitude for current scale */
            }
            if (noise_amps)
                mlx_free_float_buf(&noise_amps, &n_amps);
        }

        AGValue *fake = mlx_faciesgan_generator_forward_ag(m, z_list, z_count,
                        amp, amp_count, NULL, 0, scale);

        /* Free amplitude array */
        if (amp) {
            mlx_free_float_buf(&amp, &amp_count);
        }
        /* Free z_list container (elements are registered temps, freed by ag_reset_tape) */
        if (z_list) {
            mlx_free_ptr_array((void ***)&z_list, z_count);
        }

        AGValue *real = ag_value_from_array(facies_pyramid[scale], 0);

        /* Discriminator forward and loss */
        AGValue *d_real = mlx_faciesgan_discriminator_forward_ag(m, real, scale);
        AGValue *d_fake = NULL;
        if (fake) {
            d_fake = mlx_faciesgan_discriminator_forward_ag(m, fake, scale);
        }

        AGValue *disc_loss = NULL;
        if (d_real && d_fake) {
            /* Python does: metrics.real = -outputs["d_real"].mean()
                            metrics.fake = outputs["d_fake"].mean()
                            return metrics.real + metrics.fake + metrics.gp
               i.e. -mean(d_real) + mean(d_fake) + gp */
            AGValue *mean_real = ag_mean(d_real);
            ag_register_temp_value(mean_real);
            AGValue *mean_fake = ag_mean(d_fake);
            ag_register_temp_value(mean_fake);
            /* -mean_real + mean_fake */
            AGValue *neg_one = ag_scalar_float(-1.0f);
            ag_register_temp_value(neg_one);
            AGValue *neg_real = ag_mul(neg_one, mean_real);
            ag_register_temp_value(neg_real);
            disc_loss = ag_add(neg_real, mean_fake);
            ag_register_temp_value(disc_loss);
        }

        /* Add gradient penalty */
        if (d_real && d_fake) {
            AGValue *gp = mlx_faciesgan_compute_gradient_penalty_ag(
                              m, real, fake, scale, lambda_grad);
            if (gp) {
                ag_register_temp_value(gp);
                disc_loss = disc_loss ? ag_add(disc_loss, gp) : gp;
                ag_register_temp_value(disc_loss);
            }
        }

        /* Backprop discriminator loss to collect discriminator grads */
        if (disc_loss) {
            ag_backward(disc_loss);
            /* collect discriminator params and grads */
            int disc_param_n = 0;
            mlx_array **disc_params_list = NULL;
            MLXDiscriminator *d = mlx_faciesgan_build_discriminator(m);
            if (d) {
                disc_params_list = mlx_discriminator_get_parameters(d, &disc_param_n);
            }
            if (disc_params_list && disc_param_n > 0) {
                AGValue **disc_params_ag = NULL;
                if (mlx_alloc_ptr_array((void ***)&disc_params_ag, disc_param_n) == 0) {
                    for (int p = 0; p < disc_param_n; ++p) {
                        disc_params_ag[p] = ag_value_from_array(disc_params_list[p], 1);
                    }
                }
                mlx_discriminator_free_parameters_list(disc_params_list);
                mlx_array **disc_grads = NULL;
                if (ag_collect_grads(disc_params_ag, disc_param_n, &disc_grads) == 0) {
                    sr->disc_grads = disc_grads;
                    sr->disc_n = disc_param_n;
                }
                for (int p = 0; p < disc_param_n; ++p)
                    ag_value_free(disc_params_ag[p]);
                mlx_free_ptr_array((void ***)&disc_params_ag, disc_param_n);
            }
        }

        /* Generator loss components: keep individual AGValue pointers so
           we can both build the total loss and record per-component metrics. */
        AGValue *gen_loss = NULL;
        AGValue *adv_comp = NULL;
        AGValue *masked_comp = NULL;
        AGValue *rec_comp = NULL;
        AGValue *div_comp = NULL;
        if (d_fake) {
            /* adversarial component: -mean(d_fake) to match Python semantics */
            AGValue *mean_fake = ag_mean(d_fake);
            ag_register_temp_value(mean_fake);
            AGValue *neg = ag_scalar_float(-1.0f);
            ag_register_temp_value(neg);
            AGValue *adv = ag_mul(neg, mean_fake);
            ag_register_temp_value(adv);
            gen_loss = adv;
            adv_comp = adv;
        }

        /* masked loss */
        if (fake && facies_pyramid[scale] && masks_pyramid &&
                masks_pyramid[scale]) {
            AGValue *real_ag = ag_value_from_array(facies_pyramid[scale], 0);
            ag_register_temp_value(real_ag);
            AGValue *mask_ag = ag_value_from_array(masks_pyramid[scale], 0);
            ag_register_temp_value(mask_ag);
            AGValue *well_ag = NULL;
            if (wells_pyramid && wells_pyramid[scale]) {
                well_ag = ag_value_from_array(wells_pyramid[scale], 0);
                ag_register_temp_value(well_ag);
            }
            AGValue *masked = mlx_faciesgan_compute_masked_loss_ag(
                                  m, fake, real_ag, well_ag, mask_ag, well_loss_penalty);
            if (masked) {
                ag_register_temp_value(masked);
                masked_comp = masked;
                gen_loss = gen_loss ? ag_add(gen_loss, masked) : masked;
                ag_register_temp_value(gen_loss);
            }
        }

        /* recovery loss: if rec_in_pyramid not provided, compute it from
         * facies_pyramid */
        {
            mlx_array *tmp_rec = NULL;
            int tmp_computed = 0;
            mlx_array *use_rec = NULL;
            if (facies_pyramid && facies_pyramid[scale]) {
                if (rec_in_pyramid && rec_in_pyramid[scale]) {
                    use_rec = rec_in_pyramid[scale];
                } else {
                    if (mlx_compute_rec_input(scale, indexes, n_indexes, facies_pyramid,
                                              &tmp_rec) == 0 &&
                            tmp_rec) {
                        use_rec = tmp_rec;
                        tmp_computed = 1;
                    }
                }
            }

            if (use_rec && facies_pyramid[scale]) {
                /* if we computed a temporary rec_in, initialize rec noise amp */
                if (tmp_computed) {
                    /* best-effort: compute noise amp and set on model */
                    mlx_init_rec_noise_and_amp(m, scale, indexes, n_indexes,
                                               facies_pyramid[scale], wells_pyramid,
                                               seismic_pyramid);
                }
                AGValue *rec_in_ag = NULL;
                if (tmp_computed) {
                    /* ag_value_from_new_array makes a copy; must free original */
                    rec_in_ag = ag_value_from_new_array(tmp_rec, 0);
                } else {
                    rec_in_ag = ag_value_from_array(use_rec, 0);
                }
                ag_register_temp_value(rec_in_ag);
                AGValue *real_ag = ag_value_from_array(facies_pyramid[scale], 0);
                ag_register_temp_value(real_ag);
                AGValue *recv = mlx_faciesgan_compute_recovery_loss_ag(
                                    m, indexes, n_indexes, scale, rec_in_ag, real_ag, wells_pyramid,
                                    seismic_pyramid, alpha);
                if (recv) {
                    ag_register_temp_value(recv);
                    rec_comp = recv;
                    gen_loss = gen_loss ? ag_add(gen_loss, recv) : recv;
                    ag_register_temp_value(gen_loss);
                }
            }

            if (tmp_computed && tmp_rec) {
                /* Free the original mlx_array after copy was made */
                mlx_array_free(*tmp_rec);
                mlx_free_pod((void **)&tmp_rec);
            }
        }

        /* diversity: generate multiple samples and compute diversity loss (AG) */
        int n_div = 0;
        if (lambda_diversity > 0.0f) {
            n_div = mlx_faciesgan_get_num_diversity_samples(m);
            if (n_div < 1)
                n_div = 1;
        }
        if (n_div > 1) {
            AGValue **fake_samples = NULL;
            if (mlx_alloc_ptr_array((void ***)&fake_samples, n_div) == 0) {
                for (int di = 0; di < n_div; ++di)
                    fake_samples[di] = NULL;
                for (int di = 0; di < n_div; ++di) {
                    mlx_array **tmp_noises = NULL;
                    int tmp_n = 0;
                    if (mlx_faciesgan_get_pyramid_noise(
                                m, scale, indexes, n_indexes, &tmp_noises, &tmp_n,
                                wells_pyramid, seismic_pyramid, 0) == 0) {
                        /* Convert noise arrays to AGValues for diversity sample */
                        AGValue **tmp_z_list = NULL;
                        int tmp_z_count = tmp_n;
                        if (mlx_alloc_ptr_array((void ***)&tmp_z_list, tmp_z_count) == 0) {
                            for (int ni = 0; ni < tmp_z_count; ++ni) {
                                if (tmp_noises[ni]) {
                                    /* ag_value_from_new_array makes a copy; must free original */
                                    tmp_z_list[ni] = ag_value_from_new_array(tmp_noises[ni], 0);
                                    ag_register_temp_value(tmp_z_list[ni]);
                                    /* Free original mlx_array after copy */
                                    mlx_array_free(*tmp_noises[ni]);
                                    mlx_free_pod((void **)&tmp_noises[ni]);
                                } else {
                                    tmp_z_list[ni] = NULL;
                                }
                            }
                            mlx_free_ptr_array((void ***)&tmp_noises, tmp_n);

                            /* Build amplitude array for diversity sample */
                            float *tmp_amp = NULL;
                            int tmp_amp_count = scale + 1;
                            if (mlx_alloc_float_buf(&tmp_amp, tmp_amp_count) == 0) {
                                float *noise_amps_tmp = NULL;
                                int n_amps_tmp = 0;
                                mlx_faciesgan_get_noise_amps(m, &noise_amps_tmp, &n_amps_tmp);
                                for (int ai = 0; ai < tmp_amp_count; ++ai) {
                                    if (ai < n_amps_tmp && noise_amps_tmp)
                                        tmp_amp[ai] = noise_amps_tmp[ai];
                                    else
                                        tmp_amp[ai] = 1.0f;
                                }
                                if (noise_amps_tmp)
                                    mlx_free_float_buf(&noise_amps_tmp, &n_amps_tmp);
                            }

                            AGValue *f = mlx_faciesgan_generator_forward_ag(
                                             m, tmp_z_list, tmp_z_count, tmp_amp, tmp_amp_count,
                                             NULL, 0, scale);
                            if (f)
                                ag_register_temp_value(f);
                            fake_samples[di] = f;

                            if (tmp_amp) {
                                mlx_free_float_buf(&tmp_amp, &tmp_amp_count);
                            }
                            /* z_list elements are registered as temps, only free container */
                            mlx_free_ptr_array((void ***)&tmp_z_list, tmp_z_count);
                        } else {
                            /* Failed to allocate z_list, cleanup tmp_noises */
                            for (int ni = 0; ni < tmp_n; ++ni) {
                                if (tmp_noises[ni]) {
                                    mlx_array_free(*tmp_noises[ni]);
                                    mlx_free_pod((void **)&tmp_noises[ni]);
                                }
                            }
                            mlx_free_ptr_array((void ***)&tmp_noises, tmp_n);
                        }
                    }
                }
                /* compute diversity AG loss */
                div_comp = mlx_faciesgan_compute_diversity_loss_ag(
                               m, fake_samples, n_div, lambda_diversity);
                if (div_comp) {
                    ag_register_temp_value(div_comp);
                    gen_loss = gen_loss ? ag_add(gen_loss, div_comp) : div_comp;
                    ag_register_temp_value(gen_loss);
                }
                /* Do not free `fake_samples` AGValue entries here — they are
                 * registered as temporaries and must stay alive until the
                 * backward pass completes. Free only the container. */
                mlx_free_ptr_array((void ***)&fake_samples, n_div);
            }
        }
        /* Backprop generator loss and collect grads */
        if (gen_loss) {
            ag_backward(gen_loss);
            mlx_array **gen_grads = NULL;
            if (ag_collect_grads(gen_params_ag, gen_param_n, &gen_grads) == 0) {
                sr->gen_grads = gen_grads;
                sr->gen_n = gen_param_n;
            }
            for (int p = 0; p < gen_param_n; ++p)
                ag_value_free(gen_params_ag[p]);
            mlx_free_ptr_array((void ***)&gen_params_ag, gen_param_n);
            /* defer resetting/freeing the tape until after we extract scalar
               metrics below so AGValue temporaries remain valid while we
               inspect/evaluate them */
        }

        /* Store scalar metrics by evaluating AGValue components where available */
        if (adv_comp) {
            mlx_array *arr = ag_value_array(adv_comp);
            if (arr) {
                safe_mlx_array_eval(*arr);
                mlx_array tmp = mlx_array_new();
                if (mlx_array_set(&tmp, *arr) == 0) {
                    sr->metrics.fake = NULL;
                    if (mlx_alloc_pod((void **)&sr->metrics.fake, sizeof(mlx_array), 1) ==
                            0)
                        *sr->metrics.fake = tmp;
                } else {
                    mlx_array_free(tmp);
                }
            }
        }
        if (masked_comp) {
            mlx_array *arr = ag_value_array(masked_comp);
            if (arr) {
                safe_mlx_array_eval(*arr);
                mlx_array tmp = mlx_array_new();
                if (mlx_array_set(&tmp, *arr) == 0) {
                    sr->metrics.well = NULL;
                    if (mlx_alloc_pod((void **)&sr->metrics.well, sizeof(mlx_array), 1) ==
                            0)
                        *sr->metrics.well = tmp;
                } else {
                    mlx_array_free(tmp);
                }
            }
        }
        if (div_comp) {
            mlx_array *arr = ag_value_array(div_comp);
            if (arr) {
                safe_mlx_array_eval(*arr);
                mlx_array tmp = mlx_array_new();
                if (mlx_array_set(&tmp, *arr) == 0) {
                    sr->metrics.div = NULL;
                    if (mlx_alloc_pod((void **)&sr->metrics.div, sizeof(mlx_array), 1) ==
                            0)
                        *sr->metrics.div = tmp;
                } else {
                    mlx_array_free(tmp);
                }
            }
        }
        if (rec_comp) {
            mlx_array *arr = ag_value_array(rec_comp);
            if (arr) {
                safe_mlx_array_eval(*arr);
                mlx_array tmp = mlx_array_new();
                if (mlx_array_set(&tmp, *arr) == 0) {
                    sr->metrics.rec = NULL;
                    if (mlx_alloc_pod((void **)&sr->metrics.rec, sizeof(mlx_array), 1) ==
                            0)
                        *sr->metrics.rec = tmp;
                } else {
                    mlx_array_free(tmp);
                }
            }
        }
        if (gen_loss) {
            mlx_array *arr = ag_value_array(gen_loss);
            if (arr) {
                safe_mlx_array_eval(*arr);
                mlx_array tmp = mlx_array_new();
                if (mlx_array_set(&tmp, *arr) == 0) {
                    sr->metrics.total = NULL;
                    if (mlx_alloc_pod((void **)&sr->metrics.total, sizeof(mlx_array),
                                      1) == 0)
                        *sr->metrics.total = tmp;
                } else {
                    mlx_array_free(tmp);
                }
            }
        }
        /* All metric extraction and grad collection for this scale complete; free
           temporaries and reset the AG tape now. */
        ag_reset_tape();
    }

    *out_results = res;
    return 0;
}

void mlx_results_free(MLXResults *res) {
    if (!res)
        return;
    for (int i = 0; i < res->n_scales; ++i) {
        MLXScaleResults *sr = &res->scales[i];
        if (sr->metrics.fake) {
            mlx_array_free(*sr->metrics.fake);
            mlx_free_pod((void **)&sr->metrics.fake);
        }
        if (sr->metrics.well) {
            mlx_array_free(*sr->metrics.well);
            mlx_free_pod((void **)&sr->metrics.well);
        }
        if (sr->metrics.div) {
            mlx_array_free(*sr->metrics.div);
            mlx_free_pod((void **)&sr->metrics.div);
        }
        if (sr->metrics.rec) {
            mlx_array_free(*sr->metrics.rec);
            mlx_free_pod((void **)&sr->metrics.rec);
        }
        if (sr->metrics.total) {
            mlx_array_free(*sr->metrics.total);
            mlx_free_pod((void **)&sr->metrics.total);
        }
        if (sr->gen_grads) {
            for (int g = 0; g < sr->gen_n; ++g) {
                if (sr->gen_grads[g]) {
                    mlx_array_free(*sr->gen_grads[g]);
                    mlx_free_pod((void **)&sr->gen_grads[g]);
                }
            }
            mlx_free_ptr_array((void ***)&sr->gen_grads, sr->gen_n);
        }
        if (sr->disc_grads) {
            for (int g = 0; g < sr->disc_n; ++g) {
                if (sr->disc_grads[g]) {
                    mlx_array_free(*sr->disc_grads[g]);
                    mlx_free_pod((void **)&sr->disc_grads[g]);
                }
            }
            mlx_free_ptr_array((void ***)&sr->disc_grads, sr->disc_n);
        }
    }
    free(res->scales);
    mlx_free_pod((void **)&res);
}
