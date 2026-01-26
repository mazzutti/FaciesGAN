#include "train_step.h"
#include "array_helpers.h"
#include "autodiff.h"
#include "base_manager.h"
#include "discriminator.h"
#include "generator.h"
#include "optimizer.h"
#include "train_utils.h"
#include <stdlib.h>

/* Safe wrapper for mlx_array_eval. By default this skips device evaluation
   unless FACIESGAN_ALLOW_ARRAY_EVAL is set in the environment. This prevents
   crashes when the device/stream state is not available during debugging.
*/
static int safe_mlx_array_eval(mlx_array a) {
    const char *env = getenv("FACIESGAN_ALLOW_ARRAY_EVAL");
    if (!env)
        return 0;
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

        /* Build fake via AG using current noise (single sample) */
        AGValue *noise_in = NULL;
        mlx_array **gen_noises = NULL;
        int gen_n_noises = 0;
        int _transferred_noise_idx = -1;
        if (mlx_faciesgan_get_pyramid_noise(
                    m, scale, indexes, n_indexes, &gen_noises, &gen_n_noises,
                    wells_pyramid, seismic_pyramid, 0) == 0) {
            if (gen_n_noises > scale && gen_noises[scale]) {
                /* Transfer ownership of the underlying mlx_array to the AGValue
                   wrapper so we can free the pointer container while leaving the
                   array alive for the AG tape. */
                noise_in = ag_value_from_new_array(gen_noises[scale], 0);
                ag_register_temp_value(noise_in);
                _transferred_noise_idx = scale;
            }
            for (int ni = 0; ni < gen_n_noises; ++ni) {
                if (gen_noises[ni]) {
                    if (ni == _transferred_noise_idx) {
                        /* We've transferred ownership of the array contents into
                           the AGValue; only free the container pointer here. */
                        mlx_free_pod((void **)&gen_noises[ni]);
                    } else {
                        mlx_array_free(*gen_noises[ni]);
                        mlx_free_pod((void **)&gen_noises[ni]);
                    }
                }
            }
            mlx_free_ptr_array((void ***)&gen_noises, gen_n_noises);
        }

        AGValue *fake = mlx_faciesgan_generator_forward_ag(m, NULL, 0, NULL, 0,
                        noise_in, scale, scale);

        AGValue *real = ag_value_from_array(facies_pyramid[scale], 0);

        /* Discriminator forward and loss */
        AGValue *d_real = mlx_faciesgan_discriminator_forward_ag(m, real, scale);
        AGValue *d_fake = NULL;
        if (fake) {
            d_fake = mlx_faciesgan_discriminator_forward_ag(m, fake, scale);
        }

        AGValue *disc_loss = NULL;
        if (d_real && d_fake) {
            AGValue *sum_real = d_real;
            AGValue *sum_fake = d_fake;
            /* reduce to scalars */
            mlx_array *arr1 = ag_value_array(sum_real);
            if (arr1) {
                int ndim = (int)mlx_array_ndim(*arr1);
                for (int ax = 0; ax < ndim; ++ax)
                    sum_real = ag_sum_axis(sum_real, 0, 0);
                ag_register_temp_value(sum_real);
            }
            mlx_array *arr2 = ag_value_array(sum_fake);
            if (arr2) {
                int ndim = (int)mlx_array_ndim(*arr2);
                for (int ax = 0; ax < ndim; ++ax)
                    sum_fake = ag_sum_axis(sum_fake, 0, 0);
                ag_register_temp_value(sum_fake);
            }
            AGValue *neg = ag_scalar_float(1.0f);
            ag_register_temp_value(neg);
            AGValue *fr = ag_sub(sum_real, neg);
            ag_register_temp_value(fr);
            AGValue *ff = ag_sub(sum_fake, neg);
            ag_register_temp_value(ff);
            AGValue *rf = ag_add(fr, ff);
            ag_register_temp_value(rf);
            disc_loss = rf;
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
            /* adversarial component: (d_fake - 1)^2 as training loss but
               record metric as -mean(d_fake) to match Python semantics */
            AGValue *one = ag_scalar_float(1.0f);
            ag_register_temp_value(one);
            AGValue *err = ag_sub(d_fake, one);
            ag_register_temp_value(err);
            AGValue *sq = ag_square(err);
            ag_register_temp_value(sq);
            gen_loss = sq;

            /* compute adv metric = -mean(d_fake) */
            mlx_array *dfa = ag_value_array(d_fake);
            if (dfa) {
                AGValue *meanv = d_fake;
                int ndim = (int)mlx_array_ndim(*dfa);
                for (int ax = 0; ax < ndim; ++ax)
                    meanv = ag_sum_axis(meanv, 0, 0);
                ag_register_temp_value(meanv);
                size_t elems = mlx_array_size(*dfa);
                AGValue *den = ag_scalar_float((float)elems);
                ag_register_temp_value(den);
                meanv = ag_divide(meanv, den);
                ag_register_temp_value(meanv);
                AGValue *neg = ag_scalar_float(-1.0f);
                ag_register_temp_value(neg);
                adv_comp = ag_mul(meanv, neg);
                ag_register_temp_value(adv_comp);
            }
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
                    /* we created a temporary mlx_array `tmp_rec` and must transfer
                     * ownership into the AGValue so the tape can hold it alive
                     * after we free the container pointer below. */
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
                /* Ownership of *tmp_rec was transferred into the AGValue when
                 * we created `rec_in_ag` with `ag_value_from_new_array`. Only
                 * free the pointer container here. */
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
                        int _transferred_tmp_idx = -1;
                        if (tmp_n > scale && tmp_noises[scale]) {
                            AGValue *n_ag = ag_value_from_new_array(tmp_noises[scale], 0);
                            ag_register_temp_value(n_ag);
                            AGValue *f = mlx_faciesgan_generator_forward_ag(
                                             m, NULL, 0, NULL, 0, n_ag, scale, scale);
                            if (f)
                                ag_register_temp_value(f);
                            fake_samples[di] = f;
                            _transferred_tmp_idx = scale;
                        }
                        for (int ni = 0; ni < tmp_n; ++ni) {
                            if (tmp_noises[ni]) {
                                if (ni == _transferred_tmp_idx) {
                                    mlx_free_pod((void **)&tmp_noises[ni]);
                                } else {
                                    mlx_array_free(*tmp_noises[ni]);
                                    mlx_free_pod((void **)&tmp_noises[ni]);
                                }
                            }
                        }
                        mlx_free_ptr_array((void ***)&tmp_noises, tmp_n);
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
