#include "train_step.h"
#include "array_helpers.h"
#include "base_manager.h"
#include "discriminator.h"
#include "generator.h"
#include "mlx_native_grad.h"
#include "optimizer.h"
#include "train_utils.h"
#include <mlx/c/transforms.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Batch evaluate multiple arrays using mlx_eval() for better performance.
   This mirrors Python's mx.eval(list_of_arrays) which evaluates all arrays
   in a single batch, allowing MLX to optimize the computation graph and
   memory allocation across all arrays.
*/
static int batch_eval_arrays(mlx_array *arrays, int n) {
    if (!arrays || n <= 0)
        return 0;
    mlx_vector_array vec = mlx_vector_array_new();
    for (int i = 0; i < n; ++i) {
        if (arrays[i].ctx) {
            mlx_vector_array_append_value(vec, arrays[i]);
        }
    }
    int rc = mlx_eval(vec);
    mlx_vector_array_free(vec);
    return rc;
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

/* Scale-specific version that gets parameters for only one scale */
int mlx_faciesgan_apply_sgd_to_generator_for_scale(MLXFaciesGAN *m, MLXOptimizer *opt,
        mlx_array **grads, int n, int scale) {
    if (!m || !opt)
        return -1;
    MLXGenerator *g = mlx_faciesgan_build_generator(m);
    if (!g)
        return -1;
    int param_n = 0;
    mlx_array **params = mlx_generator_get_parameters_for_scale(g, scale, &param_n);
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

/* Scale-specific version that gets parameters for only one scale */
int mlx_faciesgan_apply_sgd_to_discriminator_for_scale(MLXFaciesGAN *m, MLXOptimizer *opt,
        mlx_array **grads, int n, int scale) {
    if (!m || !opt)
        return -1;
    MLXDiscriminator *d = mlx_faciesgan_build_discriminator(m);
    if (!d)
        return -1;
    void *disc_ptr = mlx_discriminator_get_disc_ptr(d, scale);
    if (!disc_ptr)
        return -1;
    MLXSPADEDiscriminator *spade_disc = (MLXSPADEDiscriminator *)disc_ptr;
    int param_n = 0;
    mlx_array **params = mlx_spadedisc_get_parameters(spade_disc, &param_n);
    if (!params || param_n == 0) {
        if (params)
            mlx_spadedisc_free_parameters_list(params);
        return -1;
    }
    if (n != param_n) {
        mlx_spadedisc_free_parameters_list(params);
        return -1;
    }

    int r = mlx_adam_step(opt, params, grads, param_n);
    mlx_spadedisc_free_parameters_list(params);
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
        /* Free discriminator metrics */
        if (sr->metrics.d_real) {
            mlx_array_free(*sr->metrics.d_real);
            mlx_free_pod((void **)&sr->metrics.d_real);
        }
        if (sr->metrics.d_fake) {
            mlx_array_free(*sr->metrics.d_fake);
            mlx_free_pod((void **)&sr->metrics.d_fake);
        }
        if (sr->metrics.d_gp) {
            mlx_array_free(*sr->metrics.d_gp);
            mlx_free_pod((void **)&sr->metrics.d_gp);
        }
        if (sr->metrics.d_total) {
            mlx_array_free(*sr->metrics.d_total);
            mlx_free_pod((void **)&sr->metrics.d_total);
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

/* ============================================================================
 * Native Grad Training Functions
 *
 * These use MLX's built-in value_and_grad instead of our custom AG autodiff.
 * This is more reliable and properly computes weight gradients.
 * ============================================================================ */

int mlx_faciesgan_collect_metrics_and_grads_native(
    MLXFaciesGAN *m, const int *indexes, int n_indexes,
    const int *active_scales, int n_active_scales, mlx_array **facies_pyramid,
    mlx_array **rec_in_pyramid, mlx_array **wells_pyramid,
    mlx_array **masks_pyramid, mlx_array **seismic_pyramid,
    float lambda_diversity, float well_loss_penalty, float alpha,
    float lambda_grad, int mode, MLXResults **out_results) {

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

    mlx_stream stream = mlx_default_gpu_stream_new();

    for (int si = 0; si < n_active_scales; ++si) {
        int scale = active_scales[si];
        MLXScaleResults *sr = &res->scales[si];
        sr->scale = scale;
        sr->gen_grads = NULL;
        sr->gen_n = 0;
        sr->disc_grads = NULL;
        sr->disc_n = 0;
        memset(&sr->metrics, 0, sizeof(sr->metrics));

        if (!facies_pyramid || !facies_pyramid[scale])
            continue;

        mlx_array *real = facies_pyramid[scale];
        mlx_array *wells = wells_pyramid ? wells_pyramid[scale] : NULL;
        mlx_array *masks = masks_pyramid ? masks_pyramid[scale] : NULL;
        mlx_array *seismic = seismic_pyramid ? seismic_pyramid[scale] : NULL;
        mlx_array *rec_in = rec_in_pyramid ? rec_in_pyramid[scale] : NULL;

        /* === Generate fake for discriminator training ===
         * We need a detached fake sample (no grad tracking) for disc training.
         *
         * FIX 35: Only generate noise when needed for the disc path.
         * After Fix 34 the gen_loss_closure generates its own fresh noise
         * internally via get_pyramid_noise.  Generating unused noise here
         * would advance the RNG state and break random-number parity with
         * Python, where optimize_generator never calls get_pyramid_noise
         * outside the closure. */
        MLXGenerator *gen = mlx_faciesgan_build_generator(m);
        if (!gen) continue;

        mlx_array **gen_noises_ptr = NULL;
        int gen_n_noises = 0;
        mlx_array *gen_noises = NULL;

        /* Only generate noise for disc path — gen closure generates its own */
        if (mode != MLX_COLLECT_GEN_ONLY) {
            if (mlx_faciesgan_get_pyramid_noise(
                        m, scale, indexes, n_indexes, &gen_noises_ptr, &gen_n_noises,
                        wells_pyramid, seismic_pyramid, 0) != 0) {
                continue;
            }

            gen_noises = malloc(gen_n_noises * sizeof(mlx_array));
            if (!gen_noises) {
                for (int ni = 0; ni < gen_n_noises; ++ni) {
                    if (gen_noises_ptr[ni]) {
                        mlx_array_free(*gen_noises_ptr[ni]);
                        mlx_free_pod((void **)&gen_noises_ptr[ni]);
                    }
                }
                mlx_free_ptr_array((void ***)&gen_noises_ptr, gen_n_noises);
                continue;
            }
            for (int ni = 0; ni < gen_n_noises; ++ni) {
                gen_noises[ni] = gen_noises_ptr[ni] ? *gen_noises_ptr[ni] : (mlx_array) {
                    0
                };
            }
        }

        /* Build amplitude array (always needed — used by gen closure too) */
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
                    amp[ai] = 1.0f;
            }
            if (noise_amps) mlx_free_float_buf(&noise_amps, &n_amps);
        } else {
            if (gen_noises_ptr) {
                for (int ni = 0; ni < gen_n_noises; ++ni) {
                    if (gen_noises_ptr[ni]) {
                        mlx_array_free(*gen_noises_ptr[ni]);
                        mlx_free_pod((void **)&gen_noises_ptr[ni]);
                    }
                }
                mlx_free_ptr_array((void ***)&gen_noises_ptr, gen_n_noises);
            }
            free(gen_noises);
            continue;
        }

        /* Generate fake (detached - just forward pass, no grad tracking).
         * Only needed for discriminator training; G-only mode generates its
         * own fake inside the gradient closure.
         * NOTE: Do NOT eval fake here — let value_and_grad build a fused
         * graph (gen forward → disc loss → disc grads) for much better
         * GPU throughput. */
        mlx_array fake = {0};
        if (mode != MLX_COLLECT_GEN_ONLY) {
            fake = mlx_generator_forward(gen, gen_noises, gen_n_noises,
            amp, amp_count, (mlx_array) {
                0
            }, 0, scale);
        }

        /* === Discriminator training step using native grad === */
        if (mode != MLX_COLLECT_GEN_ONLY && fake.ctx) {
            mlx_array disc_loss = {0};
            mlx_array **disc_grads = NULL;
            int disc_n = 0;
            mlx_array d_real = {0}, d_fake = {0}, d_gp = {0};

            int disc_rc = mlx_native_compute_disc_loss_and_grads(
                              m, scale, real, &fake, lambda_grad,
                              &disc_loss, &disc_grads, &disc_n,
                              &d_real, &d_fake, &d_gp);

            if (disc_rc == 0 && disc_n > 0) {
                sr->disc_grads = disc_grads;
                sr->disc_n = disc_n;

                /* Metrics and gradients are left lazy — the caller’s
                 * single-eval-per-epoch will batch-evaluate model params,
                 * optimizer state, and these metrics together, matching
                 * Python’s mx.eval(model.state, optimizer.state). */

                /* Store disc metrics (lazy) */
                if (disc_loss.ctx) {
                    sr->metrics.d_total = malloc(sizeof(mlx_array));
                    if (sr->metrics.d_total) *sr->metrics.d_total = disc_loss;
                    else mlx_array_free(disc_loss);
                }
                if (d_real.ctx) {
                    sr->metrics.d_real = malloc(sizeof(mlx_array));
                    if (sr->metrics.d_real) *sr->metrics.d_real = d_real;
                    else mlx_array_free(d_real);
                }
                if (d_fake.ctx) {
                    sr->metrics.d_fake = malloc(sizeof(mlx_array));
                    if (sr->metrics.d_fake) *sr->metrics.d_fake = d_fake;
                    else mlx_array_free(d_fake);
                }
                if (d_gp.ctx) {
                    sr->metrics.d_gp = malloc(sizeof(mlx_array));
                    if (sr->metrics.d_gp) *sr->metrics.d_gp = d_gp;
                    else mlx_array_free(d_gp);
                }
            } else {
                if (disc_grads) mlx_native_free_grads(disc_grads, disc_n);
                mlx_array_free(disc_loss);
                mlx_array_free(d_real);
                mlx_array_free(d_fake);
                mlx_array_free(d_gp);
            }
        }

        /* === Generator training step using native grad ===
         * Skipped in DISC_ONLY mode to avoid advancing the RNG state with
         * diversity noise that would be discarded anyway. */
        if (mode != MLX_COLLECT_DISC_ONLY) {
            mlx_array gen_loss = {0};
            mlx_array **gen_grads = NULL;
            int gen_n = 0;
            mlx_array g_adv = {0}, g_well = {0}, g_div = {0}, g_rec = {0};

            int gen_rc = mlx_native_compute_gen_loss_and_grads(
                             m, scale, gen_noises, gen_n_noises, amp, amp_count,
                             real, wells, masks, rec_in,
                             (int *)indexes, n_indexes, wells_pyramid, seismic_pyramid,
                             lambda_diversity, well_loss_penalty, alpha,
                             &gen_loss, &gen_grads, &gen_n,
                             &g_adv, &g_well, &g_div, &g_rec);

            if (gen_rc == 0 && gen_n > 0) {
                sr->gen_grads = gen_grads;
                sr->gen_n = gen_n;

                /* Metrics and gradients are left lazy — the caller’s
                 * single-eval-per-epoch will materialise everything. */

                /* Store gen metrics (lazy) */
                if (gen_loss.ctx) {
                    sr->metrics.total = malloc(sizeof(mlx_array));
                    if (sr->metrics.total) *sr->metrics.total = gen_loss;
                    else mlx_array_free(gen_loss);
                }
                if (g_adv.ctx) {
                    sr->metrics.fake = malloc(sizeof(mlx_array));
                    if (sr->metrics.fake) *sr->metrics.fake = g_adv;
                    else mlx_array_free(g_adv);
                }
                if (g_well.ctx) {
                    sr->metrics.well = malloc(sizeof(mlx_array));
                    if (sr->metrics.well) *sr->metrics.well = g_well;
                    else mlx_array_free(g_well);
                }
                if (g_div.ctx) {
                    sr->metrics.div = malloc(sizeof(mlx_array));
                    if (sr->metrics.div) *sr->metrics.div = g_div;
                    else mlx_array_free(g_div);
                }
                if (g_rec.ctx) {
                    sr->metrics.rec = malloc(sizeof(mlx_array));
                    if (sr->metrics.rec) *sr->metrics.rec = g_rec;
                    else mlx_array_free(g_rec);
                }
            } else {
                if (gen_grads) mlx_native_free_grads(gen_grads, gen_n);
                mlx_array_free(gen_loss);
                mlx_array_free(g_adv);
                mlx_array_free(g_well);
                mlx_array_free(g_div);
                mlx_array_free(g_rec);
            }
        }

        /* Cleanup noise arrays (only allocated in disc path) */
        if (gen_noises_ptr) {
            for (int ni = 0; ni < gen_n_noises; ++ni) {
                if (gen_noises_ptr[ni]) {
                    mlx_array_free(*gen_noises_ptr[ni]);
                    mlx_free_pod((void **)&gen_noises_ptr[ni]);
                }
            }
            mlx_free_ptr_array((void ***)&gen_noises_ptr, gen_n_noises);
        }
        free(gen_noises);
        if (amp) mlx_free_float_buf(&amp, &amp_count);
        mlx_array_free(fake);
    }

    mlx_stream_free(stream);
    *out_results = res;
    return 0;
}
