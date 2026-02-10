/* mlx_native_grad.c - Use MLX's native value_and_grad for autograd
 *
 * This replaces our custom autodiff.c implementation with MLX's built-in
 * autodiff via mlx_value_and_grad, which is more reliable and battle-tested.
 *
 * Python pattern from facies_gan.py:
 *
 *   def compute_metrics(params: dict) -> mx.array:
 *       model.update(params)  # Set traced params in model
 *       ... forward pass using model (which now uses traced params) ...
 *       return loss
 *
 *   params = model.parameters()
 *   loss, grads = mx.value_and_grad(compute_metrics)(params)
 *   optimizer.update(model, grads)
 *
 * In C, model.update(params) is achieved via mlx_array_set() on each parameter.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlx/c/mlx.h"
#include "mlx/c/compile.h"
#include "../models/facies_gan.h"
#include "../models/generator.h"
#include "../models/discriminator.h"
#include "../models/custom_layer.h"
#include "../utils.h"
#include "array_helpers.h"
#include "scalar_pool.h"
#include "mlx_native_grad.h"

/* ============================================================================
 * Payload structures for closures
 * ============================================================================ */

/* Forward declaration of metrics structure */
typedef struct MLXGenMetrics {
    mlx_array adv;    /* adversarial loss */
    mlx_array well;   /* well/masked loss */
    mlx_array div;    /* diversity loss */
    mlx_array rec;    /* recovery loss */
} MLXGenMetrics;

typedef struct MLXDiscMetrics {
    mlx_array real;   /* -mean(D(real)) */
    mlx_array fake;   /* mean(D(fake)) */
    mlx_array gp;     /* gradient penalty */
} MLXDiscMetrics;

/* Payload for generator loss closure */
typedef struct {
    MLXFaciesGAN *model;
    int scale;
    mlx_array real;           /* real facies for this scale */
    mlx_array well;           /* well conditioning (may be empty) */
    mlx_array mask;           /* mask for well loss (may be empty) */
    mlx_array rec_in;         /* reconstruction input from previous scale (may be empty) */
    mlx_array *z_list;        /* noise arrays for generator */
    int z_count;
    float *amp;               /* noise amplitudes */
    int amp_count;
    float lambda_diversity;
    float well_loss_penalty;
    float alpha;              /* recovery loss weight */
    int num_diversity_samples;
    int *indexes;             /* batch indices for rec noise generation */
    int n_indexes;
    mlx_array **wells_pyramid;   /* for rec noise generation */
    mlx_array **seismic_pyramid; /* for rec noise generation */
    /* Store parameter pointers so closure can update them */
    mlx_array **param_ptrs;
    int n_params;
    /* Output: individual metrics (filled during closure execution) */
    MLXGenMetrics *out_metrics;
} GenClosurePayload;

/* Payload for discriminator loss closure */
typedef struct {
    MLXFaciesGAN *model;
    int scale;
    mlx_array real;           /* real facies */
    mlx_array fake;           /* fake from generator (detached) */
    float lambda_grad;
    /* Store parameter pointers so closure can update them */
    mlx_array **param_ptrs;
    int n_params;
    /* Output: individual metrics (filled during closure execution) */
    MLXDiscMetrics *out_metrics;
} DiscClosurePayload;

/* ============================================================================
 * Generator closure: computes gen loss given gen params
 *
 * Mirrors Python's compute_metrics pattern:
 *   def compute_metrics(params):
 *       self.generator.gens[scale].update(params)
 *       fake_samples = self.generate_diverse_samples(...)
 *       metrics.fake = self.compute_adversarial_loss(scale, fake)
 *       metrics.well = self.compute_masked_loss(...)
 *       metrics.div = self.compute_diversity_loss(...)
 *       return total
 * ============================================================================ */

static int gen_loss_closure(mlx_vector_array *result,
                            const mlx_vector_array inputs,
                            void *payload_ptr) {
    GenClosurePayload *p = (GenClosurePayload *)payload_ptr;
    if (!p || !p->model) {
        fprintf(stderr, "[gen_loss_closure] NULL payload\n");
        return 1;
    }

    mlx_stream s = mlx_gpu_stream();

    /* === CRITICAL: Update model parameters with traced inputs ===
     * This is the C equivalent of Python's model.update(params).
     * We replace each parameter array in the model with the corresponding
     * traced input, so subsequent forward pass operations use traced arrays. */
    int vec_size = mlx_vector_array_size(inputs);
    if (vec_size != p->n_params) {
        fprintf(stderr, "[gen_loss_closure] param count mismatch: got %d, expected %d\n",
                vec_size, p->n_params);
        return 1;
    }

    for (int i = 0; i < p->n_params; ++i) {
        mlx_array traced_param = mlx_array_new();
        mlx_vector_array_get(&traced_param, inputs, i);
        /* mlx_array_set replaces the contents of param_ptrs[i] with traced_param */
        mlx_array_set(p->param_ptrs[i], traced_param);
        mlx_array_free(traced_param);
    }

    /* ================================================================
     * FIX 34: Generate ALL diversity samples FIRST, then use fakes[0]
     * for adversarial + well loss.  This matches Python exactly:
     *   fake_samples = self.generate_diverse_samples(indexes, scale, wells, seismic)
     *   fake = fake_samples[0]
     *   metrics.fake = compute_adversarial_loss(scale, fake)
     *   metrics.well = compute_masked_loss(fake, real, well, mask)
     *   metrics.div  = compute_diversity_loss(fake_samples)
     * ================================================================ */
    MLXGenerator *gen = mlx_faciesgan_build_generator(p->model);
    int n_samples = p->num_diversity_samples > 0 ? p->num_diversity_samples : 1;
    mlx_array *fakes = malloc(n_samples * sizeof(mlx_array));
    if (!fakes) {
        fprintf(stderr, "[gen_loss_closure] malloc failed for fakes\n");
        return 1;
    }
    for (int di = 0; di < n_samples; ++di) {
        /* Generate fresh pyramid noise with conditioning (like Python's
         * generate_diverse_samples → get_pyramid_noise per sample) */
        mlx_array **noise_ptrs = NULL;
        int n_noises = 0;
        if (mlx_faciesgan_get_pyramid_noise(
                    p->model, p->scale, p->indexes, p->n_indexes,
                    &noise_ptrs, &n_noises,
                    p->wells_pyramid, p->seismic_pyramid, /*rec=*/0) == 0 && n_noises > 0) {
            mlx_array *z_list_d = malloc(n_noises * sizeof(mlx_array));
            if (z_list_d) {
                for (int ni = 0; ni < n_noises; ++ni)
                    z_list_d[ni] = noise_ptrs[ni] ? *noise_ptrs[ni] : (mlx_array) {
                    0
                };
                fakes[di] = mlx_generator_forward(gen, z_list_d, n_noises,
                p->amp, p->amp_count, (mlx_array) {
                    0
                }, 0, p->scale);
                free(z_list_d);
            } else {
                fakes[di] = mlx_array_new();
            }
            for (int ni = 0; ni < n_noises; ++ni) {
                if (noise_ptrs[ni]) {
                    mlx_array_free(*noise_ptrs[ni]);
                    free(noise_ptrs[ni]);
                }
            }
            free(noise_ptrs);
        } else {
            fakes[di] = mlx_array_new();
        }
    }

    /* Python: fake = fake_samples[0] */
    mlx_array fake = fakes[0]; /* alias — freed with fakes array later */
    if (!fake.ctx) {
        fprintf(stderr, "[gen_loss_closure] generator forward failed\n");
        for (int di = 0; di < n_samples; ++di)
            if (fakes[di].ctx) mlx_array_free(fakes[di]);
        free(fakes);
        return 1;
    }

    /* Compute adversarial loss: -mean(D(fake))
     * Generator wants D(fake) to be high (look real), so minimize -D(fake) */
    MLXDiscriminator *disc = mlx_faciesgan_build_discriminator(p->model);
    mlx_array d_fake = mlx_discriminator_forward(disc, p->scale, fake);

    mlx_array mean_d_fake = mlx_array_new();
    mlx_mean(&mean_d_fake, d_fake, false, s);

    mlx_array adv_loss = mlx_array_new();
    mlx_multiply(&adv_loss, mlx_scalar_neg_one(), mean_d_fake, s);

    /* Store adversarial loss in metrics if provided */
    if (p->out_metrics) {
        mlx_array adv_copy = mlx_array_new();
        mlx_array_set(&adv_copy, adv_loss);
        p->out_metrics->adv = adv_copy;
    }

    /* Initialize total loss — direct copy avoids
     * mlx_array_new() + mlx_array_set() overhead */
    mlx_array total_loss = mlx_array_new();
    mlx_copy(&total_loss, adv_loss, s);

    /* Initialize well and div losses to zero */
    mlx_array well_loss = mlx_array_new_float(0.0f);
    mlx_array div_loss = mlx_array_new_float(0.0f);

    /* Compute well/masked loss if provided:
     * Python: well_loss_penalty * mse_loss(fake * mask, real * mask)
     *       = well_loss_penalty * mean((fake * mask - real * mask)^2) */
    if (p->well.ctx && p->mask.ctx && p->well_loss_penalty > 0.0f) {
        mlx_array fake_masked = mlx_array_new();
        mlx_multiply(&fake_masked, fake, p->mask, s);

        mlx_array real_masked = mlx_array_new();
        mlx_multiply(&real_masked, p->real, p->mask, s);

        mlx_array diff = mlx_array_new();
        mlx_subtract(&diff, fake_masked, real_masked, s);

        mlx_array diff_sq = mlx_array_new();
        mlx_square(&diff_sq, diff, s);

        mlx_array masked_mean = mlx_array_new();
        mlx_mean(&masked_mean, diff_sq, false, s);

        /* Perf: use pre-allocated penalty scalar — lives on payload, no alloc.
         * For values that match pool constants, use the pool; otherwise alloc. */
        mlx_array penalty_arr = mlx_array_new_float(p->well_loss_penalty);

        mlx_array_free(well_loss);
        well_loss = mlx_array_new();
        mlx_multiply(&well_loss, penalty_arr, masked_mean, s);

        mlx_array new_total = mlx_array_new();
        mlx_add(&new_total, total_loss, well_loss, s);
        mlx_array_free(total_loss);
        total_loss = new_total;

        mlx_array_free(fake_masked);
        mlx_array_free(real_masked);
        mlx_array_free(diff);
        mlx_array_free(diff_sq);
        mlx_array_free(masked_mean);
        mlx_array_free(penalty_arr);
    }

    /* Store well loss in metrics */
    if (p->out_metrics) {
        mlx_array well_copy = mlx_array_new();
        mlx_array_set(&well_copy, well_loss);
        p->out_metrics->well = well_copy;
    }

    /* Compute diversity loss matching Python exactly:
     *   For each pair (i,j) of diversity samples:
     *     sq_diffs = mean((fake_i - fake_j)^2)
     *     diversity = exp(-10 * sq_diffs)
     *   loss = lambda * mean(all pairwise diversity values)
     *
     * Perf: Fused pairwise approach — stack all valid fakes into a single
     * tensor along axis 0 and extract pairwise left/right operands via
     * slicing, then compute all diffs/squares/means in batch with fewer
     * MLX op dispatches.
     *
     * fakes[] was already generated above (all samples with fresh noise). */
    if (n_samples > 1 && p->lambda_diversity > 0.0f) {
        /* Count valid fakes */
        int n_valid = 0;
        for (int i = 0; i < n_samples; ++i)
            if (fakes[i].ctx) n_valid++;

        int n_pairs = n_valid * (n_valid - 1) / 2;
        if (n_valid >= 2 && n_pairs > 0) {
            /* Build left and right operand vectors for all pairs */
            mlx_vector_array left_vec = mlx_vector_array_new();
            mlx_vector_array right_vec = mlx_vector_array_new();
            for (int i = 0; i < n_samples; ++i) {
                if (!fakes[i].ctx) continue;
                for (int j = i + 1; j < n_samples; ++j) {
                    if (!fakes[j].ctx) continue;
                    mlx_vector_array_append_value(left_vec, fakes[i]);
                    mlx_vector_array_append_value(right_vec, fakes[j]);
                }
            }

            /* Stack into [n_pairs, B, H, W, C] tensors */
            mlx_array left_stack = mlx_array_new();
            mlx_array right_stack = mlx_array_new();
            mlx_stack_axis(&left_stack, left_vec, 0, s);
            mlx_stack_axis(&right_stack, right_vec, 0, s);
            mlx_vector_array_free(left_vec);
            mlx_vector_array_free(right_vec);

            /* Batched: diff = left - right, sq = diff^2 */
            mlx_array diff_all = mlx_array_new();
            mlx_subtract(&diff_all, left_stack, right_stack, s);
            mlx_array_free(left_stack);
            mlx_array_free(right_stack);

            mlx_array sq_all = mlx_array_new();
            mlx_square(&sq_all, diff_all, s);
            mlx_array_free(diff_all);

            /* Reduce over spatial+channel dims (axes 1..ndim-1) to get
             * per-pair mean squared diff.  We flatten to [n_pairs, -1]
             * then take mean along axis 1. */
            int sq_ndim = (int)mlx_array_ndim(sq_all);
            if (sq_ndim > 1) {
                /* Reshape to [n_pairs, -1] */
                int flat_shape[2] = {n_pairs, -1};
                mlx_array sq_flat = mlx_array_new();
                mlx_reshape(&sq_flat, sq_all, flat_shape, 2, s);
                mlx_array_free(sq_all);

                /* Mean along axis 1 → [n_pairs] */
                int ax1[1] = {1};
                mlx_array mean_per_pair = mlx_array_new();
                mlx_mean_axes(&mean_per_pair, sq_flat, ax1, 1, false, s);
                mlx_array_free(sq_flat);

                /* exp(-10 * mean_per_pair) → [n_pairs] */
                mlx_array neg10 = mlx_scalar_neg_ten();
                mlx_array scaled = mlx_array_new();
                mlx_multiply(&scaled, mean_per_pair, neg10, s);
                mlx_array_free(mean_per_pair);

                mlx_array expvals = mlx_array_new();
                mlx_exp(&expvals, scaled, s);
                mlx_array_free(scaled);

                /* mean over all pairs */
                mlx_array mean_div = mlx_array_new();
                mlx_mean(&mean_div, expvals, false, s);
                mlx_array_free(expvals);

                /* lambda * mean_div */
                mlx_array lambda_arr = mlx_array_new_float(p->lambda_diversity);
                mlx_array_free(div_loss);
                div_loss = mlx_array_new();
                mlx_multiply(&div_loss, lambda_arr, mean_div, s);

                /* add to total loss */
                mlx_array new_total = mlx_array_new();
                mlx_add(&new_total, total_loss, div_loss, s);
                mlx_array_free(total_loss);
                total_loss = new_total;
                mlx_array_free(mean_div);
                mlx_array_free(lambda_arr);
            } else {
                mlx_array_free(sq_all);
            }
        }
    }

    /* Store diversity loss in metrics */
    if (p->out_metrics) {
        mlx_array div_copy = mlx_array_new();
        mlx_array_set(&div_copy, div_loss);
        p->out_metrics->div = div_copy;
    }

    /* Initialize recovery loss to zero */
    mlx_array rec_loss = mlx_array_new_float(0.0f);

    /* Compute recovery loss: alpha * MSE(rec, real)
     * Reconstruction uses rec_noise (rec=true) and rec_in as in_noise */
    if (p->alpha > 0.0f && p->rec_in.ctx) {
        /* Get reconstruction noise */
        mlx_array **rec_noises_ptr = NULL;
        int n_rec_noises = 0;
        if (mlx_faciesgan_get_pyramid_noise(
                    p->model, p->scale, p->indexes, p->n_indexes,
                    &rec_noises_ptr, &n_rec_noises,
                    p->wells_pyramid, p->seismic_pyramid, 1) == 0 && n_rec_noises > 0) {

            /* Convert mlx_array** to mlx_array* */
            mlx_array *rec_noises = malloc(n_rec_noises * sizeof(mlx_array));
            if (rec_noises) {
                for (int ni = 0; ni < n_rec_noises; ++ni) {
                    rec_noises[ni] = rec_noises_ptr[ni] ? *rec_noises_ptr[ni] : (mlx_array) {
                        0
                    };
                }

                /* Forward generator with rec_in as in_noise
                 * Python uses start_scale=scale, stop_scale=scale for reconstruction */
                mlx_array rec = mlx_generator_forward(gen, rec_noises, n_rec_noises,
                                                      p->amp, p->amp_count,
                                                      p->rec_in, p->scale, p->scale);

                if (rec.ctx) {
                    /* MSE(rec, real) */
                    mlx_array diff = mlx_array_new();
                    mlx_subtract(&diff, rec, p->real, s);

                    mlx_array diff_sq = mlx_array_new();
                    mlx_square(&diff_sq, diff, s);

                    mlx_array mse = mlx_array_new();
                    mlx_mean(&mse, diff_sq, false, s);

                    /* alpha * mse */
                    mlx_array alpha_arr = mlx_array_new_float(p->alpha);
                    mlx_array_free(rec_loss);
                    rec_loss = mlx_array_new();
                    mlx_multiply(&rec_loss, alpha_arr, mse, s);

                    /* Add to total loss */
                    mlx_array new_total = mlx_array_new();
                    mlx_add(&new_total, total_loss, rec_loss, s);
                    mlx_array_free(total_loss);
                    total_loss = new_total;

                    mlx_array_free(diff);
                    mlx_array_free(diff_sq);
                    mlx_array_free(mse);
                    mlx_array_free(alpha_arr);
                    mlx_array_free(rec);
                }

                free(rec_noises);
            }

            /* Cleanup rec noise pointers */
            for (int ni = 0; ni < n_rec_noises; ++ni) {
                if (rec_noises_ptr[ni]) {
                    mlx_array_free(*rec_noises_ptr[ni]);
                    mlx_free_pod((void **)&rec_noises_ptr[ni]);
                }
            }
            mlx_free_ptr_array((void ***)&rec_noises_ptr, n_rec_noises);
        }
    }

    /* Store recovery loss in metrics */
    if (p->out_metrics) {
        mlx_array rec_copy = mlx_array_new();
        mlx_array_set(&rec_copy, rec_loss);
        p->out_metrics->rec = rec_copy;
    }

    /* Cleanup intermediate losses */
    mlx_array_free(well_loss);
    mlx_array_free(div_loss);
    mlx_array_free(rec_loss);

    /* Return total loss as vector_array with single element */
    *result = mlx_vector_array_new();
    mlx_vector_array_append_value(*result, total_loss);

    /* Cleanup */
    /* fake is an alias for fakes[0] — free the whole fakes array instead */
    for (int di = 0; di < n_samples; ++di)
        if (fakes[di].ctx) mlx_array_free(fakes[di]);
    free(fakes);
    mlx_array_free(d_fake);
    mlx_array_free(mean_d_fake);
    mlx_array_free(adv_loss);
    mlx_array_free(total_loss);

    return 0;
}

/* ============================================================================
 * Helper closure for gradient penalty: D(x) -> scalar
 * This is used with mlx_vjp to compute ∂D/∂x for true WGAN-GP
 * ============================================================================ */

typedef struct {
    MLXFaciesGAN *model;
    int scale;
} GPClosurePayload;

static int gp_disc_forward_closure(mlx_vector_array *result,
                                   const mlx_vector_array inputs,
                                   void *payload_ptr) {
    GPClosurePayload *p = (GPClosurePayload *)payload_ptr;
    if (!p || !p->model) {
        return 1;
    }

    /* inputs[0] is x_interp */
    mlx_array x_interp = mlx_array_new();
    mlx_vector_array_get(&x_interp, inputs, 0);

    /* Forward discriminator */
    MLXDiscriminator *disc = mlx_faciesgan_build_discriminator(p->model);
    mlx_array d_out = mlx_discriminator_forward(disc, p->scale, x_interp);

    if (!d_out.ctx) {
        mlx_array_free(x_interp);
        return 1;
    }

    /* Sum the output to get scalar (required for vjp) */
    mlx_stream s = mlx_gpu_stream();
    mlx_array d_sum = mlx_array_new();
    mlx_sum(&d_sum, d_out, false, s);

    *result = mlx_vector_array_new();
    mlx_vector_array_append_value(*result, d_sum);

    mlx_array_free(d_sum);
    mlx_array_free(x_interp);
    mlx_array_free(d_out);

    return 0;
}

/* ============================================================================
 * Discriminator closure: computes disc loss given disc params
 *
 * Mirrors Python's compute_metrics pattern:
 *   def compute_metrics(params):
 *       self.discriminator.discs[scale].update(params)
 *       d_real = self.discriminator(scale, real)
 *       d_fake = self.discriminator(scale, fake)
 *       metrics.real = -d_real.mean()
 *       metrics.fake = d_fake.mean()
 *       metrics.gp = gradient_penalty(...)
 *       return metrics.real + metrics.fake + metrics.gp
 * ============================================================================ */

static int disc_loss_closure(mlx_vector_array *result,
                             const mlx_vector_array inputs,
                             void *payload_ptr) {
    DiscClosurePayload *p = (DiscClosurePayload *)payload_ptr;
    if (!p || !p->model) {
        fprintf(stderr, "[disc_loss_closure] NULL payload\n");
        return 1;
    }

    mlx_stream s = mlx_gpu_stream();

    /* === CRITICAL: Update discriminator parameters with traced inputs === */
    int vec_size = mlx_vector_array_size(inputs);
    if (vec_size != p->n_params) {
        fprintf(stderr, "[disc_loss_closure] param count mismatch: got %d, expected %d\n",
                vec_size, p->n_params);
        return 1;
    }

    for (int i = 0; i < p->n_params; ++i) {
        mlx_array traced_param = mlx_array_new();
        mlx_vector_array_get(&traced_param, inputs, i);
        mlx_array_set(p->param_ptrs[i], traced_param);
        mlx_array_free(traced_param);
    }

    /* Forward pass: D(real) and D(fake) */
    MLXDiscriminator *disc = mlx_faciesgan_build_discriminator(p->model);
    mlx_array d_real = mlx_discriminator_forward(disc, p->scale, p->real);
    mlx_array d_fake = mlx_discriminator_forward(disc, p->scale, p->fake);

    if (!d_real.ctx || !d_fake.ctx) {
        fprintf(stderr, "[disc_loss_closure] discriminator forward failed\n");
        mlx_array_free(d_real);
        mlx_array_free(d_fake);
        return 1;
    }

    /* WGAN loss: -mean(D(real)) + mean(D(fake))
     * Discriminator wants D(real) high and D(fake) low */
    mlx_array mean_real = mlx_array_new();
    mlx_array mean_fake = mlx_array_new();
    mlx_mean(&mean_real, d_real, false, s);
    mlx_mean(&mean_fake, d_fake, false, s);

    mlx_array neg_real = mlx_array_new();
    mlx_multiply(&neg_real, mlx_scalar_neg_one(), mean_real, s);

    mlx_array wgan_loss = mlx_array_new();
    mlx_add(&wgan_loss, neg_real, mean_fake, s);

    /* Store individual metrics if provided */
    if (p->out_metrics) {
        /* -mean(D(real)) */
        mlx_array real_copy = mlx_array_new();
        mlx_array_set(&real_copy, neg_real);
        p->out_metrics->real = real_copy;
        /* mean(D(fake)) */
        mlx_array fake_copy = mlx_array_new();
        mlx_array_set(&fake_copy, mean_fake);
        p->out_metrics->fake = fake_copy;
    }

    /* Initialize total loss with WGAN loss — direct copy avoids
     * mlx_array_new() + mlx_array_set() overhead */
    mlx_array total_loss = mlx_array_new();
    mlx_copy(&total_loss, wgan_loss, s);

    /* Initialize gradient penalty to zero */
    mlx_array gp_loss = mlx_array_new_float(0.0f);

    /* Compute TRUE gradient penalty using mlx_vjp for WGAN-GP:
     * 1. Interpolate: x_interp = alpha * real + (1 - alpha) * fake
     * 2. Use mlx_vjp to get gradient of sum(D(x_interp)) w.r.t. x_interp
     * 3. Compute ||grad||_2 per sample
     * 4. gp = lambda * mean((||grad|| - 1)^2)
     */
    if (p->lambda_grad > 0.0f) {
        /* Get batch size from real array shape */
        int ndim = (int)mlx_array_ndim(p->real);
        int batch = 1;
        if (ndim >= 1) {
            const int *shape = mlx_array_shape(p->real);
            if (shape) batch = shape[0];
        }

        /* 1. Generate alpha ~ U(0,1) with shape [batch, 1, 1, 1] for broadcasting */
        mlx_array alpha_lo = mlx_scalar_zero();  /* Perf: use pool */
        mlx_array alpha_hi = mlx_scalar_one();    /* Perf: use pool */
        int alpha_shape[4] = {batch, 1, 1, 1};
        mlx_array alpha_rand = mlx_array_new();
        mlx_array empty_key = {0};  /* empty key means use global RNG */
        mlx_random_uniform(&alpha_rand, alpha_lo, alpha_hi, alpha_shape, 4, MLX_FLOAT32, empty_key, s);

        /* one_minus_alpha = 1 - alpha */
        mlx_array one_arr = mlx_scalar_one();  /* Perf: use pool */
        mlx_array one_minus_alpha = mlx_array_new();
        mlx_subtract(&one_minus_alpha, one_arr, alpha_rand, s);

        /* term1 = alpha * real */
        mlx_array term1 = mlx_array_new();
        mlx_multiply(&term1, alpha_rand, p->real, s);

        /* term2 = (1 - alpha) * fake */
        mlx_array term2 = mlx_array_new();
        mlx_multiply(&term2, one_minus_alpha, p->fake, s);

        /* x_interp = term1 + term2 */
        mlx_array x_interp = mlx_array_new();
        mlx_add(&x_interp, term1, term2, s);

        /* 2. Create closure for D(x) and use mlx_vjp to get ∂D/∂x_interp */
        GPClosurePayload *gp_payload = malloc(sizeof(GPClosurePayload));
        if (gp_payload) {
            gp_payload->model = p->model;
            gp_payload->scale = p->scale;

            mlx_closure gp_cls = mlx_closure_new_func_payload(
                                     gp_disc_forward_closure, gp_payload, free);

            /* Use value_and_grad (matching Python's mx.grad which uses
             * value_and_grad internally) instead of mlx_vjp. */
            int gp_argnums[1] = {0};
            mlx_closure_value_and_grad gp_vag = mlx_closure_value_and_grad_new();
            if (mlx_value_and_grad(&gp_vag, gp_cls, gp_argnums, 1) != 0) {
                mlx_closure_value_and_grad_free(gp_vag);
                mlx_closure_free(gp_cls);
            } else {

                /* Inputs: [x_interp] */
                mlx_vector_array gp_inputs = mlx_vector_array_new();
                mlx_vector_array_append_value(gp_inputs, x_interp);

                /* Apply value_and_grad to get gradients w.r.t. x_interp */
                mlx_vector_array gp_values = mlx_vector_array_new();
                mlx_vector_array gp_grads = mlx_vector_array_new();

                if (mlx_closure_value_and_grad_apply(&gp_values, &gp_grads, gp_vag, gp_inputs) == 0) {
                    /* Extract gradient w.r.t. x_interp */
                    mlx_array grad_x = mlx_array_new();
                    mlx_vector_array_get(&grad_x, gp_grads, 0);

                    if (grad_x.ctx) {
                        /* 3. Compute ||grad||_2 per pixel (matching Python)
                         * grad_x shape is [batch, H, W, C]
                         * Python: grad_norm = sqrt(sum(grad^2, axis=-1) + 1e-12)
                         * So we sum over axis 3 (channels) only, NOT H, W */

                        /* grad^2 */
                        mlx_array grad_sq = mlx_array_new();
                        mlx_square(&grad_sq, grad_x, s);

                        /* Sum over C only (axis 3) - NOT H, W
                         * This gives per-pixel gradient norms with shape [batch, H, W] */
                        int axes[1] = {3};
                        mlx_array grad_sum = mlx_array_new();
                        mlx_sum_axes(&grad_sum, grad_sq, axes, 1, false, s);

                        /* Add epsilon for numerical stability (matching Python) */
                        /* Perf: use pooled GP epsilon (1e-12f) */
                        mlx_array gp_eps_arr = mlx_scalar_gp_eps();
                        mlx_array grad_sum_eps = mlx_array_new();
                        mlx_add(&grad_sum_eps, grad_sum, gp_eps_arr, s);

                        /* sqrt to get L2 norm per pixel */
                        mlx_array grad_norm = mlx_array_new();
                        mlx_sqrt(&grad_norm, grad_sum_eps, s);

                        /* 4. (grad_norm - 1)^2 */
                        mlx_array norm_minus_1 = mlx_array_new();
                        mlx_subtract(&norm_minus_1, grad_norm, mlx_scalar_one(), s);

                        mlx_array penalty_sq = mlx_array_new();
                        mlx_square(&penalty_sq, norm_minus_1, s);

                        /* Mean over all dimensions (batch, H, W) */
                        mlx_array gp_mean = mlx_array_new();
                        mlx_mean(&gp_mean, penalty_sq, false, s);

                        /* Perf: gp_eps_arr is from scalar pool, do NOT free it */
                        mlx_array_free(grad_sum_eps);

                        /* lambda * gp_mean */
                        mlx_array lambda_arr = mlx_array_new_float(p->lambda_grad);
                        mlx_array_free(gp_loss);
                        gp_loss = mlx_array_new();
                        mlx_multiply(&gp_loss, lambda_arr, gp_mean, s);

                        /* Add to total loss */
                        mlx_array new_total = mlx_array_new();
                        mlx_add(&new_total, total_loss, gp_loss, s);
                        mlx_array_free(total_loss);
                        total_loss = new_total;

                        /* Cleanup GP computation */
                        mlx_array_free(grad_sq);
                        mlx_array_free(grad_sum);
                        mlx_array_free(grad_norm);
                        mlx_array_free(norm_minus_1);
                        mlx_array_free(penalty_sq);
                        mlx_array_free(gp_mean);
                        mlx_array_free(lambda_arr);
                    }
                    mlx_array_free(grad_x);
                }

                /* Cleanup VaG arrays */
                mlx_vector_array_free(gp_values);
                mlx_vector_array_free(gp_grads);
                mlx_vector_array_free(gp_inputs);
                mlx_closure_value_and_grad_free(gp_vag);
                mlx_closure_free(gp_cls);
            } /* end if mlx_value_and_grad succeeded */
        }

        /* Cleanup interpolation arrays — pool scalars are NOT freed */
        mlx_array_free(alpha_rand);
        mlx_array_free(one_minus_alpha);
        mlx_array_free(term1);
        mlx_array_free(term2);
        mlx_array_free(x_interp);
    }

    /* Store gradient penalty in metrics */
    if (p->out_metrics) {
        mlx_array gp_copy = mlx_array_new();
        mlx_array_set(&gp_copy, gp_loss);
        p->out_metrics->gp = gp_copy;
    }

    mlx_array_free(gp_loss);

    /* Return loss as vector_array */
    *result = mlx_vector_array_new();
    mlx_vector_array_append_value(*result, total_loss);

    /* Cleanup */
    mlx_array_free(d_real);
    mlx_array_free(d_fake);
    mlx_array_free(mean_real);
    mlx_array_free(mean_fake);
    mlx_array_free(neg_real);
    /* wgan_loss was copied into total_loss via mlx_array_set; if GP was
     * computed total_loss was replaced (and the shared ref freed) but
     * wgan_loss still holds a dangling handle.  Free it unconditionally –
     * mlx_array_free on an already-released handle is safe. */
    mlx_array_free(wgan_loss);
    mlx_array_free(total_loss);

    return 0;
}

/* Payload destructors - don't free arrays owned by caller */
static void gen_payload_dtor(void *ptr) {
    GenClosurePayload *p = (GenClosurePayload *)ptr;
    if (p) {
        /* Free noise arrays we created */
        if (p->z_list) {
            for (int i = 0; i < p->z_count; ++i) {
                mlx_array_free(p->z_list[i]);
            }
            free(p->z_list);
        }
        if (p->amp) free(p->amp);
        /* Don't free param_ptrs - owned by model */
        free(p);
    }
}

static void disc_payload_dtor(void *ptr) {
    DiscClosurePayload *p = (DiscClosurePayload *)ptr;
    if (p) {
        /* Don't free arrays - owned by caller */
        free(p);
    }
}

/* ============================================================================
 * Public API: Compute generator loss and gradients
 * ============================================================================ */

int mlx_native_compute_gen_loss_and_grads(
    MLXFaciesGAN *m,
    int scale,
    mlx_array *z_list_in, int z_count_in,
    float *amp_in, int amp_count_in,
    mlx_array *real,
    mlx_array *wells,
    mlx_array *masks,
    mlx_array *rec_in,
    int *indexes, int n_indexes,
    mlx_array **wells_pyramid,
    mlx_array **seismic_pyramid,
    float lambda_diversity,
    float well_loss_penalty,
    float alpha,
    mlx_array *out_loss,
    mlx_array ***out_grads,
    int *out_n_grads,
    mlx_array *out_adv,
    mlx_array *out_well,
    mlx_array *out_div,
    mlx_array *out_rec) {

    if (!m || !out_loss || !out_grads || !out_n_grads || !real) {
        return -1;
    }

    if (!z_list_in || z_count_in <= 0) {
        /* z_list is optional after Fix 34 — gen_loss_closure generates its
         * own noise internally.  Only amp is strictly required. */
    }

    if (!amp_in || amp_count_in <= 0) {
        fprintf(stderr, "[mlx_native_compute_gen_loss_and_grads] amp arrays required\n");
        return -1;
    }

    mlx_stream s = mlx_gpu_stream();

    /* Get generator and its parameters for this scale */
    MLXGenerator *gen = mlx_faciesgan_build_generator(m);
    if (!gen) {
        return -1;
    }

    int n_params = 0;
    mlx_array **param_ptrs = mlx_generator_get_parameters_for_scale(gen, scale, &n_params);
    if (!param_ptrs || n_params == 0) {
        if (param_ptrs) mlx_generator_free_parameters_list(param_ptrs);
        return -1;
    }

    /* Copy z_list to internal storage if provided (payload dtor handles NULL).
     * After Fix 34 the gen_loss_closure generates its own noise, so z_list
     * may be NULL when called from the GEN_ONLY path. */
    mlx_array *z_list = NULL;
    int z_count_copy = 0;
    if (z_list_in && z_count_in > 0) {
        z_list = malloc(z_count_in * sizeof(mlx_array));
        if (!z_list) {
            mlx_generator_free_parameters_list(param_ptrs);
            return -1;
        }
        for (int i = 0; i < z_count_in; ++i) {
            z_list[i] = mlx_array_new();
            mlx_array_set(&z_list[i], z_list_in[i]);
        }
        z_count_copy = z_count_in;
    }

    /* Copy amp array */
    float *amp = malloc(amp_count_in * sizeof(float));
    if (!amp) {
        for (int i = 0; i < z_count_copy; ++i) mlx_array_free(z_list[i]);
        free(z_list);
        mlx_generator_free_parameters_list(param_ptrs);
        return -1;
    }
    memcpy(amp, amp_in, amp_count_in * sizeof(float));

    /* Create payload */
    GenClosurePayload *payload = calloc(1, sizeof(GenClosurePayload));
    if (!payload) {
        for (int i = 0; i < z_count_copy; ++i) mlx_array_free(z_list[i]);
        free(z_list);
        free(amp);
        mlx_generator_free_parameters_list(param_ptrs);
        return -1;
    }

    payload->model = m;
    payload->scale = scale;
    payload->real = *real;
    payload->well = wells ? *wells : (mlx_array) {
        0
    };
    payload->mask = masks ? *masks : (mlx_array) {
        0
    };
    payload->rec_in = rec_in ? *rec_in : (mlx_array) {
        0
    };
    payload->z_list = z_list;
    payload->z_count = z_count_copy;
    payload->amp = amp;
    payload->amp_count = amp_count_in;
    payload->lambda_diversity = lambda_diversity;
    payload->well_loss_penalty = well_loss_penalty;
    payload->alpha = alpha;
    payload->num_diversity_samples = mlx_faciesgan_get_num_diversity_samples(m);
    payload->indexes = indexes;
    payload->n_indexes = n_indexes;
    payload->wells_pyramid = wells_pyramid;
    payload->seismic_pyramid = seismic_pyramid;
    payload->param_ptrs = param_ptrs;
    payload->n_params = n_params;

    /* Set up metrics output structure */
    MLXGenMetrics gen_metrics = {0};
    payload->out_metrics = &gen_metrics;

    /* Create closure with payload */
    mlx_closure forward_cls = mlx_closure_new_func_payload(
                                  gen_loss_closure, payload, gen_payload_dtor);

    /* Perf: wrap closure with mlx_compile for GPU kernel fusion.
     * shapeless=false because input shapes (parameter tensors) are fixed
     * within a given scale. This enables MLX to fuse elementwise ops
     * (multiply, add, subtract, square, mean, exp) into fewer GPU kernels. */
    mlx_closure compiled_cls = mlx_closure_new();
    if (mlx_compile(&compiled_cls, forward_cls, false) == 0) {
        /* Use compiled closure */
        mlx_closure_free(forward_cls);
        forward_cls = compiled_cls;
    } else {
        /* Fallback: use uncompiled closure */
        mlx_closure_free(compiled_cls);
    }

    /* Create value_and_grad closure
     * argnums lists all argument indices we want gradients for (all params) */
    int *argnums = malloc(n_params * sizeof(int));
    if (!argnums) {
        mlx_closure_free(forward_cls);
        mlx_generator_free_parameters_list(param_ptrs);
        return -1;
    }
    for (int i = 0; i < n_params; ++i) {
        argnums[i] = i;
    }

    mlx_closure_value_and_grad vag = mlx_closure_value_and_grad_new();
    if (mlx_value_and_grad(&vag, forward_cls, argnums, n_params) != 0) {
        fprintf(stderr, "[mlx_native_compute_gen_loss_and_grads] mlx_value_and_grad failed\n");
        free(argnums);
        mlx_closure_free(forward_cls);
        mlx_generator_free_parameters_list(param_ptrs);
        return -1;
    }
    free(argnums);

    /* Prepare inputs: current parameter values as vector_array */
    mlx_vector_array inputs = mlx_vector_array_new();
    for (int i = 0; i < n_params; ++i) {
        mlx_vector_array_append_value(inputs, *param_ptrs[i]);
    }

    /* Apply value_and_grad */
    mlx_vector_array values = mlx_vector_array_new();
    mlx_vector_array grads = mlx_vector_array_new();

    if (mlx_closure_value_and_grad_apply(&values, &grads, vag, inputs) != 0) {
        fprintf(stderr, "[mlx_native_compute_gen_loss_and_grads] value_and_grad apply failed\n");
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(values);
        mlx_vector_array_free(grads);
        mlx_closure_value_and_grad_free(vag);
        mlx_closure_free(forward_cls);
        mlx_generator_free_parameters_list(param_ptrs);
        return -1;
    }

    /* Extract loss (first element of values) */
    *out_loss = mlx_array_new();
    mlx_vector_array_get(out_loss, values, 0);

    /* Extract gradients */
    int n_grads = mlx_vector_array_size(grads);
    *out_n_grads = n_grads;

    mlx_array **grad_arr = malloc(n_grads * sizeof(mlx_array *));
    if (!grad_arr) {
        mlx_array_free(*out_loss);
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(values);
        mlx_vector_array_free(grads);
        mlx_closure_value_and_grad_free(vag);
        mlx_closure_free(forward_cls);
        mlx_generator_free_parameters_list(param_ptrs);
        return -1;
    }

    for (int i = 0; i < n_grads; ++i) {
        grad_arr[i] = malloc(sizeof(mlx_array));
        *grad_arr[i] = mlx_array_new();
        mlx_vector_array_get(grad_arr[i], grads, i);
    }
    *out_grads = grad_arr;

    /* Copy individual metrics to output if requested */
    if (out_adv && gen_metrics.adv.ctx) {
        *out_adv = gen_metrics.adv;
    } else if (out_adv) {
        *out_adv = mlx_array_new_float(0.0f);
        if (gen_metrics.adv.ctx) mlx_array_free(gen_metrics.adv);
    } else if (gen_metrics.adv.ctx) {
        mlx_array_free(gen_metrics.adv);
    }

    if (out_well && gen_metrics.well.ctx) {
        *out_well = gen_metrics.well;
    } else if (out_well) {
        *out_well = mlx_array_new_float(0.0f);
        if (gen_metrics.well.ctx) mlx_array_free(gen_metrics.well);
    } else if (gen_metrics.well.ctx) {
        mlx_array_free(gen_metrics.well);
    }

    if (out_div && gen_metrics.div.ctx) {
        *out_div = gen_metrics.div;
    } else if (out_div) {
        *out_div = mlx_array_new_float(0.0f);
        if (gen_metrics.div.ctx) mlx_array_free(gen_metrics.div);
    } else if (gen_metrics.div.ctx) {
        mlx_array_free(gen_metrics.div);
    }

    if (out_rec && gen_metrics.rec.ctx) {
        *out_rec = gen_metrics.rec;
    } else if (out_rec) {
        *out_rec = mlx_array_new_float(0.0f);
        if (gen_metrics.rec.ctx) mlx_array_free(gen_metrics.rec);
    } else if (gen_metrics.rec.ctx) {
        mlx_array_free(gen_metrics.rec);
    }

    /* Cleanup */
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(values);
    mlx_vector_array_free(grads);
    mlx_closure_value_and_grad_free(vag);
    mlx_closure_free(forward_cls);
    mlx_generator_free_parameters_list(param_ptrs);

    return 0;
}

/* ============================================================================
 * Public API: Compute discriminator loss and gradients
 * ============================================================================ */

int mlx_native_compute_disc_loss_and_grads(
    MLXFaciesGAN *m,
    int scale,
    mlx_array *real,
    mlx_array *fake,
    float lambda_grad,
    mlx_array *out_loss,
    mlx_array ***out_grads,
    int *out_n_grads,
    mlx_array *out_d_real,
    mlx_array *out_d_fake,
    mlx_array *out_d_gp) {

    if (!m || !out_loss || !out_grads || !out_n_grads || !real || !fake) {
        return -1;
    }

    mlx_stream s = mlx_gpu_stream();

    /* Get discriminator parameters for this scale */
    MLXDiscriminator *disc = mlx_faciesgan_build_discriminator(m);
    if (!disc) {
        return -1;
    }

    void *disc_ptr = mlx_discriminator_get_disc_ptr(disc, scale);
    if (!disc_ptr) {
        return -1;
    }

    MLXSPADEDiscriminator *spade_disc = (MLXSPADEDiscriminator *)disc_ptr;
    int n_params = 0;
    mlx_array **param_ptrs = mlx_spadedisc_get_parameters(spade_disc, &n_params);
    if (!param_ptrs || n_params == 0) {
        if (param_ptrs) mlx_spadedisc_free_parameters_list(param_ptrs);
        return -1;
    }

    /* Create payload */
    DiscClosurePayload *payload = calloc(1, sizeof(DiscClosurePayload));
    if (!payload) {
        mlx_spadedisc_free_parameters_list(param_ptrs);
        return -1;
    }

    payload->model = m;
    payload->scale = scale;
    payload->real = *real;
    payload->fake = *fake;
    payload->lambda_grad = lambda_grad;
    payload->param_ptrs = param_ptrs;
    payload->n_params = n_params;

    /* Set up metrics output structure */
    MLXDiscMetrics disc_metrics = {0};
    payload->out_metrics = &disc_metrics;

    /* Create closure */
    mlx_closure forward_cls = mlx_closure_new_func_payload(
                                  disc_loss_closure, payload, disc_payload_dtor);

    /* Perf: wrap closure with mlx_compile for GPU kernel fusion.
     * shapeless=false because disc parameter shapes are fixed per scale. */
    mlx_closure compiled_cls = mlx_closure_new();
    if (mlx_compile(&compiled_cls, forward_cls, false) == 0) {
        mlx_closure_free(forward_cls);
        forward_cls = compiled_cls;
    } else {
        mlx_closure_free(compiled_cls);
    }

    /* Create value_and_grad - differentiate w.r.t. all parameters */
    int *argnums = malloc(n_params * sizeof(int));
    if (!argnums) {
        mlx_closure_free(forward_cls);
        mlx_spadedisc_free_parameters_list(param_ptrs);
        return -1;
    }
    for (int i = 0; i < n_params; ++i) {
        argnums[i] = i;
    }

    mlx_closure_value_and_grad vag = mlx_closure_value_and_grad_new();
    if (mlx_value_and_grad(&vag, forward_cls, argnums, n_params) != 0) {
        fprintf(stderr, "[mlx_native_compute_disc_loss_and_grads] mlx_value_and_grad failed\n");
        free(argnums);
        mlx_closure_free(forward_cls);
        mlx_spadedisc_free_parameters_list(param_ptrs);
        return -1;
    }
    free(argnums);

    /* Prepare inputs */
    mlx_vector_array inputs = mlx_vector_array_new();
    for (int i = 0; i < n_params; ++i) {
        mlx_vector_array_append_value(inputs, *param_ptrs[i]);
    }

    /* Apply value_and_grad */
    mlx_vector_array values = mlx_vector_array_new();
    mlx_vector_array grads = mlx_vector_array_new();

    if (mlx_closure_value_and_grad_apply(&values, &grads, vag, inputs) != 0) {
        fprintf(stderr, "[mlx_native_compute_disc_loss_and_grads] value_and_grad apply failed\n");
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(values);
        mlx_vector_array_free(grads);
        mlx_closure_value_and_grad_free(vag);
        mlx_closure_free(forward_cls);
        mlx_spadedisc_free_parameters_list(param_ptrs);
        return -1;
    }

    /* Extract loss */
    *out_loss = mlx_array_new();
    mlx_vector_array_get(out_loss, values, 0);

    /* Extract gradients */
    int n_grads = mlx_vector_array_size(grads);
    *out_n_grads = n_grads;

    mlx_array **grad_arr = malloc(n_grads * sizeof(mlx_array *));
    if (!grad_arr) {
        mlx_array_free(*out_loss);
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(values);
        mlx_vector_array_free(grads);
        mlx_closure_value_and_grad_free(vag);
        mlx_closure_free(forward_cls);
        mlx_spadedisc_free_parameters_list(param_ptrs);
        return -1;
    }

    for (int i = 0; i < n_grads; ++i) {
        grad_arr[i] = malloc(sizeof(mlx_array));
        *grad_arr[i] = mlx_array_new();
        mlx_vector_array_get(grad_arr[i], grads, i);
    }
    *out_grads = grad_arr;

    /* Copy individual metrics to output if requested */
    if (out_d_real && disc_metrics.real.ctx) {
        *out_d_real = disc_metrics.real;
    } else if (out_d_real) {
        *out_d_real = mlx_array_new_float(0.0f);
        if (disc_metrics.real.ctx) mlx_array_free(disc_metrics.real);
    } else if (disc_metrics.real.ctx) {
        mlx_array_free(disc_metrics.real);
    }

    if (out_d_fake && disc_metrics.fake.ctx) {
        *out_d_fake = disc_metrics.fake;
    } else if (out_d_fake) {
        *out_d_fake = mlx_array_new_float(0.0f);
        if (disc_metrics.fake.ctx) mlx_array_free(disc_metrics.fake);
    } else if (disc_metrics.fake.ctx) {
        mlx_array_free(disc_metrics.fake);
    }

    if (out_d_gp && disc_metrics.gp.ctx) {
        *out_d_gp = disc_metrics.gp;
    } else if (out_d_gp) {
        *out_d_gp = mlx_array_new_float(0.0f);
        if (disc_metrics.gp.ctx) mlx_array_free(disc_metrics.gp);
    } else if (disc_metrics.gp.ctx) {
        mlx_array_free(disc_metrics.gp);
    }

    /* Cleanup */
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(values);
    mlx_vector_array_free(grads);
    mlx_closure_value_and_grad_free(vag);
    mlx_closure_free(forward_cls);
    mlx_spadedisc_free_parameters_list(param_ptrs);

    return 0;
}

/* ============================================================================
 * Utility: Free gradients allocated by compute functions
 * ============================================================================ */

void mlx_native_free_grads(mlx_array **grads, int n) {
    if (!grads) return;
    for (int i = 0; i < n; ++i) {
        if (grads[i]) {
            mlx_array_free(*grads[i]);
            free(grads[i]);
        }
    }
    free(grads);
}
