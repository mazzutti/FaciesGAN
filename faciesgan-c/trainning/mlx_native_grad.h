/* mlx_native_grad.h - Native MLX autodiff using value_and_grad */

#ifndef MLX_NATIVE_GRAD_H
#define MLX_NATIVE_GRAD_H

#include "mlx/c/mlx.h"
#include "../models/facies_gan.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute generator loss and gradients using MLX's native value_and_grad.
 *
 * This is the preferred way to compute gradients - it uses MLX's built-in
 * autodiff which is more reliable than our custom autodiff implementation.
 *
 * @param m             The FaciesGAN model
 * @param scale         Current pyramid scale
 * @param z_list        Noise arrays for generator (flat array, one per scale up to current)
 * @param z_count       Number of noise arrays
 * @param amp           Noise amplitudes per scale
 * @param amp_count     Number of amplitudes
 * @param real          Real facies array for this scale
 * @param wells         Well conditioning (may be NULL)
 * @param masks         Mask for well loss (may be NULL)
 * @param rec_in        Reconstruction input from previous scale (may be NULL)
 * @param indexes       Batch indices for rec noise generation
 * @param n_indexes     Number of indices
 * @param wells_pyramid Wells pyramid for rec noise (may be NULL)
 * @param seismic_pyramid Seismic pyramid for rec noise (may be NULL)
 * @param lambda_diversity   Diversity loss weight
 * @param well_loss_penalty  Well loss weight
 * @param alpha         Recovery loss weight
 * @param out_loss      Output: scalar loss value
 * @param out_grads     Output: array of gradient arrays
 * @param out_n_grads   Output: number of gradients
 * @param out_adv       Output: adversarial loss (optional, can be NULL)
 * @param out_well      Output: well/masked loss (optional, can be NULL)
 * @param out_div       Output: diversity loss (optional, can be NULL)
 * @param out_rec       Output: recovery loss (optional, can be NULL)
 *
 * @return 0 on success, -1 on error
 */
int mlx_native_compute_gen_loss_and_grads(
    MLXFaciesGAN *m,
    int scale,
    mlx_array *z_list, int z_count,
    float *amp, int amp_count,
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
    mlx_array *out_rec);

/**
 * Compute discriminator loss and gradients using MLX's native value_and_grad.
 *
 * @param m             The FaciesGAN model
 * @param scale         Current pyramid scale
 * @param real          Real facies array
 * @param fake          Fake array from generator (should be detached/not require grad)
 * @param lambda_grad   Gradient penalty weight
 * @param out_loss      Output: scalar loss value
 * @param out_grads     Output: array of gradient arrays
 * @param out_n_grads   Output: number of gradients
 * @param out_d_real    Output: -mean(D(real)) (optional, can be NULL)
 * @param out_d_fake    Output: mean(D(fake)) (optional, can be NULL)
 * @param out_d_gp      Output: gradient penalty (optional, can be NULL)
 *
 * @return 0 on success, -1 on error
 */
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
    mlx_array *out_d_gp);

/**
 * Free gradients array allocated by the compute functions.
 */
void mlx_native_free_grads(mlx_array **grads, int n);

#ifdef __cplusplus
}
#endif

#endif /* MLX_NATIVE_GRAD_H */
