#ifndef FACIESGAN_TRAIN_UTILS_H
#define FACIESGAN_TRAIN_UTILS_H

#include <mlx/c/mlx.h>
#include "models/facies_gan.h"

#ifdef __cplusplus
extern "C" {
#endif

 * Compute reconstruction input for a given scale.
 * - scale == 0 -> returns zeros_like(facies_pyramid[0])
 * - otherwise: gathers entries from facies_pyramid[scale-1] at `indexes`
 *   and upsamples (linear, align_corners=1) to match facies_pyramid[scale]
 *
 * The returned pointer is malloc'd and must be freed by the caller with
 * `mlx_array_free(*out)` followed by `free(*out)`.
int mlx_compute_rec_input(int scale, const int *indexes, int n_indexes, mlx_array **facies_pyramid, mlx_array **out);

/* Initialize reconstruction noise amplitude for a given scale.
	 - computes RMSE between generator output (given current noise amps/defaults)
		 and the provided `real` facies tensor.
	 - updates the model's stored noise amps to include the computed value
		 (caller does not own any returned memory).
	 Returns 0 on success, non-zero on error.
int mlx_init_rec_noise_and_amp(MLXFaciesGAN *m, int scale, const int *indexes, int n_indexes, const mlx_array *real, mlx_array **wells_pyramid, mlx_array **seismic_pyramid);

#ifdef __cplusplus
}
#endif

#endif
