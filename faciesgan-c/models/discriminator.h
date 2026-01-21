#ifndef MLX_C_DISCRIMINATOR_H
#define MLX_C_DISCRIMINATOR_H

#include "custom_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLXDiscriminator MLXDiscriminator;

/* Create / free */
MLXDiscriminator *mlx_discriminator_create(int num_layer, int kernel_size, int padding_size, int input_channels);
void mlx_discriminator_free(MLXDiscriminator *m);

/* Add a scale-specific discriminator (allocates an internal MLXSPADEDiscriminator) */
int mlx_discriminator_create_scale(MLXDiscriminator *m, int num_features, int min_num_features);

/* Forward: run discriminator at given scale */
mlx_array_t mlx_discriminator_forward(MLXDiscriminator *m, int scale, mlx_array_t input);

/* Parameter collection helpers for discriminator */
mlx_array **mlx_discriminator_get_parameters(MLXDiscriminator *m, int *out_count);
void mlx_discriminator_free_parameters_list(mlx_array **list);

/* Dtype handling and helpers */
void mlx_discriminator_set_dtype(MLXDiscriminator *m, mlx_dtype dtype);
/* Call wrapper and eval stub */
mlx_array_t mlx_discriminator_call(MLXDiscriminator *m, int scale, mlx_array_t input);
void mlx_discriminator_eval(MLXDiscriminator *m, int enable);

#ifdef __cplusplus
}
#endif

#endif /* MLX_C_DISCRIMINATOR_H */

/* Return pointer to internal MLXSPADEDiscriminator for given scale (opaque).
	Caller must not free the returned pointer. */
void *mlx_discriminator_get_disc_ptr(MLXDiscriminator *m, int scale);
