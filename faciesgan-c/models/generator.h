#ifndef MLX_C_GENERATOR_H
#define MLX_C_GENERATOR_H

#include "custom_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLXGenerator MLXGenerator;

/* Create / free */
MLXGenerator *mlx_generator_create(int num_layer, int kernel_size, int padding_size, int input_channels, int output_channels);
void mlx_generator_free(MLXGenerator *m);

/* Create and append a per-scale module */
int mlx_generator_create_scale(MLXGenerator *m, int scale, int num_features, int min_num_features);

/* Forward: z_list is an array of mlx_array values, z_count is its length.
 * amp is array of floats with amp_count entries. in_noise may be mlx_array_empty.
 * start_scale and stop_scale follow Python semantics; use -1 for default stop.
mlx_array_t mlx_generator_forward(
    MLXGenerator *m,
    const mlx_array *z_list,
    int z_count,
    const float *amp,
    int amp_count,
    mlx_array_t in_noise,
    int start_scale,
    int stop_scale
);

/* Convenience wrappers */
mlx_array_t mlx_generator_call(MLXGenerator *m, const mlx_array *z_list, int z_count, const float *amp, int amp_count, mlx_array_t in_noise, int start_scale, int stop_scale);
void mlx_generator_eval(MLXGenerator *m, int enable);

/* SPADE scale tracking helpers */
void mlx_generator_add_spade_scale(MLXGenerator *m, int scale);
int mlx_generator_has_spade_scale(MLXGenerator *m, int scale);
size_t mlx_generator_spade_scales_count(MLXGenerator *m);
int mlx_generator_spade_scale_at(MLXGenerator *m, size_t idx);

/* Gens introspection */
int mlx_generator_get_n_gens(MLXGenerator *m);
MLXScaleModule *mlx_generator_get_gen_ptr(MLXGenerator *m, int index);

/* Clear SPADE scales */
void mlx_generator_clear_spade_scales(MLXGenerator *m);

/* Safe accessors for per-scale modules */
int mlx_scale_is_spade(MLXGenerator *m, int index);
MLXSPADEGenerator *mlx_scale_get_spade(MLXGenerator *m, int index);
MLXConvBlock *mlx_scale_get_head(MLXGenerator *m, int index);
int mlx_scale_get_body_count(MLXGenerator *m, int index);
MLXConvBlock *mlx_scale_get_body_at(MLXGenerator *m, int index, int body_index);
int mlx_scale_has_tail_conv(MLXGenerator *m, int index);
mlx_array *mlx_scale_get_tail_conv(MLXGenerator *m, int index);

/* Parameter collection helpers */
/* Returns a malloc'd array of mlx_array* pointers and sets out_count. Caller must free the returned array (but not the arrays themselves). */
mlx_array **mlx_generator_get_parameters(MLXGenerator *m, int *out_count);
/* Returns parameters for a specific scale only (for scale-by-scale training) */
mlx_array **mlx_generator_get_parameters_for_scale(MLXGenerator *m, int scale_index, int *out_count);
void mlx_generator_free_parameters_list(mlx_array **list);

#ifdef __cplusplus
}
#endif

#endif /* MLX_C_GENERATOR_H */
