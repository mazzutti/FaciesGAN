/*
 * Pure-C port header for custom layers (models/mlx/custom_layer.py)
 * Targets the mlx-c C API. Function names and types assume the
 * mlx-c public headers provide `mlx_core` and `mlx_nn` symbols.
 *
 * This header defines opaque structs for the modules and simple
 * create/destroy/forward functions. The implementations in the
 * corresponding .c file contain explanatory notes where the mlx-c
 * C API naming may need adjustment to match the installed headers.
 */

#ifndef MLX_C_CUSTOM_LAYER_H
#define MLX_C_CUSTOM_LAYER_H

#include <stddef.h>
#include <mlx/c/mlx.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* Use the mlx-c `mlx_array` type */
    typedef mlx_array mlx_array_t;

    /* MLXLeakyReLU */
    typedef struct MLXLeakyReLU MLXLeakyReLU;
    MLXLeakyReLU *mlx_leakyrelu_create(float negative_slope);
    void mlx_leakyrelu_free(MLXLeakyReLU *m);
    mlx_array_t mlx_leakyrelu_forward(MLXLeakyReLU *m, mlx_array_t x);

    /* MLXConvBlock */
    typedef struct MLXConvBlock MLXConvBlock;
    MLXConvBlock *mlx_convblock_create(int in_ch, int out_ch, int kernel_size, int padding, int stride, int use_norm);
    void mlx_convblock_free(MLXConvBlock *m);
    mlx_array_t mlx_convblock_forward(MLXConvBlock *m, mlx_array_t x);

    /* MLXUpsample */
    typedef struct MLXUpsample MLXUpsample;
    MLXUpsample *mlx_upsample_create(int out_h, int out_w, const char *mode, int align_corners);
    void mlx_upsample_free(MLXUpsample *m);
    mlx_array_t mlx_upsample_forward(MLXUpsample *m, mlx_array_t x);

    /* MLXSPADE */
    typedef struct MLXSPADE MLXSPADE;
    MLXSPADE *mlx_spade_create(int norm_nc, int cond_nc, int hidden_nc, int kernel_size);
    void mlx_spade_free(MLXSPADE *m);
    /* forward: x and conditioning_input are mlx_array_t; returns new mlx_array_t */
    mlx_array_t mlx_spade_forward(MLXSPADE *m, mlx_array_t x, mlx_array_t conditioning_input, const char *mode, int align_corners);
    /* Accessors for internal mlp weights of MLXSPADE (useful for AG tracing) */
    mlx_array *mlx_spade_get_mlp_shared_w(MLXSPADE *m);
    mlx_array *mlx_spade_get_mlp_gamma_w(MLXSPADE *m);
    mlx_array *mlx_spade_get_mlp_beta_w(MLXSPADE *m);

    /* MLXSPADEConvBlock */
    typedef struct MLXSPADEConvBlock MLXSPADEConvBlock;
    MLXSPADEConvBlock *mlx_spadeconv_create(int in_ch, int out_ch, int cond_ch, int kernel_size, int padding, int stride, int spade_hidden);
    void mlx_spadeconv_free(MLXSPADEConvBlock *m);
    mlx_array_t mlx_spadeconv_forward(MLXSPADEConvBlock *m, mlx_array_t x, mlx_array_t cond);
    /* Introspection helpers (return pointers owned by module; do not free) */
    mlx_array *mlx_spadeconv_get_conv_weight(MLXSPADEConvBlock *m);
    /* For non-SPADE conv blocks */
    mlx_array *mlx_convblock_get_conv_weight(MLXConvBlock *m);

    /* MLXSPADEGenerator */
    typedef struct MLXSPADEGenerator MLXSPADEGenerator;
    MLXSPADEGenerator *mlx_spadegen_create(int num_layer, int kernel_size, int padding_size, int num_features, int min_num_features, int output_channels, int input_channels);
    void mlx_spadegen_free(MLXSPADEGenerator *m);
    mlx_array_t mlx_spadegen_forward(MLXSPADEGenerator *m, mlx_array_t cond);
    /* Return malloc'd array of mlx_array* pointers for init_conv, each block conv, tail_conv. Caller must free with mlx_spadegen_free_parameters_list. */
    mlx_array **mlx_spadegen_get_parameters(MLXSPADEGenerator *m, int *out_count);
    void mlx_spadegen_free_parameters_list(mlx_array **list);
    /* Additional accessors useful for AG tracing */
    int mlx_spadegen_get_n_blocks(MLXSPADEGenerator *m);
    MLXSPADEConvBlock *mlx_spadegen_get_block_at(MLXSPADEGenerator *m, int idx);
    mlx_array *mlx_spadegen_get_init_conv(MLXSPADEGenerator *m);
    mlx_array *mlx_spadegen_get_tail_conv(MLXSPADEGenerator *m);
    /* MLXSPADEConvBlock accessors */
    int mlx_spadeconv_get_padding(MLXSPADEConvBlock *m);
    MLXLeakyReLU *mlx_spadeconv_get_activation(MLXSPADEConvBlock *m);
    MLXSPADE *mlx_spadeconv_get_spade(MLXSPADEConvBlock *m);
    int mlx_spadeconv_get_stride(MLXSPADEConvBlock *m);

    /* MLXSPADE accessors */
    int mlx_spade_get_padding(MLXSPADE *m);

    /* MLXScaleModule */
    typedef struct MLXScaleModule MLXScaleModule;
    MLXScaleModule *mlx_scalemodule_create(void *head, void *body, void *tail); /* opaque pointers to modules */
    void mlx_scalemodule_free(MLXScaleModule *m);
    mlx_array_t mlx_scalemodule_forward(MLXScaleModule *m, mlx_array_t x);

    /* MLXSPADEDiscriminator */
    typedef struct MLXSPADEDiscriminator MLXSPADEDiscriminator;
    MLXSPADEDiscriminator *mlx_spadedisc_create(int num_features, int min_num_features, int num_layer, int kernel_size, int padding_size, int input_channels);
    void mlx_spadedisc_free(MLXSPADEDiscriminator *m);
    mlx_array_t mlx_spadedisc_forward(MLXSPADEDiscriminator *m, mlx_array_t x);
    /* Return malloc'd array of mlx_array* pointers for head, each body conv, tail. Caller must free with mlx_spadedisc_free_parameters_list. */
    mlx_array **mlx_spadedisc_get_parameters(MLXSPADEDiscriminator *m, int *out_count);
    void mlx_spadedisc_free_parameters_list(mlx_array **list);
    /* Accessors for discriminator internals (useful for AG tracing) */
    mlx_array *mlx_spadedisc_get_head_conv(MLXSPADEDiscriminator *m);
    int mlx_spadedisc_get_body_count(MLXSPADEDiscriminator *m);
    MLXSPADEConvBlock *mlx_spadedisc_get_body_at(MLXSPADEDiscriminator *m, int idx);
    mlx_array *mlx_spadedisc_get_tail_conv(MLXSPADEDiscriminator *m);

    /* MLXColorQuantization */
    typedef struct MLXColorQuantization MLXColorQuantization;
    MLXColorQuantization *mlx_colorquant_create(float temperature);
    void mlx_colorquant_free(MLXColorQuantization *m);
    mlx_array_t mlx_colorquant_forward(MLXColorQuantization *m, mlx_array_t x, int training);

    /* (Instrumentation removed) */

#ifdef __cplusplus
}
#endif

#endif /* MLX_C_CUSTOM_LAYER_H */
