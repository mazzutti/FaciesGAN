#include "discriminator.h"
#include <stdlib.h>
#include <string.h>

struct MLXDiscriminator {
    int num_layer;
    int kernel_size;
    int padding_size;
    int input_channels;
    void **discs; /* array of MLXSPADEDiscriminator* */
    int n_discs;
    mlx_dtype dtype;
    int eval_mode;
};

MLXDiscriminator *mlx_discriminator_create(int num_layer, int kernel_size, int padding_size, int input_channels) {
    MLXDiscriminator *m = (MLXDiscriminator *)malloc(sizeof(MLXDiscriminator));
    if (!m) return NULL;
    m->num_layer = num_layer;
    m->kernel_size = kernel_size;
    m->padding_size = padding_size;
    m->input_channels = input_channels;
    m->discs = NULL;
    m->n_discs = 0;
    /* default dtype */
    m->dtype = MLX_FLOAT32;
    m->eval_mode = 0;
    return m;
}

void mlx_discriminator_free(MLXDiscriminator *m) {
    if (!m) return;
    if (m->discs) {
        for (int i = 0; i < m->n_discs; ++i) {
            if (m->discs[i]) mlx_spadedisc_free((MLXSPADEDiscriminator *)m->discs[i]);
        }
        free(m->discs);
    }
    free(m);
}

int mlx_discriminator_create_scale(MLXDiscriminator *m, int num_features, int min_num_features) {
    if (!m) return -1;
    MLXSPADEDiscriminator *disc = mlx_spadedisc_create(num_features, min_num_features, m->num_layer, m->kernel_size, m->padding_size, m->input_channels);
    if (!disc) return -1;
    /* propagate dtype if the sub-discriminator supports it (best-effort) */
    /* If mlx_spadedisc_set_dtype exists we would call it here. For now just store on parent. */
    void **tmp = (void **)realloc(m->discs, (size_t)(m->n_discs + 1) * sizeof(void *));
    if (!tmp) {
        mlx_spadedisc_free(disc);
        return -1;
    }
    m->discs = tmp;
    m->discs[m->n_discs] = (void *)disc;
    m->n_discs += 1;
    return 0;
}

mlx_array_t mlx_discriminator_forward(MLXDiscriminator *m, int scale, mlx_array_t input) {
    if (!m) return input;
    if (scale < 0 || scale >= m->n_discs) return input;
    MLXSPADEDiscriminator *disc = (MLXSPADEDiscriminator *)m->discs[scale];
    if (!disc) return input;
    return mlx_spadedisc_forward(disc, input);
}

/* Parameter collection implementation */
mlx_array **mlx_discriminator_get_parameters(MLXDiscriminator *m, int *out_count) {
    if (!m || !out_count) return NULL;
    /* Use helper to collect parameters from each SPADE discriminator */
    int total = 0;
    /* first pass: compute total */
    for (int i = 0; i < m->n_discs; ++i) {
        MLXSPADEDiscriminator *d = (MLXSPADEDiscriminator *)m->discs[i];
        if (!d) continue;
        int t = 0;
        mlx_array **tmp = mlx_spadedisc_get_parameters(d, &t);
        if (tmp) { total += t; mlx_spadedisc_free_parameters_list(tmp); }
    }
    if (total == 0) { *out_count = 0; return NULL; }
    mlx_array **list = (mlx_array **)malloc(sizeof(mlx_array *) * total);
    if (!list) { *out_count = 0; return NULL; }
    int idx = 0;
    for (int i = 0; i < m->n_discs; ++i) {
        MLXSPADEDiscriminator *d = (MLXSPADEDiscriminator *)m->discs[i];
        if (!d) continue;
        int t = 0;
        mlx_array **tmp = mlx_spadedisc_get_parameters(d, &t);
        if (!tmp) continue;
        for (int j = 0; j < t; ++j) list[idx++] = tmp[j];
        mlx_spadedisc_free_parameters_list(tmp);
    }
    *out_count = idx;
    return list;
}

void mlx_discriminator_free_parameters_list(mlx_array **list) { if (list) free(list); }

/* Set the preferred dtype for inputs/weights. Best-effort: stored on parent. */
void mlx_discriminator_set_dtype(MLXDiscriminator *m, mlx_dtype dtype) {
    if (!m) return;
    m->dtype = dtype;
}

/* Convenience wrapper matching Python __call__ style. */
mlx_array_t mlx_discriminator_call(MLXDiscriminator *m, int scale, mlx_array_t input) {
    return mlx_discriminator_forward(m, scale, input);
}

/* Eval mode stub: enables/disables evaluation behavior (e.g., batch-norm stat use). */
void mlx_discriminator_eval(MLXDiscriminator *m, int enable) {
    if (!m) return;
    m->eval_mode = enable ? 1 : 0;
}

void *mlx_discriminator_get_disc_ptr(MLXDiscriminator *m, int scale) {
    if (!m) return NULL;
    if (scale < 0 || scale >= m->n_discs) return NULL;
    return m->discs[scale];
}
