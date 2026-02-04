#include "custom_layer.h"
#include "trainning/array_helpers.h"
#include "trainning/train_step.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mlx_array *make_synthetic_grad(const mlx_array *param) {
    if (!param)
        return NULL;
    mlx_array arr = *param;
    int ndim = (int)mlx_array_ndim(arr);
    const int *shape = mlx_array_shape(arr);
    size_t count = 1;
    for (int i = 0; i < ndim; ++i)
        count *= (size_t)shape[i];
    float *buf = NULL;
    if (count > (size_t)INT_MAX) {
        buf = (float *)malloc(sizeof(float) * count);
    } else {
        if (mlx_alloc_float_buf(&buf, (int)count) != 0)
            buf = NULL;
    }
    if (!buf)
        return NULL;
    for (size_t i = 0; i < count; ++i)
        buf[i] = 0.01f; /* small gradient */
    /* mlx_array_new_data expects a non-const int* for shape; cast away const here
    mlx_array a = mlx_array_new_data(buf, (int *)shape, ndim, MLX_FLOAT32);
    if (count > (size_t)INT_MAX)
        free(buf);
    else
        mlx_free_float_buf(&buf, NULL);
    mlx_array *ptr = NULL;
    if (mlx_alloc_pod((void **)&ptr, sizeof(mlx_array), 1) != 0) {
        mlx_array_free(a);
        return NULL;
    }
    *ptr = a;
    return ptr;
}

int main(void) {
    /* smoke-test message removed */

    MLXFaciesGAN *m = mlx_faciesgan_create(3, 3, 1, 3, 32, 8, 1, 1);
    if (!m) {
        fprintf(stderr, "failed to create faciesgan\n");
        return 1;
    }

    MLXGenerator *g = mlx_faciesgan_build_generator(m);
    if (!g) {
        fprintf(stderr, "failed to build generator\n");
        mlx_faciesgan_free(m);
        return 1;
    }
    /* create one generator scale */
    if (mlx_faciesgan_create_generator_scale(m, 0, 32, 8) != 0) {
        fprintf(stderr, "failed to create generator scale\n");
        mlx_faciesgan_free(m);
        return 1;
    }

    MLXDiscriminator *d = mlx_faciesgan_build_discriminator(m);
    if (!d) {
        fprintf(stderr, "failed to build discriminator\n");
        mlx_faciesgan_free(m);
        return 1;
    }
    /* create one discriminator scale */
    if (mlx_discriminator_create_scale(d, 32, 8) != 0) {
        fprintf(stderr, "failed to create discriminator scale\n");
        mlx_faciesgan_free(m);
        return 1;
    }

    int gen_n = 0;
    mlx_array **gen_params = mlx_generator_get_parameters(g, &gen_n);
    if (!gen_params || gen_n == 0) {
        fprintf(stderr, "no generator parameters found\n");
    }
    int disc_n = 0;
    mlx_array **disc_params = mlx_discriminator_get_parameters(d, &disc_n);
    if (!disc_params || disc_n == 0) {
        fprintf(stderr, "no discriminator parameters found\n");
    }

    /* create synthetic grads matching shapes */
    mlx_array **gen_grads = NULL;
    if (gen_n > 0) {
        if (mlx_alloc_pod((void **)&gen_grads, sizeof(mlx_array *), gen_n) != 0)
            gen_grads = NULL;
        else {
            for (int i = 0; i < gen_n; ++i)
                gen_grads[i] = make_synthetic_grad(gen_params[i]);
        }
    }
    mlx_generator_free_parameters_list(gen_params);

    mlx_array **disc_grads = NULL;
    if (disc_n > 0) {
        if (mlx_alloc_pod((void **)&disc_grads, sizeof(mlx_array *), disc_n) != 0)
            disc_grads = NULL;
        else {
            for (int i = 0; i < disc_n; ++i)
                disc_grads[i] = make_synthetic_grad(disc_params[i]);
        }
    }
    mlx_discriminator_free_parameters_list(disc_params);

    float gb1 = 0.0f, gb2 = 0.0f, geps = 0.0f;
    mlx_optimizer_get_global_adam_params(&gb1, &gb2, &geps);
    int gbc = mlx_optimizer_get_global_adam_bias_correction();
    float gwd = mlx_optimizer_get_global_adam_weight_decay();
    MLXOptimizer *opt_g = mlx_adam_create_ext(0.01f, gb1, gb2, geps, gbc, gwd);
    MLXOptimizer *opt_d = mlx_adam_create_ext(0.01f, gb1, gb2, geps, gbc, gwd);

    int r = mlx_faciesgan_train_step(m, opt_g, gen_grads, gen_n, opt_d,
                                     disc_grads, disc_n);
    /* train_step result print removed */

    /* cleanup grads */
    if (gen_grads) {
        for (int i = 0; i < gen_n; ++i) {
            if (gen_grads[i]) {
                mlx_array_free(*gen_grads[i]);
                mlx_free_pod((void **)&gen_grads[i]);
            }
        }
        mlx_free_pod((void **)&gen_grads);
    }
    if (disc_grads) {
        for (int i = 0; i < disc_n; ++i) {
            if (disc_grads[i]) {
                mlx_array_free(*disc_grads[i]);
                mlx_free_pod((void **)&disc_grads[i]);
            }
        }
        mlx_free_pod((void **)&disc_grads);
    }

    mlx_adam_free(opt_g);
    mlx_adam_free(opt_d);
    mlx_faciesgan_free(m);

    return 0;
}
