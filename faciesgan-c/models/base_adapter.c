#include "base_manager.h"
#include "facies_gan.h"
#include "options.h"
#include "trainning/array_helpers.h"
#include <stdio.h>
#include <stdlib.h>

/* Forward declarations for local callbacks */
static void *fg_build_generator(void *mgr);
static void *fg_build_discriminator(void *mgr);
static void fg_create_generator_scale(void *mgr, int scale, int num_feature,
                                      int min_num_feature);
static void fg_create_discriminator_scale(void *mgr, int num_feature,
        int min_num_feature);
static void fg_finalize_generator_scale(void *mgr, int scale, int reinit);
static void fg_finalize_discriminator_scale(void *mgr, int scale);
static void fg_save_generator_state(void *mgr, const char *path, int scale);
static void fg_save_discriminator_state(void *mgr, const char *path, int scale);
static void fg_load_generator_state(void *mgr, const char *path, int scale);
static void fg_load_discriminator_state(void *mgr, const char *path, int scale);
static void fg_save_shape(void *mgr, const char *path, int scale);
static void fg_load_shape(void *mgr, const char *path, int scale);
static void fg_load_amp(void *mgr, const char *path);
static void fg_load_wells(void *mgr, const char *path);

/* Create a manager wired to an internal MLXFaciesGAN instance */
MLXBaseManager *
mlx_base_manager_create_with_faciesgan(const TrainningOptions *opts) {
    if (!opts)
        return NULL;
    MLXFaciesGAN *fg = mlx_faciesgan_create(
                           opts->num_layer, opts->kernel_size, opts->padding_size,
                           opts->num_img_channels, opts->num_feature, opts->min_num_feature,
                           opts->discriminator_steps, opts->generator_steps);
    if (!fg)
        return NULL;

    /* Configure diversity sample count on created faciesgan for C metrics parity
     */
    if (opts->num_diversity_samples > 0)
        mlx_faciesgan_set_num_diversity_samples(fg, opts->num_diversity_samples);

    /* Configure generator input channels to include wells/seismic conditioning */
    int gen_input_ch = opts->noise_channels;
    if (opts->use_wells)
        gen_input_ch += opts->num_img_channels;
    if (opts->use_seismic)
        gen_input_ch += opts->num_img_channels;
    mlx_faciesgan_set_gen_input_channels(fg, gen_input_ch);

    MLXBaseManagerCallbacks cbs = {0};
    cbs.build_generator = fg_build_generator;
    cbs.build_discriminator = fg_build_discriminator;
    cbs.create_generator_scale = fg_create_generator_scale;
    cbs.create_discriminator_scale = fg_create_discriminator_scale;
    cbs.finalize_generator_scale = fg_finalize_generator_scale;
    cbs.finalize_discriminator_scale = fg_finalize_discriminator_scale;
    cbs.save_generator_state = fg_save_generator_state;
    cbs.save_discriminator_state = fg_save_discriminator_state;
    cbs.load_generator_state = fg_load_generator_state;
    cbs.load_discriminator_state = fg_load_discriminator_state;
    cbs.save_shape = fg_save_shape;
    cbs.load_shape = fg_load_shape;
    cbs.load_amp = fg_load_amp;
    cbs.load_wells = fg_load_wells;

    MLXBaseManager *mgr = mlx_base_manager_create(opts, &cbs);
    if (!mgr) {
        mlx_faciesgan_free(fg);
        return NULL;
    }
    mlx_base_manager_set_user_ctx(mgr, fg);
    return mgr;
}

MLXBaseManager *
mlx_base_manager_create_from_trainning(const TrainningOptions *opts) {
    if (!opts)
        return NULL;
    /* The base manager accepts `TrainningOptions` directly. Forward the
     * pointer to avoid duplicating the options struct. */
    return mlx_base_manager_create_with_faciesgan(opts);
}

/* Callbacks implementation: retrieve MLXFaciesGAN from manager user_ctx */
static MLXFaciesGAN *get_fg(void *mgr) {
    return (MLXFaciesGAN *)mlx_base_manager_get_user_ctx((MLXBaseManager *)mgr);
}

static void *fg_build_generator(void *mgr) {
    MLXFaciesGAN *fg = get_fg(mgr);
    return fg ? (void *)mlx_faciesgan_build_generator(fg) : NULL;
}

static void *fg_build_discriminator(void *mgr) {
    MLXFaciesGAN *fg = get_fg(mgr);
    return fg ? (void *)mlx_faciesgan_build_discriminator(fg) : NULL;
}

static void fg_create_generator_scale(void *mgr, int scale, int num_feature,
                                      int min_num_feature) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_create_generator_scale(fg, scale, num_feature, min_num_feature);
}

static void fg_create_discriminator_scale(void *mgr, int num_feature,
        int min_num_feature) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    MLXDiscriminator *d = mlx_faciesgan_build_discriminator(fg);
    if (d)
        mlx_discriminator_create_scale(d, num_feature, min_num_feature);
}

static void fg_finalize_generator_scale(void *mgr, int scale, int reinit) {
    /* FIX 38 — no-op for ALL scales.
     *
     * Python's MLXFaciesGAN.finalize_generator_scale does two things:
     *   if reinit: utils.init_weights(gen)        [randomize]
     *   else:      gen.update(prev.parameters())   [copy from prev scale]
     *
     * However, Python's trainer.__init__ calls reset_parameters() AFTER
     * mx.random.seed(), which reinitializes ALL scales' Conv2d weights with
     * fresh seeded random values — overwriting whatever finalize did.
     *
     * In C, the seed is set BEFORE layer creation, so creation-time weights
     * are already the correct seeded values (equivalent to Python's
     * reset_parameters).  If we copied scale-1 → scale here (as the old
     * code did for non-SPADE scale ≥ 2), we would overwrite gen2-6's
     * unique seeded weights with gen1's values, causing a parameter
     * mismatch vs Python.
     *
     * Therefore: do nothing.  The creation-time weights are final.
     */
    (void)mgr;
    (void)scale;
    (void)reinit;
}

/**
 * Re-initialise a single MLXConvBlock in place to match Python's reset_parameters:
 *   Conv2d  weight → N(0, 0.02),  bias → zeros
 *   InstanceNorm weight → ones(), bias → zeros  (Fix 37: no RNG consumed)
 */
static void reinit_convblock_weights(MLXConvBlock *cb) {
    if (!cb)
        return;
    mlx_stream s = mlx_gpu_stream();

    /* --- Conv2d weight: N(0, 0.02) --- */
    mlx_array *w = mlx_convblock_get_conv_weight(cb);
    if (w && w->ctx) {
        int ndim = mlx_array_ndim(*w);
        const int *shape = mlx_array_shape(*w);
        mlx_array new_w = mlx_array_new();
        if (mlx_random_normal(&new_w, shape, ndim, MLX_FLOAT32,
                              0.0f, 0.02f, mlx_array_empty, s) == 0) {
            mlx_array_set(w, new_w);
        }
        mlx_array_free(new_w);
    }

    /* --- Conv2d bias: zeros --- */
    mlx_array *b = mlx_convblock_get_conv_bias(cb);
    if (b && b->ctx) {
        int ndim = mlx_array_ndim(*b);
        const int *shape = mlx_array_shape(*b);
        mlx_array new_b = mlx_array_new();
        mlx_zeros(&new_b, shape, ndim, MLX_FLOAT32, s);
        mlx_array_set(b, new_b);
        mlx_array_free(new_b);
    }

    /* --- InstanceNorm weight: ones() (Fix 37 — matches Python's
     * _init_instance_norm: norm.weight = mx.ones(weight.shape)) --- */
    mlx_array *nw = mlx_convblock_get_norm_weight(cb);
    if (nw && nw->ctx) {
        int ndim = mlx_array_ndim(*nw);
        const int *shape = mlx_array_shape(*nw);
        mlx_array new_nw = mlx_array_new();
        mlx_ones(&new_nw, shape, ndim, MLX_FLOAT32, s);
        mlx_array_set(nw, new_nw);
        mlx_array_free(new_nw);
    }

    /* --- InstanceNorm bias: zeros --- */
    mlx_array *nb = mlx_convblock_get_norm_bias(cb);
    if (nb && nb->ctx) {
        int ndim = mlx_array_ndim(*nb);
        const int *shape = mlx_array_shape(*nb);
        mlx_array new_nb = mlx_array_new();
        mlx_zeros(&new_nb, shape, ndim, MLX_FLOAT32, s);
        mlx_array_set(nb, new_nb);
        mlx_array_free(new_nb);
    }

}

static void fg_finalize_discriminator_scale(void *mgr, int scale) {
    /* FIX 36: Skip reinitialisation — match Python's RNG consumption order.
     *
     * In Python, model creation happens BEFORE mx.random.seed() is set,
     * and then reset_parameters() calls init_weights() AFTER the seed.
     * So the post-seed RNG consumption for scale 0 is:
     *   gen0 init_weights → disc0 init_weights
     *
     * In C, the seed is set BEFORE model creation, so creation-time random
     * init IS the seeded init.  If we also reinitialise here, the disc
     * weights get DOUBLE-initialised:
     *   gen0 create (N_gen RNG) → disc0 create (N_disc RNG) → disc0 finalize (N_disc RNG)
     *
     * That shifts all subsequent random numbers by N_disc positions compared
     * to Python.  By skipping the reinit, disc weights come from creation-time
     * random init at the correct RNG positions:
     *   gen0 create (N_gen RNG) → disc0 create (N_disc RNG)
     *
     * The creation-time init already uses the correct distributions:
     *   Conv2d weight  → N(0, 0.02)   [consumes RNG]
     *   Conv2d bias    → zeros         [no RNG]
     *   InstanceNorm weight → ones()   [no RNG — Fix 37]
     *   InstanceNorm bias   → zeros    [no RNG]
     */
    (void)mgr;
    (void)scale;
}

static void fg_save_generator_state(void *mgr, const char *path, int scale) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    MLXGenerator *g = mlx_faciesgan_build_generator(fg);
    if (!g)
        return;
    (void)g; /* generator object created only for API parity */
    /* delegate to facies_gan implementation */
    mlx_faciesgan_save_generator_state(fg, path, scale);
}

static void fg_save_discriminator_state(void *mgr, const char *path,
                                        int scale) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_save_discriminator_state(fg, path, scale);
}

static void fg_load_generator_state(void *mgr, const char *path, int scale) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_load_generator_state(fg, path, scale);
}

static void fg_load_discriminator_state(void *mgr, const char *path,
                                        int scale) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_load_discriminator_state(fg, path, scale);
}

static void fg_save_shape(void *mgr, const char *path, int scale) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_save_shape(fg, path, scale);
}

static void fg_load_shape(void *mgr, const char *path, int scale) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_load_shape(fg, path, scale);
}

static void fg_load_amp(void *mgr, const char *path) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_load_amp(fg, path);
}
static void fg_load_wells(void *mgr, const char *path) {
    MLXFaciesGAN *fg = get_fg(mgr);
    if (!fg)
        return;
    mlx_faciesgan_load_wells(fg, path);
}
