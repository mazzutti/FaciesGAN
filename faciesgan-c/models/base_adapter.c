#include "base_manager.h"
#include "facies_gan.h"
#include "options.h"
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
  (void)mgr;
  (void)scale;
  (void)reinit; /* no-op for now */
}

static void fg_finalize_discriminator_scale(void *mgr, int scale) {
  (void)mgr;
  (void)scale; /* no-op for now */
}

static void fg_save_generator_state(void *mgr, const char *path, int scale) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  MLXGenerator *g = mlx_faciesgan_build_generator(fg);
  if (!g)
    return;
  (void)g;
  (void)scale;
  /* delegate to facies_gan implementation */
  (void)mlx_faciesgan_save_generator_state(fg, path, scale);
}

static void fg_save_discriminator_state(void *mgr, const char *path,
                                        int scale) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  (void)mlx_faciesgan_save_discriminator_state(fg, path, scale);
}

static void fg_load_generator_state(void *mgr, const char *path, int scale) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  (void)mlx_faciesgan_load_generator_state(fg, path, scale);
}

static void fg_load_discriminator_state(void *mgr, const char *path,
                                        int scale) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  (void)mlx_faciesgan_load_discriminator_state(fg, path, scale);
}

static void fg_save_shape(void *mgr, const char *path, int scale) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  (void)mlx_faciesgan_save_shape(fg, path, scale);
}

static void fg_load_shape(void *mgr, const char *path, int scale) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  (void)mlx_faciesgan_load_shape(fg, path, scale);
}

static void fg_load_amp(void *mgr, const char *path) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  (void)mlx_faciesgan_load_amp(fg, path);
}
static void fg_load_wells(void *mgr, const char *path) {
  MLXFaciesGAN *fg = get_fg(mgr);
  if (!fg)
    return;
  (void)mlx_faciesgan_load_wells(fg, path);
}
