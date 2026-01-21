#ifndef MLX_BASE_MANAGER_H
#define MLX_BASE_MANAGER_H

#include "options.h"
#include <stddef.h>

typedef struct MLXBaseManagerCallbacks {
  void *(*build_generator)(void *mgr);
  void *(*build_discriminator)(void *mgr);
  void (*create_generator_scale)(void *mgr, int scale, int num_feature,
                                 int min_num_feature);
  void (*create_discriminator_scale)(void *mgr, int num_feature,
                                     int min_num_feature);
  void (*finalize_generator_scale)(void *mgr, int scale, int reinit);
  void (*finalize_discriminator_scale)(void *mgr, int scale);
  void (*save_generator_state)(void *mgr, const char *path, int scale);
  void (*save_discriminator_state)(void *mgr, const char *path, int scale);
  void (*load_generator_state)(void *mgr, const char *path, int scale);
  void (*load_discriminator_state)(void *mgr, const char *path, int scale);
  void (*save_shape)(void *mgr, const char *path, int scale);
  void (*load_shape)(void *mgr, const char *path, int scale);
  void (*load_amp)(void *mgr, const char *path);
  void (*load_wells)(void *mgr, const char *path);
} MLXBaseManagerCallbacks;

typedef struct MLXBaseManager MLXBaseManager;

MLXBaseManager *mlx_base_manager_create(const TrainningOptions *opts,
                                        const MLXBaseManagerCallbacks *cbs);
void mlx_base_manager_free(MLXBaseManager *mgr);

/* Convenience factory: create manager wired to MLXFaciesGAN internals */
MLXBaseManager *
mlx_base_manager_create_with_faciesgan(const TrainningOptions *opts);

/* Convenience wrapper: create manager from higher-level `TrainningOptions`.
 * This avoids callers needing to manually convert their TrainningOptions into
 * the MLXTrainOptions subset. Renamed to a clearer, shorter identifier. */
MLXBaseManager *
mlx_base_manager_create_from_trainning(const TrainningOptions *opts);

void mlx_base_manager_init_scales(MLXBaseManager *mgr, int start_scale,
                                  int num_scales);
void mlx_base_manager_init_generator_for_scale(MLXBaseManager *mgr, int scale);
void mlx_base_manager_init_discriminator_for_scale(MLXBaseManager *mgr,
                                                   int scale);

int mlx_base_manager_save_scale(MLXBaseManager *mgr, int scale,
                                const char *path);
int mlx_base_manager_load(MLXBaseManager *mgr, const char *path,
                          int load_shapes, int until_scale,
                          int load_discriminator, int load_wells);

/* Simple accessors */
int mlx_base_manager_add_shape(MLXBaseManager *mgr, const int *shape,
                               size_t ndim);
size_t mlx_base_manager_shapes_count(MLXBaseManager *mgr);

/* user context accessor for adapters */
void mlx_base_manager_set_user_ctx(MLXBaseManager *mgr, void *ctx);
void *mlx_base_manager_get_user_ctx(MLXBaseManager *mgr);

#endif
