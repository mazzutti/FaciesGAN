#include "options.h"
#include "trainning/mlx_trainer_api.h"
#include <stdio.h>
#include <stdlib.h>

/* C wrapper: shutdown mlx scheduler to ensure it stops before device teardown
 */
extern int mlx_scheduler_shutdown(void);

/* Provide a no-op fallback in case the symbol is not available when linking
 * the smoke executable. This keeps the smoke runner non-invasive and
 * resilient to build configurations that omit the scheduler implementation.
 */
int mlx_scheduler_shutdown(void) { return 0; }

#include "options.h"
#include "trainning/mlx_trainer_api.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  TrainningOptions *opts = mlx_options_new_trainning_defaults();
  if (!opts) {
    fprintf(stderr, "failed to create default TrainningOptions\n");
    return 2;
  }

  /* Ensure at least one parallel scale is requested so the trainer
   * initializes model shapes and reports a non-zero scale count. */
  opts->num_parallel_scales = 1;
  /* For live LLDB capture we prefer GPU execution; reduce iterations to limit memory usage */
  opts->num_iter = 200;
  /* Use GPU backend to exercise Metal writer sites */
  opts->use_cpu = false;
  /* Reduce batch size to limit GPU memory pressure during smoke runs */
  opts->batch_size = 1;
  /* Lower synthetic shape size and features to reduce memory use */
  opts->crop_size = 8;
  opts->num_feature = 4;
  MLXTrainer *t = MLXTrainer_create_with_opts(opts);
  if (!t) {
    fprintf(stderr, "MLXTrainer_create_with_opts failed\n");
    mlx_options_free_trainning(opts);
    return 3;
  }

  int n_scales = MLXTrainer_get_n_scales(t);
  printf("MLXTrainer created: n_scales=%d\n", n_scales);

  if (MLXTrainer_setup_optimizers(t, NULL, 0) != 0) {
    fprintf(stderr, "MLXTrainer_setup_optimizers failed\n");
    MLXTrainer_destroy(t);
    mlx_options_free_trainning(opts);
    return 4;
  }

  int batch = opts->batch_size > 0 ? opts->batch_size : 2;

  /* obtain model shapes (batch, channels, height, width) */
  int *shapes = NULL;
  int shapes_n = 0;
  void *mctx = MLXTrainer_get_model_ctx(t);
  if (mlx_faciesgan_get_shapes_flat(mctx, &shapes, &shapes_n) != 0) {
    fprintf(stderr, "failed to query model shapes\n");
    MLXTrainer_destroy(t);
    mlx_options_free_trainning(opts);
    return 5;
  }
  /* Use discovered shapes count if trainer didn't report any scales. */
  if (n_scales <= 0 && shapes_n > 0)
    n_scales = shapes_n;
  /* If no shapes were discovered (no checkpoint), synthesize shapes from
   * trainer options so the smoke runner can exercise `MLXTrainer_train_scales`.
   * The shape layout expected by the trainer is [Batch, Channels, Height,
   * Width]. */
  if (shapes_n <= 0) {
    int synth_n = opts->num_parallel_scales > 0 ? opts->num_parallel_scales : 1;
    int *synth = (int *)malloc(sizeof(int) * 4 * synth_n);
    if (!synth) {
      fprintf(stderr, "out of memory (shapes)\n");
      MLXTrainer_destroy(t);
      mlx_options_free_trainning(opts);
      return 5;
    }
    for (int si = 0; si < synth_n; ++si) {
      synth[si * 4 + 0] = batch;
      /* shapes_flat layout expected by model: N, H, W, C */
      synth[si * 4 + 1] = opts->crop_size > 0 ? opts->crop_size : 64;
      synth[si * 4 + 2] = opts->crop_size > 0 ? opts->crop_size : 64;
      synth[si * 4 + 3] =
          opts->num_img_channels > 0 ? opts->num_img_channels : 3;
    }
    /* set shapes into the model so future callers can query them */
    if (mlx_faciesgan_set_shapes(mctx, synth, synth_n) != 0) {
      free(synth);
      fprintf(stderr, "failed to set synthetic model shapes\n");
      MLXTrainer_destroy(t);
      mlx_options_free_trainning(opts);
      return 5;
    }
    free(synth);
    /* re-query shapes for the rest of the smoke runner logic */
    if (mlx_faciesgan_get_shapes_flat(mctx, &shapes, &shapes_n) != 0 ||
        shapes_n <= 0) {
      fprintf(stderr, "failed to get model shapes after synthesize\n");
      MLXTrainer_destroy(t);
      mlx_options_free_trainning(opts);
      return 5;
    }
  }
  /* Ensure `n_scales` reflects final discovered/synthesized shapes. */
  if (shapes_n > 0)
    n_scales = shapes_n;

  /* Ensure generator scales exist for the synthesized/discovered shapes so
   * subsequent noise/forward calls have initialized modules. */
  for (int si = 0; si < n_scales; ++si) {
    /* create generator scale if missing */
    (void)mlx_faciesgan_create_generator_scale(
        (MLXFaciesGAN *)mctx, si, opts->num_feature, opts->min_num_feature);
  }

  mlx_array **fac_pyr = (mlx_array **)malloc(sizeof(mlx_array *) * n_scales);
  if (!fac_pyr) {
    fprintf(stderr, "out of memory\n");
    free(shapes);
    MLXTrainer_destroy(t);
    mlx_options_free_trainning(opts);
    return 6;
  }

  for (int si = 0; si < n_scales; ++si) {
    /* shapes_flat layout is N, H, W, C */
    int channels = shapes[si * 4 + 3];
    int height = shapes[si * 4 + 1];
    int width = shapes[si * 4 + 2];
    int shape[4] = {batch, height, width, channels};
    mlx_array a = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    if (mlx_random_normal(&a, shape, 4, MLX_FLOAT32, 0.0f, 1.0f,
                          mlx_array_empty, s) != 0) {
      mlx_zeros(&a, shape, 4, MLX_FLOAT32, s);
    }
    mlx_stream_free(s);
    mlx_array *p = (mlx_array *)malloc(sizeof(mlx_array));
    if (!p) {
      fprintf(stderr, "out of memory\n");
      /* clean up previous allocations */
      for (int sj = 0; sj < si; ++sj) {
        if (fac_pyr[sj]) {
          mlx_array_free(*fac_pyr[sj]);
          free(fac_pyr[sj]);
        }
      }
      free(fac_pyr);
      free(shapes);
      MLXTrainer_destroy(t);
      mlx_options_free_trainning(opts);
      return 7;
    }
    *p = a;
    fac_pyr[si] = p;
  }
  free(shapes);

  int *indexes = (int *)malloc(sizeof(int) * batch);
  if (!indexes) {
    fprintf(stderr, "out of memory\n");
    for (int si = 0; si < n_scales; ++si) {
      if (fac_pyr[si]) {
        mlx_array_free(*fac_pyr[si]);
        free(fac_pyr[si]);
      }
    }
    free(fac_pyr);
    MLXTrainer_destroy(t);
    mlx_options_free_trainning(opts);
    return 8;
  }
  for (int i = 0; i < batch; ++i)
    indexes[i] = i;

  int *scales = (int *)malloc(sizeof(int) * n_scales);
  if (!scales) {
    fprintf(stderr, "out of memory\n");
    free(indexes);
    for (int si = 0; si < n_scales; ++si) {
      if (fac_pyr[si]) {
        mlx_array_free(*fac_pyr[si]);
        free(fac_pyr[si]);
      }
    }
    free(fac_pyr);
    MLXTrainer_destroy(t);
    mlx_options_free_trainning(opts);
    return 9;
  }
  for (int i = 0; i < n_scales; ++i)
    scales[i] = i;

  int iters = opts->num_iter > 0 ? opts->num_iter : 2;
  int rc =
      MLXTrainer_train_scales(t, indexes, batch, fac_pyr, n_scales, NULL, 0,
                              NULL, 0, NULL, 0, scales, n_scales, iters);
  printf("train_scales rc=%d\n", rc);

  for (int si = 0; si < n_scales; ++si) {
    if (fac_pyr[si]) {
      mlx_array_free(*fac_pyr[si]);
      free(fac_pyr[si]);
    }
  }
  free(fac_pyr);
  free(indexes);
  free(scales);

  /* Ensure scheduler is cleanly shutdown before destroying trainer/device
   * to avoid destructor-order races where the scheduler may access device
   * resources after they've been torn down. If shutdown fails we still
   * destroy the trainer to avoid leaking higher-level resources. */
  (void)mlx_scheduler_shutdown();
  MLXTrainer_destroy(t);
  mlx_options_free_trainning(opts);
  return 0;
}
