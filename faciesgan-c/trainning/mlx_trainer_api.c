// This file is a rename of c_trainer_api.c — kept identical contents.
#include "trainning/mlx_trainer_api.h"
#include "datasets/dataloader.h"
#include "datasets/func_cache.h"
#include "datasets/mlx_dataset.h"
#include "datasets/prefetcher.h"
#include "datasets/utils.h"
#include "datasets/wells.h"
#include "io/npz_unzip.h"
#include "models/base_manager.h"
#include "models/facies_gan.h"
#include "optimizer.h"
#include "pybridge.h"
#include "trainning/pybridge.h"
#include "trainning/train_manager.h"
#include "trainning/train_step.h"
#include "trainning/train_utils.h"
#include "utils.h"
#include <dirent.h>
#include <errno.h>
#include <execinfo.h>
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/device.h>
#include <mlx/c/io.h>
#include <mlx/c/memory.h>
#include <mlx/c/ops.h>
#include <mlx/c/random.h>
#include <mlx/c/stream.h>
#include <mlx/c/vector.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

MLXTrainer *MLXTrainer_new(const TrainningOptions *opts, int fine_tuning,
                           const char *checkpoint_path) {
  MLXTrainer *trainer = (MLXTrainer *)malloc(sizeof(MLXTrainer));
  memset(trainer, 0, sizeof(*trainer));
  trainer->opts = *opts;
  /* initialize fine-tuning flag and checkpoint path from explicit args */
  trainer->fine_tuning = fine_tuning ? 1 : 0;
  trainer->checkpoint_path = NULL;
  if (checkpoint_path && checkpoint_path[0] != '\0') {
    trainer->checkpoint_path = strdup(checkpoint_path);
  } else {
    trainer->checkpoint_path = strdup(".checkpoints");
  }

  /* Derived convenience fields mirroring Python Trainer initialisation */
  trainer->start_scale = opts->start_scale;
  trainer->stop_scale = opts->stop_scale;
  trainer->output_path = NULL;
  if (opts->output_path && opts->output_path[0] != '\0')
    trainer->output_path = strdup(opts->output_path);
  trainer->num_iter = opts->num_iter;
  trainer->save_interval = opts->save_interval;
  trainer->num_parallel_scales = opts->num_parallel_scales;

  /* compute batch_size = min(options.batch_size, options.num_train_pyramids) */
  trainer->batch_size = opts->batch_size < opts->num_train_pyramids
                            ? opts->batch_size
                            : opts->num_train_pyramids;
  if (opts->wells_mask_count > 0 &&
      opts->batch_size < (int)opts->wells_mask_count)
    trainer->batch_size = (int)opts->wells_mask_count;

  /* Feature flags */
  trainer->enable_tensorboard = opts->enable_tensorboard ? 1 : 0;
  trainer->enable_plot_facies = opts->enable_plot_facies ? 1 : 0;

  trainer->num_img_channels = opts->num_img_channels;
  trainer->noise_channels = opts->noise_channels +
                            (opts->use_wells ? opts->num_img_channels : 0) +
                            (opts->use_seismic ? opts->num_img_channels : 0);
  trainer->num_real_facies = opts->num_real_facies;
  trainer->num_generated_per_real = opts->num_generated_per_real;

  /* copy wells mask columns list from options (if present) */
  trainer->wells_mask_columns = NULL;
  trainer->wells_mask_count = 0;
  if (opts->wells_mask_count > 0 && opts->wells_mask_columns) {
    trainer->wells_mask_columns =
        (int *)malloc(sizeof(int) * opts->wells_mask_count);
    if (trainer->wells_mask_columns) {
      memcpy(trainer->wells_mask_columns, opts->wells_mask_columns,
             sizeof(int) * opts->wells_mask_count);
      trainer->wells_mask_count = opts->wells_mask_count;
    }
  }

  /* Optimizer configuration (defaults from options) */
  trainer->lr_g = opts->lr_g;
  trainer->lr_d = opts->lr_d;
  trainer->beta1 = opts->beta1;
  trainer->lr_decay = opts->lr_decay;
  trainer->gamma = opts->gamma;

  /* Model parameters */
  trainer->zero_padding = opts->num_layer * (opts->kernel_size / 2);
  trainer->noise_amp = opts->noise_amp;
  trainer->min_noise_amp = opts->min_noise_amp;
  trainer->scale0_noise_amp = opts->scale0_noise_amp;

  (void)MLXTrainer_init_dataset(trainer);
  (void)MLXTrainer_init_scales(trainer);
  (void)MLXTrainer_create_dataloader(trainer);

  fprintf(stdout, "DataLoader num_workers: %d\n", trainer->opts.num_workers);

  (void)MLXTrainer_create_model(trainer);
  mlx_faciesgan_set_shapes(trainer->model, trainer->scales, trainer->n_scales);

  int alloc_n = trainer->n_scales;
  if (trainer->opts.num_parallel_scales > alloc_n)
    alloc_n = trainer->opts.num_parallel_scales;
  if (alloc_n <= 0)
    alloc_n = 1;
  trainer->gen_opts =
      (MLXOptimizer **)calloc((size_t)alloc_n, sizeof(MLXOptimizer *));
  trainer->disc_opts =
      (MLXOptimizer **)calloc((size_t)alloc_n, sizeof(MLXOptimizer *));
  trainer->gen_scheds =
      (MLXScheduler **)calloc((size_t)alloc_n, sizeof(MLXScheduler *));
  trainer->disc_scheds =
      (MLXScheduler **)calloc((size_t)alloc_n, sizeof(MLXScheduler *));

  /* Print scales table (moved out of init_scales for constructor-level
   * display). Only print when scales were populated. */
  if (trainer->scales && trainer->n_scales > 0) {
    fprintf(stdout, "Generated facie shapes:\n");
    fprintf(stdout, "╔══════════╦══════════╦══════════╦══════════╗\n");
    fprintf(stdout, "║ %8s ║ %8s ║ %8s ║ %8s ║\n", "Batch", "Height", "Width",
            "Channels");
    fprintf(stdout, "╠══════════╬══════════╬══════════╬══════════╣\n");
    for (int si = 0; si < trainer->n_scales; ++si) {
      /* stored as NHWC: [batch, height, width, channels] */
      int b = trainer->scales[si * 4 + 0];
      int h = trainer->scales[si * 4 + 1];
      int w = trainer->scales[si * 4 + 2];
      int c = trainer->scales[si * 4 + 3];
      /* print as Batch, Height, Width, Channels */
      fprintf(stdout, "║ %8d ║ %8d ║ %8d ║ %8d ║\n", b, h, w, c);
    }
    fprintf(stdout, "╚══════════╩══════════╩══════════╩══════════╝\n");
  }

  /* Initialize TensorBoard-style logging paths and print guidance when
   * enabled. We don't implement a visualizer in C; instead create the
   * directories and print the tensorboard --logdir hint (parity with
   * Python Trainer.__init__). */
  if (trainer->enable_tensorboard) {
    char viz_path[PATH_BUFSZ];
    char log_dir[PATH_BUFSZ];
    const char *base_out = trainer->output_path ? trainer->output_path : ".";
    join_path(viz_path, sizeof(viz_path), base_out, "training_visualizations");
    join_path(log_dir, sizeof(log_dir), base_out, "tensorboard_logs");
    /* dataset_info: "<num_pyramids> pyramids, <batch> batch size" */
    char dataset_info[PATH_BUFSZ];
    snprintf(dataset_info, sizeof(dataset_info), "%d pyramids, %d batch size",
             trainer->opts.num_train_pyramids, trainer->batch_size);
    if (trainer->opts.wells_mask_count > 0) {
      /* append a short wells note (full column list can be large) */
      strncat(dataset_info, ", wells present",
              sizeof(dataset_info) - strlen(dataset_info) - 1);
    }
    /* Ensure directories exist (best-effort). */
    (void)mlx_create_dirs(viz_path);
    (void)mlx_create_dirs(log_dir);
    fprintf(stdout, "📊 TensorBoard logging enabled!\n");
    fprintf(stdout, "   View training progress: tensorboard --logdir=%s\n",
            log_dir);
    fprintf(stdout, "   Then open: http://localhost:6006\n");
    /* Optionally print small dataset hint for parity with Python output */
    fprintf(stdout, "   %s\n", dataset_info);
    /* Create the Python-side visualizer (best-effort). Use update_interval=1
     * to match Python trainer default behavior. The pybridge manages a
     * global visualizer instance. */
    if (!pybridge_create_visualizer(trainer->n_scales, log_dir, log_dir, 1)) {
      fprintf(stderr,
              "warning: failed to initialize Python TensorBoard visualizer\n");
    }
  } else {
    fprintf(stdout, "📊 TensorBoard logging disabled\n");
  }

  mlx_pyramids_dataset_dump_batches_npz(trainer->dataset,
                                        "results/c_dataset_batches.npz");

  return trainer;
}

void MLXTrainer_destroy(MLXTrainer *trainer) {
  if (!trainer)
    return;

  if (trainer->model)
    mlx_faciesgan_free(trainer->model);

  /* free per-scale optimizer/scheduler instances (allocated to max of
   * discovered scales or configured parallel scales). */
  int alloc_n = trainer->n_scales;
  if (trainer->opts.num_parallel_scales > alloc_n)
    alloc_n = trainer->opts.num_parallel_scales;
  if (alloc_n <= 0)
    alloc_n = 1;

  if (trainer->gen_opts) {
    for (int i = 0; i < alloc_n; ++i) {
      if (trainer->gen_opts[i])
        mlx_adam_free(trainer->gen_opts[i]);
    }
  }
  if (trainer->disc_opts) {
    for (int i = 0; i < alloc_n; ++i) {
      if (trainer->disc_opts[i])
        mlx_adam_free(trainer->disc_opts[i]);
    }
  }
  if (trainer->gen_scheds) {
    for (int i = 0; i < alloc_n; ++i) {
      if (trainer->gen_scheds[i])
        mlx_scheduler_free(trainer->gen_scheds[i]);
    }
  }
  if (trainer->disc_scheds) {
    for (int i = 0; i < alloc_n; ++i) {
      if (trainer->disc_scheds[i])
        mlx_scheduler_free(trainer->disc_scheds[i]);
    }
  }

  if (trainer->gen_opts)
    free(trainer->gen_opts);
  if (trainer->disc_opts)
    free(trainer->disc_opts);
  if (trainer->gen_scheds)
    free(trainer->gen_scheds);
  if (trainer->disc_scheds)
    free(trainer->disc_scheds);

  if (trainer->scales)
    free(trainer->scales);

  if (trainer->checkpoint_path)
    free(trainer->checkpoint_path);

  if (trainer->output_path)
    free(trainer->output_path);

  if (trainer->wells_mask_columns)
    free(trainer->wells_mask_columns);

  /* Close Python visualizer if one was created via pybridge. */
  if (trainer->enable_tensorboard) {
    (void)pybridge_close_visualizer();
  }

  /* Free dataset/dataloader if created during MLXTrainer_new. */
  if (trainer->data_loader) {
    facies_dataloader_free(trainer->data_loader);
    trainer->data_loader = NULL;
  }
  if (trainer->dataset) {
    facies_dataset_free(trainer->dataset);
    trainer->dataset = NULL;
  }

  /* destroy prefetcher resources if any */
  if (trainer->batch_iterator) {
    prefetcher_iterator_destroy(trainer->batch_iterator);
    trainer->batch_iterator = NULL;
  }
  if (trainer->batch_prefetcher) {
    prefetcher_destroy(trainer->batch_prefetcher);
    trainer->batch_prefetcher = NULL;
  }

  free(trainer);
}

int MLXTrainer_get_shapes_flat(MLXTrainer *t, int **out_shapes, int *out_n) {
  if (!t || !out_shapes || !out_n)
    return -1;
  *out_shapes = t->scales;
  *out_n = t->n_scales;
  return 0;
}

int MLXTrainer_set_shapes(MLXTrainer *t, const int *shapes, int n_scales) {
  if (!t)
    return -1;
  if (t->scales)
    free(t->scales);
  if (!shapes || n_scales <= 0) {
    t->scales = NULL;
    t->n_scales = 0;
    return 0;
  }
  t->scales = (int *)malloc(sizeof(int) * 4 * (size_t)n_scales);
  if (!t->scales)
    return -1;
  memcpy(t->scales, shapes, sizeof(int) * 4 * (size_t)n_scales);
  t->n_scales = n_scales;
  return 0;
}

int MLXTrainer_init_dataset(MLXTrainer *trainer) {
  /* Create dataset using the canonical constructor to avoid duplicating
   * loading logic. Use channels_last=1 to match Python trainer behavior. */
  MLXPyramidsDataset *ds = NULL;
  if (mlx_pyramids_dataset_new(&ds, &trainer->opts, 0, 0, 1) != 0) {
    return -1;
  }

  trainer->dataset = ds;
  trainer->num_of_batchs = ds->n_samples > 0 && trainer->batch_size > 0
                               ? (ds->n_samples / trainer->batch_size)
                               : 0;
  return 0;
}

int MLXTrainer_create_visualizer(MLXTrainer *trainer, int update_interval) {
  if (!trainer || !trainer->enable_tensorboard)
    return 0;
  const char *base_out = trainer->output_path ? trainer->output_path : ".";
  char log_dir[PATH_BUFSZ];
  join_path(log_dir, sizeof(log_dir), base_out, "tensorboard_logs");
  return pybridge_create_visualizer(trainer->n_scales, base_out, log_dir,
                                    update_interval);
}

int MLXTrainer_update_visualizer(MLXTrainer *trainer, int epoch,
                                 const char *metrics_json,
                                 int samples_processed) {
  (void)trainer;
  if (!trainer || !trainer->enable_tensorboard)
    return 0;
  return pybridge_update_visualizer_from_json(epoch, metrics_json,
                                              samples_processed);
}

int MLXTrainer_close_visualizer(MLXTrainer *trainer) {
  (void)trainer;
  if (!trainer || !trainer->enable_tensorboard)
    return 0;
  return pybridge_close_visualizer();
}

int MLXTrainer_setup_optimizers(MLXTrainer *trainer, const int *scales,
                                int n_scales) {
  if (!trainer)
    return -1;

  int *local_scales = NULL;
  int local_n = n_scales;

  /* If caller passes NULL/0, interpret as "use configured parallel scales"
   * Prefer the explicit TrainningOptions value (`num_parallel_scales`) when
   * present; otherwise fall back to discovered model scales. This mirrors
   * the Python trainer which uses options to determine parallelism. */
  if (!scales || n_scales <= 0) {
    if (trainer->opts.num_parallel_scales > 0) {
      local_n = trainer->opts.num_parallel_scales;
    } else if (trainer->n_scales > 0) {
      local_n = trainer->n_scales;
    } else {
      return -1;
    }
    local_scales = (int *)malloc(sizeof(int) * (size_t)local_n);
    if (!local_scales)
      return -1;
    for (int i = 0; i < local_n; ++i)
      local_scales[i] = i;
    scales = local_scales;
  }

  for (int i = 0; i < local_n; ++i) {
    int sc = scales[i];
    /* create Adam optimizers with defaults (mirrors Python defaults) */
    trainer->gen_opts[sc] =
        mlx_adam_create(trainer->opts.lr_g, trainer->opts.beta1, 0.999f, 1e-8f);
    trainer->disc_opts[sc] =
        mlx_adam_create(trainer->opts.lr_d, trainer->opts.beta1, 0.999f, 1e-8f);
    /* schedulers: multistep with single milestone (lr_decay) */
    int milestones[1] = {trainer->opts.lr_decay};
    trainer->gen_scheds[sc] = mlx_scheduler_multistep_create_with_init(
        milestones, 1, trainer->opts.gamma, (const float *)&trainer->opts.lr_g,
        1);
    trainer->disc_scheds[sc] = mlx_scheduler_multistep_create_with_init(
        milestones, 1, trainer->opts.gamma, (const float *)&trainer->opts.lr_d,
        1);
    /* attach scheduler and optimizer so LR updates propagate */
    if (trainer->gen_opts[sc] && trainer->gen_scheds[sc]) {
      mlx_optimizer_attach_scheduler(trainer->gen_opts[sc],
                                     trainer->gen_scheds[sc]);
      mlx_scheduler_attach_optimizer(trainer->gen_scheds[sc],
                                     trainer->gen_opts[sc]);
    }
    if (trainer->disc_opts[sc] && trainer->disc_scheds[sc]) {
      mlx_optimizer_attach_scheduler(trainer->disc_opts[sc],
                                     trainer->disc_scheds[sc]);
      mlx_scheduler_attach_optimizer(trainer->disc_scheds[sc],
                                     trainer->disc_opts[sc]);
    }
  }
  if (local_scales)
    free(local_scales);
  return 0;
}

/* Producer thread logic is now centralized in datasets/prefetcher.c.
 * Use `prefetcher_start_from_dataloader` to spawn a producer.
 */

int MLXTrainer_compute_rec_input(MLXTrainer *trainer, int scale,
                                 const int *indexes, int n_indexes,
                                 mlx_array **facies_pyramid, mlx_array **out) {
  (void)trainer;
  return mlx_compute_rec_input(scale, indexes, n_indexes, facies_pyramid, out);
}

int MLXTrainer_init_rec_noise_and_amp(MLXTrainer *trainer, int scale,
                                      const int *indexes, int n_indexes,
                                      const mlx_array *real,
                                      mlx_array **wells_pyramid,
                                      mlx_array **seismic_pyramid) {
  if (!trainer)
    return -1;
  return mlx_init_rec_noise_and_amp(trainer->model, scale, indexes, n_indexes,
                                    real, wells_pyramid, seismic_pyramid);
}

PrefetcherIteratorHandle
MLXTrainer_create_batch_iterator(MLXTrainer *trainer, struct MLXDataloader *dl,
                                 const int *scales, int n_scales) {
  if (!trainer)
    return NULL;

  /* Lazily initialise dataset/dataloader if caller didn't provide one. */
  if (!dl) {
    /* Prefer an existing trainer-owned dataloader when available. */
    if (trainer->data_loader) {
      dl = trainer->data_loader;
    } else {
      if (!trainer->dataset) {
        if (MLXTrainer_init_dataset(trainer) != 0)
          return NULL;
      }
      unsigned int seed = (unsigned int)(trainer->opts.manual_seed >= 0
                                             ? trainer->opts.manual_seed
                                             : (int)time(NULL));
      if (MLXTrainer_create_dataloader(trainer) != 0) {
        return NULL;
      }
      dl = trainer->data_loader;
      /* Record ownership so MLXTrainer_destroy will free it. */
      trainer->data_loader = dl;
    }
  }
  int qcap = 4;
  mlx_stream s = mlx_default_cpu_stream_new();
  PrefetcherHandle ph =
      prefetcher_create_with_stream(qcap, s, (const int *)scales, n_scales);
  if (!ph) {
    if (s.ctx)
      mlx_stream_free(s);
    return NULL;
  }
  trainer->batch_prefetcher = ph;
  trainer->batch_iterator = prefetcher_iterator_create(ph);
  /* Start the centralized prefetcher producer thread which will read from
   * the dataloader and push into the prefetcher. The helper detaches the
   * thread and will free the provided stream when finished. */
  mlx_stream prod_stream = mlx_default_cpu_stream_new();
  if (prefetcher_start_from_dataloader(ph, dl, prod_stream) != 0) {
    if (prod_stream.ctx)
      mlx_stream_free(prod_stream);
    prefetcher_destroy(ph);
    trainer->batch_prefetcher = NULL;
    trainer->batch_iterator = NULL;
    return NULL;
  }
  trainer->batch_producer_running = 1;
  return trainer->batch_iterator;
}

int MLXTrainer_create_dataloader(MLXTrainer *trainer) {
  if (!trainer)
    return -1;

  /* If dataloader already exists, treat as success. */
  if (trainer->data_loader)
    return 0;

  /* Ensure dataset is initialised. */
  if (!trainer->dataset) {
    if (MLXTrainer_init_dataset(trainer) != 0)
      return -1;
  }

  MLXPyramidsDataset *ds = trainer->dataset;
  size_t batch_size =
      trainer->batch_size > 0
          ? (size_t)trainer->batch_size
          : (trainer->opts.batch_size > 0 ? (size_t)trainer->opts.batch_size
                                          : 1);
  unsigned int seed =
      (unsigned int)(trainer->opts.manual_seed >= 0 ? trainer->opts.manual_seed
                                                    : (int)time(NULL));
  int num_workers = trainer->opts.num_workers;
  int prefetch_factor = 2;
  int timeout_ms = 2000;

  struct MLXDataloader *dl = NULL;
  int rc = facies_dataloader_new_ex(
      &dl, ds, batch_size, false, false, seed, num_workers, prefetch_factor,
      num_workers > 0, timeout_ms, NULL, NULL, false, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, 0, NULL, NULL);
  if (rc != 0)
    return rc;

  trainer->data_loader = dl;
  return 0;
}

int MLXTrainer_generate_visualization_samples(
    MLXTrainer *trainer, const int *scales, int n_scales, const int *indexes,
    int n_indexes, mlx_array **wells_pyramid, int n_wells,
    mlx_array **seismic_pyramid, int n_seismic, mlx_array ***out_generated,
    int *n_out) {
  if (!trainer || !scales || n_scales <= 0 || !out_generated || !n_out)
    return -1;
  mlx_array **out = (mlx_array **)malloc(sizeof(mlx_array *) * n_scales);
  if (!out)
    return -1;
  for (int i = 0; i < n_scales; ++i) {
    int scale = scales[i];
    mlx_array **noises = NULL;
    int n_noises = 0;
    if (mlx_faciesgan_get_pyramid_noise(
            trainer->model, scale, indexes, n_indexes, &noises, &n_noises,
            wells_pyramid, seismic_pyramid, 0) != 0) {
      out[i] = NULL;
      continue;
    }
    float *use_amps = NULL;
    int use_n = 0;
    if (mlx_faciesgan_get_noise_amplitude(trainer->model, scale, &use_amps,
                                          &use_n) != 0) {
      use_amps = (float *)malloc(sizeof(float) * (scale + 1));
      if (!use_amps) {
        out[i] = NULL;
        for (int j = 0; j < n_noises; ++j) {
          if (noises[j]) {
            mlx_array_free(*noises[j]);
            free(noises[j]);
          }
        }
        free(noises);
        continue;
      }
      for (int k = 0; k < scale + 1; ++k)
        use_amps[k] = 1.0f;
      use_n = scale + 1;
    }
    /* Convert pointer array to contiguous mlx_array values expected by the
     * generator forward wrapper. */
    mlx_array *zvals = NULL;
    if (n_noises > 0) {
      zvals = (mlx_array *)malloc(sizeof(mlx_array) * (size_t)n_noises);
      if (!zvals) {
        for (int j = 0; j < n_noises; ++j) {
          if (noises[j]) {
            mlx_array_free(*noises[j]);
            free(noises[j]);
          }
        }
        free(noises);
        if (use_amps)
          free(use_amps);
        out[i] = NULL;
        continue;
      }
      for (int j = 0; j < n_noises; ++j)
        zvals[j] = *noises[j];
      /* Ensure none of the zvals are empty; replace empties with zeros. */
      for (int j = 0; j < n_noises; ++j) {
        if (mlx_array_ndim(zvals[j]) == 0) {
          mlx_stream _s = mlx_default_cpu_stream_new();
          int shape0[4] = {
              1, trainer->opts.crop_size > 0 ? trainer->opts.crop_size : 32,
              trainer->opts.crop_size > 0 ? trainer->opts.crop_size : 32,
              trainer->opts.num_img_channels > 0
                  ? trainer->opts.num_img_channels
                  : 1};
          mlx_array tmp = mlx_array_new();
          if (mlx_zeros(&tmp, shape0, 4, MLX_FLOAT32, _s) == 0) {
            zvals[j] = tmp;
          }
          mlx_stream_free(_s);
        }
      }
    }

    /* pick an initial in_noise: prefer first noise when present */
    mlx_array in_noise = mlx_array_new();
    if (n_noises > 0)
      in_noise = zvals[0];
    (void)n_noises;
    (void)zvals;
    (void)scale;
    mlx_array fake =
        mlx_faciesgan_generate_fake(trainer->model, zvals, n_noises, use_amps,
                                    use_n, in_noise, scale, scale);

    for (int j = 0; j < n_noises; ++j) {
      if (noises[j]) {
        mlx_array_free(*noises[j]);
        free(noises[j]);
      }
    }
    free(noises);
    if (zvals)
      free(zvals);
    if (use_amps)
      free(use_amps);
    mlx_array *p = (mlx_array *)malloc(sizeof(mlx_array));
    if (!p) {
      mlx_array_free(fake);
      out[i] = NULL;
      continue;
    }
    *p = fake;
    out[i] = p;
  }
  *out_generated = out;
  *n_out = n_scales;
  return 0;
}

int MLXTrainer_optimization_step(MLXTrainer *trainer, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **rec_in_pyramid,
                                 int n_rec, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *active_scales,
                                 int n_active_scales) {
  /* destroy prefetcher resources if any */
  if (trainer->batch_iterator) {
    prefetcher_iterator_destroy(trainer->batch_iterator);
    trainer->batch_iterator = NULL;
  }
  if (trainer->batch_prefetcher) {
    prefetcher_destroy(trainer->batch_prefetcher);
    trainer->batch_prefetcher = NULL;
  }
  if (!trainer)
    return -1;

  MLXResults *res = NULL;
  int rc = mlx_faciesgan_collect_metrics_and_grads(
      trainer->model, indexes, n_indexes, active_scales, n_active_scales,
      facies_pyramid, rec_in_pyramid, wells_pyramid, masks_pyramid,
      seismic_pyramid, trainer->opts.lambda_diversity,
      trainer->opts.well_loss_penalty, trainer->opts.alpha,
      trainer->opts.lambda_grad, &res);
  if (rc != 0 || !res) {
    if (res)
      mlx_results_free(res);
    return -1;
  }

  /* For each active scale: step schedulers (auto) then apply optimizer steps
   * using collected grads. This mirrors the Python Trainer per-scale logic. */
  int overall = 0;
  for (int i = 0; i < res->n_scales; ++i) {
    MLXScaleResults *sr = &res->scales[i];
    int sc = sr->scale;

    /* Step schedulers (advance by one) if present */
    if (trainer->gen_scheds && trainer->gen_scheds[sc])
      mlx_scheduler_step_auto(trainer->gen_scheds[sc],
                              trainer->gen_opts ? trainer->gen_opts[sc] : NULL);
    if (trainer->disc_scheds && trainer->disc_scheds[sc])
      mlx_scheduler_step_auto(trainer->disc_scheds[sc],
                              trainer->disc_opts ? trainer->disc_opts[sc]
                                                 : NULL);

    /* Apply generator grads */
    if (sr->gen_n > 0 && sr->gen_grads && trainer->gen_opts &&
        trainer->gen_opts[sc]) {
      int r = mlx_faciesgan_apply_sgd_to_generator(
          trainer->model, trainer->gen_opts[sc], sr->gen_grads, sr->gen_n);
      if (r != 0)
        overall = -1;
    }

    /* Apply discriminator grads */
    if (sr->disc_n > 0 && sr->disc_grads && trainer->disc_opts &&
        trainer->disc_opts[sc]) {
      int r = mlx_faciesgan_apply_sgd_to_discriminator(
          trainer->model, trainer->disc_opts[sc], sr->disc_grads, sr->disc_n);
      if (r != 0)
        overall = -1;
    }
  }

  mlx_results_free(res);
  return overall == 0 ? 0 : -1;
}

int MLXTrainer_load_model(MLXTrainer *trainer, int scale,
                          const char *checkpoint_dir) {
  if (!trainer || !checkpoint_dir)
    return -1;
  /* Reuse existing per-scale state loaders (load_*_state). The expected
   * argument is a directory path for the scale; pass `checkpoint_dir/scale`. */
  char scale_dir[PATH_MAX];
  snprintf(scale_dir, PATH_MAX, "%s/%d", checkpoint_dir, scale);
  if (mlx_faciesgan_load_generator_state(trainer->model, scale_dir, scale) != 0)
    return -1;
  if (mlx_faciesgan_load_discriminator_state(trainer->model, scale_dir,
                                             scale) != 0)
    return -1;
  return 0;
}

int MLXTrainer_save_generated_facies(MLXTrainer *trainer, int scale, int epoch,
                                     const char *results_path) {
  if (!trainer || !results_path)
    return -1;
  /* Generate noises and call numeric forward (similar to train_utils.c) */
  mlx_array **noises = NULL;
  int n_noises = 0;
  if (mlx_faciesgan_get_pyramid_noise(trainer->model, scale, NULL, 0, &noises,
                                      &n_noises, NULL, NULL, 0) != 0)
    return -1;

  /* use default amplitudes (all ones) */
  float *use_amps = (float *)malloc(sizeof(float) * (size_t)(scale + 1));
  if (!use_amps) {
    for (int i = 0; i < n_noises; ++i) {
      if (noises[i]) {
        mlx_array_free(*noises[i]);
        free(noises[i]);
      }
    }
    free(noises);
    return -1;
  }
  for (int i = 0; i < scale + 1; ++i)
    use_amps[i] = 1.0f;

  mlx_array in_noise = mlx_array_new();
  mlx_array_t fake = mlx_faciesgan_generate_fake(
      trainer->model, (const mlx_array *)noises, n_noises, use_amps, scale + 1,
      in_noise, scale, scale);

  /* free noises */
  for (int i = 0; i < n_noises; ++i) {
    if (noises[i]) {
      mlx_array_free(*noises[i]);
      free(noises[i]);
    }
  }
  free(noises);
  free(use_amps);

  char fname[PATH_MAX];
  snprintf(fname, PATH_MAX, "%s/scale_%d_epoch_%d.npy", results_path, scale,
           epoch);
  int rc = mlx_save(fname, fake);
  mlx_array_free(fake);
  mlx_array_free(in_noise);
  return rc;
}

void *MLXTrainer_get_model_ctx(MLXTrainer *trainer) {
  if (!trainer)
    return NULL;
  return (void *)trainer->model;
}

void *MLXTrainer_create_model(MLXTrainer *trainer) {
  if (!trainer)
    return NULL;

  MLXBaseManager *mgr = mlx_base_manager_create_from_trainning(&trainer->opts);
  if (!mgr)
    return NULL;
  if (trainer->opts.num_parallel_scales > 0) {
    mlx_base_manager_init_scales(mgr, 0, trainer->opts.num_parallel_scales);
  }
  /* Use provided checkpoint path when loading shapes/state; fall back to
   * literal if allocation failed. The `checkpoint_path` was initialised
   * earlier to reflect explicit args or defaults. */
  {
    const char *ckpt =
        trainer->checkpoint_path ? trainer->checkpoint_path : ".checkpoints";
    (void)mlx_base_manager_load(mgr, ckpt, 1 /*load_shapes*/,
                                -1 /*until_scale*/, 0 /*load_disc*/,
                                0 /*load_wells*/);
  }
  trainer->model = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
  if (!trainer->model) {
    mlx_base_manager_free(mgr);
    return NULL;
  }
  return (void *)trainer->model;
}

int MLXTrainer_init_scales(MLXTrainer *trainer) {
  /* If scales already set, nothing to do. */
  if (trainer->scales && trainer->n_scales > 0)
    return 0;
  DatasetScale *arr = NULL;
  int n = 0;
  if (dataset_generate_scales(&trainer->opts, 1 /*channels_last*/, &arr, &n) !=
      0) {
    return -1;
  }

  trainer->scales = (int *)malloc(sizeof(int) * 4 * (size_t)n);
  if (!trainer->scales) {
    free(arr);
    return -1;
  }
  for (int si = 0; si < n; ++si) {
    trainer->scales[si * 4 + 0] = arr[si].batch;
    trainer->scales[si * 4 + 1] = arr[si].height;
    trainer->scales[si * 4 + 2] = arr[si].width;
    trainer->scales[si * 4 + 3] = arr[si].channels;
  }
  trainer->n_scales = n;
  free(arr);
  return 0;
}

int MLXTrainer_train_scales(MLXTrainer *trainer, const int *indexes,
                            int n_indexes, mlx_array **facies_pyramid,
                            int n_facies, mlx_array **wells_pyramid,
                            int n_wells, mlx_array **masks_pyramid, int n_masks,
                            mlx_array **seismic_pyramid, int n_seismic,
                            const int *scales, int n_scales, int num_iter) {
  if (!trainer || !indexes || n_indexes <= 0 || !facies_pyramid ||
      n_facies <= 0 || !scales || n_scales <= 0 || num_iter <= 0)
    return -1;

  /* Prepare rec_in_pyramid once per iteration as Python does */
  for (int epoch = 0; epoch < num_iter; ++epoch) {
    /* compute rec inputs and init noise/amps */
    mlx_array **rec_pyr =
        (mlx_array **)calloc((size_t)n_facies, sizeof(mlx_array *));
    if (!rec_pyr)
      return -1;
    for (int si = 0; si < n_facies; ++si) {
      mlx_array *r = NULL;
      (void)mlx_compute_rec_input(si, indexes, n_indexes, facies_pyramid, &r);
      rec_pyr[si] = r;
      (void)mlx_init_rec_noise_and_amp(
          (MLXFaciesGAN *)MLXTrainer_get_model_ctx(trainer), si, indexes,
          n_indexes, facies_pyramid[si], wells_pyramid, seismic_pyramid);
    }

    int rc = MLXTrainer_optimization_step(
        trainer, indexes, n_indexes, facies_pyramid, n_facies, rec_pyr,
        n_facies, wells_pyramid, n_wells, masks_pyramid, n_masks,
        seismic_pyramid, n_seismic, scales, n_scales);

    for (int si = 0; si < n_facies; ++si) {
      if (rec_pyr[si]) {
        mlx_array_free(*rec_pyr[si]);
        free(rec_pyr[si]);
      }
    }
    free(rec_pyr);
    if (rc != 0)
      return rc;
  }
  return 0;
}

/* Dataset-runner implementation: these functions depend on dataset and
 * pybridge symbols and were merged here per project layout requirements.
 */

int MLXTrainer_run(int num_samples, int num_scales, int channels, int height,
                   int width, int batch_size) {
  if (num_samples <= 0 || num_scales <= 0 || batch_size <= 0) {
    fprintf(stderr, "invalid trainer args\n");
    return 1;
  }

  mlx_vector_vector_array facies_pyramids = mlx_vector_vector_array_new();
  for (int si = 0; si < num_samples; ++si) {
    mlx_vector_array sample = mlx_vector_array_new();
    for (int sc = 0; sc < num_scales; ++sc) {
      int shape[3] = {height, width, channels};
      mlx_array a = mlx_array_new();
      mlx_stream s = mlx_default_cpu_stream_new();
      if (mlx_random_normal(&a, shape, 3, MLX_FLOAT32, 0.0f, 1.0f,
                            mlx_array_empty, s) != 0) {
        mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
      }
      mlx_stream_free(s);
      if (mlx_vector_array_append_value(sample, a) != 0) {
        fprintf(stderr, "failed to append sample array\n");
        mlx_array_free(a);
        mlx_vector_array_free(sample);
        mlx_vector_vector_array_free(facies_pyramids);
        return 1;
      }
      mlx_array_free(a);
    }
    if (mlx_vector_vector_array_append_value(facies_pyramids, sample) != 0) {
      fprintf(stderr, "failed to append sample vector\n");
      mlx_vector_array_free(sample);
      mlx_vector_vector_array_free(facies_pyramids);
      return 1;
    }
    mlx_vector_array_free(sample);
  }

  mlx_vector_vector_array wells = mlx_vector_vector_array_new();
  mlx_vector_vector_array seismic = mlx_vector_vector_array_new();

  MLXPyramidsDataset *ds = NULL;
  if (facies_dataset_new(&ds, facies_pyramids, wells, seismic) != 0) {
    fprintf(stderr, "failed to create facies_dataset\n");
    mlx_vector_vector_array_free(facies_pyramids);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    return 1;
  }

  struct MLXDataloader *dl = NULL;
  if (facies_dataloader_new(&dl, ds, (size_t)batch_size, false, false,
                            (unsigned int)time(NULL)) != 0) {
    fprintf(stderr, "failed to create facies_dataloader\n");
    facies_dataset_free(ds);
    mlx_vector_vector_array_free(facies_pyramids);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    return 1;
  }

  printf("Starting C-native trainer: %d samples, %d scales, batch %d\n",
         num_samples, num_scales, batch_size);

  pybridge_create_visualizer(num_scales, ".", NULL, 1);
  pybridge_create_background_worker(2, 32);

  mlx_stream s = mlx_default_cpu_stream_new();

  int batch_idx = 0;
  while (1) {
    mlx_vector_array out_facies = mlx_vector_array_new();
    mlx_vector_array out_wells = mlx_vector_array_new();
    mlx_vector_array out_seismic = mlx_vector_array_new();
    int rc =
        facies_dataloader_next(dl, &out_facies, &out_wells, &out_seismic, s);
    if (rc == 2) {
      mlx_vector_array_free(out_facies);
      mlx_vector_array_free(out_wells);
      mlx_vector_array_free(out_seismic);
      break;
    } else if (rc != 0) {
      fprintf(stderr, "dataloader_next error: %d\n", rc);
      mlx_vector_array_free(out_facies);
      mlx_vector_array_free(out_wells);
      mlx_vector_array_free(out_seismic);
      break;
    }
    size_t nsc = mlx_vector_array_size(out_facies);
    printf(" batch %d: stacked scales = %zu\n", batch_idx, nsc);

    char metrics[1024];
    int off = 0;
    off += snprintf(metrics + off, sizeof(metrics) - off, "{");
    for (int sc = 0; sc < num_scales; ++sc) {
      off += snprintf(metrics + off, sizeof(metrics) - off,
                      "\"%d\":{\"d_total\":%g,\"d_real\":%g,\"d_fake\":%g,\"g_"
                      "total\":%g,\"g_adv\":%g,\"g_rec\":%g}",
                      sc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      if (sc + 1 < num_scales)
        off += snprintf(metrics + off, sizeof(metrics) - off, ",");
    }
    off += snprintf(metrics + off, sizeof(metrics) - off, "}");
    pybridge_update_visualizer_from_json(batch_idx, metrics,
                                         batch_idx * batch_size);
    mlx_vector_array_free(out_facies);
    mlx_vector_array_free(out_wells);
    mlx_vector_array_free(out_seismic);
    batch_idx++;
  }

  mlx_stream_free(s);
  facies_dataloader_free(dl);
  facies_dataset_free(ds);
  mlx_vector_vector_array_free(facies_pyramids);
  mlx_vector_vector_array_free(wells);
  mlx_vector_vector_array_free(seismic);

  printf("C-native trainer finished; processed %d batches\n", batch_idx);
  return 0;
}

int MLXTrainer_run_with_opts(const TrainningOptions *opts) {
  if (!opts)
    return -1;
  MLXTrainer *trainer = MLXTrainer_new(opts, 0, ".checkpoints");
  if (!trainer)
    return -1;
  int rc = MLXTrainer_train(trainer);
  MLXTrainer_destroy(trainer);
  return rc;
}

/* Minimal MLXTrainer_train implementation to keep file self-contained.
 * It expects an already-created `MLXTrainer*` and mirrors the previous
 * dataset-driven behavior which was based on TrainningOptions stored in
 * the trainer. */
int MLXTrainer_train(MLXTrainer *trainer) {
  if (!trainer)
    return -1;

  TrainningOptions *opts = &trainer->opts;

  int n_scales = trainer->n_scales;
  if (n_scales <= 0) {
    /* Synthesize model shapes from options when no shapes are present
     * (mirrors the smoke runner behaviour to ensure model scales exist).
     */
    int synth_n = opts->num_parallel_scales > 0 ? opts->num_parallel_scales : 1;
    int *synth = (int *)malloc(sizeof(int) * 4 * synth_n);
    if (!synth) {
      return -1;
    }
    int batch = opts->batch_size > 0 ? opts->batch_size : 1;
    for (int si = 0; si < synth_n; ++si) {
      synth[si * 4 + 0] = batch;
      synth[si * 4 + 1] = opts->crop_size > 0 ? opts->crop_size : 64;
      synth[si * 4 + 2] = opts->crop_size > 0 ? opts->crop_size : 64;
      synth[si * 4 + 3] =
          opts->num_img_channels > 0 ? opts->num_img_channels : 3;
    }
    if (mlx_faciesgan_set_shapes(trainer->model, synth, synth_n) != 0) {
      free(synth);
      fprintf(stderr, "failed to set synthetic model shapes\n");
      return -1;
    }
    free(synth);
    /* Ensure trainer reflects the synthesized shapes count */
    trainer->n_scales = synth_n;
    n_scales = trainer->n_scales;
    if (n_scales <= 0) {
      fprintf(stderr, "no model scales available after synth\n");
      return -1;
    }
  }

  /* Ensure generator scales exist for the discovered/synthesized shapes so
   * subsequent noise/forward calls have initialized modules. */
  for (int si = 0; si < n_scales; ++si) {
    (void)mlx_faciesgan_create_generator_scale(
        trainer->model, si, opts->num_feature, opts->min_num_feature);
  }

  MLXPyramidsDataset *ds = NULL;
  struct MLXDataloader *dl = NULL;
  int created_dl = 0;

  if (!trainer->dataset) {
    if (MLXTrainer_init_dataset(trainer) != 0) {
      fprintf(stderr, "failed to initialise dataset\n");
      return -1;
    }
  }
  ds = trainer->dataset;

  if (trainer->data_loader) {
    dl = trainer->data_loader;
  } else {
    unsigned int seed =
        (unsigned int)(opts->manual_seed >= 0 ? opts->manual_seed
                                              : (int)time(NULL));
    if (MLXTrainer_create_dataloader(trainer) != 0) {
      fprintf(stderr, "failed to create facies_dataloader\n");
      return -1;
    }
    dl = trainer->data_loader;
    created_dl = 1;
  }

  /* Visualizer / background worker bridge (best-effort). Honor user option */
  if (opts->enable_tensorboard) {
    pybridge_create_visualizer(
        n_scales, opts->output_path ? opts->output_path : ".", NULL, 1);
    pybridge_create_background_worker(2, 32);
  }

  /* Setup optimizers for all scales */
  int *scales = (int *)malloc(sizeof(int) * n_scales);
  for (int i = 0; i < n_scales; ++i)
    scales[i] = i;
  MLXTrainer_setup_optimizers(trainer, scales, n_scales);

  /* Create a prefetcher/iterator via the centralized helper so ownership
   * and producer thread logic is consistent. */
  PrefetcherIteratorHandle pit = NULL;
  int *scale_list = (int *)malloc(sizeof(int) * n_scales);
  if (!scale_list) {
    fprintf(stderr, "failed to allocate scale list\n");
    return -1;
  }
  for (int i = 0; i < n_scales; ++i)
    scale_list[i] = i;
  pit = MLXTrainer_create_batch_iterator(trainer, dl, scale_list, n_scales);
  free(scale_list);
  if (!pit) {
    fprintf(stderr, "failed to create prefetcher iterator\n");
    return -1;
  }

  /* Training loop: consume prepared pyramids from prefetcher iterator */
  mlx_stream s = mlx_default_cpu_stream_new();
  prefetcher_iterator_preload(pit);
  int batches = 0;
  int max_batches = opts->num_iter > 0 ? opts->num_iter : INT_MAX;
  while (batches < max_batches) {
    PrefetchedPyramidsBatch *pb = prefetcher_iterator_next(pit);
    if (!pb)
      break; /* finished */

    int nsc = pb->n_scales;
    /* copy MLX arrays out of the prefetched batch so we own them during
       the optimization step (prefetcher_free_pyramids will free pb arrays).
       Use mlx_array_set to duplicate content onto new arrays. */
    mlx_array **fac_pyr = (mlx_array **)malloc(sizeof(mlx_array *) * nsc);
    mlx_array **well_pyr = NULL;
    mlx_array **seis_pyr = NULL;
    for (int si = 0; si < nsc; ++si) {
      mlx_array tmp = mlx_array_new();
      if (pb->facies)
        mlx_copy(&tmp, pb->facies[si], s);
      mlx_array *pp = (mlx_array *)malloc(sizeof(mlx_array));
      *pp = tmp;
      fac_pyr[si] = pp;
    }
    if (pb->wells) {
      well_pyr = (mlx_array **)malloc(sizeof(mlx_array *) * nsc);
      for (int si = 0; si < nsc; ++si) {
        mlx_array tmp = mlx_array_new();
        mlx_copy(&tmp, pb->wells[si], s);
        mlx_array *pp = (mlx_array *)malloc(sizeof(mlx_array));
        *pp = tmp;
        well_pyr[si] = pp;
      }
    }
    if (pb->seismic) {
      seis_pyr = (mlx_array **)malloc(sizeof(mlx_array *) * nsc);
      for (int si = 0; si < nsc; ++si) {
        mlx_array tmp = mlx_array_new();
        mlx_copy(&tmp, pb->seismic[si], s);
        mlx_array *pp = (mlx_array *)malloc(sizeof(mlx_array));
        *pp = tmp;
        seis_pyr[si] = pp;
      }
    }

    /* we can free the prefetched batch now */
    prefetcher_free_pyramids(pb);

    int batch_n = (int)opts->batch_size;
    int *indexes = (int *)malloc(sizeof(int) * batch_n);
    for (int i = 0; i < batch_n; ++i)
      indexes[i] = i;

    int act = n_scales;

    /* Stage 3: compute recovery inputs and initialize per-scale noise
       amplitudes (parity with Python Trainer). */
    mlx_array **rec_pyr = NULL;
    if (nsc > 0) {
      rec_pyr = (mlx_array **)calloc((size_t)nsc, sizeof(mlx_array *));
      for (int si = 0; si < nsc; ++si) {
        mlx_array *tmp_rec = NULL;
        if (mlx_compute_rec_input(si, indexes, batch_n, fac_pyr, &tmp_rec) ==
                0 &&
            tmp_rec) {
          rec_pyr[si] = tmp_rec;
        } else {
          rec_pyr[si] = NULL;
        }
        /* initialize noise amplitudes based on current real sample */
        (void)mlx_init_rec_noise_and_amp(
            trainer->model, si, indexes, batch_n,
            fac_pyr && fac_pyr[si] ? fac_pyr[si] : NULL, well_pyr, seis_pyr);
      }
    }

    int step_rc = MLXTrainer_optimization_step(
        trainer, indexes, batch_n, fac_pyr, nsc, rec_pyr, nsc, well_pyr,
        well_pyr ? nsc : 0, NULL, 0, seis_pyr, seis_pyr ? nsc : 0, scales, act);

    /* free rec pyramids */
    if (rec_pyr) {
      for (int si = 0; si < nsc; ++si) {
        if (rec_pyr[si]) {
          mlx_array_free(*rec_pyr[si]);
          free(rec_pyr[si]);
        }
      }
      free(rec_pyr);
    }

    /* free pyramids */
    for (int si = 0; si < nsc; ++si) {
      if (fac_pyr && fac_pyr[si]) {
        mlx_array_free(*fac_pyr[si]);
        free(fac_pyr[si]);
      }
      if (well_pyr && well_pyr[si]) {
        mlx_array_free(*well_pyr[si]);
        free(well_pyr[si]);
      }
      if (seis_pyr && seis_pyr[si]) {
        mlx_array_free(*seis_pyr[si]);
        free(seis_pyr[si]);
      }
    }
    if (fac_pyr)
      free(fac_pyr);
    if (well_pyr)
      free(well_pyr);
    if (seis_pyr)
      free(seis_pyr);
    free(indexes);

    /* build a simple JSON metrics and update visualizer (best-effort) */
    char metrics[1024];
    int off = 0;
    off += snprintf(metrics + off, sizeof(metrics) - off, "{");
    for (int sc = 0; sc < n_scales; ++sc) {
      off += snprintf(metrics + off, sizeof(metrics) - off,
                      "\"%d\":{\"d_total\":%g}", sc, 0.0);
      if (sc + 1 < n_scales)
        off += snprintf(metrics + off, sizeof(metrics) - off, ",");
    }
    off += snprintf(metrics + off, sizeof(metrics) - off, "}");
    pybridge_update_visualizer_from_json(batches, metrics,
                                         batches * opts->batch_size);

    batches++;

    if (batches % opts->save_interval == 0) {
      MLXTrainer_save_generated_facies(
          trainer, 0, batches, opts->output_path ? opts->output_path : ".");
    }
  }

  mlx_stream_free(s);
  /* Free local dataset/dataloader only when they were created by this
   * function. If trainer owns them (created in MLXTrainer_new) they will be
   * freed by MLXTrainer_destroy below. */
  if (created_dl && dl)
    facies_dataloader_free(dl);
  free(scales);
  return 0;
}
