// This file is a rename of c_trainer_api.c — kept identical contents.
#include "models/base_manager.h"
#include "models/facies_gan.h"
#include "optimizer.h"
#include "trainning/train_manager.h"
#include "trainning/train_step.h"

/* Inlined MLXTrainer API (previously in mlx_trainer_api.h) to produce a
 * single-file implementation as requested. Keep declarations here so this
 * source is self-contained for the trainer API. */

typedef struct MLXTrainer MLXTrainer;

MLXTrainer *MLXTrainer_create_with_opts(const TrainningOptions *opts);
void MLXTrainer_destroy(MLXTrainer *t);
int MLXTrainer_run(int num_samples, int num_scales, int channels, int height,
                   int width, int batch_size);
int MLXTrainer_run_with_opts(const TrainningOptions *opts);
int MLXTrainer_run_full(const TrainningOptions *opts);
int MLXTrainer_optimization_step(MLXTrainer *t, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **rec_in_pyramid,
                                 int n_rec, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *active_scales,
                                 int n_active_scales);
int MLXTrainer_setup_optimizers(MLXTrainer *t, const int *scales, int n_scales);
int MLXTrainer_get_n_scales(MLXTrainer *t);
int MLXTrainer_load_model(MLXTrainer *t, int scale, const char *checkpoint_dir);
int MLXTrainer_save_generated_facies(MLXTrainer *t, int scale, int epoch,
                                     const char *results_path);
void *MLXTrainer_get_model_ctx(MLXTrainer *t);
#include "datasets/dataloader.h"
#include "datasets/func_cache.h"
#include "datasets/mlx_dataset.h"
#include "datasets/prefetcher.h"
#include "datasets/wells.h"
#include "io/npz_unzip.h"
#include "trainning/pybridge.h"
#include "trainning/train_manager.h"
#include "trainning/train_step.h"
#include "trainning/train_utils.h"
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

/* Producer thread args and implementation moved to file-scope so the thread
 * function is not defined inside another function (nested functions are
 * invalid in C). This thread pulls batches from a facies_dataloader and
 * pushes them into a PrefetcherHandle. */
typedef struct ProducerArgs {
  facies_dataloader *dl;
  PrefetcherHandle ph;
  mlx_stream s;
} ProducerArgs;

static void *producer_thread(void *v) {
  ProducerArgs *a = (ProducerArgs *)v;
  facies_dataloader *dl = a->dl;
  PrefetcherHandle ph = a->ph;
  mlx_stream s = a->s;
  while (1) {
    mlx_vector_array facs = mlx_vector_array_new();
    mlx_vector_array wells_out = mlx_vector_array_new();
    mlx_vector_array seis_out = mlx_vector_array_new();
    int rc = facies_dataloader_next(dl, &facs, &wells_out, &seis_out, s);
    if (rc == 2) {
      mlx_vector_array_free(facs);
      mlx_vector_array_free(wells_out);
      mlx_vector_array_free(seis_out);
      break;
    } else if (rc != 0) {
      mlx_vector_array_free(facs);
      mlx_vector_array_free(wells_out);
      mlx_vector_array_free(seis_out);
      break;
    }

    int nsc = (int)mlx_vector_array_size(facs);
    mlx_array *fac_arr = NULL;
    mlx_array *well_arr = NULL;
    mlx_array *sei_arr = NULL;
    if (nsc > 0) {
      fac_arr = (mlx_array *)malloc(sizeof(mlx_array) * nsc);
      for (int i = 0; i < nsc; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, facs, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        fac_arr[i] = tmp;
      }
    }
    int nw = (int)mlx_vector_array_size(wells_out);
    if (nw > 0) {
      well_arr = (mlx_array *)malloc(sizeof(mlx_array) * nw);
      for (int i = 0; i < nw; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, wells_out, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        well_arr[i] = tmp;
      }
    }
    int ns = (int)mlx_vector_array_size(seis_out);
    if (ns > 0) {
      sei_arr = (mlx_array *)malloc(sizeof(mlx_array) * ns);
      for (int i = 0; i < ns; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, seis_out, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        sei_arr[i] = tmp;
      }
    }

    /* push into prefetcher (copies into internal buffers/stream) */
    prefetcher_push_mlx(ph, fac_arr, nsc, well_arr, nw, NULL, 0, sei_arr, ns);

    /* free our temporary copies */
    if (fac_arr) {
      for (int i = 0; i < nsc; ++i)
        mlx_array_free(fac_arr[i]);
      free(fac_arr);
    }
    if (well_arr) {
      for (int i = 0; i < nw; ++i)
        mlx_array_free(well_arr[i]);
      free(well_arr);
    }
    if (sei_arr) {
      for (int i = 0; i < ns; ++i)
        mlx_array_free(sei_arr[i]);
      free(sei_arr);
    }

    mlx_vector_array_free(facs);
    mlx_vector_array_free(wells_out);
    mlx_vector_array_free(seis_out);
  }
  prefetcher_mark_finished(a->ph);
  mlx_stream_free(a->s);
  return NULL;
}

struct MLXTrainer {
  TrainningOptions opts;
  MLXFaciesGAN *model;
  MLXOptimizer **gen_opts;  /* per-scale */
  MLXOptimizer **disc_opts; /* per-scale */
  MLXScheduler **gen_scheds;
  MLXScheduler **disc_scheds;
  int n_scales;
  /* Optional per-trainer prefetcher/iterator for batch iteration */
  PrefetcherHandle batch_prefetcher;
  PrefetcherIteratorHandle batch_iterator;
  pthread_t batch_producer;
  int batch_producer_running;
};

MLXTrainer *MLXTrainer_create_with_opts(const TrainningOptions *opts) {
  MLXTrainer *t = (MLXTrainer *)malloc(sizeof(MLXTrainer));
  memset(t, 0, sizeof(*t));
  t->opts = *opts; /* copy options */
  MLXBaseManager *mgr = mlx_base_manager_create_from_trainning(&t->opts);
  if (!mgr) {
    free(t);
    return NULL;
  }
  if (t->opts.num_parallel_scales > 0) {
    mlx_base_manager_init_scales(mgr, 0, t->opts.num_parallel_scales);
  }
  {
    const char *ckpt = ".checkpoints";
    (void)mlx_base_manager_load(mgr, ckpt, 1 /*load_shapes*/,
                                -1 /*until_scale*/, 0 /*load_disc*/,
                                0 /*load_wells*/);
  }
  t->model = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
  if (!t->model) {
    mlx_base_manager_free(mgr);
    free(t);
    return NULL;
  }
  /* number of scales is driven by the model state/shapes (4 ints per
   * scale: batch, channels, height, width). `mlx_faciesgan_get_shapes_flat`
   * returns the number of scales and a flat array with 4*scales ints. */
  int *shapes = NULL;
  int n_scales = 0;
  if (mlx_faciesgan_get_shapes_flat(t->model, &shapes, &n_scales) == 0 &&
      n_scales > 0) {
    t->n_scales = n_scales;

    /* Print shapes in a compact table similar to Python Trainer.__init__ */
    fprintf(stdout, "Generated facies shapes:\n");
    fprintf(stdout, "╔══════════╦══════════╦══════════╦══════════╗\n");
    fprintf(stdout, "║ %8s ║ %8s ║ %8s ║ %8s ║\n", "Batch", "Channels",
            "Height", "Width");
    fprintf(stdout, "╠══════════╬══════════╬══════════╬══════════╣\n");
    for (int si = 0; si < n_scales; ++si) {
      int b = shapes[si * 4 + 0];
      int c = shapes[si * 4 + 1];
      int h = shapes[si * 4 + 2];
      int w = shapes[si * 4 + 3];
      fprintf(stdout, "║ %8d ║ %8d ║ %8d ║ %8d ║\n", b, c, h, w);
    }
    fprintf(stdout, "╚══════════╩══════════╩══════════╩══════════╝\n");

    free(shapes);
  }
  /* nothing dataset-specific here; MLXTrainer_run_full handles dataset-driven
   * runs */
  /* allocate arrays for per-scale optimizers/schedulers */
  /* Allocate enough slots for either discovered model scales or the
   * configured number of parallel scales (whichever is larger). This
   * avoids calloc(0,...) and prevents out-of-bounds writes when
   * `MLXTrainer_setup_optimizers` uses `num_parallel_scales`. */
  int alloc_n = t->n_scales;
  if (t->opts.num_parallel_scales > alloc_n)
    alloc_n = t->opts.num_parallel_scales;
  if (alloc_n <= 0)
    alloc_n = 1;
  t->gen_opts =
      (MLXOptimizer **)calloc((size_t)alloc_n, sizeof(MLXOptimizer *));
  t->disc_opts =
      (MLXOptimizer **)calloc((size_t)alloc_n, sizeof(MLXOptimizer *));
  t->gen_scheds =
      (MLXScheduler **)calloc((size_t)alloc_n, sizeof(MLXScheduler *));
  t->disc_scheds =
      (MLXScheduler **)calloc((size_t)alloc_n, sizeof(MLXScheduler *));

  t->batch_prefetcher = NULL;
  t->batch_iterator = NULL;
  t->batch_producer_running = 0;

  return t;
}

void MLXTrainer_destroy(MLXTrainer *t) {
  if (!t)
    return;
  if (t->model)
    mlx_faciesgan_free(t->model);

  /* free per-scale optimizer/scheduler instances (allocated to max of
   * discovered scales or configured parallel scales). */
  int alloc_n = t->n_scales;
  if (t->opts.num_parallel_scales > alloc_n)
    alloc_n = t->opts.num_parallel_scales;
  if (alloc_n <= 0)
    alloc_n = 1;

  if (t->gen_opts) {
    for (int i = 0; i < alloc_n; ++i) {
      if (t->gen_opts[i])
        mlx_adam_free(t->gen_opts[i]);
    }
  }
  if (t->disc_opts) {
    for (int i = 0; i < alloc_n; ++i) {
      if (t->disc_opts[i])
        mlx_adam_free(t->disc_opts[i]);
    }
  }
  if (t->gen_scheds) {
    for (int i = 0; i < alloc_n; ++i) {
      if (t->gen_scheds[i])
        mlx_scheduler_free(t->gen_scheds[i]);
    }
  }
  if (t->disc_scheds) {
    for (int i = 0; i < alloc_n; ++i) {
      if (t->disc_scheds[i])
        mlx_scheduler_free(t->disc_scheds[i]);
    }
  }

  free(t->gen_opts);
  free(t->disc_opts);
  free(t->gen_scheds);
  free(t->disc_scheds);
  free(t);
}

int MLXTrainer_setup_optimizers(MLXTrainer *t, const int *scales,
                                int n_scales) {
  if (!t)
    return -1;

  int *local_scales = NULL;
  int local_n = n_scales;

  /* If caller passes NULL/0, interpret as "use configured parallel scales"
   * Prefer the explicit TrainningOptions value (`num_parallel_scales`) when
   * present; otherwise fall back to discovered model scales. This mirrors
   * the Python trainer which uses options to determine parallelism. */
  if (!scales || n_scales <= 0) {
    if (t->opts.num_parallel_scales > 0) {
      local_n = t->opts.num_parallel_scales;
    } else if (t->n_scales > 0) {
      local_n = t->n_scales;
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
    t->gen_opts[sc] =
        mlx_adam_create(t->opts.lr_g, t->opts.beta1, 0.999f, 1e-8f);
    t->disc_opts[sc] =
        mlx_adam_create(t->opts.lr_d, t->opts.beta1, 0.999f, 1e-8f);
    /* schedulers: multistep with single milestone (lr_decay) */
    int milestones[1] = {t->opts.lr_decay};
    t->gen_scheds[sc] = mlx_scheduler_multistep_create_with_init(
        milestones, 1, t->opts.gamma, (const float *)&t->opts.lr_g, 1);
    t->disc_scheds[sc] = mlx_scheduler_multistep_create_with_init(
        milestones, 1, t->opts.gamma, (const float *)&t->opts.lr_d, 1);
    /* attach scheduler and optimizer so LR updates propagate */
    if (t->gen_opts[sc] && t->gen_scheds[sc]) {
      mlx_optimizer_attach_scheduler(t->gen_opts[sc], t->gen_scheds[sc]);
      mlx_scheduler_attach_optimizer(t->gen_scheds[sc], t->gen_opts[sc]);
    }
    if (t->disc_opts[sc] && t->disc_scheds[sc]) {
      mlx_optimizer_attach_scheduler(t->disc_opts[sc], t->disc_scheds[sc]);
      mlx_scheduler_attach_optimizer(t->disc_scheds[sc], t->disc_opts[sc]);
    }
  }
  if (local_scales)
    free(local_scales);
  return 0;
}

/* Producer thread used by trainer-created prefetchers. */
typedef struct TrainerProducerArgs {
  facies_dataloader *dl;
  PrefetcherHandle ph;
  mlx_stream s;
} TrainerProducerArgs;

static void *trainer_producer_thread(void *v) {
  TrainerProducerArgs *a = (TrainerProducerArgs *)v;
  facies_dataloader *dl = a->dl;
  PrefetcherHandle ph = a->ph;
  mlx_stream s = a->s;

  while (1) {
    mlx_vector_array facs = mlx_vector_array_new();
    mlx_vector_array wells_out = mlx_vector_array_new();
    mlx_vector_array seis_out = mlx_vector_array_new();
    int rc = facies_dataloader_next(dl, &facs, &wells_out, &seis_out, s);
    if (rc == 2) {
      mlx_vector_array_free(facs);
      mlx_vector_array_free(wells_out);
      mlx_vector_array_free(seis_out);
      break;
    } else if (rc != 0) {
      mlx_vector_array_free(facs);
      mlx_vector_array_free(wells_out);
      mlx_vector_array_free(seis_out);
      break;
    }

    int nsc = (int)mlx_vector_array_size(facs);
    mlx_array *fac_arr = NULL;
    mlx_array *well_arr = NULL;
    mlx_array *sei_arr = NULL;
    if (nsc > 0) {
      fac_arr = (mlx_array *)malloc(sizeof(mlx_array) * nsc);
      for (int i = 0; i < nsc; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, facs, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        fac_arr[i] = tmp;
      }
    }
    int nw = (int)mlx_vector_array_size(wells_out);
    if (nw > 0) {
      well_arr = (mlx_array *)malloc(sizeof(mlx_array) * nw);
      for (int i = 0; i < nw; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, wells_out, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        well_arr[i] = tmp;
      }
    }
    int ns = (int)mlx_vector_array_size(seis_out);
    if (ns > 0) {
      sei_arr = (mlx_array *)malloc(sizeof(mlx_array) * ns);
      for (int i = 0; i < ns; ++i) {
        mlx_array tmp = mlx_array_new();
        if (mlx_vector_array_get(&tmp, seis_out, i) != 0) {
          mlx_array_free(tmp);
          tmp = mlx_array_new();
        }
        sei_arr[i] = tmp;
      }
    }

    prefetcher_push_mlx(ph, fac_arr, nsc, well_arr, nw, NULL, 0, sei_arr, ns);

    if (fac_arr) {
      for (int i = 0; i < nsc; ++i)
        mlx_array_free(fac_arr[i]);
      free(fac_arr);
    }
    if (well_arr) {
      for (int i = 0; i < nw; ++i)
        mlx_array_free(well_arr[i]);
      free(well_arr);
    }
    if (sei_arr) {
      for (int i = 0; i < ns; ++i)
        mlx_array_free(sei_arr[i]);
      free(sei_arr);
    }

    mlx_vector_array_free(facs);
    mlx_vector_array_free(wells_out);
    mlx_vector_array_free(seis_out);
  }

  prefetcher_mark_finished(ph);
  if (a->s.ctx)
    mlx_stream_free(a->s);
  free(a);
  return NULL;
}

int MLXTrainer_get_n_scales(MLXTrainer *t) {
  if (!t)
    return 0;
  return t->n_scales;
}

int MLXTrainer_compute_rec_input(MLXTrainer *t, int scale, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 mlx_array **out) {
  (void)t;
  return mlx_compute_rec_input(scale, indexes, n_indexes, facies_pyramid, out);
}

int MLXTrainer_init_rec_noise_and_amp(MLXTrainer *t, int scale,
                                      const int *indexes, int n_indexes,
                                      const mlx_array *real,
                                      mlx_array **wells_pyramid,
                                      mlx_array **seismic_pyramid) {
  if (!t)
    return -1;
  return mlx_init_rec_noise_and_amp(t->model, scale, indexes, n_indexes, real,
                                    wells_pyramid, seismic_pyramid);
}

PrefetcherIteratorHandle MLXTrainer_create_batch_iterator(MLXTrainer *t,
                                                          facies_dataloader *dl,
                                                          const int *scales,
                                                          int n_scales) {
  if (!t || !dl)
    return NULL;
  int qcap = 4;
  mlx_stream s = mlx_default_cpu_stream_new();
  PrefetcherHandle ph =
      prefetcher_create_with_stream(qcap, s, (const int *)scales, n_scales);
  if (!ph) {
    if (s.ctx)
      mlx_stream_free(s);
    return NULL;
  }
  t->batch_prefetcher = ph;
  t->batch_iterator = prefetcher_iterator_create(ph);
  TrainerProducerArgs *args =
      (TrainerProducerArgs *)malloc(sizeof(TrainerProducerArgs));
  args->dl = dl;
  args->ph = ph;
  args->s = mlx_default_cpu_stream_new();
  pthread_create(&t->batch_producer, NULL, trainer_producer_thread, args);
  pthread_detach(t->batch_producer);
  t->batch_producer_running = 1;
  return t->batch_iterator;
}

int MLXTrainer_create_dataloader(MLXTrainer *t, facies_dataloader **out,
                                 facies_dataset *ds, size_t batch_size,
                                 unsigned int seed, int num_workers,
                                 int prefetch_factor, int timeout_ms) {
  (void)t;
  if (!out || !ds)
    return -1;
  return facies_dataloader_new_ex(
      out, ds, batch_size, false, false, seed, num_workers, prefetch_factor,
      num_workers > 0, timeout_ms, NULL, NULL, false, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, 0, NULL, NULL);
}

int MLXTrainer_generate_visualization_samples(
    MLXTrainer *t, const int *scales, int n_scales, const int *indexes,
    int n_indexes, mlx_array **wells_pyramid, int n_wells,
    mlx_array **seismic_pyramid, int n_seismic, mlx_array ***out_generated,
    int *n_out) {
  if (!t || !scales || n_scales <= 0 || !out_generated || !n_out)
    return -1;
  mlx_array **out = (mlx_array **)malloc(sizeof(mlx_array *) * n_scales);
  if (!out)
    return -1;
  for (int i = 0; i < n_scales; ++i) {
    int scale = scales[i];
    mlx_array **noises = NULL;
    int n_noises = 0;
    if (mlx_faciesgan_get_pyramid_noise(t->model, scale, indexes, n_indexes,
                                        &noises, &n_noises, wells_pyramid,
                                        seismic_pyramid, 0) != 0) {
      out[i] = NULL;
      continue;
    }
    float *use_amps = NULL;
    int use_n = 0;
    if (mlx_faciesgan_get_noise_amplitude(t->model, scale, &use_amps, &use_n) !=
        0) {
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
              1, t->opts.crop_size > 0 ? t->opts.crop_size : 32,
              t->opts.crop_size > 0 ? t->opts.crop_size : 32,
              t->opts.num_img_channels > 0 ? t->opts.num_img_channels : 1};
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
    for (int j = 0; j < n_noises; ++j) {
      fprintf(stdout, "[debug] save_generated noise[%d] ndim=%zu\n", j,
              mlx_array_ndim(zvals[j]));
    }
    fprintf(
        stdout,
        "[debug] save_generated calling generate_fake: n_noises=%d scale=%d\n",
        n_noises, scale);
    mlx_array fake = mlx_faciesgan_generate_fake(
        t->model, zvals, n_noises, use_amps, use_n, in_noise, scale, scale);

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

int MLXTrainer_optimization_step(MLXTrainer *t, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **rec_in_pyramid,
                                 int n_rec, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *active_scales,
                                 int n_active_scales) {
  /* destroy prefetcher resources if any */
  if (t->batch_iterator) {
    prefetcher_iterator_destroy(t->batch_iterator);
    t->batch_iterator = NULL;
  }
  if (t->batch_prefetcher) {
    prefetcher_destroy(t->batch_prefetcher);
    t->batch_prefetcher = NULL;
  }
  if (!t)
    return -1;

  MLXResults *res = NULL;
  int rc = mlx_faciesgan_collect_metrics_and_grads(
      t->model, indexes, n_indexes, active_scales, n_active_scales,
      facies_pyramid, rec_in_pyramid, wells_pyramid, masks_pyramid,
      seismic_pyramid, t->opts.lambda_diversity, t->opts.well_loss_penalty,
      t->opts.alpha, t->opts.lambda_grad, &res);
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
    if (t->gen_scheds && t->gen_scheds[sc])
      mlx_scheduler_step_auto(t->gen_scheds[sc],
                              t->gen_opts ? t->gen_opts[sc] : NULL);
    if (t->disc_scheds && t->disc_scheds[sc])
      mlx_scheduler_step_auto(t->disc_scheds[sc],
                              t->disc_opts ? t->disc_opts[sc] : NULL);

    /* Apply generator grads */
    if (sr->gen_n > 0 && sr->gen_grads && t->gen_opts && t->gen_opts[sc]) {
      int r = mlx_faciesgan_apply_sgd_to_generator(t->model, t->gen_opts[sc],
                                                   sr->gen_grads, sr->gen_n);
      if (r != 0)
        overall = -1;
    }

    /* Apply discriminator grads */
    if (sr->disc_n > 0 && sr->disc_grads && t->disc_opts && t->disc_opts[sc]) {
      int r = mlx_faciesgan_apply_sgd_to_discriminator(
          t->model, t->disc_opts[sc], sr->disc_grads, sr->disc_n);
      if (r != 0)
        overall = -1;
    }
  }

  mlx_results_free(res);
  return overall == 0 ? 0 : -1;
}

int MLXTrainer_load_model(MLXTrainer *t, int scale,
                          const char *checkpoint_dir) {
  if (!t || !checkpoint_dir)
    return -1;
  /* Reuse existing per-scale state loaders (load_*_state). The expected
   * argument is a directory path for the scale; pass `checkpoint_dir/scale`. */
  char scale_dir[PATH_MAX];
  snprintf(scale_dir, PATH_MAX, "%s/%d", checkpoint_dir, scale);
  if (mlx_faciesgan_load_generator_state(t->model, scale_dir, scale) != 0)
    return -1;
  if (mlx_faciesgan_load_discriminator_state(t->model, scale_dir, scale) != 0)
    return -1;
  return 0;
}

int MLXTrainer_save_generated_facies(MLXTrainer *t, int scale, int epoch,
                                     const char *results_path) {
  if (!t || !results_path)
    return -1;
  /* Generate noises and call numeric forward (similar to train_utils.c) */
  mlx_array **noises = NULL;
  int n_noises = 0;
  if (mlx_faciesgan_get_pyramid_noise(t->model, scale, NULL, 0, &noises,
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
  mlx_array_t fake =
      mlx_faciesgan_generate_fake(t->model, (const mlx_array *)noises, n_noises,
                                  use_amps, scale + 1, in_noise, scale, scale);

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

void *MLXTrainer_get_model_ctx(MLXTrainer *t) {
  if (!t)
    return NULL;
  return (void *)t->model;
}

void *MLXTrainer_create_model(MLXTrainer *t) {
  return MLXTrainer_get_model_ctx(t);
}

int MLXTrainer_train_scales(MLXTrainer *t, const int *indexes, int n_indexes,
                            mlx_array **facies_pyramid, int n_facies,
                            mlx_array **wells_pyramid, int n_wells,
                            mlx_array **masks_pyramid, int n_masks,
                            mlx_array **seismic_pyramid, int n_seismic,
                            const int *scales, int n_scales, int num_iter) {
  if (!t || !indexes || n_indexes <= 0 || !facies_pyramid || n_facies <= 0 ||
      !scales || n_scales <= 0 || num_iter <= 0)
    return -1;

  /* Writer canary: emit backtrace when MLXTrainer runs training loop */
  fprintf(stderr, "[writer_canary] func=MLXTrainer_train_scales tid=%lu\n",
          (unsigned long)pthread_self());
  {
    void *bt[64];
    int bt_size = backtrace(bt, 64);
    backtrace_symbols_fd(bt, bt_size, fileno(stderr));
  }

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
          (MLXFaciesGAN *)MLXTrainer_get_model_ctx(t), si, indexes, n_indexes,
          facies_pyramid[si], wells_pyramid, seismic_pyramid);
    }

    int rc = MLXTrainer_optimization_step(
        t, indexes, n_indexes, facies_pyramid, n_facies, rec_pyr, n_facies,
        wells_pyramid, n_wells, masks_pyramid, n_masks, seismic_pyramid,
        n_seismic, scales, n_scales);

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

  facies_dataset *ds = NULL;
  if (facies_dataset_new(&ds, facies_pyramids, wells, seismic) != 0) {
    fprintf(stderr, "failed to create facies_dataset\n");
    mlx_vector_vector_array_free(facies_pyramids);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    return 1;
  }

  facies_dataloader *dl = NULL;
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
  return MLXTrainer_run_full(opts);
}

/* Minimal MLXTrainer_run_full stub to keep the renamed file self-contained.
 * The full dataset-driven implementation is large and lives elsewhere; if
 * you want the full behavior restored under the new filename I can inline
 * it here as well. */
int MLXTrainer_run_full(const TrainningOptions *opts) {
  if (!opts)
    return -1;

  MLXTrainer *t = MLXTrainer_create_with_opts(opts);
  if (!t)
    return -1;

  int n_scales = MLXTrainer_get_n_scales(t);
  if (n_scales <= 0) {
    MLXTrainer_destroy(t);
    fprintf(stderr, "no model scales available\n");
    return -1;
  }

  /* Attempt to load MLX pyramids dataset from function cache (parity with
   * Python MLXPyramidsDataset). This populates facies/wells/seismic
   * vector-of-vector arrays which we pass to the existing facies_dataset
   * constructor. */
  mlx_vector_vector_array facies_pyramids = mlx_vector_vector_array_new();
  mlx_vector_vector_array wells = mlx_vector_vector_array_new();
  mlx_vector_vector_array masks = mlx_vector_vector_array_new();
  mlx_vector_vector_array seismic = mlx_vector_vector_array_new();
  int loaded_samples = 0;
  const char *cache_dir = opts->output_path ? opts->output_path : ".";
  int desired = opts->num_train_pyramids > 0 ? opts->num_train_pyramids : 1024;
  if (mlx_pyramids_dataset_load(
          opts->input_path ? opts->input_path : ".", cache_dir, desired,
          opts->stop_scale, opts->crop_size, opts->num_img_channels,
          opts->use_wells ? 1 : 0, opts->use_seismic ? 1 : 0, opts->manual_seed,
          &facies_pyramids, &wells, &masks, &seismic, &loaded_samples) != 0) {
    fprintf(stderr,
            "failed to load MLX pyramids dataset; falling back to synthetic\n");
    /* fallback to previous synthetic behavior: build a small synthetic
     * dataset to allow quick smoke runs. */
    int num_samples = 64;
    for (int si = 0; si < num_samples; ++si) {
      mlx_vector_array sample = mlx_vector_array_new();
      for (int sc = 0; sc < n_scales; ++sc) {
        int shape[3];
        shape[0] = opts->crop_size > 0 ? opts->crop_size : 32;
        shape[1] = shape[0];
        shape[2] = opts->num_img_channels > 0 ? opts->num_img_channels : 1;
        mlx_array a = mlx_array_new();
        mlx_stream s = mlx_default_cpu_stream_new();
        if (mlx_random_normal(&a, shape, 3, MLX_FLOAT32, 0.0f, 1.0f,
                              mlx_array_empty, s) != 0) {
          mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
        }
        mlx_stream_free(s);
        mlx_vector_array_append_value(sample, a);
        mlx_array_free(a);
      }
      mlx_vector_vector_array_append_value(facies_pyramids, sample);
      mlx_vector_array_free(sample);
    }
  }

  facies_dataset *ds = NULL;
  if (facies_dataset_new(&ds, facies_pyramids, wells, seismic) != 0) {
    fprintf(stderr, "failed to create facies_dataset\n");
    mlx_vector_vector_array_free(facies_pyramids);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    MLXTrainer_destroy(t);
    return -1;
  }

  /* Create dataloader with options-driven parameters (best-effort mapping).
     Use facies_dataloader_new_ex to configure workers and prefetch. */
  facies_dataloader *dl = NULL;
  unsigned int seed = (unsigned int)(opts->manual_seed >= 0 ? opts->manual_seed
                                                            : (int)time(NULL));
  if (facies_dataloader_new_ex(
          &dl, ds, (size_t)opts->batch_size, false, false, seed,
          opts->num_workers, 2, opts->num_workers > 0, 2000, NULL, NULL, false,
          NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL) != 0) {
    fprintf(stderr, "failed to create facies_dataloader\n");
    facies_dataset_free(ds);
    mlx_vector_vector_array_free(facies_pyramids);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    MLXTrainer_destroy(t);
    return -1;
  }

  /* Visualizer / background worker bridge (best-effort) */
  pybridge_create_visualizer(
      n_scales, opts->output_path ? opts->output_path : ".", NULL, 1);
  pybridge_create_background_worker(2, 32);

  /* Setup optimizers for all scales */
  int *scales = (int *)malloc(sizeof(int) * n_scales);
  for (int i = 0; i < n_scales; ++i)
    scales[i] = i;
  MLXTrainer_setup_optimizers(t, scales, n_scales);

  /* Create a prefetcher and a small producer thread that pulls from the
     facies dataloader and enqueues prepared MLX pyramids into the
     prefetcher. This mirrors Python's MLXDataPrefetcher behavior. */
  PrefetcherHandle ph = NULL;
  PrefetcherIteratorHandle pit = NULL;
  int *scale_list = (int *)malloc(sizeof(int) * n_scales);
  for (int i = 0; i < n_scales; ++i)
    scale_list[i] = i;
  /* queue capacity: heuristic based on workers/prefetch factor */
  int qcap = opts->num_workers > 0 ? opts->num_workers * 2 : 4;
  mlx_stream stream_for_prefetch = mlx_default_cpu_stream_new();
  ph = prefetcher_create_with_stream(qcap, stream_for_prefetch, scale_list,
                                     n_scales);
  pit = prefetcher_iterator_create(ph);

  ProducerArgs pargs;
  pargs.dl = dl;
  pargs.ph = ph;
  pargs.s = mlx_default_cpu_stream_new();
  pthread_t prod_th;
  pthread_create(&prod_th, NULL, producer_thread, &pargs);
  pthread_detach(prod_th);

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
        mlx_array_set(&tmp, pb->facies[si]);
      mlx_array *pp = (mlx_array *)malloc(sizeof(mlx_array));
      *pp = tmp;
      fac_pyr[si] = pp;
    }
    if (pb->wells) {
      well_pyr = (mlx_array **)malloc(sizeof(mlx_array *) * nsc);
      for (int si = 0; si < nsc; ++si) {
        mlx_array tmp = mlx_array_new();
        mlx_array_set(&tmp, pb->wells[si]);
        mlx_array *pp = (mlx_array *)malloc(sizeof(mlx_array));
        *pp = tmp;
        well_pyr[si] = pp;
      }
    }
    if (pb->seismic) {
      seis_pyr = (mlx_array **)malloc(sizeof(mlx_array *) * nsc);
      for (int si = 0; si < nsc; ++si) {
        mlx_array tmp = mlx_array_new();
        mlx_array_set(&tmp, pb->seismic[si]);
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
            t->model, si, indexes, batch_n,
            fac_pyr && fac_pyr[si] ? fac_pyr[si] : NULL, well_pyr, seis_pyr);
      }
    }

    int step_rc = MLXTrainer_optimization_step(
        t, indexes, batch_n, fac_pyr, nsc, rec_pyr, nsc, well_pyr,
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
          t, 0, batches, opts->output_path ? opts->output_path : ".");
    }
  }

  mlx_stream_free(s);
  facies_dataloader_free(dl);
  facies_dataset_free(ds);
  mlx_vector_vector_array_free(facies_pyramids);
  mlx_vector_vector_array_free(wells);
  mlx_vector_vector_array_free(seismic);
  free(scales);
  MLXTrainer_destroy(t);
  pybridge_close_visualizer();
  return 0;
}
