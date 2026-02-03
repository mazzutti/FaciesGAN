// This file is a rename of c_trainer_api.c — kept identical contents.
#include "trainning/mlx_trainer_api.h"
#include "array_helpers.h"
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
#include <mlx/c/transforms.h>
#include <mlx/c/vector.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#else
#include <sys/resource.h>
#endif
#include <time.h>
#include <unistd.h>
#include "trainning/progress.h"

static size_t mlx_get_process_rss_bytes(void) {
#if defined(__APPLE__)
    struct task_basic_info info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS) {
        return 0;
    }
    return (size_t)info.resident_size;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0;
    }
    /* Linux reports ru_maxrss in KB; convert to bytes. */
    return (size_t)ru.ru_maxrss * 1024ULL;
#endif
}

/* Helper: print epoch metrics table matching Python's handle_epoch_end format.
 * Prints at epochs 0, 49, 99, etc. (every 50 epochs) and the last epoch.
 * This provides a visual match to the Python training output. */
static void print_epoch_metrics_table(int batch_id, int total_batches, 
                                       int epoch, int num_iter, int n_scales,
                                       const double *g_totals, const double *g_advs,
                                       const double *g_recs, const double *g_wells,
                                       const double *g_divs, const double *d_totals,
                                       const double *d_reals, const double *d_fakes,
                                       const double *d_gps) {
    /* Check if we should print: epoch % 50 == 0 or epoch == 0 or epoch == num_iter - 1 */
    int should_print = ((epoch + 1) % 50 == 0) || (epoch == 0) || (epoch == num_iter - 1);
    if (!should_print || n_scales <= 0) return;
    
    /* Clear the current line (progress bar may have left content) before printing table */
    printf("\r%120s\r", "");
    printf("  Batch [%d/%d] Epoch [%4d/%d]\n", batch_id + 1, total_batches, epoch + 1, num_iter);
    printf("  \u250c");
    for (int i = 0; i < 99; ++i) printf("\u2500");
    printf("\u2510\n");
    printf("  \u2502 %5s \u2502 %8s \u2502 %7s \u2502 %7s \u2502 %7s \u2502 %7s \u2502 %8s \u2502 %7s \u2502 %7s \u2502 %7s \u2502\n",
           "Scale", "G_total", "G_adv", "G_rec", "G_well", "G_div", "D_total", "D_real", "D_fake", "D_gp");
    printf("  \u251c");
    for (int i = 0; i < 99; ++i) printf("\u2500");
    printf("\u2524\n");
    
    for (int sc = 0; sc < n_scales; ++sc) {
        printf("  \u2502 %5d \u2502 %8.3f \u2502 %7.3f \u2502 %7.3f \u2502 %7.3f \u2502 %7.3f \u2502 %8.3f \u2502 %7.3f \u2502 %7.3f \u2502 %7.3f \u2502\n",
               sc,
               g_totals ? g_totals[sc] : 0.0,
               g_advs ? g_advs[sc] : 0.0,
               g_recs ? g_recs[sc] : 0.0,
               g_wells ? g_wells[sc] : 0.0,
               g_divs ? g_divs[sc] : 0.0,
               d_totals ? d_totals[sc] : 0.0,
               d_reals ? d_reals[sc] : 0.0,
               d_fakes ? d_fakes[sc] : 0.0,
               d_gps ? d_gps[sc] : 0.0);
    }
    printf("  \u2514");
    for (int i = 0; i < 99; ++i) printf("\u2500");
    printf("\u2518\n");
    fflush(stdout);
}

/* Forward declarations for implementation functions (end in _impl).
 * These hold the original implementation bodies and will be referenced
 * by the default vtable. Wrappers below dispatch to the vtable so
 * instance-specific ops can be installed on `trainer->ops`.
 */
int MLXTrainer_train_impl(MLXTrainer *trainer);
int MLXTrainer_train_scales_impl(MLXTrainer *trainer, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *scales, int n_scales,
                                 int num_iter);
int MLXTrainer_optimization_step_impl(MLXTrainer *trainer, const int *indexes,
                                      int n_indexes, mlx_array **facies_pyramid,
                                      int n_facies, mlx_array **rec_in_pyramid,
                                      int n_rec, mlx_array **wells_pyramid,
                                      int n_wells, mlx_array **masks_pyramid,
                                      int n_masks, mlx_array **seismic_pyramid,
                                      int n_seismic, const int *active_scales,
                                      int n_active_scales);
void *MLXTrainer_create_model_impl(MLXTrainer *trainer);
void MLXTrainer_destroy_impl(MLXTrainer *trainer);
PrefetcherIteratorHandle
MLXTrainer_create_batch_iterator_impl(MLXTrainer *trainer,
                                      struct MLXDataloader *dl,
                                      const int *scales, int n_scales);
int MLXTrainer_create_dataloader_impl(MLXTrainer *trainer);
int MLXTrainer_init_dataset_impl(MLXTrainer *trainer);
int MLXTrainer_generate_visualization_samples_impl(
    MLXTrainer *trainer, const int *scales, int n_scales, const int *indexes,
    int n_indexes, mlx_array **wells_pyramid, int n_wells,
    mlx_array **seismic_pyramid, int n_seismic, mlx_array ***out_generated,
    int *n_out);
int MLXTrainer_create_visualizer_impl(MLXTrainer *trainer, int update_interval);
int MLXTrainer_update_visualizer_impl(MLXTrainer *trainer, int epoch,
                                      const char *metrics_json,
                                      int samples_processed);
int MLXTrainer_close_visualizer_impl(MLXTrainer *trainer);
int MLXTrainer_setup_optimizers_impl(MLXTrainer *trainer, const int *scales,
                                     int n_scales);
int MLXTrainer_load_model_impl(MLXTrainer *trainer, int scale,
                               const char *checkpoint_dir);
int MLXTrainer_save_generated_facies_impl(MLXTrainer *trainer, int scale,
        int epoch, const char *results_path, mlx_array real_facies,
        mlx_array masks);
void *MLXTrainer_get_model_ctx_impl(MLXTrainer *trainer);
int MLXTrainer_get_shapes_flat_impl(MLXTrainer *t, int **out_shapes,
                                    int *out_n);
int MLXTrainer_set_shapes_impl(MLXTrainer *t, const int *shapes, int n_scales);

/* Thin wrappers that dispatch through the per-instance ops vtable when
 * present. These preserve the external API while allowing per-instance
 * overrides via `trainer->ops`.
 */
int MLXTrainer_train(MLXTrainer *trainer) {
    if (trainer && trainer->ops && trainer->ops->train)
        return trainer->ops->train(trainer);
    return MLXTrainer_train_impl(trainer);
}

int MLXTrainer_train_scales(MLXTrainer *trainer, const int *indexes,
                            int n_indexes, mlx_array **facies_pyramid,
                            int n_facies, mlx_array **wells_pyramid,
                            int n_wells, mlx_array **masks_pyramid, int n_masks,
                            mlx_array **seismic_pyramid, int n_seismic,
                            const int *scales, int n_scales, int num_iter) {
    if (trainer && trainer->ops && trainer->ops->train_scales)
        return trainer->ops->train_scales(
                   trainer, indexes, n_indexes, facies_pyramid, n_facies, wells_pyramid,
                   n_wells, masks_pyramid, n_masks, seismic_pyramid, n_seismic, scales,
                   n_scales, num_iter);
    return MLXTrainer_train_scales_impl(
               trainer, indexes, n_indexes, facies_pyramid, n_facies, wells_pyramid,
               n_wells, masks_pyramid, n_masks, seismic_pyramid, n_seismic, scales,
               n_scales, num_iter);
}

int MLXTrainer_optimization_step(MLXTrainer *trainer, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **rec_in_pyramid,
                                 int n_rec, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *active_scales,
                                 int n_active_scales) {
    if (trainer && trainer->ops && trainer->ops->optimization_step)
        return trainer->ops->optimization_step(
                   trainer, indexes, n_indexes, facies_pyramid, n_facies, rec_in_pyramid,
                   n_rec, wells_pyramid, n_wells, masks_pyramid, n_masks, seismic_pyramid,
                   n_seismic, active_scales, n_active_scales);
    return MLXTrainer_optimization_step_impl(
               trainer, indexes, n_indexes, facies_pyramid, n_facies, rec_in_pyramid,
               n_rec, wells_pyramid, n_wells, masks_pyramid, n_masks, seismic_pyramid,
               n_seismic, active_scales, n_active_scales);
}

void *MLXTrainer_create_model(MLXTrainer *trainer) {
    if (trainer && trainer->ops && trainer->ops->create_model)
        return trainer->ops->create_model(trainer);
    return MLXTrainer_create_model_impl(trainer);
}

void MLXTrainer_destroy_impl(MLXTrainer *trainer) {
    /* Dispatch to instance-specific destroy if provided, but avoid
     * infinite recursion when the ops table points at this implementation.
     * If no custom destroy is installed, call the concrete `MLXTrainer_destroy`
     * which performs the actual tear-down. */
    if (trainer && trainer->ops && trainer->ops->destroy &&
            trainer->ops->destroy != MLXTrainer_destroy_impl) {
        trainer->ops->destroy(trainer);
        return;
    }
    MLXTrainer_destroy(trainer);
}

PrefetcherIteratorHandle
MLXTrainer_create_batch_iterator(MLXTrainer *trainer, struct MLXDataloader *dl,
                                 const int *scales, int n_scales) {
    if (trainer && trainer->ops && trainer->ops->create_batch_iterator)
        return trainer->ops->create_batch_iterator(trainer, dl, scales, n_scales);
    return MLXTrainer_create_batch_iterator_impl(trainer, dl, scales, n_scales);
}
int MLXTrainer_create_dataloader(MLXTrainer *trainer) {
    if (trainer && trainer->ops && trainer->ops->create_dataloader)
        return trainer->ops->create_dataloader(trainer);
    return MLXTrainer_create_dataloader_impl(trainer);
}
int MLXTrainer_init_dataset(MLXTrainer *trainer) {
    if (trainer && trainer->ops && trainer->ops->init_dataset)
        return trainer->ops->init_dataset(trainer);
    return MLXTrainer_init_dataset_impl(trainer);
}
int MLXTrainer_generate_visualization_samples(
    MLXTrainer *trainer, const int *scales, int n_scales, const int *indexes,
    int n_indexes, mlx_array **wells_pyramid, int n_wells,
    mlx_array **seismic_pyramid, int n_seismic, mlx_array ***out_generated,
    int *n_out) {
    if (trainer && trainer->ops && trainer->ops->generate_visualization_samples)
        return trainer->ops->generate_visualization_samples(
                   trainer, scales, n_scales, indexes, n_indexes, wells_pyramid, n_wells,
                   seismic_pyramid, n_scales, out_generated, n_out);
    return MLXTrainer_generate_visualization_samples_impl(
               trainer, scales, n_scales, indexes, n_indexes, wells_pyramid, n_wells,
               seismic_pyramid, n_scales, out_generated, n_out);
}
int MLXTrainer_create_visualizer(MLXTrainer *trainer, int update_interval) {
    if (trainer && trainer->ops && trainer->ops->create_visualizer)
        return trainer->ops->create_visualizer(trainer, update_interval);
    return MLXTrainer_create_visualizer_impl(trainer, update_interval);
}
int MLXTrainer_update_visualizer(MLXTrainer *trainer, int epoch,
                                 const char *metrics_json,
                                 int samples_processed) {
    if (trainer && trainer->ops && trainer->ops->update_visualizer)
        return trainer->ops->update_visualizer(trainer, epoch, metrics_json,
                                               samples_processed);
    return MLXTrainer_update_visualizer_impl(trainer, epoch, metrics_json,
            samples_processed);
}
int MLXTrainer_close_visualizer(MLXTrainer *trainer) {
    if (trainer && trainer->ops && trainer->ops->close_visualizer)
        return trainer->ops->close_visualizer(trainer);
    return MLXTrainer_close_visualizer_impl(trainer);


}
int MLXTrainer_setup_optimizers(MLXTrainer *trainer, const int *scales,
                                int n_scales) {
    if (trainer && trainer->ops && trainer->ops->setup_optimizers)
        return trainer->ops->setup_optimizers(trainer, scales, n_scales);
    return MLXTrainer_setup_optimizers_impl(trainer, scales, n_scales);
}
int MLXTrainer_load_model(MLXTrainer *trainer, int scale,
                          const char *checkpoint_dir) {
    if (trainer && trainer->ops && trainer->ops->load_model)
        return trainer->ops->load_model(trainer, scale, checkpoint_dir);
    return MLXTrainer_load_model_impl(trainer, scale, checkpoint_dir);
}
int MLXTrainer_save_generated_facies(MLXTrainer *trainer, int scale, int epoch,
                                     const char *results_path, mlx_array real_facies,
                                     mlx_array masks) {
    if (trainer && trainer->ops && trainer->ops->save_generated_facies)
        return trainer->ops->save_generated_facies(trainer, scale, epoch,
                results_path, real_facies, masks);
    return MLXTrainer_save_generated_facies_impl(trainer, scale, epoch,
            results_path, real_facies, masks);
}

void *MLXTrainer_get_model_ctx(MLXTrainer *trainer) {
    if (trainer && trainer->ops && trainer->ops->get_model_ctx)
        return trainer->ops->get_model_ctx(trainer);
    return MLXTrainer_get_model_ctx_impl(trainer);
}

int MLXTrainer_get_shapes_flat(MLXTrainer *t, int **out_shapes, int *out_n) {
    if (t && t->ops && t->ops->get_shapes_flat)
        return t->ops->get_shapes_flat(t, out_shapes, out_n);
    return MLXTrainer_get_shapes_flat_impl(t, out_shapes, out_n);
}
int MLXTrainer_set_shapes(MLXTrainer *t, const int *shapes, int n_scales) {
    if (t && t->ops && t->ops->set_shapes)
        return t->ops->set_shapes(t, shapes, n_scales);
    return MLXTrainer_set_shapes_impl(t, shapes, n_scales);
}

/* Define the default ops vtable mapping to the implementation functions. */
MLXTrainerOps default_mlx_trainer_ops = {
    .train = MLXTrainer_train_impl,
    .train_scales = MLXTrainer_train_scales_impl,
    .optimization_step = MLXTrainer_optimization_step_impl,
    .create_model = MLXTrainer_create_model_impl,
    .destroy = MLXTrainer_destroy_impl,
    .create_batch_iterator = MLXTrainer_create_batch_iterator_impl,
    .create_dataloader = MLXTrainer_create_dataloader_impl,
    .init_dataset = MLXTrainer_init_dataset_impl,
    .generate_visualization_samples =
    MLXTrainer_generate_visualization_samples_impl,
    .create_visualizer = MLXTrainer_create_visualizer_impl,
    .update_visualizer = MLXTrainer_update_visualizer_impl,
    .close_visualizer = MLXTrainer_close_visualizer_impl,
    .setup_optimizers = MLXTrainer_setup_optimizers_impl,
    .load_model = MLXTrainer_load_model_impl,
    .save_generated_facies = MLXTrainer_save_generated_facies_impl,
    .get_model_ctx = MLXTrainer_get_model_ctx_impl,
    .get_shapes_flat = MLXTrainer_get_shapes_flat_impl,
    .set_shapes = MLXTrainer_set_shapes_impl,
};

MLXTrainer *MLXTrainer_new(const TrainningOptions *opts, int fine_tuning,
                           const char *checkpoint_path) {
    MLXTrainer *trainer = NULL;
    if (mlx_alloc_pod((void **)&trainer, sizeof(MLXTrainer), 1) != 0)
        return NULL;
    memset(trainer, 0, sizeof(*trainer));

    /* Print training banner (parity with Python main.py) */
    printf("\n============================================================\n");
    printf("PARALLEL LAPGAN TRAINING\n");
    printf("============================================================\n");
    printf("Device: MLX (gpu, 0)\n");
    printf("Training scales: %d to %d\n", opts->start_scale, opts->stop_scale);
    printf("Parallel scales: %d\n", opts->num_parallel_scales);
    printf("Iterations per scale: %d\n", opts->num_iter);
    printf("Output path: %s\n", opts->output_path ? opts->output_path : ".");
    printf("============================================================\n\n");

    fprintf(stderr, "MLX Configuration:\n");
    
    /* Configure MLX memory management (parity with Python Trainer):
     * Set memory limit to 48GB and use GPU device by default. */
    size_t mem_limit_result = 0;
    size_t mem_limit = 48UL * 1024UL * 1024UL * 1024UL;  /* 48GB */
    mlx_set_memory_limit(&mem_limit_result, mem_limit);
    fprintf(stderr, "  Memory limit: %.1f GB\n", (double)mem_limit_result / (1024.0 * 1024.0 * 1024.0));

    /* Check device type being used - should be GPU if Metal is available */
    mlx_device default_dev = mlx_device_new();
    mlx_get_default_device(&default_dev);
    mlx_device_type dev_type;
    mlx_device_get_type(&dev_type, default_dev);
    fprintf(stderr, "  Default device: %s\n", dev_type == MLX_GPU ? "GPU" : "CPU");
    mlx_device_free(default_dev);

    /* Set MLX random seed for reproducibility (parity with Python mx.random.seed) */
    if (opts->manual_seed >= 0) {
        mlx_random_seed((uint64_t)opts->manual_seed);
        fprintf(stderr, "  Random seed: %d\n", opts->manual_seed);
    }

    /* attach default ops vtable so callers can use trainer->ops->... if
     * desired. The ops struct references the existing function-based API so
     * this is a non-invasive, backward-compatible enhancement. */
    extern MLXTrainerOps default_mlx_trainer_ops;
    trainer->ops = &default_mlx_trainer_ops;
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
        if (mlx_alloc_int_array(&trainer->wells_mask_columns,
                                opts->wells_mask_count) == 0) {
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

    MLXTrainer_init_dataset(trainer);
    MLXTrainer_init_scales(trainer);
    MLXTrainer_create_dataloader(trainer);

    printf("DataLoader num_workers: %d\n", opts->num_workers);

    MLXTrainer_create_model(trainer);
    mlx_faciesgan_set_shapes(trainer->model, trainer->scales, trainer->n_scales);

    if (trainer->scales && trainer->n_scales > 0) {
        printf("Generated facie shapes:\n");
        printf("╔══════════╦══════════╦══════════╦══════════╗\n");
        printf("║ %8s ║ %8s ║ %8s ║ %8s ║\n", "Batch", "Channels",
               "Height", "Width");
        printf("╠══════════╬══════════╬══════════╬══════════╣\n");
        for (int si = 0; si < trainer->n_scales; ++si) {
            /* stored as NHWC: [batch, height, width, channels] */
            int b = trainer->scales[si * 4 + 0];
            int h = trainer->scales[si * 4 + 1];
            int w = trainer->scales[si * 4 + 2];
            int c = trainer->scales[si * 4 + 3];
            /* print as Batch, Channels, Height, Width (NCHW display) */
            printf("║ %8d ║ %8d ║ %8d ║ %8d ║\n", b, h, w, c);
        }
        printf("╚══════════╩══════════╩══════════╩══════════╝\n");
    }

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

    /* Allocate per-scale metrics arrays for logging */
    trainer->last_g_total = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_g_adv = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_g_rec = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_g_well = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_g_div = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_d_total = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_d_real = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_d_fake = (double *)calloc((size_t)alloc_n, sizeof(double));
    trainer->last_d_gp = (double *)calloc((size_t)alloc_n, sizeof(double));

    /* Print TensorBoard status message (parity with Python) */
    if (trainer->enable_tensorboard) {
        printf("📊 TensorBoard logging enabled!\n");
    } else {
        printf("📊 TensorBoard logging disabled\n");
    }

    /* Print MLX Metal Configuration (parity with Python MLXTrainer.__init__) */
    {
        size_t active_mem = 0, peak_mem = 0;
        mlx_get_active_memory(&active_mem);
        mlx_get_peak_memory(&peak_mem);
        printf("MLX Metal Configuration:\n");
        printf("  Device: Device(gpu, 0)\n");
        printf("  Active memory: %.3f GB\n", (double)active_mem / (1024.0 * 1024.0 * 1024.0));
        printf("  Peak memory: %.3f GB\n", (double)peak_mem / (1024.0 * 1024.0 * 1024.0));
        printf("  Compilation: %s\n", opts->compile_backend ? "Enabled" : "Disabled");
    }
    printf("Using MLX backend for training.\n");

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
        mlx_create_dirs(viz_path);
        mlx_create_dirs(log_dir);
        /* TensorBoard guidance prints removed; visualizer still created when
         * available. */
        /* Create the Python-side visualizer (best-effort). Use update_interval=1
         * to match Python trainer default behavior. The pybridge manages a
         * global visualizer instance. */
        if (!pybridge_create_visualizer(trainer->n_scales, log_dir, log_dir, 1)) {
            fprintf(stderr,
                    "warning: failed to initialize Python TensorBoard visualizer\n");
        }
    } else {
        /* TensorBoard disabled message removed */
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

    /* Free per-scale metrics arrays */
    if (trainer->last_g_total) free(trainer->last_g_total);
    if (trainer->last_g_adv) free(trainer->last_g_adv);
    if (trainer->last_g_rec) free(trainer->last_g_rec);
    if (trainer->last_g_well) free(trainer->last_g_well);
    if (trainer->last_g_div) free(trainer->last_g_div);
    if (trainer->last_d_total) free(trainer->last_d_total);
    if (trainer->last_d_real) free(trainer->last_d_real);
    if (trainer->last_d_fake) free(trainer->last_d_fake);
    if (trainer->last_d_gp) free(trainer->last_d_gp);

    if (trainer->scales)
        mlx_free_int_array(&trainer->scales, &trainer->n_scales);

    if (trainer->checkpoint_path)
        free(trainer->checkpoint_path);

    if (trainer->output_path)
        free(trainer->output_path);

    if (trainer->wells_mask_columns) {
        mlx_free_int_array(&trainer->wells_mask_columns, NULL);
        trainer->wells_mask_count = 0;
    }

    /* Close Python visualizer if one was created via pybridge. */
    if (trainer->enable_tensorboard) {
        pybridge_close_visualizer();
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

    /* Note: do NOT destroy prefetcher/iterator here — the caller still
     * holds pointers into prefetched batches which would be invalidated
     * by destroying those resources. Prefetcher teardown happens at
     * trainer shutdown or when explicitly requested. */

    mlx_free_pod((void **)&trainer);
}

int MLXTrainer_get_shapes_flat_impl(MLXTrainer *t, int **out_shapes,
                                    int *out_n) {
    if (!t || !out_shapes || !out_n)
        return -1;
    *out_shapes = t->scales;
    *out_n = t->n_scales;
    return 0;
}

int MLXTrainer_set_shapes_impl(MLXTrainer *t, const int *shapes, int n_scales) {
    if (!t)
        return -1;
    if (t->scales)
        mlx_free_int_array(&t->scales, &t->n_scales);
    if (!shapes || n_scales <= 0) {
        t->scales = NULL;
        t->n_scales = 0;
        return 0;
    }
    if (mlx_alloc_int_array(&t->scales, 4 * n_scales) != 0)
        return -1;
    memcpy(t->scales, shapes, sizeof(int) * 4 * (size_t)n_scales);
    t->n_scales = n_scales;
    return 0;
}

int MLXTrainer_init_dataset_impl(MLXTrainer *trainer) {
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

int MLXTrainer_create_visualizer_impl(MLXTrainer *trainer,
                                      int update_interval) {
    if (!trainer || !trainer->enable_tensorboard)
        return 0;
    const char *base_out = trainer->output_path ? trainer->output_path : ".";
    char log_dir[PATH_BUFSZ];
    join_path(log_dir, sizeof(log_dir), base_out, "tensorboard_logs");
    return pybridge_create_visualizer(trainer->n_scales, base_out, log_dir,
                                      update_interval);
}

int MLXTrainer_update_visualizer_impl(MLXTrainer *trainer, int epoch,
                                      const char *metrics_json,
                                      int samples_processed) {
    if (!trainer || !trainer->enable_tensorboard)
        return 0;
    return pybridge_update_visualizer_from_json(epoch, metrics_json,
            samples_processed);
}

int MLXTrainer_close_visualizer_impl(MLXTrainer *trainer) {
    if (!trainer || !trainer->enable_tensorboard)
        return 0;
    return pybridge_close_visualizer();
}

int MLXTrainer_setup_optimizers_impl(MLXTrainer *trainer, const int *scales,
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
        if (mlx_alloc_int_array(&local_scales, local_n) != 0)
            return -1;
        for (int i = 0; i < local_n; ++i)
            local_scales[i] = i;
        scales = local_scales;
    }

    /* To avoid possible races / use-after-free when other threads may
     * destruct transient MLX objects (events, callbacks) while we are
     * populating per-scale pointers, allocate temporary arrays and fill
     * them before swapping into the trainer. This reduces the window of
     * concurrent writes into trainer-owned memory. */
    /* Determine required container size: ensure we have room for the
     * largest scale index present in `scales` (scales may be non-sequential).
     */
    int needed_alloc_n = trainer->n_scales;
    if (trainer->opts.num_parallel_scales > needed_alloc_n)
        needed_alloc_n = trainer->opts.num_parallel_scales;
    if (needed_alloc_n <= 0)
        needed_alloc_n = 1;
    /* If caller passed an explicit `scales` array, ensure the container is
     * large enough to hold the highest index referenced. */
    int max_scale_idx = -1;
    for (int i = 0; i < local_n; ++i)
        if (scales[i] > max_scale_idx)
            max_scale_idx = scales[i];
    if (max_scale_idx >= 0 && max_scale_idx + 1 > needed_alloc_n)
        needed_alloc_n = max_scale_idx + 1;

    MLXOptimizer **new_gen_opts = calloc((size_t)needed_alloc_n, sizeof(MLXOptimizer *));
    MLXOptimizer **new_disc_opts = calloc((size_t)needed_alloc_n, sizeof(MLXOptimizer *));
    MLXScheduler **new_gen_scheds = calloc((size_t)needed_alloc_n, sizeof(MLXScheduler *));
    MLXScheduler **new_disc_scheds = calloc((size_t)needed_alloc_n, sizeof(MLXScheduler *));
    if (!new_gen_opts || !new_disc_opts || !new_gen_scheds || !new_disc_scheds) {
        free(new_gen_opts);
        free(new_disc_opts);
        free(new_gen_scheds);
        free(new_disc_scheds);
        if (local_scales)
            mlx_free_int_array(&local_scales, &local_n);
        return -1;
    }

    for (int i = 0; i < local_n; ++i) {
        int sc = scales[i];
        /* create Adam optimizers with defaults (mirrors Python defaults) */
        new_gen_opts[sc] = mlx_adam_create(trainer->opts.lr_g, trainer->opts.beta1, 0.999f, 1e-8f);
        new_disc_opts[sc] = mlx_adam_create(trainer->opts.lr_d, trainer->opts.beta1, 0.999f, 1e-8f);
        /* schedulers: multistep with single milestone (lr_decay) */
        int milestones[1] = {trainer->opts.lr_decay};
        new_gen_scheds[sc] = mlx_scheduler_multistep_create_with_init(
                                 milestones, 1, trainer->opts.gamma, (const float *)&trainer->opts.lr_g,
                                 1);
        new_disc_scheds[sc] = mlx_scheduler_multistep_create_with_init(
                                  milestones, 1, trainer->opts.gamma, (const float *)&trainer->opts.lr_d,
                                  1);
        /* attach scheduler and optimizer so LR updates propagate */
        if (new_gen_opts[sc] && new_gen_scheds[sc]) {
            mlx_optimizer_attach_scheduler(new_gen_opts[sc], new_gen_scheds[sc]);
            mlx_scheduler_attach_optimizer(new_gen_scheds[sc], new_gen_opts[sc]);
        }
        if (new_disc_opts[sc] && new_disc_scheds[sc]) {
            mlx_optimizer_attach_scheduler(new_disc_opts[sc], new_disc_scheds[sc]);
            mlx_scheduler_attach_optimizer(new_disc_scheds[sc], new_disc_opts[sc]);
        }
    }

    /* Swap populated arrays into trainer atomically (replace any existing
     * arrays). We intentionally only free the old container buffers here; the
     * newly-created optimizer/scheduler objects are owned by the trainer and
     * will be cleaned up during `MLXTrainer_destroy`. */
    MLXOptimizer **old_gen_opts = trainer->gen_opts;
    MLXOptimizer **old_disc_opts = trainer->disc_opts;
    MLXScheduler **old_gen_scheds = trainer->gen_scheds;
    MLXScheduler **old_disc_scheds = trainer->disc_scheds;

    trainer->gen_opts = new_gen_opts;
    trainer->disc_opts = new_disc_opts;
    trainer->gen_scheds = new_gen_scheds;
    trainer->disc_scheds = new_disc_scheds;

    if (old_gen_opts)
        free(old_gen_opts);
    if (old_disc_opts)
        free(old_disc_opts);
    if (old_gen_scheds)
        free(old_gen_scheds);
    if (old_disc_scheds)
        free(old_disc_scheds);

    if (local_scales)
        mlx_free_int_array(&local_scales, &local_n);
    return 0;
}

/* Producer thread logic is now centralized in datasets/prefetcher.c.
 * Use `prefetcher_start_from_dataloader` to spawn a producer.
 */

int MLXTrainer_compute_rec_input(MLXTrainer *trainer, int scale,
                                 const int *indexes, int n_indexes,
                                 mlx_array **facies_pyramid, mlx_array **out) {
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
MLXTrainer_create_batch_iterator_impl(MLXTrainer *trainer,
                                      struct MLXDataloader *dl,
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
    mlx_stream s = mlx_default_gpu_stream_new();
    /* scales and n_scales are used below when creating prefetcher */

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
    mlx_stream prod_stream = mlx_default_gpu_stream_new();
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

int MLXTrainer_create_dataloader_impl(MLXTrainer *trainer) {
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

    /* dataloader created */

    trainer->data_loader = dl;
    return 0;
}

int MLXTrainer_generate_visualization_samples_impl(
    MLXTrainer *trainer, const int *scales, int n_scales, const int *indexes,
    int n_indexes, mlx_array **wells_pyramid, int n_wells,
    mlx_array **seismic_pyramid, int n_seismic, mlx_array ***out_generated,
    int *n_out) {
    if (!trainer || !scales || n_scales <= 0 || !out_generated || !n_out)
        return -1;
    mlx_array **out = NULL;
    if (mlx_alloc_mlx_array_ptrs(&out, n_scales) != 0)
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
            if (mlx_alloc_float_buf(&use_amps, scale + 1) != 0) {
                out[i] = NULL;
                mlx_free_mlx_array_ptrs(&noises, n_noises);
                continue;
            }
            for (int k = 0; k < scale + 1; ++k)
                use_amps[k] = 1.0f;
            use_n = scale + 1;
        }

        /* Convert pointer array to contiguous mlx_array values expected by the
         * generator forward wrapper. We use mlx_alloc_mlx_array_raw to avoid
         * creating empty arrays that we'd need to free before overwriting. */
        mlx_array *zvals = NULL;
        int zvals_count = n_noises;
        if (n_noises > 0) {
            zvals = (mlx_array *)malloc(sizeof(mlx_array) * (size_t)n_noises);
            if (!zvals) {
                mlx_free_mlx_array_ptrs(&noises, n_noises);
                if (use_amps)
                    mlx_free_float_buf(&use_amps, &use_n);
                out[i] = NULL;
                continue;
            }
            for (int j = 0; j < n_noises; ++j) {
                zvals[j] = *noises[j];
            }
            /* Ensure none of the zvals are empty; replace empties with zeros. */
            for (int j = 0; j < n_noises; ++j) {
                if (mlx_array_ndim(zvals[j]) == 0) {
                    mlx_stream _s = mlx_default_gpu_stream_new();
                    int shape0[4] = {
                        1, trainer->opts.crop_size > 0 ? trainer->opts.crop_size : 32,
                        trainer->opts.crop_size > 0 ? trainer->opts.crop_size : 32,
                        trainer->opts.num_img_channels > 0
                        ? trainer->opts.num_img_channels
                        : 1
                    };
                    mlx_array tmp = mlx_array_new();
                    if (mlx_zeros(&tmp, shape0, 4, MLX_FLOAT32, _s) == 0) {
                        mlx_array_free(zvals[j]);  /* free the empty array */
                        zvals[j] = tmp;
                    } else {
                        mlx_array_free(tmp);
                    }
                    mlx_stream_free(_s);
                }
            }
            /* Free the noises container (pointer wrappers) but NOT the arrays,
             * since ownership transferred to zvals */
            for (int j = 0; j < n_noises; ++j) {
                if (noises[j]) {
                    free(noises[j]);  /* free mlx_array* struct, not its content */
                }
            }
            free(noises);
            noises = NULL;
        }

        /* pick an initial in_noise: prefer first noise when present */
        mlx_array in_noise = mlx_array_new();
        if (zvals && zvals_count > 0)
            in_noise = zvals[0];

        mlx_array fake =
            mlx_faciesgan_generate_fake(trainer->model, zvals, zvals_count, use_amps,
                                        use_n, in_noise, scale, scale);

        /* Free zvals (which now owns the array data) */
        if (zvals) {
            for (int j = 0; j < zvals_count; ++j) {
                mlx_array_free(zvals[j]);
            }
            free(zvals);
            zvals = NULL;
        }
        if (use_amps)
            mlx_free_float_buf(&use_amps, &use_n);

        mlx_array *p = NULL;
        if (mlx_alloc_pod((void **)&p, sizeof(mlx_array), 1) != 0) {
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

int MLXTrainer_optimization_step_impl(MLXTrainer *trainer, const int *indexes,
                                      int n_indexes, mlx_array **facies_pyramid,
                                      int n_facies, mlx_array **rec_in_pyramid,
                                      int n_rec, mlx_array **wells_pyramid,
                                      int n_wells, mlx_array **masks_pyramid,
                                      int n_masks, mlx_array **seismic_pyramid,
                                      int n_seismic, const int *active_scales,
                                      int n_active_scales) {
    if (!trainer)
        return -1;

    MLXResults *res = NULL;
    /* Ensure optional per-scale pyramid pointer arrays are at least as large
     * as the maximum active scale index referenced. Some call sites (the
     * prefetcher) produce arrays sized to the number of parallel scales; the
     * active scales may reference higher absolute indices. Pad with NULLs if
     * necessary to avoid out-of-bounds reads inside model helpers. */
    int max_scale = 0;
    if (active_scales && n_active_scales > 0) {
        for (int ai = 0; ai < n_active_scales; ++ai)
            if (active_scales[ai] > max_scale)
                max_scale = active_scales[ai];
    }
    int need_n = max_scale + 1;
    mlx_array **wells_arg = wells_pyramid;
    mlx_array **seismic_arg = seismic_pyramid;
    mlx_array **masks_arg = masks_pyramid;
    mlx_array **rec_arg = rec_in_pyramid;
    mlx_array **facies_arg = facies_pyramid;

    mlx_array **tmp_wells = NULL;
    mlx_array **tmp_seismic = NULL;
    mlx_array **tmp_masks = NULL;
    mlx_array **tmp_rec = NULL;
    mlx_array **tmp_facies = NULL;

    if (need_n > 0) {
        if (wells_pyramid && n_wells < need_n) {
            if (mlx_alloc_ptr_array((void ***)&tmp_wells, need_n) == 0) {
                for (int i = 0; i < need_n; ++i)
                    tmp_wells[i] = (i < n_wells) ? wells_pyramid[i] : NULL;
                wells_arg = tmp_wells;
            }
        }
        if (seismic_pyramid && n_seismic < need_n) {
            if (mlx_alloc_ptr_array((void ***)&tmp_seismic, need_n) == 0) {
                for (int i = 0; i < need_n; ++i)
                    tmp_seismic[i] = (i < n_seismic) ? seismic_pyramid[i] : NULL;
                seismic_arg = tmp_seismic;
            }
        }
        if (masks_pyramid && n_masks < need_n) {
            if (mlx_alloc_ptr_array((void ***)&tmp_masks, need_n) == 0) {
                for (int i = 0; i < need_n; ++i)
                    tmp_masks[i] = (i < n_masks) ? masks_pyramid[i] : NULL;
                masks_arg = tmp_masks;
            }
        }
        if (facies_pyramid && n_facies < need_n) {
            if (mlx_alloc_ptr_array((void ***)&tmp_facies, need_n) == 0) {
                for (int i = 0; i < need_n; ++i)
                    tmp_facies[i] = (i < n_facies) ? facies_pyramid[i] : NULL;
                facies_arg = tmp_facies;
            }
        }
        if (rec_in_pyramid && n_rec < need_n) {
            if (mlx_alloc_ptr_array((void ***)&tmp_rec, need_n) == 0) {
                for (int i = 0; i < need_n; ++i)
                    tmp_rec[i] = (i < n_rec) ? rec_in_pyramid[i] : NULL;
                rec_arg = tmp_rec;
            }
        }
    }

    /* Defensive: sanitize `active_scales` to ensure we pass valid scale
     * indices into the collector. Upstream bugs were observed where shape
     * heights (e.g. 12,20) were passed as scale ids; attempt a best-effort
     * mapping by matching those values to trainer shapes (height or width).
     */
    int *san_active = NULL;
    if (active_scales && n_active_scales > 0) {
        if (mlx_alloc_int_array(&san_active, n_active_scales) == 0) {
            for (int ai = 0; ai < n_active_scales; ++ai) {
                int val = active_scales[ai];
                int mapped = val;
                if (trainer->n_scales > 0 && (val < 0 || val >= trainer->n_scales)) {
                    /* try to match against configured shapes (height/width) */
                    for (int tsi = 0; tsi < trainer->n_scales; ++tsi) {
                        int h = trainer->scales[tsi * 4 + 1];
                        int w = trainer->scales[tsi * 4 + 2];
                        if (val == h || val == w) {
                            mapped = tsi;
                            break;
                        }
                    }
                    /* mapped/val used for defensive mapping above */
                }
                san_active[ai] = mapped;
            }
        } else {
            san_active = NULL; /* allocation failed; fall back to original */
        }
    }

    int rc = mlx_faciesgan_collect_metrics_and_grads(
                 trainer->model, indexes, n_indexes,
                 san_active ? san_active : active_scales, n_active_scales,
                 facies_arg, rec_arg, wells_arg, masks_arg,
                 seismic_arg, trainer->opts.lambda_diversity,
                 trainer->opts.well_loss_penalty, trainer->opts.alpha,
                 trainer->opts.lambda_grad, &res);
    /* Diagnostics: dump active_scales and configured trainer shapes so we can
     * correlate any unexpected scale IDs with the configured shapes. */
    /* diagnostics handled via logging above when needed */
    /* free temporary sanitized active scales copy if we created one */
    if (san_active) {
        int _tmpn = n_active_scales;
        mlx_free_int_array(&san_active, &_tmpn);
        san_active = NULL;
    }
    /* Debug: report collection outcome */
    if (rc != 0 || !res) {

        if (res)
            mlx_results_free(res);
        if (tmp_wells)
            mlx_free_ptr_array((void ***)&tmp_wells, need_n);
        if (tmp_facies)
            mlx_free_ptr_array((void ***)&tmp_facies, need_n);
        if (tmp_seismic)
            mlx_free_ptr_array((void ***)&tmp_seismic, need_n);
        if (tmp_masks)
            mlx_free_ptr_array((void ***)&tmp_masks, need_n);
        if (tmp_rec)
            mlx_free_ptr_array((void ***)&tmp_rec, need_n);
        return -1;
    }

    /* Dump collected scale results for correlation with active_scales */
    if (rc != 0 || !res) {
        if (res)
            mlx_results_free(res);
        if (tmp_wells)
            mlx_free_ptr_array((void ***)&tmp_wells, need_n);
        if (tmp_facies)
            mlx_free_ptr_array((void ***)&tmp_facies, need_n);
        if (tmp_seismic)
            mlx_free_ptr_array((void ***)&tmp_seismic, need_n);
        if (tmp_masks)
            mlx_free_ptr_array((void ***)&tmp_masks, need_n);
        if (tmp_rec)
            mlx_free_ptr_array((void ***)&tmp_rec, need_n);
        return -1;
    }

    /* For each active scale: step schedulers (auto) then apply optimizer steps
     * using collected grads. This mirrors the Python Trainer per-scale logic. */
    int overall = 0;
    for (int i = 0; i < res->n_scales; ++i) {
        MLXScaleResults *sr = &res->scales[i];
        int sc = sr->scale;

        /* Store metrics for this scale into trainer arrays for later printing.
         * The metrics are stored in MLXScaleMetrics: fake (g_adv), well, div, rec, total.
         * We need to evaluate the mlx_array to get the scalar value. */
        if (sc >= 0 && sc < trainer->n_scales && trainer->last_g_total) {
            /* Helper macro to extract scalar value from mlx_array */
            #define EXTRACT_SCALAR(arr_ptr) \
                ((arr_ptr) && (arr_ptr)->ctx ? ({ \
                    float val = 0.0f; \
                    mlx_array_item_float32(&val, *(arr_ptr)); \
                    (double)val; \
                }) : 0.0)
            
            /* Generator metrics */
            trainer->last_g_adv[sc] = EXTRACT_SCALAR(sr->metrics.fake);
            trainer->last_g_well[sc] = EXTRACT_SCALAR(sr->metrics.well);
            trainer->last_g_div[sc] = EXTRACT_SCALAR(sr->metrics.div);
            trainer->last_g_rec[sc] = EXTRACT_SCALAR(sr->metrics.rec);
            trainer->last_g_total[sc] = EXTRACT_SCALAR(sr->metrics.total);
            
            /* Discriminator metrics - now stored separately in struct */
            trainer->last_d_real[sc] = EXTRACT_SCALAR(sr->metrics.d_real);
            trainer->last_d_fake[sc] = EXTRACT_SCALAR(sr->metrics.d_fake);
            trainer->last_d_gp[sc] = EXTRACT_SCALAR(sr->metrics.d_gp);
            trainer->last_d_total[sc] = EXTRACT_SCALAR(sr->metrics.d_total);
            
            #undef EXTRACT_SCALAR
        }

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
            if (r != 0) {
                overall = -1;
            } else {
                ;
            }
        } else if (sr->gen_n <= 0 || !sr->gen_grads) {
            ;
        } else if (!trainer->gen_opts || !trainer->gen_opts[sc]) {
            ;
        }

        /* Apply discriminator grads */
        if (sr->disc_n > 0 && sr->disc_grads && trainer->disc_opts &&
                trainer->disc_opts[sc]) {
            int r = mlx_faciesgan_apply_sgd_to_discriminator(
                        trainer->model, trainer->disc_opts[sc], sr->disc_grads, sr->disc_n);
            if (r != 0) {
                overall = -1;
            } else {
                ;
            }
        } else if (sr->disc_n <= 0 || !sr->disc_grads) {
            ;
        } else if (!trainer->disc_opts || !trainer->disc_opts[sc]) {
            ;
        }
    }

    /* Evaluate all model parameters (parity with Python mx.eval(self.model.state)).
     * This forces lazy computation graphs to materialize and releases intermediate
     * arrays, preventing memory accumulation. */
    mlx_faciesgan_eval_all_parameters(trainer->model);

    /* Evaluate optimizer state arrays (m, v momentum) to materialize computation
     * graphs and prevent memory accumulation - same as Python evaluating optimizer
     * state after each step. */
    for (int sc = 0; sc < trainer->n_scales; ++sc) {
        if (trainer->gen_opts && trainer->gen_opts[sc])
            mlx_optimizer_eval_state(trainer->gen_opts[sc]);
        if (trainer->disc_opts && trainer->disc_opts[sc])
            mlx_optimizer_eval_state(trainer->disc_opts[sc]);
    }

    /* Synchronize and clear Metal cache after each optimization step to avoid
     * accumulating resources (command buffers/encoders) and hitting the
     * Metal resource limit. */
    mlx_stream sync_s = mlx_default_gpu_stream_new();
    mlx_synchronize(sync_s);
    mlx_clear_cache();
    mlx_stream_free(sync_s);

    mlx_results_free(res);
    if (tmp_wells)
        mlx_free_ptr_array((void ***)&tmp_wells, need_n);
    if (tmp_facies)
        mlx_free_ptr_array((void ***)&tmp_facies, need_n);
    if (tmp_seismic)
        mlx_free_ptr_array((void ***)&tmp_seismic, need_n);
    if (tmp_masks)
        mlx_free_ptr_array((void ***)&tmp_masks, need_n);
    if (tmp_rec)
        mlx_free_ptr_array((void ***)&tmp_rec, need_n);
    return overall == 0 ? 0 : -1;
}

int MLXTrainer_load_model_impl(MLXTrainer *trainer, int scale,
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

int MLXTrainer_save_generated_facies_impl(MLXTrainer *trainer, int scale,
        int epoch, const char *results_path, mlx_array real_facies,
        mlx_array masks) {
    if (!trainer || !results_path)
        return -1;

    /* Use options from trainer (respects CLI args) */
    const int num_real = trainer->num_real_facies > 0
                         ? trainer->num_real_facies : 5;
    const int num_gen_per_real = trainer->num_generated_per_real > 0
                                 ? trainer->num_generated_per_real : 5;
    const int cell_size = 256; /* Cell size in grid visualization - matches Python */

    mlx_stream s = mlx_default_cpu_stream_new();  /* I/O needs CPU stream */

    /* Get batch size from real_facies */
    int real_ndim = mlx_array_ndim(real_facies);
    const int *real_shape = mlx_array_shape(real_facies);
    int batch_size = (real_ndim == 4) ? real_shape[0] : 1;

    /* Select num_real random indices from batch (like Python: indexes = randint(batch_size, (num_real_facies,))) */
    int *selected_indices = (int *)malloc(num_real * sizeof(int));
    if (!selected_indices) {
        mlx_stream_free(s);
        return -1;
    }
    for (int i = 0; i < num_real; i++) {
        selected_indices[i] = rand() % batch_size;
    }

    /* Total number of generated samples = num_real * num_gen_per_real */
    int total_gen = num_real * num_gen_per_real;

    /* Allocate array to store all fake samples (one per generation) */
    mlx_array *all_fakes = (mlx_array *)malloc(total_gen * sizeof(mlx_array));
    if (!all_fakes) {
        free(selected_indices);
        mlx_stream_free(s);
        return -1;
    }
    for (int i = 0; i < total_gen; i++)
        all_fakes[i] = mlx_array_new();

    /* Generate total_gen fake samples with different noise */
    for (int gen_idx = 0; gen_idx < total_gen; gen_idx++) {
        /* Generate new random noises for each sample */
        mlx_array **noises = NULL;
        int n_noises = 0;
        if (mlx_faciesgan_get_pyramid_noise(trainer->model, scale, NULL, 0, &noises,
                                            &n_noises, NULL, NULL, 0) != 0) {
            /* Clean up on error */
            for (int j = 0; j < gen_idx; j++)
                mlx_array_free(all_fakes[j]);
            free(all_fakes);
            free(selected_indices);
            mlx_stream_free(s);
            return -1;
        }

        /* use default amplitudes (all ones) */
        float *use_amps = NULL;
        if (mlx_alloc_float_buf(&use_amps, scale + 1) != 0) {
            mlx_free_mlx_array_ptrs(&noises, n_noises);
            for (int j = 0; j < gen_idx; j++)
                mlx_array_free(all_fakes[j]);
            free(all_fakes);
            free(selected_indices);
            mlx_stream_free(s);
            return -1;
        }
        for (int i = 0; i < scale + 1; ++i)
            use_amps[i] = 1.0f;

        /* Convert pointer array to contiguous mlx_array values.
         * Transfer ownership from noises to zvals. */
        mlx_array *zvals = NULL;
        int zvals_count = n_noises;
        if (n_noises > 0) {
            zvals = (mlx_array *)malloc(sizeof(mlx_array) * (size_t)n_noises);
            if (!zvals) {
                mlx_free_mlx_array_ptrs(&noises, n_noises);
                mlx_free_float_buf(&use_amps, NULL);
                for (int j = 0; j < gen_idx; j++)
                    mlx_array_free(all_fakes[j]);
                free(all_fakes);
                free(selected_indices);
                mlx_stream_free(s);
                return -1;
            }
            for (int j = 0; j < n_noises; ++j) {
                zvals[j] = *noises[j];
            }
            /* Free the noises container (pointer wrappers) but NOT the arrays,
             * since ownership transferred to zvals */
            for (int j = 0; j < n_noises; ++j) {
                if (noises[j]) {
                    free(noises[j]);  /* free mlx_array* struct, not its content */
                }
            }
            free(noises);
            noises = NULL;
        }

        mlx_array in_noise = mlx_array_new();
        mlx_array_t fake = mlx_faciesgan_generate_fake(
                               trainer->model, zvals, zvals_count, use_amps, scale + 1,
                               in_noise, scale, scale);

        /* IMPORTANT: Evaluate the fake array BEFORE freeing noise inputs!
         * MLX uses lazy evaluation, so the computation graph references the noise
         * arrays. If we free them before evaluation, we get zeros/garbage. */
        mlx_array_eval(fake);
        mlx_synchronize(s);

        /* Free in_noise now that it's no longer needed */
        mlx_array_free(in_noise);

        /* Now safe to free zvals (which owns the array data) */
        if (zvals) {
            for (int j = 0; j < zvals_count; ++j) {
                mlx_array_free(zvals[j]);
            }
            free(zvals);
            zvals = NULL;
        }
        mlx_free_float_buf(&use_amps, NULL);

        /* Clamp to [-1, 1] */
        mlx_array fake_clamped = mlx_array_new();
        if (mlx_clamp(&fake_clamped, fake, -1.0f, 1.0f, s) == 0) {
            mlx_array_free(fake);
            fake = fake_clamped;
        }

        /* Denormalize from [-1, 1] to [0, 1]: (x + 1) / 2 */
        mlx_array one = mlx_array_new_float(1.0f);
        mlx_array two = mlx_array_new_float(2.0f);
        mlx_array fake_plus_one = mlx_array_new();
        mlx_array fake_denorm = mlx_array_new();
        if (mlx_add(&fake_plus_one, fake, one, s) == 0) {
            if (mlx_divide(&fake_denorm, fake_plus_one, two, s) == 0) {
                mlx_array_free(fake);
                fake = fake_denorm;
            } else {
                mlx_array_free(fake_denorm);
            }
            mlx_array_free(fake_plus_one);
        }
        mlx_array_free(one);
        mlx_array_free(two);
        mlx_array_free(in_noise);

        all_fakes[gen_idx] = fake;
    }

    /* Build path to match Python: output_path/<scale>/real_x_generated_facies/ */
    char scale_results_path[PATH_MAX];
    snprintf(scale_results_path, PATH_MAX, "%s/%d/real_x_generated_facies", results_path, scale);
    mlx_create_dirs(scale_results_path);

    int rc = 0;

    /* Denormalize real_facies for both paths */
    mlx_array real_denorm = mlx_array_new();
    if (real_facies.ctx != NULL) {
        mlx_array real_plus_one = mlx_array_new();
        mlx_array one2 = mlx_array_new_float(1.0f);
        mlx_array two2 = mlx_array_new_float(2.0f);
        if (mlx_add(&real_plus_one, real_facies, one2, s) == 0) {
            mlx_divide(&real_denorm, real_plus_one, two2, s);
            mlx_array_free(real_plus_one);
        }
        mlx_array_free(one2);
        mlx_array_free(two2);
    }

    /* Check if we should use Python bridge for plotting (matches matplotlib fonts) */
    if (trainer->opts.use_pybridge_plot && real_facies.ctx != NULL) {
        /* Build 5D fake array: (num_real, num_gen_per_real, H, W, C) */
        int fake_h = 1, fake_w = 1, fake_c = 3;
        if (total_gen > 0 && all_fakes[0].ctx != NULL) {
            int fndim = mlx_array_ndim(all_fakes[0]);
            const int *fshape = mlx_array_shape(all_fakes[0]);
            if (fndim == 4) {
                fake_h = fshape[1];
                fake_w = fshape[2];
                fake_c = fshape[3];
            } else if (fndim == 3) {
                fake_h = fshape[0];
                fake_w = fshape[1];
                fake_c = fshape[2];
            }
        }

        /* Create 5D array by stacking: (num_real, num_gen_per_real, H, W, C) */
        int shape_5d[5] = {num_real, num_gen_per_real, fake_h, fake_w, fake_c};
        mlx_array fake_5d = mlx_array_new();
        if (mlx_zeros(&fake_5d, shape_5d, 5, MLX_FLOAT32, s) == 0) {
            /* Copy each fake into the 5D array */
            for (int ri = 0; ri < num_real; ri++) {
                for (int gi = 0; gi < num_gen_per_real; gi++) {
                    int idx = ri * num_gen_per_real + gi;
                    if (idx < total_gen && all_fakes[idx].ctx != NULL) {
                        /* Extract the first sample if batched (B, H, W, C) -> (H, W, C) */
                        mlx_array sample = all_fakes[idx];
                        int sndim = mlx_array_ndim(sample);
                        if (sndim == 4) {
                            /* Slice first element from batch dimension */
                            mlx_array sliced = mlx_array_new();
                            int starts[4] = {0, 0, 0, 0};
                            int stops[4] = {1, fake_h, fake_w, fake_c};
                            int strides[4] = {1, 1, 1, 1};
                            if (mlx_slice(&sliced, sample, starts, 4, stops, 4, strides, 4, s) == 0) {
                                /* Squeeze batch dimension (axis 0) */
                                mlx_array squeezed = mlx_array_new();
                                if (mlx_squeeze_axis(&squeezed, sliced, 0, s) == 0) {
                                    /* Insert into 5D array at position [ri, gi, :, :, :] */
                                    int ins_starts[5] = {ri, gi, 0, 0, 0};
                                    int ins_stops[5] = {ri + 1, gi + 1, fake_h, fake_w, fake_c};
                                    int ins_strides[5] = {1, 1, 1, 1, 1};
                                    /* Expand squeezed to 5D for assignment: (H,W,C) -> (1,1,H,W,C) */
                                    mlx_array exp1 = mlx_array_new();
                                    mlx_array exp2 = mlx_array_new();
                                    if (mlx_expand_dims(&exp1, squeezed, 0, s) == 0 &&
                                            mlx_expand_dims(&exp2, exp1, 0, s) == 0) {
                                        /* Use slice_update to place data into the 5D array */
                                        mlx_array updated = mlx_array_new();
                                        if (mlx_slice_update(&updated, fake_5d, exp2,
                                                             ins_starts, 5, ins_stops, 5, ins_strides, 5, s) == 0) {
                                            mlx_array_free(fake_5d);
                                            fake_5d = updated;
                                        } else {
                                            mlx_array_free(updated);
                                        }
                                        mlx_array_free(exp1);
                                        mlx_array_free(exp2);
                                    } else {
                                        mlx_array_free(exp1);
                                        mlx_array_free(exp2);
                                    }
                                    mlx_array_free(squeezed);
                                } else {
                                    mlx_array_free(squeezed);
                                }
                                mlx_array_free(sliced);
                            } else {
                                mlx_array_free(sliced);
                            }
                        }
                    }
                }
            }

            /* Evaluate the 5D array before saving */
            mlx_array_eval(fake_5d);
            mlx_synchronize(s);

            /* Save NPY files for pybridge - use Python naming convention */
            char fake_npy_path[PATH_MAX];
            char real_npy_path[PATH_MAX];
            char masks_npy_path[PATH_MAX];
            snprintf(fake_npy_path, PATH_MAX, "%s/scale_%d_epoch_%d_fake.npy", scale_results_path, scale, epoch);
            snprintf(real_npy_path, PATH_MAX, "%s/scale_%d_epoch_%d_real.npy", scale_results_path, scale, epoch);
            snprintf(masks_npy_path, PATH_MAX, "%s/scale_%d_epoch_%d_masks.npy", scale_results_path, scale, epoch);

            /* Save fake array */
            if (mlx_save(fake_npy_path, fake_5d) == 0) {
                /* Save real array (use selected_indices to create subset) */
                int rndim = mlx_array_ndim(real_denorm);
                const int *rshape = mlx_array_shape(real_denorm);
                int real_h = (rndim >= 2) ? rshape[1] : 1;
                int real_w = (rndim >= 3) ? rshape[2] : 1;
                int real_c = (rndim >= 4) ? rshape[3] : 3;

                /* Create subset of real facies using selected_indices */
                int sub_shape[4] = {num_real, real_h, real_w, real_c};
                mlx_array real_subset = mlx_array_new();
                if (mlx_zeros(&real_subset, sub_shape, 4, MLX_FLOAT32, s) == 0) {
                    for (int ri = 0; ri < num_real; ri++) {
                        int idx = selected_indices[ri];
                        /* Slice real_denorm[idx:idx+1, :, :, :] */
                        mlx_array sliced = mlx_array_new();
                        int starts[4] = {idx, 0, 0, 0};
                        int stops[4] = {idx + 1, real_h, real_w, real_c};
                        int strides[4] = {1, 1, 1, 1};
                        if (mlx_slice(&sliced, real_denorm, starts, 4, stops, 4, strides, 4, s) == 0) {
                            /* Insert into real_subset at position [ri, :, :, :] */
                            int ins_starts[4] = {ri, 0, 0, 0};
                            int ins_stops[4] = {ri + 1, real_h, real_w, real_c};
                            int ins_strides[4] = {1, 1, 1, 1};
                            mlx_array updated = mlx_array_new();
                            if (mlx_slice_update(&updated, real_subset, sliced,
                                                 ins_starts, 4, ins_stops, 4, ins_strides, 4, s) == 0) {
                                mlx_array_free(real_subset);
                                real_subset = updated;
                            } else {
                                mlx_array_free(updated);
                            }
                            mlx_array_free(sliced);
                        } else {
                            mlx_array_free(sliced);
                        }
                    }

                    mlx_array_eval(real_subset);
                    mlx_synchronize(s);

                    if (mlx_save(real_npy_path, real_subset) == 0) {
                        /* Save masks if present */
                        const char *masks_path_arg = NULL;
                        if (masks.ctx != NULL) {
                            /* Create subset of masks using selected_indices
                             * Masks can be 3D (B, H, W) or 4D (B, H, W, 1) */
                            int mndim = mlx_array_ndim(masks);
                            const int *mshape = mlx_array_shape(masks);

                            /* Determine mask dimensions based on actual shape */
                            if (mndim == 4) {
                                /* 4D mask: (B, H, W, C) - typically (B, H, W, 1) */
                                int mask_h = mshape[1];
                                int mask_w = mshape[2];
                                int mask_c = mshape[3];

                                int msub_shape[4] = {num_real, mask_h, mask_w, mask_c};
                                mlx_array masks_subset = mlx_array_new();
                                if (mlx_zeros(&masks_subset, msub_shape, 4, MLX_FLOAT32, s) == 0) {
                                    for (int ri = 0; ri < num_real; ri++) {
                                        int idx = selected_indices[ri];
                                        mlx_array sliced = mlx_array_new();
                                        int starts[4] = {idx, 0, 0, 0};
                                        int stops[4] = {idx + 1, mask_h, mask_w, mask_c};
                                        int strides[4] = {1, 1, 1, 1};
                                        if (mlx_slice(&sliced, masks, starts, 4, stops, 4, strides, 4, s) == 0) {
                                            int ins_starts[4] = {ri, 0, 0, 0};
                                            int ins_stops[4] = {ri + 1, mask_h, mask_w, mask_c};
                                            int ins_strides[4] = {1, 1, 1, 1};
                                            mlx_array updated = mlx_array_new();
                                            if (mlx_slice_update(&updated, masks_subset, sliced,
                                                                 ins_starts, 4, ins_stops, 4, ins_strides, 4, s) == 0) {
                                                mlx_array_free(masks_subset);
                                                masks_subset = updated;
                                            } else {
                                                mlx_array_free(updated);
                                            }
                                            mlx_array_free(sliced);
                                        } else {
                                            mlx_array_free(sliced);
                                        }
                                    }

                                    mlx_array_eval(masks_subset);
                                    mlx_synchronize(s);

                                    if (mlx_save(masks_npy_path, masks_subset) == 0) {
                                        masks_path_arg = masks_npy_path;
                                    }
                                    mlx_array_free(masks_subset);
                                } else {
                                    mlx_array_free(masks_subset);
                                }
                            } else if (mndim == 3) {
                                /* 3D mask: (B, H, W) */
                                int mask_h = mshape[1];
                                int mask_w = mshape[2];

                                int msub_shape[3] = {num_real, mask_h, mask_w};
                                mlx_array masks_subset = mlx_array_new();
                                if (mlx_zeros(&masks_subset, msub_shape, 3, MLX_FLOAT32, s) == 0) {
                                    for (int ri = 0; ri < num_real; ri++) {
                                        int idx = selected_indices[ri];
                                        /* Slice masks[idx:idx+1, :, :] */
                                        mlx_array sliced = mlx_array_new();
                                        int starts[3] = {idx, 0, 0};
                                        int stops[3] = {idx + 1, mask_h, mask_w};
                                        int strides[3] = {1, 1, 1};
                                        if (mlx_slice(&sliced, masks, starts, 3, stops, 3, strides, 3, s) == 0) {
                                            int ins_starts[3] = {ri, 0, 0};
                                            int ins_stops[3] = {ri + 1, mask_h, mask_w};
                                            int ins_strides[3] = {1, 1, 1};
                                            mlx_array updated = mlx_array_new();
                                            if (mlx_slice_update(&updated, masks_subset, sliced,
                                                                 ins_starts, 3, ins_stops, 3, ins_strides, 3, s) == 0) {
                                                mlx_array_free(masks_subset);
                                                masks_subset = updated;
                                            } else {
                                                mlx_array_free(updated);
                                            }
                                            mlx_array_free(sliced);
                                        } else {
                                            mlx_array_free(sliced);
                                        }
                                    }

                                    mlx_array_eval(masks_subset);
                                    mlx_synchronize(s);

                                    if (mlx_save(masks_npy_path, masks_subset) == 0) {
                                        masks_path_arg = masks_npy_path;
                                    }
                                    mlx_array_free(masks_subset);
                                } else {
                                    mlx_array_free(masks_subset);
                                }
                            } /* end 3D mask handling */
                        } /* end masks.ctx != NULL */

                        /* Call pybridge to generate the plot with matplotlib fonts */
                        rc = pybridge_submit_plot_generated_facies(
                                 fake_npy_path, real_npy_path, scale, epoch,
                                 scale_results_path, masks_path_arg) ? 0 : -1;

                        if (rc == 0) {
                        } else {
                            /* Fall through to C rendering below */
                        }
                    }
                    mlx_array_free(real_subset);
                }
            }
            mlx_array_free(fake_5d);
        } else {
            mlx_array_free(fake_5d);
        }

        /* If pybridge succeeded, skip C rendering */
        if (rc == 0) {
            goto cleanup;
        }
        /* Otherwise fall through to C rendering */
        rc = 0; /* Reset for C rendering attempt */
    }

    /* Save grid PNG with C rendering (bitmap font) */
    if (real_facies.ctx != NULL && real_denorm.ctx != NULL) {
        char png_path[PATH_MAX];
        snprintf(png_path, PATH_MAX, "%s/gen_%d_%d.png", scale_results_path, scale, epoch);
        rc = mlx_save_facies_grid_png_v2(png_path, all_fakes, total_gen,
                                         real_denorm, selected_indices, num_real,
                                         num_gen_per_real, cell_size, scale, epoch, masks);
        if (rc == 0) {
        } else {
        }
    } else if (real_facies.ctx == NULL) {
        /* No real facies, save first fake sample as standalone PNG */
        char png_path[PATH_MAX];
        snprintf(png_path, PATH_MAX, "%s/gen_%d_%d_fake.png", scale_results_path, scale, epoch);
        rc = mlx_save_png(png_path, all_fakes[0]);
    }

cleanup:
    /* Clean up */
    mlx_array_free(real_denorm);
    for (int i = 0; i < total_gen; i++)
        mlx_array_free(all_fakes[i]);
    free(all_fakes);
    free(selected_indices);

    mlx_stream_free(s);
    return rc;
}

void *MLXTrainer_get_model_ctx_impl(MLXTrainer *trainer) {
    if (!trainer)
        return NULL;
    return (void *)trainer->model;
}

void *MLXTrainer_create_model_impl(MLXTrainer *trainer) {
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
        mlx_base_manager_load(mgr, ckpt, 1 /*load_shapes*/,
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

    /* Populate trainer->scales with the flattened shape tuples (batch,height,width,channels)
     * returned by `dataset_generate_scales`. Many callers expect `trainer->scales`
     * to be a contiguous int array of length `4 * n_scales` containing NHWC shape
     * entries; restore that contract here to avoid treating shapes as scale IDs. */
    if (mlx_alloc_int_array(&trainer->scales, 4 * n) != 0) {
        mlx_free_pod((void **)&arr);
        return -1;
    }
    for (int si = 0; si < n; ++si) {
        /* stored as NHWC: [batch, height, width, channels] */
        trainer->scales[si * 4 + 0] = arr[si].batch;
        trainer->scales[si * 4 + 1] = arr[si].height;
        trainer->scales[si * 4 + 2] = arr[si].width;
        trainer->scales[si * 4 + 3] = arr[si].channels;
    }
    /* record number of scales for callers */
    trainer->n_scales = n;
    mlx_free_pod((void **)&arr);
    return 0;
}

int MLXTrainer_train_scales_impl(MLXTrainer *trainer, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *scales, int n_scales,
                                 int num_iter) {
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
            /* Diagnostic block intentionally empty in release build */
            mlx_compute_rec_input(si, indexes, n_indexes, facies_pyramid, &r);
            rec_pyr[si] = r;
            mlx_init_rec_noise_and_amp(
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
                mlx_free_pod((void **)&rec_pyr[si]);
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
            mlx_stream s = mlx_default_gpu_stream_new();
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

    /* Starting C-native trainer print removed */

    pybridge_create_visualizer(num_scales, ".", NULL, 1);
    pybridge_create_background_worker(2, 32);

    mlx_stream s = mlx_default_gpu_stream_new();

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
        /* per-batch stacked scales print removed */

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

    /* C-native trainer finished message removed */
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
int MLXTrainer_train_impl(MLXTrainer *trainer) {
    if (!trainer)
        return -1;

    TrainningOptions *opts = &trainer->opts;

    int n_scales = trainer->n_scales;
    if (n_scales <= 0) {
        /* Prefer to initialise scales from existing dataset/model state rather
         * than forcing synthetic shapes. Call the init helper which will
         * populate `trainer->scales`/`trainer->n_scales` when possible. If
         * this fails or leaves zero scales, abort rather than synthesizing
         * shapes implicitly. */
        if (MLXTrainer_init_scales(trainer) != 0) {
            fprintf(stderr, "failed to initialise scales from dataset/model\n");
            return -1;
        }
        n_scales = trainer->n_scales;
        if (n_scales <= 0) {
            fprintf(stderr, "no model scales available (init_scales returned 0)\n");
            return -1;
        }
    }

    /* Ensure generator scales exist for the discovered/synthesized shapes so
     * subsequent noise/forward calls have initialized modules. */
    for (int si = 0; si < n_scales; ++si) {
        mlx_faciesgan_create_generator_scale(
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
    }
    /* Always create background worker for PNG plotting */
    pybridge_create_background_worker(2, 32);

    /* Build an explicit contiguous list of scale indices (0..n_scales-1)
     * to mirror Python trainer behavior where scales are integer indices.
     * This avoids accidentally passing flattened shape tuples (NHWC) into
     * routines that expect scale IDs. `scale_idxs` is freed before
     * function exit (or on early return). */
    int *scale_idxs = NULL;
    if (mlx_alloc_int_array(&scale_idxs, n_scales) != 0) {
        fprintf(stderr, "failed to allocate scale index array\n");
        return -1;
    }
    for (int si = 0; si < n_scales; ++si)
        scale_idxs[si] = si;
    MLXTrainer_setup_optimizers(trainer, scale_idxs, n_scales);

    /* Create a prefetcher/iterator via the centralized helper so ownership
     * and producer thread logic is consistent. */
    PrefetcherIteratorHandle pit = NULL;
    int *scale_list = NULL;
    pit = MLXTrainer_create_batch_iterator(trainer, dl, scale_idxs, n_scales);
    if (!pit) {
        fprintf(stderr, "failed to create prefetcher iterator\n");
        if (scale_idxs) {
            int _sn = n_scales;
            mlx_free_int_array(&scale_idxs, &_sn);
        }
        return -1;
    }

    /* Training loop: consume prepared pyramids from prefetcher iterator */
    mlx_stream s = mlx_default_gpu_stream_new();
    prefetcher_iterator_preload(pit);
    
    /* Print training scales banner (parity with Python Trainer.train) */
    printf("\n============================================================\n");
    printf("Training scales (");
    for (int si = 0; si < n_scales; ++si) {
        printf("%d", si);
        if (si + 1 < n_scales) printf(", ");
    }
    printf(") in parallel\n");
    printf("============================================================\n\n");
    int batches = 0;
    int max_batches = opts->num_iter > 0 ? opts->num_iter : INT_MAX;
#ifdef FG_MEM_DEBUG
    size_t mem_before_first = 0;
#endif
    /* Initialize terminal progress bar if enabled */
#ifdef FG_PROGRESS
    fg_progress_init("Train", (size_t)max_batches, 0);
#endif
    const char *clear_cache_env = getenv("MLX_MEM_CLEAR_CACHE");
    /* Default to clearing cache unless explicitly disabled (0 or false) */
    bool clear_cache = !clear_cache_env || 
                       !(strcmp(clear_cache_env, "0") == 0 ||
                         strcmp(clear_cache_env, "false") == 0);
    
    /* Set a cache limit to prevent unbounded GPU memory growth.
     * This limits how much memory MLX keeps in its cache for reuse. */
    size_t cache_limit_result = 0;
    size_t cache_limit = 512 * 1024 * 1024;  /* 512 MB cache limit */
    mlx_set_cache_limit(&cache_limit_result, cache_limit);
    
    while (batches < max_batches) {
#ifdef FG_MEM_DEBUG
        /* Memory tracking: measure before batch */
        size_t mem_before_active = 0;
        size_t mem_before_cache = 0;
        size_t mem_before_peak = 0;
        size_t rss_before = 0;
        mlx_get_active_memory(&mem_before_active);
        mlx_get_cache_memory(&mem_before_cache);
        mlx_get_peak_memory(&mem_before_peak);
        size_t mem_before = mem_before_active + mem_before_cache;
        rss_before = mlx_get_process_rss_bytes();
        if (batches == 0) mem_before_first = mem_before;
        mlx_reset_peak_memory();
#endif

        PrefetchedPyramidsBatch *pb = prefetcher_iterator_next(pit);
        if (!pb)
            break; /* finished */

        int nsc = pb->n_scales;
        mlx_array **fac_pyr = pb->facies_ptrs;
        mlx_array **well_pyr = pb->wells_ptrs;
        mlx_array **seis_pyr = pb->seismic_ptrs;
        mlx_array **mask_pyr = pb->masks_ptrs;

        /* Evaluate lazy arrays in batch (matches Python mx.eval pattern) */
        mlx_global_lock(); /* protect all MLX operations */
        /* update progress bar (show current batch index) */
    #ifdef FG_PROGRESS
        fg_progress_update((size_t)batches);
    #endif
        mlx_vector_array pyr_vec = mlx_vector_array_new();
        for (int i = 0; i < nsc; ++i) {
            if (fac_pyr && fac_pyr[i])
                mlx_vector_array_append_value(pyr_vec, *fac_pyr[i]);
            if (well_pyr && well_pyr[i])
                mlx_vector_array_append_value(pyr_vec, *well_pyr[i]);
            if (seis_pyr && seis_pyr[i])
                mlx_vector_array_append_value(pyr_vec, *seis_pyr[i]);
            if (mask_pyr && mask_pyr[i])
                mlx_vector_array_append_value(pyr_vec, *mask_pyr[i]);
        }
        mlx_eval(pyr_vec);
        mlx_vector_array_free(pyr_vec);

        /* Compute masks from wells on main thread (thread-safe MLX ops).
         * masks = greater(sum(abs(wells), axis=channels, keepdims=true), 0)
         * This was previously done in the producer thread but caused crashes
         * due to MLX thread-safety issues. */
        mlx_array **computed_masks = NULL;
        int n_computed_masks = 0;
        if (well_pyr && nsc > 0 && (!mask_pyr || !mask_pyr[0] || !(*mask_pyr[0]).ctx)) {
            computed_masks = (mlx_array **)calloc((size_t)nsc, sizeof(mlx_array *));
            if (computed_masks) {
                mlx_stream mask_s = mlx_default_gpu_stream_new();
                for (int wi = 0; wi < nsc; ++wi) {
                    if (!well_pyr[wi] || !(*well_pyr[wi]).ctx) {
                        computed_masks[wi] = NULL;
                        continue;
                    }
                    mlx_array well = *well_pyr[wi];
                    int well_ndim = (int)mlx_array_ndim(well);
                    if (well_ndim <= 0) {
                        computed_masks[wi] = NULL;
                        continue;
                    }

                    /* abs(well) */
                    mlx_array abs_arr = mlx_array_new();
                    if (mlx_abs(&abs_arr, well, mask_s) != 0) {
                        mlx_array_free(abs_arr);
                        computed_masks[wi] = NULL;
                        continue;
                    }

                    /* sum over channel axis (axis=3 for BHWC format) */
                    int axis = (well_ndim >= 4) ? 3 : (well_ndim - 1);
                    mlx_array sum_arr = mlx_array_new();
                    if (mlx_sum_axis(&sum_arr, abs_arr, axis, true, mask_s) != 0) {
                        mlx_array_free(abs_arr);
                        mlx_array_free(sum_arr);
                        computed_masks[wi] = NULL;
                        continue;
                    }
                    mlx_array_free(abs_arr);

                    /* zeros like sum_arr */
                    mlx_array zero = mlx_array_new();
                    if (mlx_zeros_like(&zero, sum_arr, mask_s) != 0) {
                        mlx_array_free(sum_arr);
                        mlx_array_free(zero);
                        computed_masks[wi] = NULL;
                        continue;
                    }

                    /* mask = greater(sum_arr, zero) */
                    mlx_array *mask_ptr = (mlx_array *)malloc(sizeof(mlx_array));
                    if (!mask_ptr) {
                        mlx_array_free(sum_arr);
                        mlx_array_free(zero);
                        computed_masks[wi] = NULL;
                        continue;
                    }
                    *mask_ptr = mlx_array_new();
                    if (mlx_greater(mask_ptr, sum_arr, zero, mask_s) != 0) {
                        mlx_array_free(*mask_ptr);
                        free(mask_ptr);
                        mlx_array_free(sum_arr);
                        mlx_array_free(zero);
                        computed_masks[wi] = NULL;
                        continue;
                    }
                    mlx_array_free(sum_arr);
                    mlx_array_free(zero);

                    /* Evaluate immediately on main thread */
                    mlx_array_eval(*mask_ptr);
                    computed_masks[wi] = mask_ptr;
                    n_computed_masks++;
                }
                mlx_stream_free(mask_s);
            }
        }

        /* Use computed masks if we created them, otherwise use prefetched masks */
        mlx_array **effective_mask_pyr = (computed_masks && n_computed_masks > 0) ? computed_masks : mask_pyr;

        int batch_n = (int)opts->batch_size;
        int *indexes = NULL;
        if (mlx_alloc_int_array(&indexes, batch_n) != 0) {
            mlx_global_unlock();
            if (pb)
                prefetcher_free_pyramids(pb);
            return -1;
        }
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
                if (mlx_compute_rec_input(si, indexes, batch_n, fac_pyr, &tmp_rec) == 0 && tmp_rec) {
                    rec_pyr[si] = tmp_rec;
                } else {
                    rec_pyr[si] = NULL;
                }
                /* initialize noise amplitudes based on current real sample */
                mlx_init_rec_noise_and_amp(
                    trainer->model, si, indexes, batch_n,
                    fac_pyr && fac_pyr[si] ? fac_pyr[si] : NULL, well_pyr, seis_pyr);
            }
        }

        /* Pre-sync and clear before optimization step to prevent Metal resource
         * accumulation when many arrays are created during gradient computation */
        mlx_synchronize(s);
        mlx_clear_cache();

        int step_rc = MLXTrainer_optimization_step(
                          trainer, indexes, batch_n, fac_pyr, nsc, rec_pyr, nsc, well_pyr,
                          well_pyr ? nsc : 0, NULL, 0, seis_pyr, seis_pyr ? nsc : 0,
                          scale_idxs, act);

        /* Print metrics table at specific intervals (parity with Python handle_epoch_end) */
        print_epoch_metrics_table(
            0, 1,  /* batch_id, total_batches - C has single batch iterator */
            batches, max_batches, n_scales,
            trainer->last_g_total, trainer->last_g_adv,
            trainer->last_g_rec, trainer->last_g_well,
            trainer->last_g_div, trainer->last_d_total,
            trainer->last_d_real, trainer->last_d_fake,
            trainer->last_d_gp);

        /* Keep a copy of all scales' real facies and masks for visualization before freeing batch */
        mlx_array *real_facies_all_scales = NULL;
        mlx_array *masks_all_scales = NULL;
        if (fac_pyr && nsc > 0) {
            real_facies_all_scales = (mlx_array *)calloc(nsc, sizeof(mlx_array));
            masks_all_scales = (mlx_array *)calloc(nsc, sizeof(mlx_array));
            if (real_facies_all_scales && masks_all_scales) {
                mlx_stream copy_s = mlx_default_gpu_stream_new();
                for (int sc = 0; sc < nsc; ++sc) {
                    real_facies_all_scales[sc] = mlx_array_new();
                    masks_all_scales[sc] = mlx_array_new();
                    if (fac_pyr[sc]) {
                        mlx_copy(&real_facies_all_scales[sc], *fac_pyr[sc], copy_s);
                    }
                    /* Use effective_mask_pyr which may be computed_masks or prefetched masks */
                    if (effective_mask_pyr && effective_mask_pyr[sc]) {
                        mlx_copy(&masks_all_scales[sc], *effective_mask_pyr[sc], copy_s);
                    }
                }
                mlx_stream_free(copy_s);
            }
        }

        /* Now that we've consumed pb and created owned copies, free the
         * prefetched batch so the prefetcher can reuse its buffers. */
        if (pb)
            prefetcher_free_pyramids(pb);

        /* Free computed masks (created on main thread) */
        if (computed_masks) {
            for (int cm = 0; cm < nsc; ++cm) {
                if (computed_masks[cm]) {
                    mlx_array_free(*computed_masks[cm]);
                    free(computed_masks[cm]);
                }
            }
            free(computed_masks);
            computed_masks = NULL;
        }

        /* free rec pyramids */
        if (rec_pyr) {
            for (int si = 0; si < nsc; ++si) {
                if (rec_pyr[si]) {
                    mlx_free_mlx_array_vals(&rec_pyr[si], 1);
                }
            }
            free(rec_pyr);
        }

        /* pointer arrays were stack-allocated and reference pb entries; the
         * underlying buffers were freed by `prefetcher_free_pyramids(pb)` above,
         * so there is no heap memory to release here. */
        mlx_free_int_array(&indexes, &batch_n);

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

        /* Save policy matches Python exactly:
         * Python: if (epoch % save_interval == 0 or epoch == num_iter - 1) and (epoch != 0 or num_iter == 1)
         * - epoch is 0-based in Python (batches is also 0-based in C)
         * - Save at intervals (batches % save_interval == 0)
         * - Save on last epoch (batches == max_batches - 1)
         * - Skip first epoch (batches == 0) unless num_iter == 1 */
        int is_save_interval = (batches % opts->save_interval == 0);
        int is_last_epoch = (batches == max_batches - 1);
        int not_first_or_single = (batches != 0 || max_batches == 1);
        int should_save = (is_save_interval || is_last_epoch) && not_first_or_single;

        if (should_save) {
            const char *spath = opts->output_path ? opts->output_path : ".";
            /* Save generated facies for ALL active scales (matching Python behavior) */
            for (int sc = 0; sc < n_scales; ++sc) {
                mlx_array real_for_scale = (real_facies_all_scales && sc < nsc)
                                           ? real_facies_all_scales[sc]
                                           : mlx_array_new();
                mlx_array mask_for_scale = (masks_all_scales && sc < nsc)
                                           ? masks_all_scales[sc]
                                           : mlx_array_new();
                /* Python uses 0-based epoch in filename (gen_{scale}_{epoch}.png) */
                int save_rc = MLXTrainer_save_generated_facies(
                                  trainer, sc, batches, spath, real_for_scale, mask_for_scale);
                (void)save_rc;
            }
        }

        /* Free saved real facies and masks copies for all scales BEFORE memory measurement */
        if (real_facies_all_scales) {
            for (int sc = 0; sc < nsc; ++sc) {
                mlx_array_free(real_facies_all_scales[sc]);
            }
            free(real_facies_all_scales);
            real_facies_all_scales = NULL;
        }
        if (masks_all_scales) {
            for (int sc = 0; sc < nsc; ++sc) {
                mlx_array_free(masks_all_scales[sc]);
            }
            free(masks_all_scales);
            masks_all_scales = NULL;
        }

        /* Sync and clear cache for memory management */
        mlx_synchronize(s);
        if (clear_cache) {
            mlx_clear_cache();
        }
#ifdef FG_MEM_DEBUG
        /* Memory tracking: measure after ALL batch cleanup including array frees */
        size_t mem_after_active = 0;
        size_t mem_after_cache = 0;
        size_t mem_after_peak = 0;
        size_t rss_after = 0;
        mlx_get_active_memory(&mem_after_active);
        mlx_get_cache_memory(&mem_after_cache);
        mlx_get_peak_memory(&mem_after_peak);
        size_t mem_after = mem_after_active + mem_after_cache;
        rss_after = mlx_get_process_rss_bytes();
        /* Cast to signed to handle negative differences (memory freed) */
        double leak_mb = (double)((int64_t)mem_after - (int64_t)mem_before) / (1024.0 * 1024.0);
        double total_mb = (double)mem_after / (1024.0 * 1024.0);
        double before_mb = (double)mem_before / (1024.0 * 1024.0);
        double cumul_mb = (double)((int64_t)mem_after - (int64_t)mem_before_first) / (1024.0 * 1024.0);
        double before_active_mb = (double)mem_before_active / (1024.0 * 1024.0);
        double before_cache_mb = (double)mem_before_cache / (1024.0 * 1024.0);
        double after_active_mb = (double)mem_after_active / (1024.0 * 1024.0);
        double after_cache_mb = (double)mem_after_cache / (1024.0 * 1024.0);
        double peak_mb = (double)mem_after_peak / (1024.0 * 1024.0);
        double rss_before_mb = (double)rss_before / (1024.0 * 1024.0);
        double rss_after_mb = (double)rss_after / (1024.0 * 1024.0);
        fprintf(stderr,
            "[MEM] batch=%d: before=%.2fMB after=%.2fMB leak=%.2fMB cumulative=%.2fMB "
            "(active %.2f->%.2f MB, cache %.2f->%.2f MB, peak %.2f MB, rss %.2f->%.2f MB)\n",
            batches, before_mb, total_mb, leak_mb, cumul_mb,
            before_active_mb, after_active_mb, before_cache_mb, after_cache_mb,
            peak_mb, rss_before_mb, rss_after_mb);
#endif

        batches++;

        /* Release MLX lock before looping to get next batch */
        mlx_global_unlock();
    }

    mlx_stream_free(s);
    /* Free local dataset/dataloader only when they were created by this
     * function. If trainer owns them (created in MLXTrainer_new) they will be
     * freed by MLXTrainer_destroy below. */
    if (created_dl && dl)
        facies_dataloader_free(dl);
    /* Do not destroy the trainer here; ownership/deallocation is the caller's
     * responsibility (avoids double-free when callers also destroy). */
    pybridge_close_visualizer();
    /* Shutdown background worker and wait for pending tasks (e.g., pybridge
     * plot submissions) to complete before exiting. */
    pybridge_shutdown_background_worker(1);
    /* If a batch prefetcher/iterator were created, stop producers and
     * destroy iterator/prefetcher to avoid leaving background threads
     * running after trainer teardown. Use `prefetcher_stop` to join
     * producer threads cleanly before destroying the prefetcher. */
    if (trainer->batch_iterator) {
        prefetcher_iterator_destroy(trainer->batch_iterator);
        trainer->batch_iterator = NULL;
    }
    if (trainer->batch_prefetcher) {
        /* attempt graceful stop of background producers first */
        prefetcher_stop(trainer->batch_prefetcher);
        prefetcher_destroy(trainer->batch_prefetcher);
        trainer->batch_prefetcher = NULL;
        trainer->batch_producer_running = 0;
    }
    return 0;
}
