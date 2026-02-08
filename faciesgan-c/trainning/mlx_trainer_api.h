#ifndef MLX_TRAINER_API_H
#define MLX_TRAINER_API_H

#include "datasets/dataloader.h"
#include "datasets/mlx_dataset.h"
#include "options.h"
#include "trainning/train_manager.h"
#include "trainning/train_step.h"
#include <pthread.h>
typedef void *PrefetcherIteratorHandle;
struct MLXFaciesGAN;
struct MLXOptimizer;
struct MLXScheduler;
typedef void *PrefetcherHandle;

/* Forward declare trainer for use in ops/vtable. */
struct MLXTrainer;

/* Optional operations vtable to allow instance-method style calls. */
typedef struct MLXTrainerOps {
    int (*train)(struct MLXTrainer *trainer);
    int (*train_scales)(struct MLXTrainer *trainer, const int *indexes,
                        int n_indexes, mlx_array **facies_pyramid, int n_facies,
                        mlx_array **wells_pyramid, int n_wells,
                        mlx_array **masks_pyramid, int n_masks,
                        mlx_array **seismic_pyramid, int n_seismic,
                        const int *scales, int n_scales, int num_iter);
    int (*optimization_step)(struct MLXTrainer *trainer, const int *indexes,
                             int n_indexes, mlx_array **facies_pyramid,
                             int n_facies, mlx_array **rec_in_pyramid, int n_rec,
                             mlx_array **wells_pyramid, int n_wells,
                             mlx_array **masks_pyramid, int n_masks,
                             mlx_array **seismic_pyramid, int n_seismic,
                             const int *active_scales, int n_active_scales);
    void *(*create_model)(struct MLXTrainer *trainer);
    void (*destroy)(struct MLXTrainer *trainer);
    PrefetcherIteratorHandle (*create_batch_iterator)(struct MLXTrainer *trainer,
            struct MLXDataloader *dl,
            const int *scales,
            int n_scales);
    int (*create_dataloader)(struct MLXTrainer *trainer);
    int (*init_dataset)(struct MLXTrainer *trainer);
    int (*generate_visualization_samples)(struct MLXTrainer *trainer,
                                          const int *scales, int n_scales,
                                          const int *indexes, int n_indexes,
                                          mlx_array **wells_pyramid, int n_wells,
                                          mlx_array **seismic_pyramid,
                                          int n_seismic,
                                          mlx_array ***out_generated, int *n_out);
    int (*create_visualizer)(struct MLXTrainer *trainer, int update_interval);
    int (*update_visualizer)(struct MLXTrainer *trainer, int epoch,
                             const char *metrics_json, int samples_processed);
    int (*close_visualizer)(struct MLXTrainer *trainer);
    int (*setup_optimizers)(struct MLXTrainer *trainer, const int *scales,
                            int n_scales);
    int (*load_model)(struct MLXTrainer *trainer, int scale,
                      const char *checkpoint_dir);
    int (*save_generated_facies)(struct MLXTrainer *trainer, int scale, int epoch,
                                 const char *results_path, mlx_array real_facies,
                                 mlx_array masks,
                                 mlx_array **wells_pyramid, int n_wells,
                                 mlx_array **seismic_pyramid, int n_seismic);
    void *(*get_model_ctx)(struct MLXTrainer *trainer);
    int (*get_shapes_flat)(struct MLXTrainer *t, int **out_shapes, int *out_n);
    int (*set_shapes)(struct MLXTrainer *t, const int *shapes, int n_scales);
} MLXTrainerOps;

typedef struct MLXTrainer {
    TrainningOptions opts;
    struct MLXFaciesGAN *model;
    struct MLXOptimizer **gen_opts;
    struct MLXOptimizer **disc_opts;
    struct MLXScheduler **gen_scheds;
    struct MLXScheduler **disc_scheds;
    int n_scales;
    int *scales;
    int start_scale;
    int stop_scale;
    char *output_path;
    int num_iter;
    int save_interval;
    int num_parallel_scales;
    int batch_size;
    int num_img_channels;
    int noise_channels;
    int num_real_facies;
    int num_generated_per_real;
    int *wells_mask_columns;
    size_t wells_mask_count;
    int enable_tensorboard;
    int enable_plot_facies;
    float lr_g;
    float lr_d;
    float beta1;
    int lr_decay;
    float gamma;
    int zero_padding;
    float noise_amp;
    float min_noise_amp;
    float scale0_noise_amp;
    int fine_tuning;
    char *checkpoint_path;
    PrefetcherHandle batch_prefetcher;
    PrefetcherIteratorHandle batch_iterator;
    pthread_t batch_producer;
    int batch_producer_running;
    MLXPyramidsDataset *dataset;
    struct MLXDataloader *data_loader;
    int num_of_batchs;
    /* Last computed metrics for logging (per-scale arrays, length = n_scales) */
    double *last_g_total;
    double *last_g_adv;
    double *last_g_rec;
    double *last_g_well;
    double *last_g_div;
    double *last_d_total;
    double *last_d_real;
    double *last_d_fake;
    double *last_d_gp;
    /* Optional ops/vtable for OO-like instance methods. If NULL, callers can
     * continue to use the MLXTrainer_* functions declared in this header. */
    MLXTrainerOps *ops;
} MLXTrainer;

int MLXTrainer_compute_rec_input(MLXTrainer *trainer, int scale,
                                 const int *indexes, int n_indexes,
                                 mlx_array **facies_pyramid, mlx_array **out);

/* Initialize recovery noise and set noise amplitudes for `scale`. Returns 0
 * on success. Mirrors Python `init_rec_noise_and_amp`. */
int MLXTrainer_init_rec_noise_and_amp(MLXTrainer *trainer, int scale,
                                      const int *indexes, int n_indexes,
                                      const mlx_array *real,
                                      mlx_array **wells_pyramid,
                                      mlx_array **seismic_pyramid);

/* Create a batch iterator (prefetcher-backed) from an existing
 * `facies_dataloader`. Returns an opaque `PrefetcherIteratorHandle` or NULL
 * on failure. Caller must call `prefetcher_iterator_destroy` when done. */
PrefetcherIteratorHandle
MLXTrainer_create_batch_iterator(MLXTrainer *trainer, struct MLXDataloader *dl,
                                 const int *scales, int n_scales);

/* Convenience: create a dataloader configured according to trainer options.
 * This forwards to `facies_dataloader_new_ex`. Returns 0 on success. */
/* Create a dataloader configured from trainer options (dataset must be
 * present or will be initialised). Returns 0 on success and sets `*out`.
 */
/* Create a dataloader configured from trainer options. On success the
 * trainer->data_loader will be set (if not already) and 0 returned. */
int MLXTrainer_create_dataloader(MLXTrainer *trainer);

/* Initialize dataset by loading MLX pyramids from function cache and
 * constructing a facies_dataset stored on the trainer. Returns 0 on
 * success, non-zero on failure. */
int MLXTrainer_init_dataset(MLXTrainer *trainer);

/* Generate visualization samples for the provided `scales` and `indexes`.
 * On success returns 0 and sets `out_generated` to a newly-allocated array
 * of `mlx_array` pointers (length `n_out`); caller must free the arrays and
 * the outer pointer. */
int MLXTrainer_generate_visualization_samples(
    MLXTrainer *trainer, const int *scales, int n_scales, const int *indexes,
    int n_indexes, mlx_array **wells_pyramid, int n_wells,
    mlx_array **seismic_pyramid, int n_seismic, mlx_array ***out_generated,
    int *n_out);

/* Create a trainer from TrainningOptions. Returns NULL on failure. */
/* Create a trainer from TrainningOptions.
 * Additional explicit args `fine_tuning` and `checkpoint_path` may be
 * provided to override values in `opts`.
 */
MLXTrainer *MLXTrainer_new(const TrainningOptions *opts, int fine_tuning,
                           const char *checkpoint_path);

/* Destroy trainer and free resources. */
void MLXTrainer_destroy(MLXTrainer *trainer);

/* Run full dataset-driven trainer using TrainningOptions (existing C
 * implementation). Returns 0 on success. */
/* Low-level convenience: run a synthetic in-memory trainer (useful for
 * unit tests). Returns 0 on success. */
int MLXTrainer_run(int num_samples, int num_scales, int channels, int height,
                   int width, int batch_size);

/* Run trainer using a prepared TrainningOptions struct (may be used by
 * executables to run the full dataset-driven training). */
int MLXTrainer_run_with_opts(const TrainningOptions *opts);

/* Run full dataset-driven trainer using TrainningOptions (existing C
 * implementation). Returns 0 on success. */
int MLXTrainer_train(MLXTrainer *trainer);

/* Run a single optimization step; accepts per-scale arrays mirroring the
 * Python API: facies_pyramid, rec_in_pyramid, wells_pyramid, masks_pyramid,
 * seismic_pyramid. `active_scales` is an array of scale indices to operate on.
 */
int MLXTrainer_optimization_step(MLXTrainer *trainer, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **rec_in_pyramid,
                                 int n_rec, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *active_scales,
                                 int n_active_scales);

/* Setup optimizers and schedulers for provided scales. */
int MLXTrainer_setup_optimizers(MLXTrainer *trainer, const int *scales,
                                int n_scales);

/* Load model weights for a scale from checkpoint dir. */
int MLXTrainer_load_model(MLXTrainer *trainer, int scale,
                          const char *checkpoint_dir);

/* Save generated facies for visualization (best-effort). */
int MLXTrainer_save_generated_facies(MLXTrainer *trainer, int scale, int epoch,
                                     const char *results_path, mlx_array real_facies,
                                     mlx_array masks,
                                     mlx_array **wells_pyramid, int n_wells,
                                     mlx_array **seismic_pyramid, int n_seismic);

/* Expose underlying MLXFaciesGAN pointer for advanced use. */
void *MLXTrainer_get_model_ctx(MLXTrainer *trainer);

/* Create/return the underlying model pointer (opaque). Same as
 * `MLXTrainer_get_model_ctx` but provided for API parity with Python.
 */
void *MLXTrainer_create_model(MLXTrainer *trainer);

/* Get/set model shapes stored by the trainer. Shapes are flat arrays with 4
 * integers per scale (Batch, Channels, Height, Width). `get` returns a pointer
 * to the trainer-owned array; do not free it. */
int MLXTrainer_get_shapes_flat(MLXTrainer *t, int **out_shapes, int *out_n);
int MLXTrainer_set_shapes(MLXTrainer *t, const int *shapes, int n_scales);

/* Initialize trainer scale info from the model shapes. This queries the
 * underlying `MLXFaciesGAN` for flat shapes and stores them into
 * `trainer->scales`/`trainer->scales_n` and `trainer->n_scales`. Returns 0
 * on success (scales found), non-zero otherwise. */
int MLXTrainer_init_scales(MLXTrainer *trainer);

/* Visualizer helpers that forward to the Python bridge (when available).
 * These let C code create/update/close the TensorBoardVisualizer implemented
 * in Python via `pybridge`. */
int MLXTrainer_create_visualizer(MLXTrainer *trainer, int update_interval);
int MLXTrainer_update_visualizer(MLXTrainer *trainer, int epoch,
                                 const char *metrics_json,
                                 int samples_processed);
int MLXTrainer_close_visualizer(MLXTrainer *trainer);

/* Train scales wrapper: run `num_iter` iterations of optimization over the
 * provided `scales`. This is a thin wrapper that computes recovery inputs
 * and calls `MLXTrainer_optimization_step` per iteration. Returns 0 on
 * success.
 */
int MLXTrainer_train_scales(MLXTrainer *trainer, const int *indexes,
                            int n_indexes, mlx_array **facies_pyramid,
                            int n_facies, mlx_array **wells_pyramid,
                            int n_wells, mlx_array **masks_pyramid, int n_masks,
                            mlx_array **seismic_pyramid, int n_seismic,
                            const int *scales, int n_scales, int num_iter);

#endif
