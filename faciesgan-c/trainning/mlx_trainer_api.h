#ifndef MLX_TRAINER_API_H
#define MLX_TRAINER_API_H

#include "options.h"
#include "trainning/train_manager.h"
#include "trainning/train_step.h"

/* Forward declarations for dataset/prefetcher handles to avoid heavy includes
 * in this public header. The concrete types live in datasets/*.h. */
typedef struct facies_dataset_ facies_dataset;
typedef struct facies_dataloader_ facies_dataloader;
typedef void *PrefetcherIteratorHandle;

/* Additional MLXTrainer helpers mirroring Python Trainer methods. These are
 * thin wrappers around existing C helpers and convenience functions used by
 * the combined C trainer implementation. */

typedef struct MLXTrainer MLXTrainer;

/* Compute reconstruction input for `scale` given `facies_pyramid` and
 * `indexes`. On success returns 0 and sets `out` to a newly-allocated
 * `mlx_array*` (caller must free). */
int MLXTrainer_compute_rec_input(MLXTrainer *t, int scale, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 mlx_array **out);

/* Initialize recovery noise and set noise amplitudes for `scale`. Returns 0
 * on success. Mirrors Python `init_rec_noise_and_amp`. */
int MLXTrainer_init_rec_noise_and_amp(MLXTrainer *t, int scale,
                                      const int *indexes, int n_indexes,
                                      const mlx_array *real,
                                      mlx_array **wells_pyramid,
                                      mlx_array **seismic_pyramid);

/* Create a batch iterator (prefetcher-backed) from an existing
 * `facies_dataloader`. Returns an opaque `PrefetcherIteratorHandle` or NULL
 * on failure. Caller must call `prefetcher_iterator_destroy` when done. */
PrefetcherIteratorHandle MLXTrainer_create_batch_iterator(MLXTrainer *t,
                                                          facies_dataloader *dl,
                                                          const int *scales,
                                                          int n_scales);

/* Convenience: create a dataloader configured according to trainer options.
 * This forwards to `facies_dataloader_new_ex`. Returns 0 on success. */
int MLXTrainer_create_dataloader(MLXTrainer *t, facies_dataloader **out,
                                 facies_dataset *ds, size_t batch_size,
                                 unsigned int seed, int num_workers,
                                 int prefetch_factor, int timeout_ms);

/* Generate visualization samples for the provided `scales` and `indexes`.
 * On success returns 0 and sets `out_generated` to a newly-allocated array
 * of `mlx_array` pointers (length `n_out`); caller must free the arrays and
 * the outer pointer. */
int MLXTrainer_generate_visualization_samples(
    MLXTrainer *t, const int *scales, int n_scales, const int *indexes,
    int n_indexes, mlx_array **wells_pyramid, int n_wells,
    mlx_array **seismic_pyramid, int n_seismic, mlx_array ***out_generated,
    int *n_out);

/* Create a trainer from TrainningOptions. Returns NULL on failure. */
MLXTrainer *MLXTrainer_create_with_opts(const TrainningOptions *opts);

/* Destroy trainer and free resources. */
void MLXTrainer_destroy(MLXTrainer *t);

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
int MLXTrainer_run_full(const TrainningOptions *opts);

/* Run a single optimization step; accepts per-scale arrays mirroring the
 * Python API: facies_pyramid, rec_in_pyramid, wells_pyramid, masks_pyramid,
 * seismic_pyramid. `active_scales` is an array of scale indices to operate on.
 */
int MLXTrainer_optimization_step(MLXTrainer *t, const int *indexes,
                                 int n_indexes, mlx_array **facies_pyramid,
                                 int n_facies, mlx_array **rec_in_pyramid,
                                 int n_rec, mlx_array **wells_pyramid,
                                 int n_wells, mlx_array **masks_pyramid,
                                 int n_masks, mlx_array **seismic_pyramid,
                                 int n_seismic, const int *active_scales,
                                 int n_active_scales);

/* Setup optimizers and schedulers for provided scales. */
int MLXTrainer_setup_optimizers(MLXTrainer *t, const int *scales, int n_scales);

/* Return number of scales available in the underlying model. */
int MLXTrainer_get_n_scales(MLXTrainer *t);
/* Load model weights for a scale from checkpoint dir. */
int MLXTrainer_load_model(MLXTrainer *t, int scale, const char *checkpoint_dir);

/* Save generated facies for visualization (best-effort). */
int MLXTrainer_save_generated_facies(MLXTrainer *t, int scale, int epoch,
                                     const char *results_path);

/* Expose underlying MLXFaciesGAN pointer for advanced use. */
void *MLXTrainer_get_model_ctx(MLXTrainer *t);

/* Create/return the underlying model pointer (opaque). Same as
 * `MLXTrainer_get_model_ctx` but provided for API parity with Python.
 */
void *MLXTrainer_create_model(MLXTrainer *t);

/* Train scales wrapper: run `num_iter` iterations of optimization over the
 * provided `scales`. This is a thin wrapper that computes recovery inputs
 * and calls `MLXTrainer_optimization_step` per iteration. Returns 0 on
 * success.
 */
int MLXTrainer_train_scales(MLXTrainer *t, const int *indexes, int n_indexes,
                            mlx_array **facies_pyramid, int n_facies,
                            mlx_array **wells_pyramid, int n_wells,
                            mlx_array **masks_pyramid, int n_masks,
                            mlx_array **seismic_pyramid, int n_seismic,
                            const int *scales, int n_scales, int num_iter);

#endif
