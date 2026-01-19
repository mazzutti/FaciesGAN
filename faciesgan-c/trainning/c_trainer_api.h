#ifndef C_TRAINER_API_H
#define C_TRAINER_API_H

#include "trainning/c_trainer.h"
#include "trainning/train_step.h"
#include "trainning/train_manager.h"
#include "options.h"

/* Lightweight C-side Trainer API that mirrors key methods from the
 * Python `MLXTrainer`. This file exposes a `CTrainer` opaque handle and
 * convenience wrappers that call into the existing C implementation
 * (train_manager, train_step, datasets, checkpointing, etc.).
 */

typedef struct CTrainer CTrainer;

/* Create a trainer from TrainningOptions. Returns NULL on failure. */
CTrainer *c_trainer_create_with_opts(const TrainningOptions *opts);

/* Destroy trainer and free resources. */
void c_trainer_destroy(CTrainer *t);

/* Run a single optimization step; a thin wrapper around train_step. */
int c_trainer_optimization_step(CTrainer *t, int *indexes, int n_indexes);

/* Setup optimizers and schedulers for provided scales. */
int c_trainer_setup_optimizers(CTrainer *t, const int *scales, int n_scales);

/* Load model weights for a scale from checkpoint dir. */
int c_trainer_load_model(CTrainer *t, int scale, const char *checkpoint_dir);

/* Save generated facies for visualization (best-effort). */
int c_trainer_save_generated_facies(CTrainer *t, int scale, int epoch, const char *results_path);

/* Expose underlying MLXFaciesGAN pointer for advanced use. */
void *c_trainer_get_model_ctx(CTrainer *t);

#endif
