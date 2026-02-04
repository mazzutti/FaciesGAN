#ifndef FACIESGAN_TRAINER_H
#define FACIESGAN_TRAINER_H

#include "options.h"

typedef struct FaciesGANTrainer FaciesGANTrainer;

FaciesGANTrainer *facies_trainer_new(TrainningOptions *topt, int gpu_device,
                                     const char *checkpoints_dir);

int facies_trainer_run(FaciesGANTrainer *trainer);
void facies_trainer_destroy(FaciesGANTrainer *trainer);

#endif
