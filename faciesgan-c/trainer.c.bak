#include "trainer.h"
#include "trainning/array_helpers.h"
#include "trainning/mlx_trainer_api.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct FaciesGANTrainer {
    TrainningOptions *topt;
    MLXTrainer *trainer;
    char *checkpoints_dir;
};

FaciesGANTrainer *facies_trainer_new(TrainningOptions *topt, int gpu_device,
                                     const char *checkpoints_dir) {
    if (!topt)
        return NULL;
    FaciesGANTrainer *trainer = NULL;
    if (mlx_alloc_pod((void **)&trainer, sizeof(*trainer), 1) != 0)
        return NULL;
    trainer->topt = topt;
    trainer->trainer = NULL;
    trainer->checkpoints_dir = checkpoints_dir ? strdup(checkpoints_dir) : NULL;

    char device_str[128];
    if (topt->use_mlx)
        snprintf(device_str, sizeof(device_str), "MLX (gpu %d)", topt->gpu_device);
    else if (topt->use_cpu)
        snprintf(device_str, sizeof(device_str), "cpu");
    else
        snprintf(device_str, sizeof(device_str), "gpu:%d", topt->gpu_device);

    /* banner and verbose training info removed to reduce console noise */

    char logpath[PATH_BUFSZ];
    join_path(logpath, sizeof(logpath), topt->output_path, "log.txt");
    FILE *lf = fopen(logpath, "a");
    if (lf) {
        fprintf(lf, "FaciesGAN run initialized: %s\n", topt->output_path);
        fclose(lf);
    }

    trainer->trainer = MLXTrainer_new(topt, gpu_device, checkpoints_dir);
    return trainer;
}

int facies_trainer_run(FaciesGANTrainer *trainer) {
    if (!trainer || !trainer->trainer)
        return 1;
    return MLXTrainer_train(trainer->trainer);
}

void facies_trainer_destroy(FaciesGANTrainer *trainer) {
    if (!trainer)
        return;
    if (trainer->trainer)
        MLXTrainer_destroy(trainer->trainer);
    free(trainer->checkpoints_dir);
    mlx_free_pod((void **)&trainer);
}
