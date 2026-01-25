#include "trainer.h"
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
  FaciesGANTrainer *t = (FaciesGANTrainer *)malloc(sizeof(*t));
  if (!t)
    return NULL;
  t->topt = topt;
  t->trainer = NULL;
  t->checkpoints_dir = checkpoints_dir ? strdup(checkpoints_dir) : NULL;

  char device_str[128];
  if (topt->use_mlx)
    snprintf(device_str, sizeof(device_str), "MLX (gpu %d)", topt->gpu_device);
  else if (topt->use_cpu)
    snprintf(device_str, sizeof(device_str), "cpu");
  else
    snprintf(device_str, sizeof(device_str), "gpu:%d", topt->gpu_device);

  printf("\n============================================================\n");
  printf("PARALLEL LAPGAN TRAINING\n");
  printf("============================================================\n");
  printf("Device: %s\n", device_str);
  printf("Training scales: %d to %d\n", 0, topt->stop_scale);
  printf("Parallel scales: %d\n", topt->num_parallel_scales);
  printf("Iterations per scale: %d\n", topt->num_iter);
  printf("Output path: %s\n", topt->output_path);
  printf("============================================================\n\n");

  char logpath[PATH_BUFSZ];
  join_path(logpath, sizeof(logpath), topt->output_path, "log.txt");
  FILE *lf = fopen(logpath, "a");
  if (lf) {
    fprintf(lf, "FaciesGAN run initialized: %s\n", topt->output_path);
    fclose(lf);
  }

  t->trainer = MLXTrainer_new(topt, gpu_device, checkpoints_dir);
  return t;
}

int facies_trainer_run(FaciesGANTrainer *t) {
  if (!t || !t->trainer)
    return 1;
  return MLXTrainer_train(t->trainer);
}

void facies_trainer_destroy(FaciesGANTrainer *t) {
  if (!t)
    return;
  if (t->trainer)
    MLXTrainer_destroy(t->trainer);
  free(t->checkpoints_dir);
  free(t);
}
