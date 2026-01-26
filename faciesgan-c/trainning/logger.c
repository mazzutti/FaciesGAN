#include "logger.h"
#include "trainning/array_helpers.h"
#include "trainning/common_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct MLXLogger {
  FILE *f;
};

MLXLogger *mlx_logger_create(const char *path) {
  if (!path)
    return NULL;
  MLXLogger *l = NULL;
  if (mlx_alloc_pod((void **)&l, sizeof(MLXLogger), 1) != 0)
    return NULL;
  l->f = fopen(path, "a");
  if (!l->f) {
    mlx_free_pod((void **)&l);
    return NULL;
  }
  return l;
}

void mlx_logger_free(MLXLogger *l) {
  if (!l)
    return;
  safe_fclose(&l->f);
  mlx_free_pod((void **)&l);
}

void mlx_logger_log(MLXLogger *l, const char *msg) {
  if (!l || !l->f || !msg)
    return;
  fprintf(l->f, "%s\n", msg);
  fflush(l->f);
}
