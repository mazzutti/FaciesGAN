#include "logger.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct MLXLogger { FILE *f; };

MLXLogger *mlx_logger_create(const char *path) {
    if (!path) return NULL;
    MLXLogger *l = (MLXLogger *)malloc(sizeof(MLXLogger));
    if (!l) return NULL;
    l->f = fopen(path, "a");
    if (!l->f) { free(l); return NULL; }
    return l;
}

void mlx_logger_free(MLXLogger *l) { if (!l) return; if (l->f) fclose(l->f); free(l); }

void mlx_logger_log(MLXLogger *l, const char *msg) {
    if (!l || !l->f || !msg) return;
    fprintf(l->f, "%s\n", msg);
    fflush(l->f);
}
