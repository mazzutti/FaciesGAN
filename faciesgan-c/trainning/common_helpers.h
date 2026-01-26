#ifndef TRAINNING_COMMON_HELPERS_H
#define TRAINNING_COMMON_HELPERS_H

#include <mlx/c/array.h>
#include <stdio.h>
#include <stdlib.h>

/* Safely free a heap pointer and set it to NULL. */
static inline void safe_free(void **p) {
  if (!p)
    return;
  if (*p) {
    free(*p);
    *p = NULL;
  }
}

/* Safely fclose and null the FILE pointer. */
static inline void safe_fclose(FILE **f) {
  if (!f)
    return;
  if (*f) {
    fclose(*f);
    *f = NULL;
  }
}

/* Free an mlx_array value only when a caller-provided "initialized" flag
   indicates it owns valid storage. */
static inline void mlx_array_free_if_init(mlx_array *a, int initialized) {
  if (!a)
    return;
  if (initialized) {
    mlx_array_free(*a);
  }
}

#endif
