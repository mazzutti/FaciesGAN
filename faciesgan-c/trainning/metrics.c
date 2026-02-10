#include "metrics.h"
#include "trainning/array_helpers.h"
#include "trainning/common_helpers.h"
#include <stdlib.h>

struct MLXMetricAccumulator {
  mlx_array sum;
  size_t count;
  int initialized;
};

MLXMetricAccumulator *mlx_metric_accumulator_create(void) {
  MLXMetricAccumulator *a = NULL;
  if (mlx_alloc_pod((void **)&a, sizeof(MLXMetricAccumulator), 1) != 0)
    return NULL;
  a->sum = mlx_array_new();
  a->count = 0;
  a->initialized = 0;
  return a;
}

void mlx_metric_accumulator_free(MLXMetricAccumulator *acc) {
  if (!acc)
    return;
  mlx_array_free_if_init(&acc->sum, acc->initialized);
  mlx_free_pod((void **)&acc);
}

void mlx_metric_accumulator_add(MLXMetricAccumulator *acc, mlx_array val) {
  if (!acc)
    return;
  mlx_stream s = mlx_gpu_stream();
  if (!acc->initialized) {
    mlx_array_set(&acc->sum, val);
    acc->initialized = 1;
    acc->count = 1;
    return;
  }
  mlx_array tmp = mlx_array_new();
  if (mlx_add(&tmp, acc->sum, val, s) == 0) {
    mlx_array_set(&acc->sum, tmp);
    acc->count += 1;
  }
  mlx_array_free(tmp);
}

mlx_array mlx_metric_accumulator_mean(MLXMetricAccumulator *acc) {
  mlx_array out = mlx_array_new();
  if (!acc || acc->count == 0)
    return out;
  mlx_stream s = mlx_gpu_stream();
  mlx_array denom = mlx_array_new_float((float)acc->count);
  if (mlx_divide(&out, acc->sum, denom, s) != 0) {
    mlx_array_free(out);
    out = mlx_array_new();
  }
  mlx_array_free(denom);
  return out;
}

void mlx_metric_accumulator_reset(MLXMetricAccumulator *acc) {
  if (!acc)
    return;
  if (acc->initialized) {
    mlx_array_free(acc->sum);
    acc->sum = mlx_array_new();
    acc->initialized = 0;
  }
  acc->count = 0;
}
