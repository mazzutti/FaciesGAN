#ifndef MLX_C_METRICS_H
#define MLX_C_METRICS_H

#include "custom_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLXMetricAccumulator MLXMetricAccumulator;

MLXMetricAccumulator *mlx_metric_accumulator_create(void);
void mlx_metric_accumulator_free(MLXMetricAccumulator *acc);
void mlx_metric_accumulator_add(MLXMetricAccumulator *acc, mlx_array val);
mlx_array mlx_metric_accumulator_mean(MLXMetricAccumulator *acc);
void mlx_metric_accumulator_reset(MLXMetricAccumulator *acc);

#ifdef __cplusplus
}
#endif

#endif
