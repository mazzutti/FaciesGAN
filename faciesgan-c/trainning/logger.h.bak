#ifndef MLX_C_LOGGER_H
#define MLX_C_LOGGER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLXLogger MLXLogger;

MLXLogger *mlx_logger_create(const char *path);
void mlx_logger_free(MLXLogger *l);
void mlx_logger_log(MLXLogger *l, const char *msg);

#ifdef __cplusplus
}
#endif

#endif
