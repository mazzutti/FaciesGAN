#ifndef MEM_DEBUG_H
#define MEM_DEBUG_H

#include <mlx/c/mlx.h>
#include <stdbool.h>
#include <stddef.h>

/*
 * MLX Array Memory Leak Tracking
 * ==============================
 *
 * This module provides utilities to track mlx_array allocations and detect
 * memory leaks. Enable tracking by defining MLX_MEM_DEBUG before including
 * this header, or by setting the environment variable MLX_MEM_DEBUG=1.
 *
 * Usage:
 *   1. Replace mlx_array_new() with MLX_ARRAY_NEW()
 *   2. Replace mlx_array_free() with MLX_ARRAY_FREE()
 *   3. Call mlx_mem_print_leaks() at program exit
 *
 * The tracker records file/line info for each allocation to help identify
 * the source of leaks.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of concurrent tracked allocations */
#ifndef MLX_MEM_MAX_TRACKED
#define MLX_MEM_MAX_TRACKED 65536
#endif

/* Allocation record for tracking */
typedef struct {
    void *ctx;           /* mlx_array.ctx pointer (unique identifier) */
    const char *file;    /* Source file where allocated */
    int line;            /* Line number where allocated */
    const char *func;    /* Function name where allocated */
    const char *varname; /* Variable name (if provided) */
    size_t alloc_id;     /* Sequential allocation ID */
    int ndim;            /* Number of dimensions (for debugging) */
    int shape[4];        /* First 4 dims of shape (for debugging) */
} mlx_mem_record_t;

/* Initialize the memory tracker. Called automatically on first use. */
void mlx_mem_init(void);

/* Shutdown and free tracker resources. Optionally prints remaining leaks. */
void mlx_mem_shutdown(bool print_leaks);

/* Check if memory debugging is enabled (via env var or compile-time flag) */
bool mlx_mem_is_enabled(void);

/* Register a new mlx_array allocation */
void mlx_mem_register(mlx_array arr, const char *file, int line,
                      const char *func, const char *varname);

/* Unregister an mlx_array before freeing */
void mlx_mem_unregister(mlx_array arr, const char *file, int line,
                        const char *func);

/* Print all currently tracked (leaked) allocations */
void mlx_mem_print_leaks(void);

/* Get count of currently tracked allocations */
size_t mlx_mem_get_count(void);

/* Get total number of allocations since init */
size_t mlx_mem_get_total_allocs(void);

/* Get total number of frees since init */
size_t mlx_mem_get_total_frees(void);

/* Print memory statistics */
void mlx_mem_print_stats(void);

/* Enable/disable tracking at runtime */
void mlx_mem_set_enabled(bool enabled);

/* Set verbosity level (0=quiet, 1=leaks only, 2=all alloc/free) */
void mlx_mem_set_verbosity(int level);

/*
 * Convenience macros for automatic file/line tracking.
 * These wrap the standard MLX functions to add tracking.
 */

#if defined(MLX_MEM_DEBUG) || defined(MLX_MEM_DEBUG_RUNTIME)

/* Track a new mlx_array allocation */
#define MLX_ARRAY_NEW()                                                        \
    mlx_mem_tracked_new(__FILE__, __LINE__, __func__, NULL)

/* Track a new mlx_array allocation with variable name */
#define MLX_ARRAY_NEW_NAMED(name)                                              \
    mlx_mem_tracked_new(__FILE__, __LINE__, __func__, #name)

/* Track freeing an mlx_array */
#define MLX_ARRAY_FREE(arr)                                                    \
    mlx_mem_tracked_free(arr, __FILE__, __LINE__, __func__)

/* Register an existing array (e.g., returned from MLX function) */
#define MLX_ARRAY_TRACK(arr)                                                   \
    mlx_mem_register(arr, __FILE__, __LINE__, __func__, #arr)

/* Untrack without freeing (e.g., when transferring ownership) */
#define MLX_ARRAY_UNTRACK(arr)                                                 \
    mlx_mem_unregister(arr, __FILE__, __LINE__, __func__)

/* Tracked version of detach_and_free */
#define MLX_DETACH_AND_FREE(arr)                                               \
    mlx_mem_tracked_detach_free(arr, __FILE__, __LINE__, __func__)

/* Optional global wrapping of mlx_array_new/free when enabled and safe. */
#ifndef MLX_MEM_NO_WRAP
#define mlx_array_new() MLX_ARRAY_NEW()
#define mlx_array_free(arr) MLX_ARRAY_FREE(arr)
#define mlx_array_new_float(v)                                                 \
    mlx_mem_tracked_new_float((v), __FILE__, __LINE__, __func__)
#define mlx_array_new_int(v)                                                   \
    mlx_mem_tracked_new_int((v), __FILE__, __LINE__, __func__)
#define mlx_array_new_bool(v)                                                  \
    mlx_mem_tracked_new_bool((v), __FILE__, __LINE__, __func__)
#define mlx_array_new_data(buf, shape, ndim, dtype)                            \
    mlx_mem_tracked_new_data((buf), (shape), (ndim), (dtype), __FILE__, __LINE__, __func__)
#endif

#else /* MLX_MEM_DEBUG not defined */

/* Pass-through to standard functions when debugging is disabled */
#define MLX_ARRAY_NEW()          mlx_array_new()
#define MLX_ARRAY_NEW_NAMED(n)   mlx_array_new()
#define MLX_ARRAY_FREE(arr)      mlx_array_free(arr)
#define MLX_ARRAY_TRACK(arr)     ((void)0)
#define MLX_ARRAY_UNTRACK(arr)   ((void)0)
#define MLX_DETACH_AND_FREE(arr) detach_and_free(arr)

#endif /* MLX_MEM_DEBUG */

/* Helper functions used by macros (always available) */
mlx_array mlx_mem_tracked_new(const char *file, int line, const char *func,
                              const char *varname);

mlx_array mlx_mem_tracked_new_float(float v, const char *file, int line,
                                    const char *func);
mlx_array mlx_mem_tracked_new_int(int v, const char *file, int line,
                                  const char *func);
mlx_array mlx_mem_tracked_new_bool(bool v, const char *file, int line,
                                   const char *func);
mlx_array mlx_mem_tracked_new_data(void *buf, const int *shape, int ndim,
                                   mlx_dtype dtype, const char *file, int line,
                                   const char *func);

void mlx_mem_tracked_free(mlx_array arr, const char *file, int line,
                          const char *func);

void mlx_mem_tracked_detach_free(mlx_array arr, const char *file, int line,
                                 const char *func);

/*
 * Scope-based leak checking.
 * Call mlx_mem_scope_begin() at the start of a function or block,
 * and mlx_mem_scope_end() at the end to check for leaks in that scope.
 */
size_t mlx_mem_scope_begin(void);
void mlx_mem_scope_end(size_t start_count, const char *scope_name);

#define MLX_MEM_SCOPE_BEGIN() size_t _mlx_scope_start = mlx_mem_scope_begin()
#define MLX_MEM_SCOPE_END(name) mlx_mem_scope_end(_mlx_scope_start, name)

#ifdef __cplusplus
}
#endif

#endif /* MEM_DEBUG_H */
