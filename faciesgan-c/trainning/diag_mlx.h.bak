/* Diagnostics wrapper for mlx_array_free to log stack traces and array info.
 * This is project-local and does not modify third-party code.
 */
#ifndef FACIESGAN_DIAG_MLX_H
#define FACIESGAN_DIAG_MLX_H

#include <execinfo.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"

typedef int (*diag_mlx_binary_op)(mlx_array *out, const mlx_array a,
                                  const mlx_array b, const mlx_stream s);

static inline int diag_mlx_align_binary(mlx_array *out, mlx_array a, mlx_array b,
                                        mlx_stream s, diag_mlx_binary_op op) {
    return op(out, a, b, s);
}

static inline void diag_mlx_array_free_wrapper(mlx_array arr, const char *file, int line) {
    if (!arr.ctx) {
        return;
    }
    /* Call the real C API free (macro defined after this function). */
    mlx_array_free(arr);
}

/* Override mlx_array_free in files that include trainning/train_utils.h */
#define mlx_array_free(a) diag_mlx_array_free_wrapper(a, __FILE__, __LINE__)

/* Binary op wrappers disabled to avoid interfering with MLX behavior. */

#endif
