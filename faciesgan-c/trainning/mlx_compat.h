#ifndef MLX_COMPAT_H
#define MLX_COMPAT_H

#include <mlx/c/mlx.h>
#include <stdio.h>

static inline int safe_mlx_conv2d(
        mlx_array* res,
        const mlx_array input,
        const mlx_array weight,
        int stride0,
        int stride1,
        int pad0,
        int pad1,
        int dil0,
        int dil1,
        int groups,
        const mlx_stream s) {
    const int *in_sh = mlx_array_shape(input);
    const int *w_sh = mlx_array_shape(weight);
    if (!in_sh || !w_sh) {
        return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1,
                                            dil0, dil1, groups, s);
    }
    if (w_sh[3] == in_sh[3]) {
        return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1,
                                            dil0, dil1, groups, s);
    }
    if (w_sh[0] == in_sh[3]) {
        int axes[4] = {3, 1, 2, 0};
        mlx_array trans = mlx_array_new();
        if (mlx_transpose_axes(&trans, weight, axes, 4, s) != 0) {
            mlx_array_free(trans);
            return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1,
                                                dil0, dil1, groups, s);
        }
        int rc = mlx_conv2d(res, input, trans, stride0, stride1, pad0, pad1,
                                                dil0, dil1, groups, s);
        mlx_array_free(trans);
        return rc;
    }
    fprintf(stderr,
                    "[safe_mlx_conv2d] unexpected weight layout input.ch=%d weight=[%d,%d,%d,%d]\n",
                    in_sh[3], w_sh[0], w_sh[1], w_sh[2], w_sh[3]);
    return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1,
                                        dil0, dil1, groups, s);
}

static inline int safe_mlx_conv_transpose2d(
        mlx_array* res,
        const mlx_array input,
        const mlx_array weight,
        int stride0,
        int stride1,
        int pad0,
        int pad1,
        int dil0,
        int dil1,
        int groups,
        const mlx_stream s) {
    const int *in_sh = mlx_array_shape(input);
    const int *w_sh = mlx_array_shape(weight);
    if (!in_sh || !w_sh) {
        return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0,
                        pad1, dil0, dil1, 0, 0, groups, s);
    }
    if (w_sh[3] == in_sh[3]) {
        return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0,
                        pad1, dil0, dil1, 0, 0, groups, s);
    }
    if (w_sh[0] == in_sh[3]) {
        int axes[4] = {3, 1, 2, 0};
        mlx_array trans = mlx_array_new();
        if (mlx_transpose_axes(&trans, weight, axes, 4, s) != 0) {
            mlx_array_free(trans);
            return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0,
                                        pad1, dil0, dil1, 0, 0, groups, s);
        }
        int rc = mlx_conv_transpose2d(res, input, trans, stride0, stride1, pad0,
                          pad1, dil0, dil1, 0, 0, groups, s);
        mlx_array_free(trans);
        return rc;
    }
    fprintf(stderr,
                    "[safe_mlx_conv_transpose2d] unexpected weight layout input.ch=%d weight=[%d,%d,%d,%d]\n",
                    in_sh[3], w_sh[0], w_sh[1], w_sh[2], w_sh[3]);
    return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0,
                                pad1, dil0, dil1, 0, 0, groups, s);
}

#endif
