#include "mlx_compat.h"
#include <stdio.h>

/* -----------------------------------------------------------------------
 * safe_mlx_conv2d  –  always inspects shapes (backward-compat / autodiff)
 * ----------------------------------------------------------------------- */
int safe_mlx_conv2d(mlx_array *res, const mlx_array input,
                    const mlx_array weight, int stride0, int stride1, int pad0,
                    int pad1, int dil0, int dil1, int groups,
                    const mlx_stream s) {
    int layout = CONV_LAYOUT_UNKNOWN;
    return cached_mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1,
                             dil0, dil1, groups, &layout, s);
}

/* -----------------------------------------------------------------------
 * cached_mlx_conv2d  –  detect layout once, skip shape check afterwards
 * ----------------------------------------------------------------------- */
int cached_mlx_conv2d(mlx_array *res, const mlx_array input,
                      const mlx_array weight, int stride0, int stride1,
                      int pad0, int pad1, int dil0, int dil1, int groups,
                      int *layout_state, const mlx_stream s) {
    /* Fast path: layout already known */
    if (layout_state && *layout_state == CONV_LAYOUT_DIRECT) {
        return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1, dil0,
                          dil1, groups, s);
    }
    if (layout_state && *layout_state == CONV_LAYOUT_TRANSPOSE) {
        int axes[4] = {3, 1, 2, 0};
        mlx_array trans = mlx_array_new();
        if (mlx_transpose_axes(&trans, weight, axes, 4, s) != 0) {
            mlx_array_free(trans);
            return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1, dil0,
                              dil1, groups, s);
        }
        int rc = mlx_conv2d(res, input, trans, stride0, stride1, pad0, pad1, dil0,
                            dil1, groups, s);
        mlx_array_free(trans);
        return rc;
    }

    /* Detection path: inspect shapes and cache the result */
    const int *in_sh = mlx_array_shape(input);
    const int *w_sh = mlx_array_shape(weight);
    if (!in_sh || !w_sh) {
        /* Shapes unavailable — fall through without caching */
        return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1, dil0,
                          dil1, groups, s);
    }
    if (w_sh[3] == in_sh[3]) {
        if (layout_state) *layout_state = CONV_LAYOUT_DIRECT;
        return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1, dil0,
                          dil1, groups, s);
    }
    if (w_sh[0] == in_sh[3]) {
        if (layout_state) *layout_state = CONV_LAYOUT_TRANSPOSE;
        int axes[4] = {3, 1, 2, 0};
        mlx_array trans = mlx_array_new();
        if (mlx_transpose_axes(&trans, weight, axes, 4, s) != 0) {
            mlx_array_free(trans);
            return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1, dil0,
                              dil1, groups, s);
        }
        int rc = mlx_conv2d(res, input, trans, stride0, stride1, pad0, pad1, dil0,
                            dil1, groups, s);
        mlx_array_free(trans);
        return rc;
    }
    fprintf(stderr,
            "[cached_mlx_conv2d] unexpected weight layout input.ch=%d "
            "weight=[%d,%d,%d,%d]\n",
            in_sh[3], w_sh[0], w_sh[1], w_sh[2], w_sh[3]);
    return mlx_conv2d(res, input, weight, stride0, stride1, pad0, pad1, dil0,
                      dil1, groups, s);
}

/* -----------------------------------------------------------------------
 * safe_mlx_conv_transpose2d  –  always inspects shapes
 * ----------------------------------------------------------------------- */
int safe_mlx_conv_transpose2d(mlx_array *res, const mlx_array input,
                              const mlx_array weight, int stride0, int stride1,
                              int pad0, int pad1, int dil0, int dil1,
                              int groups, const mlx_stream s) {
    int layout = CONV_LAYOUT_UNKNOWN;
    return cached_mlx_conv_transpose2d(res, input, weight, stride0, stride1,
                                       pad0, pad1, dil0, dil1, groups, &layout,
                                       s);
}

/* -----------------------------------------------------------------------
 * cached_mlx_conv_transpose2d  –  detect layout once, skip afterwards
 * ----------------------------------------------------------------------- */
int cached_mlx_conv_transpose2d(mlx_array *res, const mlx_array input,
                                const mlx_array weight, int stride0,
                                int stride1, int pad0, int pad1, int dil0,
                                int dil1, int groups, int *layout_state,
                                const mlx_stream s) {
    /* Fast path: layout already known */
    if (layout_state && *layout_state == CONV_LAYOUT_DIRECT) {
        return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0,
                                    pad1, dil0, dil1, 0, 0, groups, s);
    }
    if (layout_state && *layout_state == CONV_LAYOUT_TRANSPOSE) {
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

    /* Detection path */
    const int *in_sh = mlx_array_shape(input);
    const int *w_sh = mlx_array_shape(weight);
    if (!in_sh || !w_sh) {
        return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0,
                                    pad1, dil0, dil1, 0, 0, groups, s);
    }
    if (w_sh[3] == in_sh[3]) {
        if (layout_state) *layout_state = CONV_LAYOUT_DIRECT;
        return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0,
                                    pad1, dil0, dil1, 0, 0, groups, s);
    }
    if (w_sh[0] == in_sh[3]) {
        if (layout_state) *layout_state = CONV_LAYOUT_TRANSPOSE;
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
            "[cached_mlx_conv_transpose2d] unexpected weight layout input.ch=%d "
            "weight=[%d,%d,%d,%d]\n",
            in_sh[3], w_sh[0], w_sh[1], w_sh[2], w_sh[3]);
    return mlx_conv_transpose2d(res, input, weight, stride0, stride1, pad0, pad1,
                                dil0, dil1, 0, 0, groups, s);
}
