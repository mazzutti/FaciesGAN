#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "autodiff.h"
#include <mlx/c/mlx.h>

/* forward */
static void run_wgan_gp_test(void);
static void run_wgan_gp_trials(int trials);

/* Numerical gradient check for conv2d weight using ag_conv2d */

static float get_scalar_from_array(const mlx_array a)
{
    mlx_array_eval(a);
    const float *p = mlx_array_data_float32(a);
    return p ? p[0] : 0.0f;
}

int main(void)
{
    mlx_stream s = mlx_default_cpu_stream_new();
    /* small test shapes */
    int N = 1, H = 5, W = 5, C_in = 1;
    int C_out = 1, KH = 3, KW = 3;
    int ish[4] = {N, H, W, C_in};
    int wsh[4] = {C_out, KH, KW, C_in};
    /* fill input */
    float *ibuf = calloc(N * H * W * C_in, sizeof(float));
    for (int i = 0; i < N * H * W * C_in; ++i)
        ibuf[i] = (float)(i % 7 - 3);
    mlx_array in = mlx_array_new_data(ibuf, ish, 4, MLX_FLOAT32);
    free(ibuf);
    /* weight init */
    float *wbuf = calloc(C_out * KH * KW * C_in, sizeof(float));
    for (int i = 0; i < C_out * KH * KW * C_in; ++i)
        wbuf[i] = 0.1f * (i + 1);
    mlx_array w = mlx_array_new_data(wbuf, wsh, 4, MLX_FLOAT32);
    free(wbuf);

    /* AG wrappers */
    AGValue *ain = ag_value_from_array(&in, 0);
    AGValue *aw = ag_value_from_array(&w, 1);
    AGValue *out = ag_conv2d(ain, aw, 1, 1, 1, 1, 1, 1, 1);
    /* reduce to scalar loss = sum(out)
       repeatedly sum axis 0 until scalar */
    AGValue *loss = out;
    while (1)
    {
        const mlx_array *a = ag_value_array(loss);
        if (!a)
            break;
        int ndim = (int)mlx_array_ndim(*a);
        if (ndim == 0)
            break;
        loss = ag_sum_axis(loss, 0, 0);
    }

    /* backward */
    ag_backward(loss);
    /* Debug: print shapes and a few values */
    const mlx_array *out_arr = ag_value_array(out);
    const int *in_shape_dbg = mlx_array_shape(in);
    const int *out_shape_dbg = mlx_array_shape(*out_arr);
    const int *w_shape_dbg = mlx_array_shape(w);
    printf("in shape: %d %d %d %d\n", in_shape_dbg[0], in_shape_dbg[1], in_shape_dbg[2], in_shape_dbg[3]);
    printf("out shape: %d %d %d %d\n", out_shape_dbg[0], out_shape_dbg[1], out_shape_dbg[2], out_shape_dbg[3]);
    printf("w shape: %d %d %d %d\n", w_shape_dbg[0], w_shape_dbg[1], w_shape_dbg[2], w_shape_dbg[3]);
    mlx_array_eval(*out_arr);
    const float *out_vals_dbg = mlx_array_data_float32(*out_arr);
    printf("out[0..5]: ");
    for (int i = 0; i < 5; i++)
        printf("%.3f ", out_vals_dbg[i]);
    printf("\n");

    mlx_array *g = ag_value_get_grad(aw);
    if (!g)
    {
        printf("No gradient computed for weight\n");
        return 2;
    }
    mlx_array_eval(*g);
    const float *gdata = mlx_array_data_float32(*g);

    /* Compute reference weight gradient using MLX patch-unfold + matmul approach */
    mlx_array in_padded = mlx_array_new();
    int axes[2] = {1, 2};
    int low_pad[2] = {1, 1};
    int high_pad[2] = {1, 1};
    mlx_array pad_val = mlx_array_new_float(0.0f);
    mlx_pad(&in_padded, in, axes, 2, low_pad, 2, high_pad, 2, pad_val, "constant", s);
    /* Build patches shape: (N, H_out, W_out, KH, KW, C_in) */
    int H_out = out_shape_dbg[1];
    int W_out = out_shape_dbg[2];
    int patches_shape[6] = {N, H_out, W_out, KH, KW, C_in};
    /* Compute in_padded contiguous strides */
    const int *inpad_shape = mlx_array_shape(in_padded);
    int64_t inpad_strides[4];
    inpad_strides[3] = 1;
    for (int i = 2; i >= 0; --i)
        inpad_strides[i] = inpad_strides[i + 1] * inpad_shape[i + 1];
    /* kernel_strides = {stride0, stride1} */
    int kernel_strides[2] = {1, 1};
    int64_t patches_strides[6];
    patches_strides[0] = inpad_strides[0];
    patches_strides[1] = inpad_strides[1] * kernel_strides[0];
    patches_strides[2] = inpad_strides[2] * kernel_strides[1];
    patches_strides[3] = inpad_strides[1];
    patches_strides[4] = inpad_strides[2];
    patches_strides[5] = inpad_strides[3];
    mlx_array in_patches = mlx_array_new();
    mlx_as_strided(&in_patches, in_padded, patches_shape, 6, patches_strides, 6, 0, s);
    /* reshape cotan (out.grad) to (-1, O) where O=C_out */
    mlx_array *cotan = ag_value_get_grad(out);
    int O = C_out;
    int cotan_shape[2] = {-1, O};
    mlx_array cotan_mat = mlx_array_new();
    mlx_reshape(&cotan_mat, *cotan, cotan_shape, 2, s);
    /* reshape in_patches to (cotan_mat.shape(0), -1) */
    int inpatch_shape0 = (int)((size_t)(cotan_mat.ctx ? 0 : 0)); /* placeholder, reshape with -1 works */
    int resh_shape[2] = {-1, KH * KW * C_in};
    mlx_array in_patches_mat = mlx_array_new();
    mlx_reshape(&in_patches_mat, in_patches, resh_shape, 2, s);
    /* grad = transpose(cotan_mat) @ in_patches_mat */
    mlx_array cotan_t = mlx_array_new();
    mlx_transpose(&cotan_t, cotan_mat, s);
    mlx_array grad_mat = mlx_array_new();
    mlx_matmul(&grad_mat, cotan_t, in_patches_mat, s);
    /* reshape grad_mat to weight shape */
    int wshape_ref[4] = {C_out, KH, KW, C_in};
    mlx_array grad_ref = mlx_array_new();
    mlx_reshape(&grad_ref, grad_mat, wshape_ref, 4, s);
    mlx_array_eval(grad_ref);
    const float *ref_data = mlx_array_data_float32(grad_ref);

    /* numerical gradient for a few indices */
    float eps = 1e-3f;
    int total = C_out * KH * KW * C_in;
    for (int idx = 0; idx < total; ++idx)
    {
        /* perturb weight copy using a host-backed buffer to ensure visibility */
        const float *w_orig = mlx_array_data_float32(w);
        size_t wcount = (size_t)C_out * KH * KW * C_in;
        float *wbuf_pos = malloc(sizeof(float) * wcount);
        if (!wbuf_pos)
            continue;
        for (size_t ii = 0; ii < wcount; ++ii)
            wbuf_pos[ii] = w_orig[ii];
        wbuf_pos[idx] += eps;
        mlx_array wpos = mlx_array_new_data(wbuf_pos, wsh, 4, MLX_FLOAT32);
        free(wbuf_pos);
        mlx_array outp = mlx_array_new();
        mlx_conv2d(&outp, in, wpos, 1, 1, 1, 1, 1, 1, 1, s);
        mlx_array_eval(outp);
        const float *pdata = mlx_array_data_float32(outp);
        size_t pcount = mlx_array_size(outp);
        double lpsum = 0.0;
        for (size_t ii = 0; ii < pcount; ++ii)
            lpsum += pdata[ii];
        float lp = (float)lpsum;
        mlx_array_free(outp);
        mlx_array_free(wpos);

        float *wbuf_neg = malloc(sizeof(float) * wcount);
        if (!wbuf_neg)
            continue;
        for (size_t ii = 0; ii < wcount; ++ii)
            wbuf_neg[ii] = w_orig[ii];
        wbuf_neg[idx] -= eps;
        mlx_array wneg = mlx_array_new_data(wbuf_neg, wsh, 4, MLX_FLOAT32);
        free(wbuf_neg);
        mlx_array outn = mlx_array_new();
        mlx_conv2d(&outn, in, wneg, 1, 1, 1, 1, 1, 1, 1, s);
        mlx_array_eval(outn);
        const float *ndata = mlx_array_data_float32(outn);
        size_t ncount = mlx_array_size(outn);
        double lnsum = 0.0;
        for (size_t ii = 0; ii < ncount; ++ii)
            lnsum += ndata[ii];
        float ln = (float)lnsum;
        mlx_array_free(outn);
        mlx_array_free(wneg);

        float num = (lp - ln) / (2.0f * eps);
        float autod = gdata[idx];
        float refv = ref_data[idx];
        float rel = fabsf(autod - num) / (fmaxf(fabsf(num), 1e-6f));
        float rel_ref = fabsf(refv - num) / (fmaxf(fabsf(num), 1e-6f));
        printf("idx %d: autod=%.6f ref=%.6f num=%.6f rel_autod=%.6f rel_ref=%.6f\n", idx, autod, refv, num, rel, rel_ref);
        /* avoid freeing wpos/wneg because they may alias underlying buffers
            managed by `w` and lead to double-free in this simple harness */
    }

    /* Note: skip ag_reset_tape() here to avoid freeing shared internal temporaries
       which can cause double-free in this lightweight test harness. In longer
       running code, use ag_reset_tape() carefully. */
    /* skip freeing in/w here to avoid double-free during AD debugging */
    /* run GP comparison test (multi-trial) */
    run_wgan_gp_trials(8);

    mlx_stream_free(s);
    return 0;
}

/* WGAN-GP comparison: AG create-graph vs finite-difference numeric gradient */
static void run_wgan_gp_test_once(mlx_stream s, int N, int H, int W, int C_in, int C_out, int KH, int KW, float *out_gp_ag, double *out_gp_num)
{
    int ish[4] = {N, H, W, C_in};
    int wsh[4] = {C_out, KH, KW, C_in};
    size_t xcount = (size_t)N * H * W * C_in;

    float *xb = malloc(sizeof(float) * xcount);
    for (size_t i = 0; i < xcount; ++i)
        xb[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    mlx_array x = mlx_array_new_data(xb, ish, 4, MLX_FLOAT32);
    free(xb);

    size_t wcount = (size_t)C_out * KH * KW * C_in;
    float *wb = malloc(sizeof(float) * wcount);
    for (size_t i = 0; i < wcount; ++i)
        wb[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    mlx_array w = mlx_array_new_data(wb, wsh, 4, MLX_FLOAT32);
    free(wb);

    AGValue *x_ag = ag_value_from_array(&x, 0);
    AGValue *w_ag = ag_value_from_array(&w, 0);
    AGValue *d = ag_conv2d(x_ag, w_ag, 1, 1, 1, 1, 1, 1, 1);
    AGValue *sval = d;
    while (1)
    {
        mlx_array *a = ag_value_array(sval);
        if (!a)
            break;
        if (mlx_array_ndim(*a) == 0)
            break;
        sval = ag_sum_axis(sval, 0, 0);
    }

    if (ag_backward_create_graph(sval) != 0)
    {
        *out_gp_ag = NAN;
        *out_gp_num = NAN;
        return;
    }
    AGValue *gx = ag_value_get_grad_ag(x_ag);
    if (!gx)
    {
        *out_gp_ag = NAN;
        *out_gp_num = NAN;
        return;
    }

    AGValue *g2 = ag_square(gx);
    AGValue *reduced = g2;
    for (int ax = 3; ax >= 1; --ax)
        reduced = ag_sum_axis(reduced, ax, 1);
    AGValue *sum_batch = ag_sum_axis(reduced, 0, 0);
    AGValue *norm = ag_sqrt(sum_batch);
    AGValue *one = ag_scalar_float(1.0f);
    AGValue *dif = ag_sub(norm, one);
    AGValue *gp_ag = ag_square(dif);
    mlx_array *gp_arr = ag_value_array(gp_ag);
    float gp_ag_val = gp_arr ? (get_scalar_from_array(*gp_arr)) : NAN;

    /* numeric finite-diff for norm of grad */
    float eps = 1e-3f;
    float *gnum = malloc(sizeof(float) * xcount);
    for (size_t idx = 0; idx < xcount; ++idx)
    {
        float *xbp = malloc(sizeof(float) * xcount);
        const float *xdata = mlx_array_data_float32(x);
        for (size_t j = 0; j < xcount; ++j)
            xbp[j] = xdata[j];
        xbp[idx] += eps;
        mlx_array xpos = mlx_array_new_data(xbp, ish, 4, MLX_FLOAT32);
        free(xbp);
        mlx_array outp = mlx_array_new();
        mlx_conv2d(&outp, xpos, w, 1, 1, 1, 1, 1, 1, 1, s);
        mlx_array_eval(outp);
        const float *pdata = mlx_array_data_float32(outp);
        size_t pcount = mlx_array_size(outp);
        double psum = 0.0;
        for (size_t ii = 0; ii < pcount; ++ii)
            psum += pdata[ii];
        mlx_array_free(outp);

        float *xbn = malloc(sizeof(float) * xcount);
        for (size_t j = 0; j < xcount; ++j)
            xbn[j] = mlx_array_data_float32(x)[j];
        xbn[idx] -= eps;
        mlx_array xneg = mlx_array_new_data(xbn, ish, 4, MLX_FLOAT32);
        free(xbn);
        mlx_array outn = mlx_array_new();
        mlx_conv2d(&outn, xneg, w, 1, 1, 1, 1, 1, 1, 1, s);
        mlx_array_eval(outn);
        const float *ndata = mlx_array_data_float32(outn);
        size_t ncount = mlx_array_size(outn);
        double nsum = 0.0;
        for (size_t ii = 0; ii < ncount; ++ii)
            nsum += ndata[ii];
        mlx_array_free(outn);

        gnum[idx] = (float)((psum - nsum) / (2.0 * eps));
    }
    double sumsq = 0.0;
    for (size_t i = 0; i < xcount; ++i)
        sumsq += (double)gnum[i] * gnum[i];
    double norm_num = sqrt(sumsq);
    double gp_num = (norm_num - 1.0) * (norm_num - 1.0);
    *out_gp_ag = gp_ag_val;
    *out_gp_num = gp_num;
    free(gnum);
}

static void run_wgan_gp_trials(int trials)
{
    printf("\n--- WGAN-GP multi-trial compare (%d trials) ---\n", trials);
    fflush(stdout);
    mlx_stream s = mlx_default_cpu_stream_new();
    int N = 1, H = 4, W = 4, C_in = 1;
    int C_out = 1, KH = 3, KW = 3;
    double sum_rel = 0.0;
    double max_rel = 0.0;
    int good = 0;
    for (int t = 0; t < trials; ++t)
    {
        float gp_ag = NAN;
        double gp_num = NAN;
        run_wgan_gp_test_once(s, N, H, W, C_in, C_out, KH, KW, &gp_ag, &gp_num);
        if (!isnan(gp_ag) && !isnan(gp_num))
        {
            double rel = fabs(gp_ag - gp_num) / fmax(fabs(gp_num), 1e-6);
            printf("trial %d: GP AG=%.9f GP NUM=%.9f rel=%.6g\n", t, gp_ag, gp_num, rel);
            sum_rel += rel;
            if (rel > max_rel)
                max_rel = rel;
            good++;
        }
        else
        {
            printf("trial %d: failed (nan)\n", t);
        }
        fflush(stdout);
    }
    if (good > 0)
        printf("summary: trials=%d good=%d mean_rel=%g max_rel=%g\n", trials, good, sum_rel / good, max_rel);
    mlx_stream_free(s);
}
