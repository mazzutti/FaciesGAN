#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "autodiff.h"
#include <mlx/c/mlx.h>

/* Compare numerical GP (finite differences) vs AG-based create-graph GP
   using a tiny conv discriminator D(x) = sum(conv2d(x, w)). */

static float eval_scalar_from_array(const mlx_array a)
{
    mlx_array_eval(a);
    const float *p = mlx_array_data_float32(a);
    return p ? p[0] : 0.0f;
}

int main(void)
{
    printf("[test] main start\n");
    fflush(stdout);
    mlx_stream s = mlx_default_cpu_stream_new();
    int N = 1, H = 4, W = 4, C = 1;
    int KH = 3, KW = 3, Cout = 1;
    int xshape[4] = {N, H, W, C};
    int wshape[4] = {Cout, KH, KW, C};

    /* create input x and weight w */
    size_t xcount = (size_t)N * H * W * C;
    float *xbuf = malloc(sizeof(float) * xcount);
    for (size_t i = 0; i < xcount; ++i)
        xbuf[i] = 0.01f * (float)(i + 1);
    mlx_array x = mlx_array_new_data(xbuf, xshape, 4, MLX_FLOAT32);
    free(xbuf);

    size_t wcount = (size_t)Cout * KH * KW * C;
    float *wbuf = malloc(sizeof(float) * wcount);
    for (size_t i = 0; i < wcount; ++i)
        wbuf[i] = 0.1f * (float)(i + 1);
    mlx_array w = mlx_array_new_data(wbuf, wshape, 4, MLX_FLOAT32);
    free(wbuf);

    /* AG wrappers */
    AGValue *x_ag = ag_value_from_array(&x, 0);
    AGValue *w_ag = ag_value_from_array(&w, 0);

    /* discriminator forward: conv + sum to scalar */
    AGValue *d = ag_conv2d(x_ag, w_ag, 1, 1, 1, 1, 1, 1, 1);
    ag_register_temp_value(d);
    /* reduce to scalar */
    AGValue *sval = d;
    while (1)
    {
        mlx_array *a = ag_value_array(sval);
        if (!a)
            break;
        if (mlx_array_ndim(*a) == 0)
            break;
        sval = ag_sum_axis(sval, 0, 0);
        ag_register_temp_value(sval);
    }

    /* AG-based GP: create_graph over D to get grad_ag on x */
    if (ag_backward_create_graph(sval) != 0)
    {
        printf("ag_backward_create_graph failed\n");
        fflush(stdout);
        return 2;
    }
    printf("[test] ag_backward_create_graph succeeded\n");
    fflush(stdout);
    AGValue *gx = ag_value_get_grad_ag(x_ag);
    if (!gx)
    {
        printf("no grad_ag on x\n");
        return 3;
    }

    /* compute AG-based GP value: norm over spatial+channel dims per-sample -> sqrt(sum(g^2,H,W,C)) then (norm-1)^2 average over batch */
    printf("[test] computing g2\n");
    fflush(stdout);
    AGValue *g2 = ag_square(gx);
    ag_register_temp_value(g2);
    /* reduce over axes 3,2,1 (NHWC) */
    AGValue *reduced = g2;
    for (int ax = 3; ax >= 1; --ax)
    {
        reduced = ag_sum_axis(reduced, ax, 1);
        ag_register_temp_value(reduced);
    }
    /* now reduced shape is (N,1,1,1) or (N,) - reduce batch to scalar by sum over axis 0 */
    AGValue *sum_batch = ag_sum_axis(reduced, 0, 0);
    ag_register_temp_value(sum_batch);
    printf("[test] computed sum_batch\n");
    fflush(stdout);
    AGValue *norm = ag_sqrt(sum_batch);
    ag_register_temp_value(norm);
    AGValue *one = ag_scalar_float(1.0f);
    ag_register_temp_value(one);
    AGValue *dif = ag_sub(norm, one);
    ag_register_temp_value(dif);
    AGValue *gp_ag = ag_square(dif);
    ag_register_temp_value(gp_ag);

    /* evaluate AG-based GP numeric value */
    mlx_array *gp_arr = ag_value_array(gp_ag);
    if (!gp_arr)
    {
        printf("gp_ag has no array\n");
        return 4;
    }
    float gp_ag_val = eval_scalar_from_array(*gp_arr);
    printf("[test] gp_ag_val computed OK: %.8f\n", gp_ag_val);
    fflush(stdout);

    /* Numerical GP via finite differences on x */
    printf("[test] starting numeric finite-diff loop (xcount=%zu)\n", xcount);
    float eps = 1e-3f;
    /* compute D(x) baseline */
    mlx_array out_base = mlx_array_new();
    mlx_conv2d(&out_base, x, w, 1, 1, 1, 1, 1, 1, 1, s);
    /* evaluate out_base and sum in host */
    mlx_array_eval(out_base);
    const float *outbase_data = mlx_array_data_float32(out_base);
    size_t out_n = mlx_array_size(out_base);
    double base_sum = 0.0;
    for (size_t i = 0; i < out_n; ++i)
        base_sum += outbase_data[i];

    /* gradient array numeric */
    float *gnum = malloc(sizeof(float) * xcount);
    for (size_t idx = 0; idx < xcount; ++idx)
    {
        /* create perturbed copy */
        float *xb = malloc(sizeof(float) * xcount);
        const float *xdata = mlx_array_data_float32(x);
        for (size_t j = 0; j < xcount; ++j)
            xb[j] = xdata[j];
        xb[idx] += eps;
        mlx_array xpos = mlx_array_new_data(xb, xshape, 4, MLX_FLOAT32);
        free(xb);
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
        mlx_array xneg = mlx_array_new_data(xbn, xshape, 4, MLX_FLOAT32);
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

        float grad = (float)((psum - nsum) / (2.0 * eps));
        gnum[idx] = grad;
    }

    /* compute norm and GP numeric */
    double sumsq = 0.0;
    for (size_t i = 0; i < xcount; ++i)
        sumsq += (double)gnum[i] * (double)gnum[i];
    double norm_num = sqrt(sumsq);
    double gp_num = (norm_num - 1.0) * (norm_num - 1.0);

    printf("GP AG: %.8f\nGP NUM: %.8f\n", gp_ag_val, (double)gp_num);
    double rel = fabs((double)gp_ag_val - gp_num) / (fmax(fabs(gp_num), 1e-6));
    printf("relative error = %.6g\n", rel);

    free(gnum);
    mlx_stream_free(s);
    return 0;
}
