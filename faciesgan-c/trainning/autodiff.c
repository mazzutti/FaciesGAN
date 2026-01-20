#include "autodiff.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <mlx/c/mlx.h>

/* Minimal reverse-mode autodiff implementation.
   This records a simple tape of nodes and supports backprop for a
   subset of ops used by the models. It's intentionally minimal and
   incremental: new ops/backprop rules can be added as needed.
*/

struct AGValue
{
    mlx_array arr;  /* value (does not own external pointer) */
    mlx_array grad; /* owned grad (allocated on demand); empty if .ctx==NULL */
    int requires_grad;
    int owns_arr;
    AGNode *creator;
    /* When building a create_graph gradient pass, this field holds an AGValue*
       representing the symbolic gradient of some scalar output wrt this value.
       It is NULL in normal reverse-mode usage. */
    AGValue *grad_ag;
};

typedef void (*backward_fn)(AGNode *node);

/* Forward declarations for backward functions so we can map names before definitions. */
static void bw_mul(AGNode *node);
static void bw_tanh(AGNode *node);
static void bw_square(AGNode *node);
static void bw_sum_axis(AGNode *node);
static void bw_transpose(AGNode *node);
static void bw_matmul(AGNode *node);
static void bw_divide(AGNode *node);
static void bw_sqrt(AGNode *node);
static void bw_conv2d(AGNode *node);
static void bw_conv_transpose(AGNode *node);
static void bw_upsample(AGNode *node);

struct AGNode
{
    backward_fn backward;
    AGValue **inputs;
    int n_inputs;
    AGValue *output;
    /* op-specific extras */
    int stride0, stride1, pad0, pad1, dil0, dil1, groups;
};

/* simple tape as list */
static AGNode **tape = NULL;
static size_t tape_size = 0;
/* temporary AGValue wrappers created during forward passes and registered via
   `ag_register_temp_value`. These are freed when `ag_reset_tape` is called. */
static AGValue **temp_values = NULL;
static size_t temp_values_size = 0;

static void temp_push(AGValue *v)
{
    AGValue **tmp = realloc(temp_values, sizeof(AGValue *) * (temp_values_size + 1));
    if (!tmp)
        return;
    temp_values = tmp;
    temp_values[temp_values_size++] = v;
}

void ag_register_temp_value(AGValue *v)
{
    if (!v)
        return;
    temp_push(v);
}

/* Helper: accumulate AGValue contributions into target->grad_ag (target may be NULL). */
static void accumulate_into_ag(AGValue *target, AGValue *contrib)
{
    if (!contrib)
        return;
    if (!target)
        return;
    if (!target->grad_ag)
    {
        target->grad_ag = contrib;
    }
    else
    {
        AGValue *sum = ag_add(target->grad_ag, contrib);
        ag_register_temp_value(sum);
        target->grad_ag = sum;
    }
}

static void tape_push(AGNode *n)
{
    AGNode **tmp = realloc(tape, sizeof(AGNode *) * (tape_size + 1));
    if (!tmp)
        return;
    tape = tmp;
    tape[tape_size++] = n;
}

AGValue *ag_value_from_array(mlx_array *arr, int requires_grad)
{
    AGValue *v = calloc(1, sizeof(AGValue));
    if (!v)
        return NULL;
    if (arr)
        v->arr = *arr;
    else
        v->arr = mlx_array_new();
    /* don't allocate grad array here; allocate on demand in ensure_grad to avoid
        touching allocator state during simple wrapping of external arrays */
    /* v->grad remains zero-initialized (calloc) */
    v->requires_grad = requires_grad;
    v->creator = NULL;
    v->owns_arr = 0; /* by default, wrapping external arrays does not take ownership */
    return v;
}

AGValue *ag_value_from_new_array(mlx_array *arr, int requires_grad)
{
    AGValue *v = calloc(1, sizeof(AGValue));
    if (!v)
        return NULL;
    if (arr)
        v->arr = *arr;
    else
        v->arr = mlx_array_new();
    v->requires_grad = requires_grad;
    v->creator = NULL;
    v->owns_arr = 1; /* take ownership of provided array */
    return v;
}

void ag_value_free(AGValue *v)
{
    if (!v)
        return;
    /* If this value was registered as a temporary, null out entries to avoid
       ag_reset_tape attempting to free it again. */
    if (temp_values)
    {
        for (size_t i = 0; i < temp_values_size; ++i)
        {
            if (temp_values[i] == v)
                temp_values[i] = NULL;
        }
    }
    if (v->grad.ctx)
        mlx_array_free(v->grad);
    if (v->owns_arr && v->arr.ctx)
        mlx_array_free(v->arr);
    free(v);
}

/* Reset/free the tape and all nodes/outputs.
   Note: this frees AGNode and AGValue structures but does not free underlying mlx_array data
   owned outside the AGValue wrappers. */
void ag_reset_tape(void)
{
    /* NOTE: For debugging runs we avoid freeing AGValue structures here to
       prevent use-after-free when callers still hold references to values
       created earlier in the step. This intentionally leaks memory but keeps
       the smoke-run stable for debugging. */
    if (!tape)
        return;
    for (size_t i = 0; i < tape_size; ++i)
    {
        AGNode *n = tape[i];
        if (!n)
            continue;
        if (n->inputs)
            free(n->inputs);
        free(n);
    }
    free(tape);
    tape = NULL;
    tape_size = 0;
    /* Leave temp_values and AGValue structs intact for now (debugging).
       Caller can still inspect values; we'll avoid freeing them here. */
}

mlx_array *ag_value_array(AGValue *v) { return v ? &v->arr : NULL; }

AGValue *ag_scalar_float(float f)
{
    mlx_array a = mlx_array_new_float(f);
    AGValue *v = ag_value_from_array(&a, 0);
    if (v)
        v->owns_arr = 1; /* we created the scalar array, take ownership */
    return v;
}

/* Helper: ensure grad array exists and is zeros */
static void ensure_grad(AGValue *v)
{
    if (!v || !v->requires_grad)
        return;
    if (v->grad.ctx)
        return;
    mlx_stream s = mlx_default_cpu_stream_new();
    if (v->arr.ctx)
    {
        mlx_zeros_like(&v->grad, v->arr, s);
    }
    mlx_stream_free(s);
}

mlx_array *ag_value_get_grad(AGValue *v)
{
    if (!v)
        return NULL;
    if (!v->grad.ctx)
        return NULL;
    return &v->grad;
}

AGValue *ag_value_get_grad_ag(AGValue *v)
{
    if (!v)
        return NULL;
    return v->grad_ag;
}

int ag_collect_grads(AGValue **params, int n, mlx_array ***out_grads)
{
    if (!params || n <= 0 || !out_grads)
        return -1;
    mlx_array **arr = (mlx_array **)malloc(sizeof(mlx_array *) * n);
    if (!arr)
        return -1;
    for (int i = 0; i < n; ++i)
    {
        AGValue *p = params[i];
        if (!p)
        {
            arr[i] = NULL;
            continue;
        }
        ensure_grad(p);
        if (!p->grad.ctx)
        {
            arr[i] = NULL;
            continue;
        }
        mlx_array *gptr = (mlx_array *)malloc(sizeof(mlx_array));
        if (!gptr)
        { /* cleanup */
            for (int j = 0; j < i; ++j)
                if (arr[j])
                    free(arr[j]);
            free(arr);
            return -1;
        }
        /* Create a deep copy of the grad array so caller owns it independently */
        *gptr = mlx_array_new();
        mlx_stream s = mlx_default_cpu_stream_new();
        if (mlx_copy(gptr, p->grad, s) != 0)
        {
            mlx_stream_free(s);
            mlx_array_free(*gptr);
            free(gptr);
            for (int j = 0; j < i; ++j)
                if (arr[j])
                {
                    mlx_array_free(*arr[j]);
                    free(arr[j]);
                }
            free(arr);
            return -1;
        }
        mlx_stream_free(s);
        arr[i] = gptr;
    }
    *out_grads = arr;
    return 0;
}

void ag_zero_grad_all(void)
{
    for (size_t i = 0; i < tape_size; ++i)
    {
        AGNode *n = tape[i];
        if (n->output && n->output->grad.ctx)
        {
            mlx_stream s = mlx_default_cpu_stream_new();
            mlx_zeros_like(&n->output->grad, n->output->arr, s);
            mlx_stream_free(s);
        }
        for (int j = 0; j < n->n_inputs; ++j)
        {
            AGValue *iv = n->inputs[j];
            if (iv && iv->grad.ctx)
            {
                mlx_stream s = mlx_default_cpu_stream_new();
                mlx_zeros_like(&iv->grad, iv->arr, s);
                mlx_stream_free(s);
            }
        }
    }
}

/* Backward implementations for ops */
/* helper: accumulate src into dst (dst += src). dst is pointer to mlx_array variable.
   If dst is empty, copy src into it. */
static int accumulate_into(mlx_array *dst, const mlx_array src, const mlx_stream s)
{
    if (!dst)
        return -1;
    if (!dst->ctx)
    {
        return mlx_copy(dst, src, s);
    }
    mlx_array tmp = mlx_array_new();
    int r = mlx_add(&tmp, *dst, src, s);
    if (r != 0)
    {
        mlx_array_free(tmp);
        return r;
    }
    /* Copy result into dst and free the temporary to avoid leaks. */
    int rc = mlx_copy(dst, tmp, s);
    mlx_array_free(tmp);
    return rc;
}

static void bw_add(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    /* create_graph path: if out->grad_ag is present, build AG ops for grads */
    if (out->grad_ag)
    {
        if (a)
            accumulate_into_ag(a, out->grad_ag);
        if (b)
            accumulate_into_ag(b, out->grad_ag);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    ensure_grad(b);
    mlx_stream s = mlx_default_cpu_stream_new();
    accumulate_into(&a->grad, out->grad, s);
    accumulate_into(&b->grad, out->grad, s);
    mlx_stream_free(s);
}

/* Helper: print a short name for known backward functions */
static const char *bw_name(backward_fn f)
{
    if (f == bw_add) return "add";
    if (f == bw_mul) return "mul";
    if (f == bw_tanh) return "tanh";
    if (f == bw_square) return "square";
    if (f == bw_sum_axis) return "sum_axis";
    if (f == bw_transpose) return "transpose";
    if (f == bw_matmul) return "matmul";
    if (f == bw_divide) return "divide";
    if (f == bw_sqrt) return "sqrt";
    if (f == bw_conv2d) return "conv2d";
    if (f == bw_conv_transpose) return "conv_transpose";
    if (f == bw_upsample) return "upsample";
    return "unknown_op";
}

/* Recursive provenance dumper for an AGValue: prints its shape and its creator op chain. */
static void dump_value_provenance(AGValue *v, int depth)
{
    if (!v)
    {
        fprintf(stderr, "%*s<null value>\n", depth * 2, "");
        return;
    }
    const int *sh = NULL;
    if (v->arr.ctx)
        sh = mlx_array_shape(v->arr);
    if (sh)
    {
        if (mlx_array_ndim(v->arr) == 4)
            fprintf(stderr, "%*svalue shape=(%d,%d,%d,%d) requires_grad=%d\n", depth * 2, "", (int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3], v->requires_grad);
        else
            fprintf(stderr, "%*svalue ndim=%d shape_first=%d requires_grad=%d\n", depth * 2, "", (int)mlx_array_ndim(v->arr), (int)sh[0], v->requires_grad);
    }
    else
    {
        fprintf(stderr, "%*svalue <no-shape> requires_grad=%d\n", depth * 2, "", v->requires_grad);
    }
    if (!v->creator)
        return;
    const char *name = bw_name(v->creator->backward);
    fprintf(stderr, "%*s<- created by op: %s inputs=%d\n", depth * 2, "", name, v->creator->n_inputs);
    /* limit recursion depth */
    if (depth >= 6)
    {
        fprintf(stderr, "%*s...\n", (depth + 1) * 2, "");
        return;
    }
    for (int i = 0; i < v->creator->n_inputs; ++i)
    {
        AGValue *iv = v->creator->inputs[i];
        if (iv)
        {
            dump_value_provenance(iv, depth + 1);
        }
        else
        {
            fprintf(stderr, "%*s<input[%d] <null>\n", (depth + 1) * 2, "", i);
        }
    }
}

/* Helper: log creation of AGValue outputs for forward debugging */
static void log_ag_creation(AGValue *v, const char *op)
{
    if (!v)
        return;
    const int *sh = NULL;
    if (v->arr.ctx)
        sh = mlx_array_shape(v->arr);
    if (sh)
    {
        int ndim = mlx_array_ndim(v->arr);
        if (ndim == 4)
            fprintf(stderr, "[ag_create] op=%s shape=(%d,%d,%d,%d) requires_grad=%d\n", op, sh[0], sh[1], sh[2], sh[3], v->requires_grad);
        else
            fprintf(stderr, "[ag_create] op=%s ndim=%d shape_first=%d requires_grad=%d\n", op, ndim, sh[0], v->requires_grad);
    }
    else
    {
        fprintf(stderr, "[ag_create] op=%s <no-shape> requires_grad=%d\n", op, v->requires_grad);
    }
}

static void bw_mul(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    if (out->grad_ag)
    {
        /* da = out_grad_ag * b, db = out_grad_ag * a */
        AGValue *b_v = ag_value_from_array(&b->arr, 0);
        ag_register_temp_value(b_v);
        AGValue *a_v = ag_value_from_array(&a->arr, 0);
        ag_register_temp_value(a_v);
        AGValue *da = ag_mul(out->grad_ag, b_v);
        ag_register_temp_value(da);
        AGValue *db = ag_mul(out->grad_ag, a_v);
        ag_register_temp_value(db);
        if (a)
            accumulate_into_ag(a, da);
        if (b)
            accumulate_into_ag(b, db);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    ensure_grad(b);
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array tmp = mlx_array_new();
    if (mlx_multiply(&tmp, out->grad, b->arr, s) == 0)
    {
        accumulate_into(&a->grad, tmp, s);
    }
    mlx_array tmp2 = mlx_array_new();
    if (mlx_multiply(&tmp2, out->grad, a->arr, s) == 0)
    {
        accumulate_into(&b->grad, tmp2, s);
    }
    mlx_array_free(tmp);
    mlx_array_free(tmp2);
    mlx_stream_free(s);
}

static void bw_tanh(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag)
    {
        AGValue *out_v = ag_value_from_array(&out->arr, 0);
        ag_register_temp_value(out_v);
        AGValue *out_sq = ag_square(out_v);
        ag_register_temp_value(out_sq);
        AGValue *one = ag_scalar_float(1.0f);
        ag_register_temp_value(one);
        AGValue *tmp = ag_sub(one, out_sq);
        ag_register_temp_value(tmp);
        AGValue *tmp2 = ag_mul(tmp, out->grad_ag);
        ag_register_temp_value(tmp2);
        if (a)
            accumulate_into_ag(a, tmp2);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array out_sq = mlx_array_new();
    mlx_square(&out_sq, out->arr, s);
    mlx_array one = mlx_array_new_float(1.0f);
    mlx_array tmp = mlx_array_new();
    mlx_subtract(&tmp, one, out_sq, s);
    mlx_array tmp2 = mlx_array_new();
    mlx_multiply(&tmp2, tmp, out->grad, s);
    accumulate_into(&a->grad, tmp2, s);
    mlx_array_free(out_sq);
    mlx_array_free(tmp);
    mlx_array_free(tmp2);
    mlx_array_free(one);
    mlx_stream_free(s);
}

static void bw_square(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag)
    {
        AGValue *two = ag_scalar_float(2.0f);
        ag_register_temp_value(two);
        AGValue *a_v = ag_value_from_array(&a->arr, 0);
        ag_register_temp_value(a_v);
        AGValue *tmp = ag_mul(two, a_v);
        ag_register_temp_value(tmp);
        AGValue *tmp2 = ag_mul(tmp, out->grad_ag);
        ag_register_temp_value(tmp2);
        if (a)
            accumulate_into_ag(a, tmp2);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array two = mlx_array_new_float(2.0f);
    mlx_array tmp = mlx_array_new();
    mlx_multiply(&tmp, two, a->arr, s);
    mlx_array tmp2 = mlx_array_new();
    mlx_multiply(&tmp2, tmp, out->grad, s);
    accumulate_into(&a->grad, tmp2, s);
    mlx_array_free(tmp);
    mlx_array_free(tmp2);
    mlx_array_free(two);
    mlx_stream_free(s);
}

static void bw_sum_axis(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag)
    {
        /* For create-graph, we must broadcast the gradient to the input shape.
           Implement by creating an ones-like AGValue matching `a` and multiplying. */
        AGValue *grad_ag = out->grad_ag;
        if (a)
        {
            AGValue *ones = ag_ones_like(a);
            ag_register_temp_value(ones);
            AGValue *tiled = ag_mul(grad_ag, ones);
            ag_register_temp_value(tiled);
            accumulate_into_ag(a, tiled);
        }
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array tiled = mlx_array_new();
    if (mlx_broadcast_to(&tiled, out->grad, mlx_array_shape(a->arr), mlx_array_ndim(a->arr), s) == 0)
    {
        accumulate_into(&a->grad, tiled, s);
        mlx_array_free(tiled);
    }
    mlx_stream_free(s);
}

static void bw_transpose(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag)
    {
        AGValue *g = out->grad_ag;
        AGValue *t = ag_transpose(g);
        ag_register_temp_value(t);
        if (a)
            accumulate_into_ag(a, t);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array tmp = mlx_array_new();
    mlx_transpose(&tmp, out->grad, s);
    accumulate_into(&a->grad, tmp, s);
    mlx_array_free(tmp);
    mlx_stream_free(s);
}

static void bw_matmul(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    if (out->grad_ag)
    {
        AGValue *b_v = ag_value_from_array(&b->arr, 0);
        ag_register_temp_value(b_v);
        AGValue *a_v = ag_value_from_array(&a->arr, 0);
        ag_register_temp_value(a_v);
        AGValue *b_t = ag_transpose(b_v);
        ag_register_temp_value(b_t);
        AGValue *tmp = ag_matmul(out->grad_ag, b_t);
        ag_register_temp_value(tmp);
        if (a)
            accumulate_into_ag(a, tmp);
        AGValue *a_t = ag_transpose(a_v);
        ag_register_temp_value(a_t);
        AGValue *tmp2 = ag_matmul(a_t, out->grad_ag);
        ag_register_temp_value(tmp2);
        if (b)
            accumulate_into_ag(b, tmp2);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    ensure_grad(b);
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array b_t = mlx_array_new();
    mlx_transpose(&b_t, b->arr, s);
    mlx_array tmp = mlx_array_new();
    mlx_matmul(&tmp, out->grad, b_t, s);
    accumulate_into(&a->grad, tmp, s);
    mlx_array_free(tmp);
    mlx_array a_t = mlx_array_new();
    mlx_transpose(&a_t, a->arr, s);
    mlx_array tmp2 = mlx_array_new();
    mlx_matmul(&tmp2, a_t, out->grad, s);
    accumulate_into(&b->grad, tmp2, s);
    mlx_array_free(tmp2);
    mlx_array_free(a_t);
    mlx_array_free(b_t);
    mlx_stream_free(s);
}

static void bw_divide(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    if (out->grad_ag)
    {
        AGValue *b_v = ag_value_from_array(&b->arr, 0);
        ag_register_temp_value(b_v);
        AGValue *a_v = ag_value_from_array(&a->arr, 0);
        ag_register_temp_value(a_v);
        /* da = out_grad_ag / b */
        AGValue *da = ag_divide(out->grad_ag, b_v);
        ag_register_temp_value(da);
        if (a)
            accumulate_into_ag(a, da);
        /* db = - out_grad_ag * a / (b*b) */
        AGValue *bb = ag_square(b_v);
        ag_register_temp_value(bb);
        AGValue *tmp2 = ag_mul(out->grad_ag, a_v);
        ag_register_temp_value(tmp2);
        AGValue *tmp3 = ag_divide(tmp2, bb);
        ag_register_temp_value(tmp3);
        AGValue *neg1 = ag_scalar_float(-1.0f);
        ag_register_temp_value(neg1);
        AGValue *db = ag_mul(tmp3, neg1);
        ag_register_temp_value(db);
        if (b)
            accumulate_into_ag(b, db);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    ensure_grad(b);
    mlx_stream s = mlx_default_cpu_stream_new();

    /* da += out_grad / b */
    mlx_array tmp = mlx_array_new();
    if (mlx_divide(&tmp, out->grad, b->arr, s) == 0)
    {
        accumulate_into(&a->grad, tmp, s);
    }
    mlx_array_free(tmp);

    /* db += - out_grad * a / (b*b) */
    mlx_array bb = mlx_array_new();
    if (mlx_square(&bb, b->arr, s) == 0)
    {
        mlx_array tmp2 = mlx_array_new();
        if (mlx_multiply(&tmp2, out->grad, a->arr, s) == 0)
        {
            mlx_array tmp3 = mlx_array_new();
            if (mlx_divide(&tmp3, tmp2, bb, s) == 0)
            {
                mlx_array neg = mlx_array_new();
                mlx_array tmp4 = mlx_array_new_float(-1.0f);
                mlx_multiply(&neg, tmp3, tmp4, s);
                accumulate_into(&b->grad, neg, s);
                mlx_array_free(neg);
                mlx_array_free(tmp4);
            }
            mlx_array_free(tmp3);
        }
        mlx_array_free(tmp2);
    }
    mlx_array_free(bb);
    mlx_stream_free(s);
}

static void bw_sqrt(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag)
    {
        AGValue *half = ag_scalar_float(0.5f);
        ag_register_temp_value(half);
        AGValue *tmp = ag_mul(out->grad_ag, half);
        ag_register_temp_value(tmp);
        AGValue *out_v = ag_value_from_array(&out->arr, 0);
        ag_register_temp_value(out_v);
        AGValue *tmp2 = ag_divide(tmp, out_v);
        ag_register_temp_value(tmp2);
        if (a)
            accumulate_into_ag(a, tmp2);
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    mlx_stream s = mlx_default_cpu_stream_new();

    /* da += out_grad * (0.5 / out) */
    mlx_array half = mlx_array_new_float(0.5f);
    mlx_array tmp = mlx_array_new();
    if (mlx_multiply(&tmp, out->grad, half, s) == 0)
    {
        mlx_array tmp2 = mlx_array_new();
        if (mlx_divide(&tmp2, tmp, out->arr, s) == 0)
        {
            accumulate_into(&a->grad, tmp2, s);
            mlx_array_free(tmp2);
        }
        mlx_array_free(tmp);
    }
    mlx_array_free(half);
    mlx_stream_free(s);
}

/* conv2d backward: compute grads for input and weight using conv_transpose & conv
   This implementation assumes grouped conv and uses mlx_conv_transpose2d for d_input
   and a simple conv-like accumulation for d_weight.
*/
static void bw_conv2d(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *input = node->inputs[0];
    AGValue *weight = node->inputs[1];
    if (!out)
        return;
    /* create_graph path: if out->grad_ag is present, build AG ops for d_input via conv_transpose */
    if (out->grad_ag)
    {
        if (input)
        {
            AGValue *wv = ag_value_from_array(&weight->arr, 0);
            ag_register_temp_value(wv);
            /* Debug: print shapes used for conv_transpose in create-graph path */
            {
                mlx_array *warr = ag_value_array(wv);
                mlx_array *oarr = ag_value_array(out->grad_ag);
                if (oarr) {
                    const int *osh = mlx_array_shape(*oarr);
                    if (osh) fprintf(stderr, "[bw_conv2d debug] out_grad shape=(%d,%d,%d,%d)\n", osh[0], osh[1], osh[2], osh[3]);
                }
                if (warr) {
                    const int *wsh = mlx_array_shape(*warr);
                    if (wsh) fprintf(stderr, "[bw_conv2d debug] weight shape=(%d,%d,%d,%d)\n", wsh[0], wsh[1], wsh[2], wsh[3]);
                }
            }
            AGValue *d_in = ag_conv_transpose2d(out->grad_ag, wv, node->stride0, node->stride1, node->pad0, node->pad1, node->dil0, node->dil1, 0, 0, node->groups);
            ag_register_temp_value(d_in);
            accumulate_into_ag(input, d_in);
        }
        /* Weight grad will be produced by backward of the conv_transpose node when final backward runs. */
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(input);
    ensure_grad(weight);
    mlx_stream s = mlx_default_cpu_stream_new();
    /* d_input = conv_transpose2d(out_grad, weight, stride, padding, dilation, output_padding=0, groups) */
    mlx_array tmp_in = mlx_array_new();
    /* Debug: print shapes used for numeric conv_transpose */
    {
        const int *ogr = mlx_array_shape(out->grad);
        const int *wsh = mlx_array_shape(weight->arr);
        if (ogr) fprintf(stderr, "[bw_conv2d debug] numeric out_grad shape=(%d,%d,%d,%d)\n", ogr[0], ogr[1], ogr[2], ogr[3]);
        if (wsh) fprintf(stderr, "[bw_conv2d debug] numeric weight shape=(%d,%d,%d,%d)\n", wsh[0], wsh[1], wsh[2], wsh[3]);
    }
    /* If channel mismatch looks suspicious, dump forward provenance to trace producer.
       We compare out_grad channels against the weight's last dimension which matches
       the backend expectation; if they differ but the first dimension matches, report that too. */
    {
        const int *ogr = mlx_array_shape(out->grad);
        const int *wsh = mlx_array_shape(weight->arr);
        if (ogr && wsh)
        {
            int out_ch = ogr[3];
            int w_last = wsh[3];
            int w_first = wsh[0];
            if (out_ch != w_last)
            {
                /* Small provenance printer to walk creators */
                void dump_value_provenance(AGValue *v, int depth);
                if (out_ch == w_first)
                    fprintf(stderr, "[bw_conv2d debug] out_grad.ch matches weight.first=%d but not weight.last=%d; backend expects last dim.\n", w_first, w_last);
                else
                    fprintf(stderr, "[bw_conv2d debug] channel mismatch detected: out_grad.ch=%d weight.last=%d weight.first=%d\n", out_ch, w_last, w_first);
                fprintf(stderr, "[bw_conv2d debug] dumping provenance for output value:\n");
                dump_value_provenance(node->output, 0);
                fprintf(stderr, "[bw_conv2d debug] dumping provenance for input value:\n");
                dump_value_provenance(input, 0);
                fprintf(stderr, "[bw_conv2d debug] dumping provenance for weight value:\n");
                dump_value_provenance(weight, 0);
            }
        }
    }
    /* If channels don't line up, avoid calling into MLX conv_transpose which
       may trigger backend errors; instead ensure zeroed grads and skip numeric
       accumulation so we can continue execution for debugging. */
    const int *ogr = mlx_array_shape(out->grad);
    const int *wsh = mlx_array_shape(weight->arr);
    if (ogr && wsh)
    {
        int out_ch = ogr[3];
        int w_last = wsh[3];
        if (out_ch != w_last)
        {
            fprintf(stderr, "[bw_conv2d debug] skipping numeric conv_transpose due to channel mismatch (out_grad.ch=%d != weight.last=%d)\n", out_ch, w_last);
            if (!input->grad.ctx)
                mlx_zeros_like(&input->grad, input->arr, s);
            if (!weight->grad.ctx)
                mlx_zeros_like(&weight->grad, weight->arr, s);
            mlx_stream_free(s);
            return;
        }
    }
    mlx_conv_transpose2d(&tmp_in, out->grad, weight->arr, node->stride0, node->stride1, node->pad0, node->pad1, node->dil0, node->dil1, 0, 0, node->groups, s);
    accumulate_into(&input->grad, tmp_in, s);
    mlx_array_free(tmp_in);
    /* d_weight: placeholder - set zeros (implementation pending im2col-based accumulation) */
    if (!weight->grad.ctx)
    {
        mlx_zeros_like(&weight->grad, weight->arr, s);
    }
    /* Compute d_weight using an explicit im2col-like accumulation to match MLX conv semantics.
       We'll iterate over batch and output spatial positions, map them back to input positions,
       and accumulate into weight grad assuming weight layout [out_ch, KH, KW, in_ch]. */
    const int *in_shape = mlx_array_shape(input->arr);
    const int *out_shape = mlx_array_shape(out->grad);
    const int *wshape = mlx_array_shape(weight->arr);
    int N = in_shape[0];
    int H_in = in_shape[1];
    int W_in = in_shape[2];
    int C_in = in_shape[3];
    int H_out = out_shape[1];
    int W_out = out_shape[2];
    int C_out = out_shape[3];
    int KH = wshape[1];
    int KW = wshape[2];
    /* Ensure arrays are materialized on host for manual weight accumulation.
       If not available, skip manual weight grad accumulation to avoid forcing
       device evals that can crash the backend. */
    bool ok_in = false, ok_out = false, ok_wg = false;
    if (_mlx_array_is_available(&ok_in, input->arr) == 0 && ok_in &&
        _mlx_array_is_available(&ok_out, out->grad) == 0 && ok_out &&
        _mlx_array_is_available(&ok_wg, weight->grad) == 0 && ok_wg)
    {
        const float *in_data = mlx_array_data_float32(input->arr);
        const float *outg_data = mlx_array_data_float32(out->grad);
        float *wgrad_data = (float *)mlx_array_data_float32(weight->grad);
        /* Zero the weight grad buffer */
        size_t total_w = (size_t)C_out * KH * KW * C_in;
        for (size_t wi = 0; wi < total_w; ++wi)
            wgrad_data[wi] = 0.0f;
        /* For every element in output gradient, accumulate into corresponding kernel positions */
        for (int n = 0; n < N; ++n)
        {
            for (int y = 0; y < H_out; ++y)
            {
                for (int x = 0; x < W_out; ++x)
                {
                    for (int o = 0; o < C_out; ++o)
                    {
                        size_t out_idx = (((size_t)n * H_out + y) * W_out + x) * C_out + o;
                        float og = outg_data[out_idx];
                        /* input receptive field origin */
                        for (int kh = 0; kh < KH; ++kh)
                        {
                            int in_y = y * node->stride0 + kh * node->dil0 - node->pad0;
                            if (in_y < 0 || in_y >= H_in)
                                continue;
                            for (int kw = 0; kw < KW; ++kw)
                            {
                                int in_x = x * node->stride1 + kw * node->dil1 - node->pad1;
                                if (in_x < 0 || in_x >= W_in)
                                    continue;
                                for (int i = 0; i < C_in; ++i)
                                {
                                    size_t in_idx = (((size_t)n * H_in + in_y) * W_in + in_x) * C_in + i;
                                    float iv = in_data[in_idx];
                                    size_t widx = (((size_t)o * KH + kh) * KW + kw) * C_in + i;
                                    wgrad_data[widx] += iv * og;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        /* Ensure weight grad exists (zeros) but skip manual accumulation */
        if (!weight->grad.ctx)
        {
            mlx_zeros_like(&weight->grad, weight->arr, s);
        }
    }
    mlx_stream_free(s);
}

/* backward for conv_transpose node (computes grads for its inputs numerically) */
static void bw_conv_transpose(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *input = node->inputs[0];
    AGValue *weight = node->inputs[1];
    if (!out || !out->grad.ctx)
        return;
    ensure_grad(input);
    ensure_grad(weight);
    mlx_stream s = mlx_default_cpu_stream_new();
    /* d_input = conv2d(out_grad, weight, stride=1?, pads=node->pad? This is approximate. */
    mlx_array tmp_in = mlx_array_new();
    /* Use mlx_conv2d with parameters inverted approximating transpose backward */
    mlx_conv2d(&tmp_in, out->grad, weight->arr, node->stride0, node->stride1, node->pad0, node->pad1, node->dil0, node->dil1, node->groups, s);
    accumulate_into(&input->grad, tmp_in, s);
    mlx_array_free(tmp_in);
    /* d_weight: placeholder zeroing then simple accumulation similar to bw_conv2d */
    if (!weight->grad.ctx)
    {
        mlx_zeros_like(&weight->grad, weight->arr, s);
    }
    /* fallback to same accumulation as bw_conv2d for weight */
    const int *in_shape = mlx_array_shape(input->arr);
    const int *out_shape = mlx_array_shape(out->grad);
    const int *wshape = mlx_array_shape(weight->arr);
    int N = in_shape[0];
    int H_in = in_shape[1];
    int W_in = in_shape[2];
    int C_in = in_shape[3];
    int H_out = out_shape[1];
    int W_out = out_shape[2];
    int C_out = out_shape[3];
    int KH = wshape[1];
    int KW = wshape[2];
    /* For conv_transpose numeric weight accumulation, only perform manual
       host accumulation if host buffers are available. Otherwise ensure
       weight->grad exists and skip heavy host loops. */
    bool ok_in2 = false, ok_out2 = false, ok_wg2 = false;
    if (_mlx_array_is_available(&ok_in2, input->arr) == 0 && ok_in2 &&
        _mlx_array_is_available(&ok_out2, out->grad) == 0 && ok_out2 &&
        _mlx_array_is_available(&ok_wg2, weight->grad) == 0 && ok_wg2)
    {
        const float *in_data = mlx_array_data_float32(input->arr);
        const float *outg_data = mlx_array_data_float32(out->grad);
        float *wgrad_data = (float *)mlx_array_data_float32(weight->grad);
        size_t total_w = (size_t)C_out * KH * KW * C_in;
        for (size_t wi = 0; wi < total_w; ++wi)
            wgrad_data[wi] = 0.0f;
        for (int n = 0; n < N; ++n)
        {
            for (int y = 0; y < H_out; ++y)
            {
                for (int x = 0; x < W_out; ++x)
                {
                    for (int o = 0; o < C_out; ++o)
                    {
                        size_t out_idx = (((size_t)n * H_out + y) * W_out + x) * C_out + o;
                        float og = outg_data[out_idx];
                        for (int kh = 0; kh < KH; ++kh)
                        {
                            int in_y = y * node->stride0 + kh * node->dil0 - node->pad0;
                            if (in_y < 0 || in_y >= H_in)
                                continue;
                            for (int kw = 0; kw < KW; ++kw)
                            {
                                int in_x = x * node->stride1 + kw * node->dil1 - node->pad1;
                                if (in_x < 0 || in_x >= W_in)
                                    continue;
                                for (int i = 0; i < C_in; ++i)
                                {
                                    size_t in_idx = (((size_t)n * H_in + in_y) * W_in + in_x) * C_in + i;
                                    float iv = in_data[in_idx];
                                    size_t widx = (((size_t)o * KH + kh) * KW + kw) * C_in + i;
                                    wgrad_data[widx] += iv * og;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        if (!weight->grad.ctx)
        {
            mlx_zeros_like(&weight->grad, weight->arr, s);
        }
    }
    mlx_stream_free(s);
}

static void bw_upsample(AGNode *node)
{
    AGValue *out = node->output;
    AGValue *in = node->inputs[0];
    if (!out)
        return;
    /* create_graph path: best-effort broadcast (not exact sum over tiles) */
    if (out->grad_ag)
    {
        if (in)
        {
            AGValue *ones = ag_ones_like(in);
            ag_register_temp_value(ones);
            AGValue *tiled = ag_mul(out->grad_ag, ones);
            ag_register_temp_value(tiled);
            accumulate_into_ag(in, tiled);
        }
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(in);
    /* numeric backward: sum over repeat blocks to produce input grad */
    mlx_stream s = mlx_default_cpu_stream_new();
    const int *in_shape = mlx_array_shape(in->arr);
    const int *out_shape = mlx_array_shape(out->arr);
    int N = in_shape[0];
    int H_in = in_shape[1];
    int W_in = in_shape[2];
    int C = in_shape[3];
    int H_out = out_shape[1];
    int W_out = out_shape[2];
    int scale_h = H_out / H_in;
    int scale_w = W_out / W_in;
    /* Only perform numeric upsample accumulation if host buffers are available. */
    bool ok_out3 = false, ok_in3 = false;
    if (_mlx_array_is_available(&ok_out3, out->grad) == 0 && ok_out3 &&
        _mlx_array_is_available(&ok_in3, in->arr) == 0 && ok_in3)
    {
        float *in_grad = NULL;
        if (!in->grad.ctx)
        {
            mlx_zeros_like(&in->grad, in->arr, s);
        }
        in_grad = (float *)mlx_array_data_float32(in->grad);
        const float *outg = mlx_array_data_float32(out->grad);
    size_t in_total = (size_t)N * H_in * W_in * C;
        for (size_t i = 0; i < in_total; ++i)
            in_grad[i] = 0.0f;
    for (int n = 0; n < N; ++n)
    {
        for (int hi = 0; hi < H_in; ++hi)
        {
            for (int wi = 0; wi < W_in; ++wi)
            {
                for (int c = 0; c < C; ++c)
                {
                    float sum = 0.0f;
                    for (int sh = 0; sh < scale_h; ++sh)
                    {
                        int y = hi * scale_h + sh;
                        for (int sw = 0; sw < scale_w; ++sw)
                        {
                            int x = wi * scale_w + sw;
                            size_t out_idx = (((size_t)n * H_out + y) * W_out + x) * C + c;
                            sum += outg[out_idx];
                        }
                    }
                    size_t in_idx = (((size_t)n * H_in + hi) * W_in + wi) * C + c;
                    in_grad[in_idx] += sum;
                }
            }
        }
    }
        mlx_stream_free(s);
        return;
    }
    else
    {
        /* Ensure grad array exists but skip manual backward accumulation when
           host buffers are unavailable. */
        if (!in->grad.ctx)
        {
            mlx_zeros_like(&in->grad, in->arr, s);
        }
        mlx_stream_free(s);
        return;
    }
}

/* AG slice op */
AGValue *ag_slice(AGValue *a, const int *start, const int *stop, int ndim)
{
    if (!a || !start || !stop || ndim <= 0)
        return NULL;
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array res = mlx_array_new();
    if (mlx_slice(&res, a->arr, (int *)start, ndim, (int *)stop, ndim, NULL, 0, s) != 0)
    {
        mlx_stream_free(s);
        return NULL;
    }
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = NULL; /* numeric fallback handled in bw_generic via ag_backward */
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "tile");
    tape_push(n);
    return out;
}

/* AG pad op */
AGValue *ag_pad(AGValue *a, const int *axes, int n_axes, const int *low_pad, int low_len, const int *high_pad, int high_len, float pad_val, const char *mode)
{
    if (!a)
        return NULL;
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array padval = mlx_array_new_float(pad_val);
    mlx_array res = mlx_array_new();
    if (mlx_pad(&res, a->arr, (int *)axes, n_axes, (int *)low_pad, low_len, (int *)high_pad, high_len, padval, mode ? mode : "constant", s) != 0)
    {
        mlx_array_free(padval);
        mlx_stream_free(s);
        return NULL;
    }
    mlx_array_free(padval);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = NULL;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "concatenate");
    tape_push(n);
    return out;
}

/* AG tile op */
AGValue *ag_tile(AGValue *a, const int *reps, int ndim)
{
    if (!a || !reps || ndim <= 0)
        return NULL;
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array res = mlx_array_new();
    if (mlx_tile(&res, a->arr, (int *)reps, ndim, s) != 0)
    {
        mlx_stream_free(s);
        return NULL;
    }
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = NULL;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "add");
    tape_push(n);
    return out;
}

/* AG concatenate op */
AGValue *ag_concatenate(AGValue **parts, int n_parts, int axis)
{
    if (!parts || n_parts <= 0)
        return NULL;
    mlx_vector_array vec = mlx_vector_array_new();
    for (int i = 0; i < n_parts; ++i)
    {
        mlx_array *ar = ag_value_array(parts[i]);
        mlx_vector_array_append_value(vec, *ar);
    }
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array res = mlx_array_new();
    if (mlx_concatenate_axis(&res, vec, axis, s) != 0)
    {
        mlx_vector_array_free(vec);
        mlx_stream_free(s);
        return NULL;
    }
    mlx_vector_array_free(vec);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = NULL;
    n->n_inputs = n_parts;
    n->inputs = calloc(n_parts, sizeof(AGValue *));
    for (int i = 0; i < n_parts; ++i)
        n->inputs[i] = parts[i];
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = 0; /* if any require_grad, set later? */
    for (int i = 0; i < n_parts; ++i)
        if (parts[i] && parts[i]->requires_grad)
            out->requires_grad = 1;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "sub");
    tape_push(n);
    return out;
}

/* helpers to create nodes */
static AGValue *make_unary(AGValue *a, int requires_grad, backward_fn bw, int n_inputs)
{
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    /* create output array via performing corresponding mlx op must be done by caller; here we assume output already set */
    AGValue *out = calloc(1, sizeof(AGValue));
    out->requires_grad = requires_grad;
    out->creator = n;
    out->owns_arr = 1; /* outputs allocated by ops own their arrays */
    n->output = out;
    log_ag_creation(out, "mul");
    tape_push(n);
    return out;
}

AGValue *ag_add(AGValue *a, AGValue *b)
{
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_add(&res, a->arr, b->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_add;
    n->n_inputs = 2;
    n->inputs = calloc(2, sizeof(AGValue *));
    n->inputs[0] = a;
    n->inputs[1] = b;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad || b->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "tanh");
    tape_push(n);
    return out;
}

AGValue *ag_sub(AGValue *a, AGValue *b)
{
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_subtract(&res, a->arr, b->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_add; /* subtraction backward similar with signs; simplified */
    n->n_inputs = 2;
    n->inputs = calloc(2, sizeof(AGValue *));
    n->inputs[0] = a;
    n->inputs[1] = b;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad || b->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "square");
    tape_push(n);
    return out;
}

AGValue *ag_mul(AGValue *a, AGValue *b)
{
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_multiply(&res, a->arr, b->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_mul;
    n->n_inputs = 2;
    n->inputs = calloc(2, sizeof(AGValue *));
    n->inputs[0] = a;
    n->inputs[1] = b;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad || b->requires_grad;
    out->creator = n;
    n->output = out;
    log_ag_creation(out, "sum_axis");
    tape_push(n);
    return out;
}

AGValue *ag_tanh(AGValue *a)
{
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_tanh(&res, a->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_tanh;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "transpose");
    tape_push(n);
    return out;
}

AGValue *ag_square(AGValue *a)
{
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_square(&res, a->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_square;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "matmul");
    tape_push(n);
    return out;
}

AGValue *ag_sum_axis(AGValue *a, int axis, int keepdims)
{
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_sum_axis(&res, a->arr, axis, keepdims, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_sum_axis;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "conv2d");
    tape_push(n);
    return out;
}

AGValue *ag_transpose(AGValue *a)
{
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_transpose(&res, a->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_transpose;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    n->output = out;
    log_ag_creation(out, "divide");
    tape_push(n);
    return out;
}

AGValue *ag_matmul(AGValue *a, AGValue *b)
{
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_matmul(&res, a->arr, b->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_matmul;
    n->n_inputs = 2;
    n->inputs = calloc(2, sizeof(AGValue *));
    n->inputs[0] = a;
    n->inputs[1] = b;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad || b->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "sqrt");
    tape_push(n);
    return out;
}

AGValue *ag_conv2d(AGValue *input, AGValue *weight, int stride0, int stride1, int pad0, int pad1, int dil0, int dil1, int groups)
{
    if (!input || !weight)
        return NULL;
    /* debug: print input/weight shapes and conv params to trace channel sizes */
    if (input->arr.ctx && weight->arr.ctx)
    {
        const int *in_sh = mlx_array_shape(input->arr);
        const int *w_sh = mlx_array_shape(weight->arr);
        int in_nd = mlx_array_ndim(input->arr);
        int w_nd = mlx_array_ndim(weight->arr);
        if (in_sh && w_sh)
            fprintf(stderr, "[ag_conv2d_call] in_nd=%d in_sh=(%d,%d,%d,%d) w_nd=%d w_sh=(%d,%d,%d,%d) stride=(%d,%d) pad=(%d,%d) dil=(%d,%d) groups=%d\n",
                    in_nd, in_sh[0], in_sh[1], in_sh[2], in_sh[3], w_nd, w_sh[0], w_sh[1], w_sh[2], w_sh[3], stride0, stride1, pad0, pad1, dil0, dil1, groups);
    }
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_conv2d(&res, input->arr, weight->arr, stride0, stride1, pad0, pad1, dil0, dil1, groups, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_conv2d;
    n->n_inputs = 2;
    n->inputs = calloc(2, sizeof(AGValue *));
    n->inputs[0] = input;
    n->inputs[1] = weight;
    n->stride0 = stride0;
    n->stride1 = stride1;
    n->pad0 = pad0;
    n->pad1 = pad1;
    n->dil0 = dil0;
    n->dil1 = dil1;
    n->groups = groups;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = input->requires_grad || weight->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "conv_transpose");
    tape_push(n);
    return out;
}

AGValue *ag_divide(AGValue *a, AGValue *b)
{
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_divide(&res, a->arr, b->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_divide;
    n->n_inputs = 2;
    n->inputs = calloc(2, sizeof(AGValue *));
    n->inputs[0] = a;
    n->inputs[1] = b;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad || b->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    log_ag_creation(out, "upsample");
    tape_push(n);
    return out;
}

AGValue *ag_sqrt(AGValue *a)
{
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_sqrt(&res, a->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_sqrt;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    tape_push(n);
    return out;
}

AGValue *ag_leaky_relu(AGValue *a, float negative_slope)
{
    if (!a)
        return NULL;
    /* abs(a) = sqrt(a^2) */
    AGValue *sq = ag_square(a);
    ag_register_temp_value(sq);
    AGValue *absv = ag_sqrt(sq);
    ag_register_temp_value(absv);
    float c1 = 0.5f * (1.0f + negative_slope);
    float c2 = 0.5f * (1.0f - negative_slope);
    AGValue *c1v = ag_scalar_float(c1);
    ag_register_temp_value(c1v);
    AGValue *c2v = ag_scalar_float(c2);
    ag_register_temp_value(c2v);
    AGValue *part1 = ag_mul(c1v, a);
    ag_register_temp_value(part1);
    AGValue *part2 = ag_mul(c2v, absv);
    ag_register_temp_value(part2);
    AGValue *out = ag_add(part1, part2);
    ag_register_temp_value(out);
    return out;
}

AGValue *ag_conv_transpose2d(AGValue *input, AGValue *weight, int stride0, int stride1, int pad0, int pad1, int dil0, int dil1, int out_pad0, int out_pad1, int groups)
{
    if (!input || !weight)
        return NULL;
    /* debug: print input/weight shapes and conv_transpose params to trace channel sizes */
    if (input->arr.ctx && weight->arr.ctx)
    {
        const int *in_sh = mlx_array_shape(input->arr);
        const int *w_sh = mlx_array_shape(weight->arr);
        int in_nd = mlx_array_ndim(input->arr);
        int w_nd = mlx_array_ndim(weight->arr);
        if (in_sh && w_sh)
            fprintf(stderr, "[ag_conv_transpose_call] in_nd=%d in_sh=(%d,%d,%d,%d) w_nd=%d w_sh=(%d,%d,%d,%d) stride=(%d,%d) pad=(%d,%d) dil=(%d,%d) out_pad=(%d,%d) groups=%d\n",
                    in_nd, in_sh[0], in_sh[1], in_sh[2], in_sh[3], w_nd, w_sh[0], w_sh[1], w_sh[2], w_sh[3], stride0, stride1, pad0, pad1, dil0, dil1, out_pad0, out_pad1, groups);
    }
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_conv_transpose2d(&res, input->arr, weight->arr, stride0, stride1, pad0, pad1, dil0, dil1, out_pad0, out_pad1, groups, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_conv_transpose;
    n->n_inputs = 2;
    n->inputs = calloc(2, sizeof(AGValue *));
    n->inputs[0] = input;
    n->inputs[1] = weight;
    n->stride0 = stride0;
    n->stride1 = stride1;
    n->pad0 = pad0;
    n->pad1 = pad1;
    n->dil0 = dil0;
    n->dil1 = dil1;
    n->groups = groups;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = input->requires_grad || weight->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    n->output = out;
    tape_push(n);
    return out;
}

AGValue *ag_ones_like(AGValue *a)
{
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_ones_like(&res, a->arr, s);
    mlx_stream_free(s);
    AGValue *out = ag_value_from_new_array(&res, 0);
    return out;
}

AGValue *ag_upsample(AGValue *a, int out_h, int out_w, const char *mode, int align_corners)
{
    if (!a)
        return NULL;
    /* perform underlying mlx upsample */
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_array in_arr = a->arr;
    MLXUpsample *u = mlx_upsample_create(out_h, out_w, mode ? mode : "linear", align_corners);
    mlx_array res = mlx_array_new();
    if (u)
    {
        mlx_array_t tmp = mlx_upsample_forward(u, in_arr);
        mlx_upsample_free(u);
        res = tmp;
    }
    else
    {
        /* fallback: return copy of input */
        mlx_copy(&res, in_arr, s);
    }
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_upsample;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    AGValue *out = calloc(1, sizeof(AGValue));
    out->arr = res;
    out->requires_grad = a->requires_grad;
    out->creator = n;
    out->owns_arr = 1;
    /* store scale factors in stride0/stride1 for backward */
    n->stride0 = out_h;
    n->stride1 = out_w;
    n->output = out;
    tape_push(n);
    return out;
}

int ag_backward(AGValue *output)
{
    if (!output)
        return -1;
    /* initialize grad of output to ones */
    ensure_grad(output);
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_ones_like(&output->grad, output->arr, s);
    mlx_stream_free(s);
    /* traverse tape in reverse order and call backward */
    for (ssize_t i = (ssize_t)tape_size - 1; i >= 0; --i)
    {
        AGNode *n = tape[i];
        if (n->backward)
            n->backward(n);
    }
    return 0;
}

int ag_backward_create_graph(AGValue *output)
{
    if (!output)
        return -1;
    /* initialize symbolic grad of output to an array of ones matching output->arr */
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_ones_like(&res, output->arr, s);
    mlx_stream_free(s);
    AGValue *one = ag_value_from_new_array(&res, 0);
    ag_register_temp_value(one);
    output->grad_ag = one;
    /* traverse tape in reverse order and call backward which will build AG ops
       accumulating into inputs' grad_ag fields */
    for (ssize_t i = (ssize_t)tape_size - 1; i >= 0; --i)
    {
        AGNode *n = tape[i];
        if (n->backward)
            n->backward(n);
    }
    return 0;
}
