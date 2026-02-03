#include "autodiff.h"
#include "array_helpers.h"
#include "mlx_compat.h"
#include <mlx/c/mlx.h>
#include <mlx/c/memory.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>

/* Debug helper to print current GPU memory usage */
static void ag_debug_mem(const char *label) {
    static int debug_enabled = -1;
    if (debug_enabled < 0) {
        debug_enabled = getenv("FG_MEM_STEP_TRACE") ? 1 : 0;
    }
    if (!debug_enabled) return;
    size_t active = 0;
    mlx_get_active_memory(&active);
    fprintf(stderr, "[MEM_AG] %s: %.2f MB\n", label, (double)active / (1024.0 * 1024.0));
}

/* Minimal reverse-mode autodiff implementation.
   This records a simple tape of nodes and supports backprop for a
   subset of ops used by the models. It's intentionally minimal and
   incremental: new ops/backprop rules can be added as needed.
*/

struct AGValue {
    mlx_array arr;  /* value (does not own external pointer) */
    mlx_array grad; /* owned grad (allocated on demand); empty if .ctx==NULL */
    int requires_grad;
    int owns_arr;
    AGNode *creator;
    mlx_array
    *external_ptr; /* optional pointer to original mlx_array when wrapping */
    /* When building a create_graph gradient pass, this field holds an AGValue*
       representing the symbolic gradient of some scalar output wrt this value.
       It is NULL in normal reverse-mode usage. */
    AGValue *grad_ag;
};

typedef void (*backward_fn)(AGNode *node);

/* Forward declarations for backward functions so we can map names before
 * definitions. */
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

struct AGNode {
    backward_fn backward;
    AGValue **inputs;
    int n_inputs;
    AGValue *output;
    /* op-specific extras */
    int stride0, stride1, pad0, pad1, dil0, dil1, groups;
    int axis;  /* axis for sum_axis */
};

/* simple tape as list */
static AGNode **tape = NULL;
static size_t tape_size = 0;
/* temporary AGValue wrappers created during forward passes and registered via
   `ag_register_temp_value`. These are freed when `ag_reset_tape` is called. */
static AGValue **temp_values = NULL;
static size_t temp_values_size = 0;

static void temp_push(AGValue *v) {
    AGValue **tmp =
        realloc(temp_values, sizeof(AGValue *) * (temp_values_size + 1));
    if (!tmp)
        return;
    temp_values = tmp;
    temp_values[temp_values_size++] = v;
}

void ag_register_temp_value(AGValue *v) {
    if (!v)
        return;
    temp_push(v);
}

/* Helper: reduce gradient contribution to match target shape for broadcasting.
 * If contrib has more dimensions or larger sizes than target, sum over the
 * excess dimensions to produce a gradient matching target's shape. */
static AGValue *reduce_to_shape(AGValue *contrib, AGValue *target) {
    if (!contrib || !target)
        return contrib;
    if (!contrib->arr.ctx || !target->arr.ctx)
        return contrib;

    int contrib_ndim = mlx_array_ndim(contrib->arr);
    int target_ndim = mlx_array_ndim(target->arr);
    const int *contrib_shape = mlx_array_shape(contrib->arr);
    const int *target_shape = mlx_array_shape(target->arr);

    if (!contrib_shape || !target_shape)
        return contrib;

    /* If shapes match, no reduction needed */
    if (contrib_ndim == target_ndim) {
        int same = 1;
        for (int i = 0; i < contrib_ndim; i++) {
            if (contrib_shape[i] != target_shape[i]) {
                same = 0;
                break;
            }
        }
        if (same)
            return contrib;
    }

    /* Need to reduce contrib to match target shape.
     * Sum over leading dimensions if contrib has more dims,
     * and sum over dimensions where target is smaller (broadcast dims). */
    AGValue *result = contrib;

    /* First, if contrib has more dimensions, sum over leading axes */
    while (mlx_array_ndim(result->arr) > target_ndim) {
        AGValue *reduced = ag_sum_axis(result, 0, 0);  /* sum & drop leading axis */
        ag_register_temp_value(reduced);
        result = reduced;
    }

    /* Now both have same ndim, reduce where target dim is 1 but contrib dim > 1,
     * or where target shape is smaller. */
    const int *result_shape = mlx_array_shape(result->arr);
    int result_ndim = mlx_array_ndim(result->arr);

    /* Align from the right: compare corresponding dims */
    for (int i = result_ndim - 1; i >= 0; i--) {
        int target_idx = target_ndim - (result_ndim - i);
        if (target_idx < 0) {
            /* This axis doesn't exist in target, sum over it */
            AGValue *reduced = ag_sum_axis(result, i, 0);
            ag_register_temp_value(reduced);
            result = reduced;
            result_shape = mlx_array_shape(result->arr);
            result_ndim = mlx_array_ndim(result->arr);
            i = result_ndim;  /* restart from the end */
            continue;
        }
        if (target_shape[target_idx] == 1 && result_shape[i] > 1) {
            /* Target was broadcast on this axis, sum and keep dim */
            AGValue *reduced = ag_sum_axis(result, i, 1);
            ag_register_temp_value(reduced);
            result = reduced;
            result_shape = mlx_array_shape(result->arr);
        }
    }

    return result;
}

/* Helper: accumulate AGValue contributions into target->grad_ag (target may be
 * NULL). Handles shape reduction for broadcasting. */
static void accumulate_into_ag(AGValue *target, AGValue *contrib) {
    if (!contrib)
        return;
    if (!target)
        return;

    /* Reduce contrib to match target's shape if needed for broadcasting */
    AGValue *reduced_contrib = reduce_to_shape(contrib, target);

    if (!target->grad_ag) {
        target->grad_ag = reduced_contrib;
    } else {
        /* Before adding, ensure both grad_ag and reduced_contrib have compatible shapes.
         * If they don't match (e.g., one is 3D, other is 4D), we need to broadcast
         * one of them to match the other. Use the LARGER shape as the target. */
        int grad_ndim = mlx_array_ndim(target->grad_ag->arr);
        int contrib_ndim = mlx_array_ndim(reduced_contrib->arr);

        AGValue *grad_to_add = target->grad_ag;
        AGValue *contrib_to_add = reduced_contrib;

        if (grad_ndim != contrib_ndim) {
            mlx_stream s = mlx_default_gpu_stream_new();

            if (contrib_ndim > grad_ndim) {
                /* Broadcast grad_ag to match contrib shape */
                const int *contrib_shape = mlx_array_shape(reduced_contrib->arr);
                mlx_array grad_arr = target->grad_ag->arr;

                /* Expand dims at end to match ndim */
                for (int i = grad_ndim; i < contrib_ndim; i++) {
                    mlx_array tmp = mlx_array_new();
                    mlx_expand_dims(&tmp, grad_arr, -1, s);
                    if (grad_arr.ctx != target->grad_ag->arr.ctx)
                        mlx_array_free(grad_arr);
                    grad_arr = tmp;
                }

                int expanded_ndim = mlx_array_ndim(grad_arr);
                const int *expanded_shape = mlx_array_shape(grad_arr);
                fprintf(stderr, "  after expand: ndim=%d shape=", expanded_ndim);
                for (int i = 0; i < expanded_ndim; i++) fprintf(stderr, "%d ", expanded_shape[i]);
                fprintf(stderr, "\n  target shape=");
                for (int i = 0; i < contrib_ndim; i++) fprintf(stderr, "%d ", contrib_shape[i]);
                fprintf(stderr, "\n");

                mlx_array bcast = mlx_array_new();
                if (mlx_broadcast_to(&bcast, grad_arr, contrib_shape, contrib_ndim, s) == 0) {
                    grad_to_add = ag_value_from_new_array(&bcast, 0);
                    ag_register_temp_value(grad_to_add);
                    mlx_array_free(bcast);  /* free original after copy */
                } else {
                    fprintf(stderr, "  broadcast FAILED!\n");
                }
                if (grad_arr.ctx != target->grad_ag->arr.ctx)
                    mlx_array_free(grad_arr);
            } else {
                /* Broadcast contrib to match grad_ag shape */
                const int *grad_shape = mlx_array_shape(target->grad_ag->arr);
                mlx_array contrib_arr = reduced_contrib->arr;

                fprintf(stderr, "  expanding contrib from %d to %d dims\n", contrib_ndim, grad_ndim);

                for (int i = contrib_ndim; i < grad_ndim; i++) {
                    mlx_array tmp = mlx_array_new();
                    mlx_expand_dims(&tmp, contrib_arr, -1, s);
                    if (contrib_arr.ctx != reduced_contrib->arr.ctx)
                        mlx_array_free(contrib_arr);
                    contrib_arr = tmp;
                }

                int expanded_ndim = mlx_array_ndim(contrib_arr);
                const int *expanded_shape = mlx_array_shape(contrib_arr);
                fprintf(stderr, "  after expand: ndim=%d shape=", expanded_ndim);
                for (int i = 0; i < expanded_ndim; i++) fprintf(stderr, "%d ", expanded_shape[i]);
                fprintf(stderr, "\n  target shape=");
                for (int i = 0; i < grad_ndim; i++) fprintf(stderr, "%d ", grad_shape[i]);
                fprintf(stderr, "\n");

                mlx_array bcast = mlx_array_new();
                if (mlx_broadcast_to(&bcast, contrib_arr, grad_shape, grad_ndim, s) == 0) {
                    contrib_to_add = ag_value_from_new_array(&bcast, 0);
                    ag_register_temp_value(contrib_to_add);
                    mlx_array_free(bcast);  /* free original after copy */
                } else {
                    fprintf(stderr, "  broadcast FAILED!\n");
                }
                if (contrib_arr.ctx != reduced_contrib->arr.ctx)
                    mlx_array_free(contrib_arr);
            }
            mlx_stream_free(s);
        }

        AGValue *sum = ag_add(grad_to_add, contrib_to_add);
        ag_register_temp_value(sum);
        target->grad_ag = sum;
    }
}

static void tape_push(AGNode *n) {
    AGNode **tmp = realloc(tape, sizeof(AGNode *) * (tape_size + 1));
    if (!tmp)
        return;
    tape = tmp;
    tape[tape_size++] = n;
}

AGValue *ag_value_from_array(mlx_array *arr, int requires_grad) {
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
    v->owns_arr =
        0; /* by default, wrapping external arrays does not take ownership */
    v->external_ptr = arr; /* keep original pointer for possible in-place fixes */
    return v;
}

AGValue *ag_value_from_new_array(mlx_array *arr, int requires_grad) {
    AGValue *v = calloc(1, sizeof(AGValue));
    if (!v)
        return NULL;
    /* Make a deep copy of the provided array so the AGValue owns an
       independent buffer. This avoids reuse-after-free when callers free or
       reuse the original array/ctx while asynchronous kernels may still be
       operating on it. */
    if (arr) {
        v->arr = mlx_array_new();
        mlx_stream s = mlx_default_gpu_stream_new();
        if (mlx_copy(&v->arr, *arr, s) != 0) {
            mlx_stream_free(s);
            free(v);
            return NULL;
        }
        mlx_stream_free(s);
    } else {
        v->arr = mlx_array_new();
    }
    v->requires_grad = requires_grad;
    v->creator = NULL;
    v->owns_arr = 1; /* owns the copied array */
    v->external_ptr = NULL;
    /* Debug logging removed */
    return v;
}

void ag_value_free(AGValue *v) {
    if (!v)
        return;
    /* Debug: log AGValue freeing info to help trace double-free/ownership.
     * Use a runtime environment variable `FG_ENABLE_DEBUG` so verbose frees can
     * be enabled without rebuilding. This keeps debug prints available when
     * needed but silent by default for ASAN runs. */
    /* Debug logging removed */
    /* If this value was registered as a temporary, null out entries to avoid
       ag_reset_tape attempting to free it again. */
    if (temp_values) {
        for (size_t i = 0; i < temp_values_size; ++i) {
            if (temp_values[i] == v)
                temp_values[i] = NULL;
        }
    }
    /* Free gradient array if it exists */
    if (v->grad.ctx) {
        mlx_array_free(v->grad);
    }
    if (v->owns_arr && v->arr.ctx) {
        void *actx2 = v->arr.ctx;
        mlx_array_free(v->arr);
    }
    free(v);
}

/* Reset/free the tape and all nodes/outputs.
   Note: this frees AGNode and AGValue structures but does not free underlying
   mlx_array data owned outside the AGValue wrappers. */
void ag_reset_tape(void) {
    if (!tape)
        return;
    ag_debug_mem("ag_reset_tape: start");
    /* Free AGNode entries, their outputs, and their input arrays */
    for (size_t i = 0; i < tape_size; ++i) {
        AGNode *n = tape[i];
        if (!n)
            continue;
        /* Free the output AGValue created by this operation.
         * This is critical: each ag_ operation allocates an AGValue with
         * owns_arr=1, meaning it owns its underlying mlx_array. If we don't
         * free it here, the array leaks every batch. */
        if (n->output) {
            ag_value_free(n->output);
            n->output = NULL;
        }
        if (n->inputs)
            free(n->inputs);
        free(n);
    }
    free(tape);
    tape = NULL;
    tape_size = 0;
    ag_debug_mem("ag_reset_tape: after nodes");
    /* Free temporary AGValue wrappers that were registered during forward.
       These may own underlying mlx_array buffers (owns_arr) and will be
       freed by ag_value_free. This restores correct lifetime semantics so
       the tape no longer leaks memory. */
    if (temp_values) {
        for (size_t i = 0; i < temp_values_size; ++i) {
            AGValue *v = temp_values[i];
            if (v)
                ag_value_free(v);
        }
        free(temp_values);
        temp_values = NULL;
        temp_values_size = 0;
    }
    ag_debug_mem("ag_reset_tape: after temps");
}

mlx_array *ag_value_array(AGValue *v) {
    return v ? &v->arr : NULL;
}

AGValue *ag_scalar_float(float f) {
    mlx_array a = mlx_array_new_float(f);
    AGValue *v = ag_value_from_new_array(&a, 0);
    /* Free the original array - ag_value_from_new_array made a copy via mlx_copy
     * which increments the reference count. We need to release our local reference. */
    mlx_array_free(a);
    return v;
}

/* Helper: ensure grad array exists and is zeros */
static void ensure_grad(AGValue *v) {
    if (!v || !v->requires_grad)
        return;
    if (v->grad.ctx)
        return;
    mlx_stream s = mlx_default_gpu_stream_new();
    if (v->arr.ctx) {
        mlx_zeros_like(&v->grad, v->arr, s);
    }
    mlx_stream_free(s);
}

mlx_array *ag_value_get_grad(AGValue *v) {
    if (!v)
        return NULL;
    if (!v->grad.ctx)
        return NULL;
    return &v->grad;
}

AGValue *ag_value_get_grad_ag(AGValue *v) {
    if (!v)
        return NULL;
    return v->grad_ag;
}

/* Helper: Find an AGValue in the tape or temp_values that wraps the same
 * underlying mlx_array (by comparing ctx pointers) and has a computed gradient.
 * This allows collecting gradients from AGValues that were used in the
 * computation graph, even if the caller has different AGValue instances
 * wrapping the same underlying arrays. */
static AGValue *find_ag_with_matching_array_and_grad(mlx_array target_arr) {
    if (!target_arr.ctx)
        return NULL;
    
    /* Search temp_values first (these are the requires_grad=1 params used in forward) */
    if (temp_values) {
        for (size_t i = 0; i < temp_values_size; ++i) {
            AGValue *v = temp_values[i];
            if (v && v->arr.ctx == target_arr.ctx) {
                if (v->grad.ctx) {
                    return v;
                }
            }
        }
    }
    
    /* Search tape outputs */
    if (tape) {
        for (size_t i = 0; i < tape_size; ++i) {
            AGNode *n = tape[i];
            if (!n)
                continue;
            if (n->output && n->output->arr.ctx == target_arr.ctx) {
                if (n->output->grad.ctx) {
                    return n->output;
                }
            }
            /* Also check inputs */
            for (int j = 0; j < n->n_inputs; ++j) {
                AGValue *iv = n->inputs[j];
                if (iv && iv->arr.ctx == target_arr.ctx) {
                    if (iv->grad.ctx) {
                        return iv;
                    }
                }
            }
        }
    }
    
    return NULL;
}

/* DEBUG: Check grad value from found AGValue */
static void debug_check_matched_grad(AGValue *source, int idx) {
    if (!source || !source->grad.ctx) {
        fprintf(stderr, "[find_match] param[%d] source is NULL or grad.ctx is NULL\n", idx);
        return;
    }
    /* Check non-zero grad */
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array abs_arr = mlx_array_new();
    mlx_array sum_arr = mlx_array_new();
    mlx_abs(&abs_arr, source->grad, s);
    mlx_sum(&sum_arr, abs_arr, false, s);
    mlx_array_eval(sum_arr);
    float val = 0.0f;
    mlx_array_item_float32(&val, sum_arr);
    fprintf(stderr, "[find_match] param[%d] source->grad sum_abs=%.8g source->requires_grad=%d\n", 
            idx, val, source->requires_grad);
    mlx_array_free(abs_arr);
    mlx_array_free(sum_arr);
    mlx_stream_free(s);
}

int ag_collect_grads(AGValue **params, int n, mlx_array ***out_grads) {
    if (!params || n <= 0 || !out_grads)
        return -1;
    mlx_array **arr = NULL;
    if (mlx_alloc_mlx_array_ptrs(&arr, n) != 0)
        return -1;
    
    for (int i = 0; i < n; ++i) {
        AGValue *p = params[i];
        if (!p) {
            arr[i] = NULL;
            continue;
        }
        
        /* First try to find an AGValue with matching underlying array that has grad */
        AGValue *source = find_ag_with_matching_array_and_grad(p->arr);
        
        /* Debug: log what we found */
        if (i == 0) {
            if (source) {
                fprintf(stderr, "[ag_collect_grads] param[0] arr.ctx=%p found source=%p grad.ctx=%p req_grad=%d\n",
                        p->arr.ctx, (void*)source, source->grad.ctx, source->requires_grad);
                if (source->grad.ctx) {
                    mlx_stream s_dbg = mlx_default_gpu_stream_new();
                    mlx_array abs_dbg = mlx_array_new();
                    mlx_array sum_dbg = mlx_array_new();
                    mlx_abs(&abs_dbg, source->grad, s_dbg);
                    mlx_sum(&sum_dbg, abs_dbg, false, s_dbg);
                    mlx_array_eval(sum_dbg);
                    float grad_sum = 0.0f;
                    mlx_array_item_float32(&grad_sum, sum_dbg);
                    mlx_array_free(abs_dbg);
                    mlx_array_free(sum_dbg);
                    mlx_stream_free(s_dbg);
                    fprintf(stderr, "[ag_collect_grads] param[0] grad_sum_abs=%.8f\n", grad_sum);
                }
            } else {
                fprintf(stderr, "[ag_collect_grads] param[0] arr.ctx=%p NOT FOUND in tape/temps\n", p->arr.ctx);
            }
        }
        
        if (!source) {
            /* Fall back to original behavior: use the provided AGValue */
            ensure_grad(p);
            source = p;
        }
        
        if (!source->grad.ctx) {
            arr[i] = NULL;
            continue;
        }
        mlx_array *gptr = NULL;
        if (mlx_alloc_pod((void **)&gptr, sizeof(mlx_array), 1) != 0) {
            /* cleanup */
            for (int j = 0; j < i; ++j)
                if (arr[j]) {
                    mlx_array_free(*arr[j]);
                    mlx_free_pod((void **)&arr[j]);
                }
            mlx_free_mlx_array_ptrs(&arr, n);
            return -1;
        }
        /* Create a copy of the grad array so caller owns it independently */
        *gptr = mlx_array_new();
        mlx_stream s = mlx_default_gpu_stream_new();
        if (mlx_copy(gptr, source->grad, s) != 0) {
            mlx_stream_free(s);
            mlx_array_free(*gptr);
            mlx_free_pod((void **)&gptr);
            for (int j = 0; j < i; ++j)
                if (arr[j]) {
                    mlx_array_free(*arr[j]);
                    mlx_free_pod((void **)&arr[j]);
                }
            mlx_free_mlx_array_ptrs(&arr, n);
            return -1;
        }
        mlx_stream_free(s);
        arr[i] = gptr;
    }
    
    *out_grads = arr;
    return 0;
}

void ag_zero_grad_all(void) {
    for (size_t i = 0; i < tape_size; ++i) {
        AGNode *n = tape[i];
        if (n->output && n->output->grad.ctx) {
            mlx_stream s = mlx_default_gpu_stream_new();
            mlx_zeros_like(&n->output->grad, n->output->arr, s);
            mlx_stream_free(s);
        }
        for (int j = 0; j < n->n_inputs; ++j) {
            AGValue *iv = n->inputs[j];
            if (iv && iv->grad.ctx) {
                mlx_stream s = mlx_default_gpu_stream_new();
                mlx_zeros_like(&iv->grad, iv->arr, s);
                mlx_stream_free(s);
            }
        }
    }
}

/* Clear all grad_ag pointers on the tape (for after create-graph backward) */
void ag_clear_grad_ag_all(void) {
    for (size_t i = 0; i < tape_size; ++i) {
        AGNode *n = tape[i];
        if (n->output) {
            n->output->grad_ag = NULL;
        }
        for (int j = 0; j < n->n_inputs; ++j) {
            AGValue *iv = n->inputs[j];
            if (iv) {
                iv->grad_ag = NULL;
            }
        }
    }
}/* Backward implementations for ops */
/* helper: accumulate src into dst (dst += src). dst is pointer to mlx_array
   variable. If dst is empty, copy src into it. */
static int accumulate_into(mlx_array *dst, const mlx_array src,
                           const mlx_stream s) {
    if (!dst)
        return -1;
    if (!dst->ctx) {
        return mlx_copy(dst, src, s);
    }
    mlx_array tmp = mlx_array_new();
    int r = mlx_add(&tmp, *dst, src, s);
    if (r != 0) {
        detach_and_free(tmp);
        return r;
    }
    /* Copy result into dst and free the temporary to avoid leaks. */
    int rc = mlx_copy(dst, tmp, s);
    if (s.ctx)
        mlx_synchronize(s);
    detach_and_free(tmp);
    return rc;
}

/* Helper to reduce gradient to match a target shape (numeric version).
 * This is needed when backward through broadcast ops like add/mul where
 * one operand was broadcast to match the other. */
static mlx_array reduce_to_target_shape(mlx_array src, const mlx_array target, mlx_stream s) {
    int src_ndim = mlx_array_ndim(src);
    int tgt_ndim = mlx_array_ndim(target);
    const int *src_sh = mlx_array_shape(src);
    const int *tgt_sh = mlx_array_shape(target);

    /* If shapes already match, return src as-is */
    int match = (src_ndim == tgt_ndim);
    if (match) {
        for (int i = 0; i < src_ndim; i++) {
            if (src_sh[i] != tgt_sh[i]) {
                match = 0;
                break;
            }
        }
    }
    if (match) return src;

    mlx_array reduced = mlx_array_new();
    mlx_copy(&reduced, src, s);

    /* First, sum over leading axes if src has more dims */
    while (mlx_array_ndim(reduced) > tgt_ndim) {
        mlx_array tmp = mlx_array_new();
        mlx_sum_axis(&tmp, reduced, 0, false, s);
        mlx_array_free(reduced);
        reduced = tmp;
    }

    /* Now both have same ndim, sum where tgt dim is smaller or 1.
     * We need to be careful: after reducing with keepdims=0, ndim changes
     * and axis indices shift. Use a while loop and restart when needed. */
    int changed = 1;
    while (changed) {
        changed = 0;
        int r_ndim = mlx_array_ndim(reduced);
        const int *r_sh = mlx_array_shape(reduced);

        /* Compare from right (trailing dims) */
        for (int i = 0; i < r_ndim && i < tgt_ndim; i++) {
            int ri = r_ndim - 1 - i;   /* index in reduced, from right */
            int ti = tgt_ndim - 1 - i;  /* index in target, from right */
            if (tgt_sh[ti] < r_sh[ri]) {
                /* Need to reduce this axis. If tgt_sh[ti] == 1, keepdims=1 */
                int keepdims = (tgt_sh[ti] == 1);
                mlx_array tmp = mlx_array_new();
                mlx_sum_axis(&tmp, reduced, ri, keepdims, s);
                mlx_array_free(reduced);
                reduced = tmp;
                changed = 1;
                break;  /* restart the check with new shape */
            }
        }
    }

    return reduced;
}

static void bw_add(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    /* create_graph path: if out->grad_ag is present, build AG ops for grads */
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();
    /* Reduce gradients if broadcast was used in forward */
    mlx_array grad_a = reduce_to_target_shape(out->grad, a->arr, s);
    mlx_array grad_b = reduce_to_target_shape(out->grad, b->arr, s);
    accumulate_into(&a->grad, grad_a, s);
    accumulate_into(&b->grad, grad_b, s);
    if (grad_a.ctx != out->grad.ctx) mlx_array_free(grad_a);
    if (grad_b.ctx != out->grad.ctx) mlx_array_free(grad_b);
    mlx_stream_free(s);
}

/* Helper: print a short name for known backward functions */
static const char *bw_name(backward_fn f) {
    if (f == bw_add)
        return "add";
    if (f == bw_mul)
        return "mul";
    if (f == bw_tanh)
        return "tanh";
    if (f == bw_square)
        return "square";
    if (f == bw_sum_axis)
        return "sum_axis";
    if (f == bw_transpose)
        return "transpose";
    if (f == bw_matmul)
        return "matmul";
    if (f == bw_divide)
        return "divide";
    if (f == bw_sqrt)
        return "sqrt";
    if (f == bw_conv2d)
        return "conv2d";
    if (f == bw_conv_transpose)
        return "conv_transpose";
    if (f == bw_upsample)
        return "upsample";
    return "unknown_op";
}

static void bw_mul(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array tmp = mlx_array_new();
    if (mlx_multiply(&tmp, out->grad, b->arr, s) == 0) {
        /* Reduce tmp to match a's shape before accumulating */
        mlx_array reduced = reduce_to_target_shape(tmp, a->arr, s);
        accumulate_into(&a->grad, reduced, s);
        if (reduced.ctx != tmp.ctx) mlx_array_free(reduced);
    }
    mlx_array tmp2 = mlx_array_new();
    if (mlx_multiply(&tmp2, out->grad, a->arr, s) == 0) {
        /* Reduce tmp2 to match b's shape before accumulating */
        mlx_array reduced = reduce_to_target_shape(tmp2, b->arr, s);
        accumulate_into(&b->grad, reduced, s);
        if (reduced.ctx != tmp2.ctx) mlx_array_free(reduced);
    }
    detach_and_free(tmp);
    detach_and_free(tmp2);
    mlx_stream_free(s);
}

static void bw_tanh(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array out_sq = mlx_array_new();
    mlx_square(&out_sq, out->arr, s);
    mlx_array one = mlx_array_new_float(1.0f);
    mlx_array tmp = mlx_array_new();
    mlx_subtract(&tmp, one, out_sq, s);
    mlx_array tmp2 = mlx_array_new();
    mlx_multiply(&tmp2, tmp, out->grad, s);
    accumulate_into(&a->grad, tmp2, s);
    detach_and_free(out_sq);
    detach_and_free(tmp);
    detach_and_free(tmp2);
    detach_and_free(one);
    mlx_stream_free(s);
}

static void bw_square(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array two = mlx_array_new_float(2.0f);
    mlx_array tmp = mlx_array_new();
    mlx_multiply(&tmp, two, a->arr, s);
    mlx_array tmp2 = mlx_array_new();
    mlx_multiply(&tmp2, tmp, out->grad, s);
    accumulate_into(&a->grad, tmp2, s);
    detach_and_free(tmp);
    detach_and_free(tmp2);
    detach_and_free(two);
    mlx_stream_free(s);
}

static void bw_sum_axis(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;

    int reduced_axis = node->axis;  /* axis that was reduced (normalized positive) */

    if (out->grad_ag) {
        /* For create-graph, we must broadcast the gradient to the input shape.
         * When sum_axis was called with keepdims=0, the output has fewer dims
         * than the input. We need to expand_dims on the reduced axis before
         * broadcasting to restore the proper alignment.
         */
        AGValue *grad_ag = out->grad_ag;
        if (a && a->arr.ctx) {
            int in_ndim = mlx_array_ndim(a->arr);
            int out_ndim = mlx_array_ndim(out->arr);

            AGValue *expanded = grad_ag;
            /* If output has fewer dims, we need to expand dims at the axis that was reduced */
            if (out_ndim < in_ndim) {
                mlx_stream s = mlx_default_gpu_stream_new();
                mlx_array grad_arr = grad_ag->arr;

                /* Expand at the CORRECT axis that was reduced */
                mlx_array tmp = mlx_array_new();
                mlx_expand_dims(&tmp, grad_arr, reduced_axis, s);
                grad_arr = tmp;

                const int *in_shape = mlx_array_shape(a->arr);

                /* Now broadcast to input shape */
                mlx_array bcast = mlx_array_new();
                if (mlx_broadcast_to(&bcast, grad_arr, in_shape, in_ndim, s) == 0) {
                    expanded = ag_value_from_new_array(&bcast, 0);
                    ag_register_temp_value(expanded);
                    mlx_array_free(bcast);  /* free original after copy */
                }
                if (grad_arr.ctx != grad_ag->arr.ctx)
                    mlx_array_free(grad_arr);
                mlx_stream_free(s);
            } else {
                /* keepdims was 1, just multiply with ones */
                AGValue *ones = ag_ones_like(a);
                ag_register_temp_value(ones);
                expanded = ag_mul(grad_ag, ones);
                ag_register_temp_value(expanded);
            }
            accumulate_into_ag(a, expanded);
        }
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(a);
    mlx_stream s = mlx_default_gpu_stream_new();

    /* Numeric backward: broadcast out->grad to a->arr shape.
     * If out has fewer dims than input (keepdims=0 was used), we need to
     * expand dims first at the reduced axis before broadcasting. */
    int in_ndim = (int)mlx_array_ndim(a->arr);
    int out_ndim = (int)mlx_array_ndim(out->arr);
    const int *in_shape = mlx_array_shape(a->arr);

    mlx_array grad_to_broadcast = out->grad;
    int need_free_expanded = 0;

    if (out_ndim < in_ndim) {
        /* Expand dims at the CORRECT axis that was reduced */
        mlx_array tmp = mlx_array_new();
        if (mlx_expand_dims(&tmp, out->grad, reduced_axis, s) != 0) {
            mlx_stream_free(s);
            return;
        }
        grad_to_broadcast = tmp;
        need_free_expanded = 1;
    }

    mlx_array tiled = mlx_array_new();
    if (mlx_broadcast_to(&tiled, grad_to_broadcast, in_shape, in_ndim, s) == 0) {
        accumulate_into(&a->grad, tiled, s);
        detach_and_free(tiled);
    }
    if (need_free_expanded) {
        mlx_array_free(grad_to_broadcast);
    }
    mlx_stream_free(s);
}

/* Backward for ag_mean: grad_out / size broadcast to input shape. */
static void bw_mean(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out || !a)
        return;
    mlx_stream s = mlx_default_gpu_stream_new();
    /* Compute total element count (size) of the input */
    size_t ndim = mlx_array_ndim(a->arr);
    const int *shape = mlx_array_shape(a->arr);
    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
        size *= (size_t)shape[i];

    if (out->grad_ag) {
        /* Create-graph mode: (grad_out / size) broadcast to input shape. */
        AGValue *sz = ag_scalar_float((float)size);
        ag_register_temp_value(sz);
        AGValue *grad_scaled = ag_divide(out->grad_ag, sz);
        ag_register_temp_value(grad_scaled);
        AGValue *ones = ag_ones_like(a);
        ag_register_temp_value(ones);
        AGValue *bcast = ag_mul(grad_scaled, ones);
        ag_register_temp_value(bcast);
        accumulate_into_ag(a, bcast);
        mlx_stream_free(s);
        return;
    }
    if (!out->grad.ctx) {
        mlx_stream_free(s);
        return;
    }
    ensure_grad(a);
    /* Compute grad/size and broadcast to input shape */
    mlx_array sz_arr = mlx_array_new_float((float)size);
    mlx_array scaled = mlx_array_new();
    mlx_divide(&scaled, out->grad, sz_arr, s);
    mlx_array tiled = mlx_array_new();
    if (mlx_broadcast_to(&tiled, scaled, shape, ndim, s) == 0) {
        accumulate_into(&a->grad, tiled, s);
        detach_and_free(tiled);
    }
    detach_and_free(scaled);
    detach_and_free(sz_arr);
    mlx_stream_free(s);
}

/* Backward for ag_sum: broadcast grad (ones) to input shape.
 * grad_input = grad_output * ones_like(input) = grad_output broadcast to input shape */
static void bw_sum(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out || !a)
        return;
    mlx_stream s = mlx_default_gpu_stream_new();
    size_t ndim = mlx_array_ndim(a->arr);
    const int *shape = mlx_array_shape(a->arr);

    if (out->grad_ag) {
        /* Create-graph mode: broadcast grad_out to input shape. */
        AGValue *ones = ag_ones_like(a);
        ag_register_temp_value(ones);
        AGValue *bcast = ag_mul(out->grad_ag, ones);
        ag_register_temp_value(bcast);
        accumulate_into_ag(a, bcast);
        mlx_stream_free(s);
        return;
    }
    if (!out->grad.ctx) {
        mlx_stream_free(s);
        return;
    }
    ensure_grad(a);
    /* Broadcast grad to input shape */
    mlx_array tiled = mlx_array_new();
    if (mlx_broadcast_to(&tiled, out->grad, shape, ndim, s) == 0) {
        accumulate_into(&a->grad, tiled, s);
        detach_and_free(tiled);
    }
    mlx_stream_free(s);
}

static void bw_transpose(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array tmp = mlx_array_new();
    mlx_transpose(&tmp, out->grad, s);
    accumulate_into(&a->grad, tmp, s);
    detach_and_free(tmp);
    mlx_stream_free(s);
}

static void bw_matmul(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array b_t = mlx_array_new();
    mlx_transpose(&b_t, b->arr, s);
    mlx_array tmp = mlx_array_new();
    mlx_matmul(&tmp, out->grad, b_t, s);
    accumulate_into(&a->grad, tmp, s);
    detach_and_free(tmp);
    mlx_array a_t = mlx_array_new();
    mlx_transpose(&a_t, a->arr, s);
    mlx_array tmp2 = mlx_array_new();
    mlx_matmul(&tmp2, a_t, out->grad, s);
    accumulate_into(&b->grad, tmp2, s);
    detach_and_free(tmp2);
    detach_and_free(a_t);
    detach_and_free(b_t);
    mlx_stream_free(s);
}

static void bw_divide(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    AGValue *b = node->inputs[1];
    if (!out)
        return;
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();

    /* da += out_grad / b */
    mlx_array tmp = mlx_array_new();
    if (mlx_divide(&tmp, out->grad, b->arr, s) == 0) {
        accumulate_into(&a->grad, tmp, s);
    }
    detach_and_free(tmp);

    /* db += - out_grad * a / (b*b) */
    mlx_array bb = mlx_array_new();
    if (mlx_square(&bb, b->arr, s) == 0) {
        mlx_array tmp2 = mlx_array_new();
        if (mlx_multiply(&tmp2, out->grad, a->arr, s) == 0) {
            mlx_array tmp3 = mlx_array_new();
            if (mlx_divide(&tmp3, tmp2, bb, s) == 0) {
                mlx_array neg = mlx_array_new();
                mlx_array tmp4 = mlx_array_new_float(-1.0f);
                mlx_multiply(&neg, tmp3, tmp4, s);
                accumulate_into(&b->grad, neg, s);
                detach_and_free(neg);
                detach_and_free(tmp4);
            }
            detach_and_free(tmp3);
        }
        detach_and_free(tmp2);
    }
    detach_and_free(bb);
    mlx_stream_free(s);
}

static void bw_sqrt(AGNode *node) {
    AGValue *out = node->output;
    AGValue *a = node->inputs[0];
    if (!out)
        return;
    if (out->grad_ag) {
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
    mlx_stream s = mlx_default_gpu_stream_new();

    /* da += out_grad * (0.5 / out) */
    mlx_array half = mlx_array_new_float(0.5f);
    mlx_array tmp = mlx_array_new();
    if (mlx_multiply(&tmp, out->grad, half, s) == 0) {
        mlx_array tmp2 = mlx_array_new();
        if (mlx_divide(&tmp2, tmp, out->arr, s) == 0) {
            accumulate_into(&a->grad, tmp2, s);
            detach_and_free(tmp2);
        }
        detach_and_free(tmp);
    }
    detach_and_free(half);
    mlx_stream_free(s);
}

/* conv2d backward: compute grads for input and weight using conv_transpose &
   conv This implementation assumes grouped conv and uses mlx_conv_transpose2d
   for d_input and a simple conv-like accumulation for d_weight.
*/
static void bw_conv2d(AGNode *node) {
    AGValue *out = node->output;
    AGValue *input = node->inputs[0];
    AGValue *weight = node->inputs[1];
    if (!out)
        return;
    /* create_graph path: if out->grad_ag is present, build AG ops for d_input via
     * conv_transpose */
    if (out->grad_ag) {
        if (input) {
            /* MLX conv backward for input: wt_trans = swapaxes(wt, 0, -1)
             * This converts weight from (out_ch, KH, KW, in_ch) to (in_ch, KH, KW, out_ch)
             * matching Python MLX's Convolution::vjp behavior */
            mlx_stream s = mlx_default_gpu_stream_new();
            mlx_array wt_trans = mlx_array_new();
            int axes[4] = {3, 1, 2, 0}; /* swap axis 0 and 3 (last) */
            mlx_transpose_axes(&wt_trans, weight->arr, axes, 4, s);
            mlx_stream_free(s);

            AGValue *wv = ag_value_from_new_array(&wt_trans, 0);
            mlx_array_free(wt_trans);  /* free original after copy */
            ag_register_temp_value(wv);
            AGValue *d_in = ag_conv_transpose2d(
                                out->grad_ag, wv, node->stride0, node->stride1, node->pad0,
                                node->pad1, node->dil0, node->dil1, 0, 0, node->groups);
            ag_register_temp_value(d_in);
            accumulate_into_ag(input, d_in);
        }
        /* Weight grad will be produced by backward of the conv_transpose node when
         * final backward runs. */
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(input);
    ensure_grad(weight);
    mlx_stream s = mlx_default_gpu_stream_new();
    /* d_input = conv_transpose2d(out_grad, weight, stride, padding, dilation,
     * output_padding=0, groups) */
    mlx_array tmp_in = mlx_array_new();
    /* If channels don't line up, avoid calling into MLX conv_transpose which
       may trigger backend errors; instead ensure zeroed grads and skip numeric
       accumulation so we can continue execution for debugging. */
    const int *ogr = mlx_array_shape(out->grad);
    const int *wsh = mlx_array_shape(weight->arr);
    /* Attempt a defensive transpose fallback when weight is stored with
       channels in the first axis (e.g. MLX-layout [C_in,KH,KW,C_out]) but
       numeric backward expects the last axis to be output channels. If a
       transpose succeeds we'll use the transposed temporary for numeric
       conv_transpose and manual accumulation. Otherwise, fall back to the
       existing safe skip to avoid backend crashes. */
    mlx_array use_warr = weight->arr;
    int trans_used = 0;
    mlx_array trans = mlx_array_new();
    if (ogr && wsh) {
        int out_ch = ogr[3];
        int w_last = wsh[3];
        int w_first = wsh[0];
        if (out_ch != w_last) {
            if (w_first == out_ch) {
                int axes[4] = {3, 1, 2, 0};
                if (mlx_transpose_axes(&trans, weight->arr, axes, 4, s) == 0) {
                    /* use transposed view for subsequent numeric ops */
                    use_warr = trans;
                    trans_used = 1;
                    /* refresh shape pointer to transposed array */
                    wsh = mlx_array_shape(use_warr);
                } else {
                    /* transpose failed: free trans and fall through to skip */
                    detach_and_free(trans);
                }
            }
            /* If transpose wasn't used (or failed), and channels still mismatch,
               skip numeric accumulation as before. */
            if (!trans_used) {
                int wlast_now = wsh ? wsh[3] : -1;
                if (!input->grad.ctx)
                    mlx_zeros_like(&input->grad, input->arr, s);
                if (!weight->grad.ctx)
                    mlx_zeros_like(&weight->grad, weight->arr, s);
                if (trans_used) {
                    if (s.ctx)
                        mlx_synchronize(s);
                    detach_and_free(trans);
                }
                mlx_stream_free(s);
                return;
            }
        }
    }
    mlx_conv_transpose2d(&tmp_in, out->grad, use_warr, node->stride0,
                         node->stride1, node->pad0, node->pad1, node->dil0,
                         node->dil1, 0, 0, node->groups, s);
    accumulate_into(&input->grad, tmp_in, s);
    detach_and_free(tmp_in);
    
    /* Compute d_weight on GPU using mlx_conv_general.
     * Weight gradient = conv(input^T, grad_out^T) with appropriate settings.
     * For NHWC layout: input [N,H,W,Cin], grad [N,Ho,Wo,Cout], weight [Cout,Kh,Kw,Cin]
     * 
     * The weight gradient is computed by treating input as the "weight" and 
     * grad_output as the "input" in a correlation operation. We use mlx_conv_general
     * with flip=false to compute correlation instead of convolution.
     */
    const int *in_shape = mlx_array_shape(input->arr);
    const int *out_shape = mlx_array_shape(out->grad);
    const int *wshape = mlx_array_shape(use_warr);
    
    if (!in_shape || !out_shape || !wshape) {
        if (!weight->grad.ctx)
            mlx_zeros_like(&weight->grad, weight->arr, s);
        if (trans_used) detach_and_free(trans);
        else mlx_array_free(trans);
        mlx_stream_free(s);
        return;
    }
    
    int N = in_shape[0];
    int H_in = in_shape[1];
    int W_in = in_shape[2];
    int C_in = in_shape[3];
    int H_out = out_shape[1];
    int W_out = out_shape[2];
    int C_out = out_shape[3];
    int KH = wshape[1];
    int KW = wshape[2];
    
    /* Compute weight gradient on GPU using einsum-like tensor contractions.
     * d_weight[o,kh,kw,i] = sum over n,y,x of: input[n,y*s+kh*d-p,x*s+kw*d-p,i] * grad[n,y,x,o]
     * 
     * We can compute this by:
     * 1. Transpose input to [Cin, H, W, N]
     * 2. Transpose grad to [Cout, Ho, Wo, N]  
     * 3. Use conv2d with grad as "input" and transposed-input as "weight"
     * 4. Reshape/transpose result to match weight shape
     * 
     * However, for simplicity and correctness, we use the im2col approach on GPU:
     * - Extract patches from input using unfold
     * - Reshape grad appropriately  
     * - Compute outer product and sum
     * 
     * Actually, the simplest GPU approach is to use conv2d with swapped roles:
     * Transpose input: [N,H,W,Cin] -> [Cin,H,W,N]
     * Transpose grad:  [N,Ho,Wo,Cout] -> [Cout,Ho,Wo,N]
     * Then weight_grad = conv2d(transposed_input_as_input, transposed_grad_as_filter)
     * 
     * TODO: Implement proper GPU-based weight gradient using MLX's native
     * value_and_grad API. For now, weight grads are zeros which means only
     * input gradients propagate. This is a known limitation.
     */
    
    /* Initialize weight grad to zeros - proper implementation pending */
    if (!weight->grad.ctx) {
        mlx_zeros_like(&weight->grad, weight->arr, s);
    }
    
    /* Free the transposed weight array if we created one */
    if (trans_used) {
        detach_and_free(trans);
    } else {
        /* trans was allocated but never used - free the empty array */
        mlx_array_free(trans);
    }
    mlx_stream_free(s);
}

/* backward for conv_transpose node (computes grads for its inputs numerically)
 */
static void bw_conv_transpose(AGNode *node) {
    AGValue *out = node->output;
    AGValue *input = node->inputs[0];
    AGValue *weight = node->inputs[1];
    if (!out || !out->grad.ctx)
        return;
    ensure_grad(input);
    ensure_grad(weight);
    mlx_stream s = mlx_default_gpu_stream_new();
    
    /* d_input = conv2d(out_grad, weight) approximately */
    mlx_array tmp_in = mlx_array_new();
    safe_mlx_conv2d(&tmp_in, out->grad, weight->arr, node->stride0, node->stride1,
                    node->pad0, node->pad1, node->dil0, node->dil1, node->groups,
                    s);
    accumulate_into(&input->grad, tmp_in, s);
    detach_and_free(tmp_in);
    
    /* Compute weight gradient on GPU (same approach as bw_conv2d) */
    /* TODO: Implement proper weight gradient. For now, zeros. */
    if (!weight->grad.ctx) {
        mlx_zeros_like(&weight->grad, weight->arr, s);
    }
    
    mlx_stream_free(s);
}

static void bw_upsample(AGNode *node) {
    AGValue *out = node->output;
    AGValue *in = node->inputs[0];
    if (!out)
        return;
    /* create_graph path: downsample gradient back to input shape */
    if (out->grad_ag) {
        if (in && in->arr.ctx) {
            /* Get input shape to determine target size for downsampling */
            const int *in_shape = mlx_array_shape(in->arr);
            if (in_shape) {
                int target_h = in_shape[1];
                int target_w = in_shape[2];
                /* Use ag_upsample to resize gradient to input shape
                 * (downsample if out > in, upsample if out < in) */
                AGValue *resized_grad = ag_upsample(out->grad_ag, target_h, target_w, "linear", 1);
                ag_register_temp_value(resized_grad);
                accumulate_into_ag(in, resized_grad);
            }
        }
        return;
    }
    if (!out->grad.ctx)
        return;
    ensure_grad(in);
    /* numeric backward: sum over repeat blocks to produce input grad */
    mlx_stream s = mlx_default_gpu_stream_new();
    const int *in_shape = mlx_array_shape(in->arr);
    const int *out_shape = mlx_array_shape(out->grad);  /* Use out->grad shape, not out->arr */
    if (!in_shape || !out_shape) {
        mlx_stream_free(s);
        return;
    }
    int N = in_shape[0];
    int H_in = in_shape[1];
    int W_in = in_shape[2];
    int C = in_shape[3];
    int H_out = out_shape[1];
    int W_out = out_shape[2];
    int scale_h = H_out / H_in;
    int scale_w = W_out / W_in;
    /* Only perform numeric upsample accumulation if host buffers are available.
     */
    bool ok_out3 = false, ok_in3 = false;
    int r1 = _mlx_array_is_available(&ok_out3, out->grad);
    int r2 = _mlx_array_is_available(&ok_in3, in->arr);
    if (r1 == 0 && ok_out3 && r2 == 0 && ok_in3) {
        float *in_grad = NULL;
        if (!in->grad.ctx) {
            mlx_zeros_like(&in->grad, in->arr, s);
        }
        in_grad = (float *)mlx_array_data_float32(in->grad);
        const float *outg = mlx_array_data_float32(out->grad);
        if (!in_grad || !outg) {
            mlx_stream_free(s);
            return;
        }
        size_t in_total = (size_t)N * H_in * W_in * C;
        for (size_t i = 0; i < in_total; ++i)
            in_grad[i] = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int hi = 0; hi < H_in; ++hi) {
                for (int wi = 0; wi < W_in; ++wi) {
                    for (int c = 0; c < C; ++c) {
                        float sum = 0.0f;
                        for (int sh = 0; sh < scale_h; ++sh) {
                            int y = hi * scale_h + sh;
                            for (int sw = 0; sw < scale_w; ++sw) {
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
    } else {
        /* Ensure grad array exists but skip manual backward accumulation when
           host buffers are unavailable. */
        if (!in->grad.ctx) {
            mlx_zeros_like(&in->grad, in->arr, s);
        }
        mlx_stream_free(s);
        return;
    }
}

/* AG slice op */
AGValue *ag_slice(AGValue *a, const int *start, const int *stop, int ndim) {
    if (!a || !start || !stop || ndim <= 0)
        return NULL;
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array res = mlx_array_new();
    if (mlx_slice(&res, a->arr, (int *)start, ndim, (int *)stop, ndim, NULL, 0,
                  s) != 0) {
        mlx_stream_free(s);
        return NULL;
    }
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward =
        NULL; /* numeric fallback handled in bw_generic via ag_backward */
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

/* AG pad op */
AGValue *ag_pad(AGValue *a, const int *axes, int n_axes, const int *low_pad,
                int low_len, const int *high_pad, int high_len, float pad_val,
                const char *mode) {
    if (!a)
        return NULL;
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array padval = mlx_array_new_float(pad_val);
    mlx_array res = mlx_array_new();
    if (mlx_pad(&res, a->arr, (int *)axes, n_axes, (int *)low_pad, low_len,
                (int *)high_pad, high_len, padval, mode ? mode : "constant",
                s) != 0) {
        detach_and_free(padval);
        mlx_stream_free(s);
        return NULL;
    }
    detach_and_free(padval);
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
    tape_push(n);
    return out;
}

/* AG tile op */
AGValue *ag_tile(AGValue *a, const int *reps, int ndim) {
    if (!a || !reps || ndim <= 0)
        return NULL;
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array res = mlx_array_new();
    if (mlx_tile(&res, a->arr, (int *)reps, ndim, s) != 0) {
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
    tape_push(n);
    return out;
}

/* AG concatenate op */
AGValue *ag_concatenate(AGValue **parts, int n_parts, int axis) {
    if (!parts || n_parts <= 0)
        return NULL;
    mlx_vector_array vec = mlx_vector_array_new();
    for (int i = 0; i < n_parts; ++i) {
        mlx_array *ar = ag_value_array(parts[i]);
        mlx_vector_array_append_value(vec, *ar);
    }
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array res = mlx_array_new();
    if (mlx_concatenate_axis(&res, vec, axis, s) != 0) {
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
    tape_push(n);
    return out;
}

/* helpers to create nodes */
static AGValue *make_unary(AGValue *a, int requires_grad, backward_fn bw,
                           int n_inputs) {
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw;
    n->n_inputs = 1;
    n->inputs = calloc(1, sizeof(AGValue *));
    n->inputs[0] = a;
    /* create output array via performing corresponding mlx op must be done by
     * caller; here we assume output already set */
    AGValue *out = calloc(1, sizeof(AGValue));
    out->requires_grad = requires_grad;
    out->creator = n;
    out->owns_arr = 1; /* outputs allocated by ops own their arrays */
    n->output = out;
    tape_push(n);
    return out;
}

AGValue *ag_add(AGValue *a, AGValue *b) {
    if (!a || !b)
        return NULL;

    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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
    tape_push(n);
    return out;
}

AGValue *ag_sub(AGValue *a, AGValue *b) {
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_subtract(&res, a->arr, b->arr, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward =
        bw_add; /* subtraction backward similar with signs; simplified */
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
    tape_push(n);
    return out;
}

AGValue *ag_mul(AGValue *a, AGValue *b) {
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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
    out->owns_arr = 1;
    n->output = out;
    tape_push(n);
    return out;
}

AGValue *ag_tanh(AGValue *a) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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
    tape_push(n);
    return out;
}

AGValue *ag_square(AGValue *a) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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
    tape_push(n);
    return out;
}

AGValue *ag_sum_axis(AGValue *a, int axis, int keepdims) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    /* Normalize negative axis to positive */
    int in_ndim = (int)mlx_array_ndim(a->arr);
    int norm_axis = axis < 0 ? in_ndim + axis : axis;
    mlx_sum_axis(&res, a->arr, axis, keepdims, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_sum_axis;
    n->axis = norm_axis;  /* store which axis was reduced */
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

AGValue *ag_mean(AGValue *a) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    /* mlx_mean with keepdims=false reduces all elements to a scalar */
    mlx_mean(&res, a->arr, false, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_mean;
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

AGValue *ag_sum(AGValue *a) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    /* mlx_sum with keepdims=false reduces all elements to a scalar */
    mlx_sum(&res, a->arr, false, s);
    mlx_stream_free(s);
    AGNode *n = calloc(1, sizeof(AGNode));
    n->backward = bw_sum;
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

AGValue *ag_transpose(AGValue *a) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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
    out->owns_arr = 1;
    n->output = out;
    tape_push(n);
    return out;
}

AGValue *ag_matmul(AGValue *a, AGValue *b) {
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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
    tape_push(n);
    return out;
}

AGValue *ag_conv2d(AGValue *input, AGValue *weight, int stride0, int stride1,
                   int pad0, int pad1, int dil0, int dil1, int groups) {
    if (!input || !weight)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    safe_mlx_conv2d(&res, input->arr, weight->arr, stride0, stride1, pad0, pad1,
                    dil0, dil1, groups, s);
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
    tape_push(n);
    return out;
}

AGValue *ag_divide(AGValue *a, AGValue *b) {
    if (!a || !b)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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
    tape_push(n);
    return out;
}

AGValue *ag_sqrt(AGValue *a) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
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

AGValue *ag_leaky_relu(AGValue *a, float negative_slope) {
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

AGValue *ag_conv_transpose2d(AGValue *input, AGValue *weight, int stride0,
                             int stride1, int pad0, int pad1, int dil0,
                             int dil1, int out_pad0, int out_pad1, int groups) {
    if (!input || !weight)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_conv_transpose2d(&res, input->arr, weight->arr, stride0, stride1, pad0,
                         pad1, dil0, dil1, out_pad0, out_pad1, groups, s);
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

AGValue *ag_ones_like(AGValue *a) {
    if (!a)
        return NULL;
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_ones_like(&res, a->arr, s);
    mlx_stream_free(s);
    AGValue *out = ag_value_from_new_array(&res, 0);
    mlx_array_free(res);  /* free original after copy */
    return out;
}

AGValue *ag_upsample(AGValue *a, int out_h, int out_w, const char *mode,
                     int align_corners) {
    if (!a)
        return NULL;
    /* perform underlying mlx upsample */
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array in_arr = a->arr;
    MLXUpsample *u =
        mlx_upsample_create(out_h, out_w, mode ? mode : "linear", align_corners);
    mlx_array res = mlx_array_new();
    if (u) {
        mlx_array_t tmp = mlx_upsample_forward(u, in_arr);
        mlx_upsample_free(u);
        res = tmp;
    } else {
        /* fallback: return copy of input */
        mlx_copy(&res, in_arr, s);
    }

    /* Check if upsample returned the same array as input (identity case).
     * If so, create a deep copy to ensure separate ownership.
     * Note: We do NOT free res here because when same_array is true,
     * res is an alias/view of in_arr which we don't own. */
    int same_array = (res.ctx == in_arr.ctx);
    if (same_array) {
        mlx_array copy = mlx_array_new();
        mlx_copy(&copy, res, s);
        res = copy;  /* res was a view, no need to free */
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

int ag_backward(AGValue *output) {
    if (!output)
        return -1;
    /* initialize grad of output to ones */
    ensure_grad(output);
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_ones_like(&output->grad, output->arr, s);
    mlx_stream_free(s);
    /* traverse tape in reverse order and call backward */
    for (ssize_t i = (ssize_t)tape_size - 1; i >= 0; --i) {
        AGNode *n = tape[i];
        if (!n)
            continue;
        if (n->backward)
            n->backward(n);
    }
    return 0;
}
int ag_backward_create_graph(AGValue *output) {
    if (!output)
        return -1;
    /* initialize symbolic grad of output to an array of ones matching output->arr
     */
    mlx_array res = mlx_array_new();
    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_ones_like(&res, output->arr, s);
    mlx_stream_free(s);
    AGValue *one = ag_value_from_new_array(&res, 0);
    mlx_array_free(res);  /* free original after copy */
    ag_register_temp_value(one);
    output->grad_ag = one;
    /* traverse tape in reverse order and call backward which will build AG ops
       accumulating into inputs' grad_ag fields */
    for (ssize_t i = (ssize_t)tape_size - 1; i >= 0; --i) {
        AGNode *n = tape[i];
        if (n->backward)
            n->backward(n);
    }
    return 0;
}
