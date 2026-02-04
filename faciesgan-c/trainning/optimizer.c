#include "optimizer.h"
#include "array_helpers.h"
#include "scheduler.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
/* IO helpers for .npz serialization */
#include "io/npz_create.h"
#include "io/npz_unzip.h"
/* Note: includes that reference mlx-c are left unchanged (mlx-c submodule) */
#include "../mlx-c/mlx/c/io.h"
#include "../mlx-c/mlx/c/io_types.h"
#include <mlx/c/transforms.h>

/* Global default Adam hyperparameters (modifiable by launcher). */
static float g_def_beta1 = 0.9f;
static float g_def_beta2 = 0.999f;
static float g_def_eps = 1e-8f;
/* Global default for bias-correction (0 = disabled, 1 = enabled). */
static int g_def_bias_correction = 0;
/* global default weight decay for AdamW */
static float g_def_weight_decay = 0.0f;

struct MLXOptimizer {
    float lr;
    /* optional attached scheduler to sample LR per-step */
    struct MLXScheduler *attached_scheduler;
    /* last LR used (for inspection/testing) */
    float last_used_lr;
    /* Adam hyperparameters */
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int bias_correction; /* whether to apply bias correction */
    int step;            /* time step for bias correction */

    /* per-parameter state (allocated on first step) */
    int param_n;
    mlx_array **m; /* first moment */
    mlx_array **v; /* second moment */
};

MLXOptimizer *mlx_adam_create(float lr, float beta1, float beta2, float eps) {
    MLXOptimizer *o = NULL;
    if (mlx_alloc_pod((void **)&o, sizeof(MLXOptimizer), 1) != 0)
        return NULL;
    o->lr = lr;
    o->beta1 = beta1;
    o->beta2 = beta2;
    o->eps = eps;
    o->bias_correction = g_def_bias_correction;
    o->weight_decay = g_def_weight_decay;
    o->attached_scheduler = NULL;
    o->last_used_lr = o->lr;
    o->step = 0;
    o->param_n = 0;
    o->m = NULL;
    o->v = NULL;
    return o;
}

MLXOptimizer *mlx_adam_create_ext(float lr, float beta1, float beta2, float eps,
                                  int bias_correction, float weight_decay) {
    MLXOptimizer *o = NULL;
    if (mlx_alloc_pod((void **)&o, sizeof(MLXOptimizer), 1) != 0)
        return NULL;
    o->lr = lr;
    o->beta1 = beta1;
    o->beta2 = beta2;
    o->eps = eps;
    o->bias_correction = bias_correction ? 1 : 0;
    o->weight_decay = weight_decay;
    o->attached_scheduler = NULL;
    o->last_used_lr = o->lr;
    o->step = 0;
    o->param_n = 0;
    o->m = NULL;
    o->v = NULL;
    return o;
}

/* Convenience: create an Adam optimizer using the current global defaults. */
MLXOptimizer *mlx_adam_create_with_defaults(float lr) {
    return mlx_adam_create(lr, g_def_beta1, g_def_beta2, g_def_eps);
}

void mlx_adam_free(MLXOptimizer *opt) {
    if (!opt)
        return;
    if (opt->m) {
        mlx_free_mlx_array_ptrs(&opt->m, opt->param_n);
    }
    if (opt->v) {
        mlx_free_mlx_array_ptrs(&opt->v, opt->param_n);
    }
    mlx_free_pod((void **)&opt);
}

void mlx_optimizer_eval_state(MLXOptimizer *opt) {
    if (!opt)
        return;
    /* Batch evaluate all optimizer state arrays using mlx_eval() for better
       performance. This mirrors Python's mx.eval(optimizer.state) pattern. */
    mlx_vector_array vec = mlx_vector_array_new();

    /* Collect all m (first moment) arrays */
    if (opt->m) {
        for (int i = 0; i < opt->param_n; ++i) {
            if (opt->m[i] && opt->m[i]->ctx) {
                mlx_vector_array_append_value(vec, *opt->m[i]);
            }
        }
    }
    /* Collect all v (second moment) arrays */
    if (opt->v) {
        for (int i = 0; i < opt->param_n; ++i) {
            if (opt->v[i] && opt->v[i]->ctx) {
                mlx_vector_array_append_value(vec, *opt->v[i]);
            }
        }
    }

    /* Batch evaluate all optimizer state at once */
    if (mlx_vector_array_size(vec) > 0) {
        mlx_eval(vec);
    }
    mlx_vector_array_free(vec);
}

int mlx_adam_step(MLXOptimizer *opt, mlx_array **params, mlx_array **grads,
                  int n) {
    if (!opt)
        return -1;
    if (!params || !grads)
        return -1;

    /* Lazy init per-parameter state on first call (or explicit init via wrapper)
    mlx_stream s = mlx_default_gpu_stream_new();
    if (opt->param_n == 0) {
        mlx_stream tmp_s = s; /* pass stream into helper */
        mlx_optimizer_init_from_params(opt, params, n);
    }

    opt->step += 1; /* increment time step */

    /* determine effective LR: prefer attached scheduler if present */
    float effective_lr = opt->lr;
    if (opt->attached_scheduler) {
        float tmp_lr[4] = {0.0f, 0.0f, 0.0f, 0.0f};  /* Initialize to avoid garbage */
        int got = mlx_scheduler_lr_for_step(opt->attached_scheduler, opt->step,
                                            tmp_lr, 1);
        if (got > 0)
            effective_lr = tmp_lr[0];
    }
    opt->last_used_lr = effective_lr;

    /* Pre-allocate scalar temporaries for reuse to reduce allocation churn */
    mlx_array scal_b1 = mlx_array_new();
    mlx_array scal_1mb1 = mlx_array_new();
    mlx_array scal_b2 = mlx_array_new();
    mlx_array scal_1mb2 = mlx_array_new();
    mlx_array eps_arr = mlx_array_new();
    mlx_array lr_arr = mlx_array_new();
    mlx_array decay_scale_arr = mlx_array_new();
    if (opt->beta1 != 0.0f) {
        mlx_array_free(scal_b1);
        scal_b1 = mlx_array_new_float(opt->beta1);
        mlx_array_free(scal_1mb1);
        scal_1mb1 = mlx_array_new_float(1.0f - opt->beta1);
    }
    if (opt->beta2 != 0.0f) {
        mlx_array_free(scal_b2);
        scal_b2 = mlx_array_new_float(opt->beta2);
        mlx_array_free(scal_1mb2);
        scal_1mb2 = mlx_array_new_float(1.0f - opt->beta2);
    }
    mlx_array_free(eps_arr);
    eps_arr = mlx_array_new_float(opt->eps);
    mlx_array_free(lr_arr);
    lr_arr = mlx_array_new_float(effective_lr);
    if (opt->weight_decay != 0.0f) {
        mlx_array_free(decay_scale_arr);
        decay_scale_arr = mlx_array_new_float(opt->lr * opt->weight_decay);
    }

    /* Force eval of all gradients before using them to ensure lazy computations
     * are materialized. This is critical for MLX's lazy evaluation model. */
    /* Force eval of all gradients in-place before using them. */
    for (int i = 0; i < n; ++i) {
        if (grads[i] && grads[i]->ctx) {
            mlx_array_eval(*grads[i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        mlx_array *p = params[i];
        mlx_array *g = grads[i];
        if (!p || !g)
            continue;

        /* ensure state arrays exist for this param */
        if (!opt->m[i] || !opt->v[i])
            continue;

        /* m = beta1 * m + (1-beta1) * g */
        mlx_array tmp1 = mlx_array_new();
        mlx_array tmp2 = mlx_array_new();
        if (mlx_multiply(&tmp1, *opt->m[i], scal_b1, s) != 0) {
            mlx_array_free(tmp1);
            mlx_array_free(tmp2);
            continue;
        }
        if (mlx_multiply(&tmp2, *g, scal_1mb1, s) != 0) {
            mlx_array_free(tmp1);
            mlx_array_free(tmp2);
            continue;
        }
        mlx_array m_new = mlx_array_new();
        if (mlx_add(&m_new, tmp1, tmp2, s) != 0) {
            mlx_array_free(tmp1);
            mlx_array_free(tmp2);
            mlx_array_free(m_new);
            continue;
        }
        mlx_array_free(tmp1);
        mlx_array_free(tmp2);
        mlx_array_set(opt->m[i], m_new);
        mlx_array_free(m_new);

        /* v = beta2 * v + (1-beta2) * (g * g) */
        mlx_array g_sq = mlx_array_new();
        if (mlx_multiply(&g_sq, *g, *g, s) != 0) {
            mlx_array_free(g_sq);
            continue;
        }
        mlx_array tmp3 = mlx_array_new();
        mlx_array tmp4 = mlx_array_new();
        if (mlx_multiply(&tmp3, *opt->v[i], scal_b2, s) != 0) {
            mlx_array_free(g_sq);
            mlx_array_free(tmp3);
            mlx_array_free(tmp4);
            continue;
        }
        if (mlx_multiply(&tmp4, g_sq, scal_1mb2, s) != 0) {
            mlx_array_free(g_sq);
            mlx_array_free(tmp3);
            mlx_array_free(tmp4);
            continue;
        }
        mlx_array v_new = mlx_array_new();
        if (mlx_add(&v_new, tmp3, tmp4, s) != 0) {
            mlx_array_free(g_sq);
            mlx_array_free(tmp3);
            mlx_array_free(tmp4);
            mlx_array_free(v_new);
            continue;
        }
        mlx_array_free(g_sq);
        mlx_array_free(tmp3);
        mlx_array_free(tmp4);
        mlx_array_set(opt->v[i], v_new);
        mlx_array_free(v_new);

        /* compute the `update` array (lr * adjusted moment / denom) */
        mlx_array update = mlx_array_new();
        if (opt->bias_correction) {
            /* bias-corrected moments */
            float bias_correction1 = 1.0f - powf(opt->beta1, (float)opt->step);
            float bias_correction2 = 1.0f - powf(opt->beta2, (float)opt->step);
            mlx_array denom1 = mlx_array_new_float(bias_correction1);
            mlx_array denom2 = mlx_array_new_float(bias_correction2);
            mlx_array m_hat = mlx_array_new();
            mlx_array v_hat = mlx_array_new();
            if (mlx_divide(&m_hat, *opt->m[i], denom1, s) != 0) {
                mlx_array_free(denom1);
                mlx_array_free(denom2);
                mlx_array_free(m_hat);
                mlx_array_free(v_hat);
                continue;
            }
            if (mlx_divide(&v_hat, *opt->v[i], denom2, s) != 0) {
                mlx_array_free(denom1);
                mlx_array_free(denom2);
                mlx_array_free(m_hat);
                mlx_array_free(v_hat);
                continue;
            }
            mlx_array_free(denom1);
            mlx_array_free(denom2);

            /* denom = sqrt(v_hat) + eps */
            mlx_array sqrt_v = mlx_array_new();
            if (mlx_sqrt(&sqrt_v, v_hat, s) != 0) {
                mlx_array_free(m_hat);
                mlx_array_free(v_hat);
                mlx_array_free(sqrt_v);
                continue;
            }
            mlx_array denom = mlx_array_new();
            if (mlx_add(&denom, sqrt_v, eps_arr, s) != 0) {
                mlx_array_free(m_hat);
                mlx_array_free(v_hat);
                mlx_array_free(sqrt_v);
                mlx_array_free(denom);
                continue;
            }
            mlx_array_free(sqrt_v);

            /* update = (m_hat / denom) * lr */
            mlx_array num = mlx_array_new();
            if (mlx_divide(&num, m_hat, denom, s) != 0) {
                mlx_array_free(m_hat);
                mlx_array_free(v_hat);
                mlx_array_free(denom);
                mlx_array_free(num);
                continue;
            }
            if (mlx_multiply(&update, num, lr_arr, s) != 0) {
                mlx_array_free(num);
                mlx_array_free(m_hat);
                mlx_array_free(v_hat);
                mlx_array_free(denom);
                mlx_array_free(update);
                continue;
            }
            mlx_array_free(num);
            mlx_array_free(m_hat);
            mlx_array_free(v_hat);
            mlx_array_free(denom);
        } else {
            /* No bias correction: simple Adam-style step */
            /* denom = sqrt(v) + eps */
            mlx_array sqrt_v = mlx_array_new();
            if (mlx_sqrt(&sqrt_v, *opt->v[i], s) != 0) {
                mlx_array_free(sqrt_v);
                continue;
            }
            mlx_array denom = mlx_array_new();
            if (mlx_add(&denom, sqrt_v, eps_arr, s) != 0) {
                mlx_array_free(sqrt_v);
                mlx_array_free(denom);
                continue;
            }
            mlx_array_free(sqrt_v);

            mlx_array num = mlx_array_new();
            if (mlx_divide(&num, *opt->m[i], denom, s) != 0) {
                mlx_array_free(denom);
                mlx_array_free(num);
                continue;
            }
            if (mlx_multiply(&update, num, lr_arr, s) != 0) {
                mlx_array_free(num);
                mlx_array_free(denom);
                mlx_array_free(update);
                continue;
            }
            mlx_array_free(num);
            mlx_array_free(denom);
        }

        /* new_p = p - update */
        /* Apply AdamW decoupled weight decay: add lr * weight_decay * p to update
        if (opt->weight_decay != 0.0f) {
            mlx_array decay = mlx_array_new();
            if (mlx_multiply(&decay, *p, decay_scale_arr, s) == 0) {
                mlx_array update2 = mlx_array_new();
                if (mlx_add(&update2, update, decay, s) == 0) {
                    mlx_array_free(update);
                    update = update2;
                } else {
                    mlx_array_free(update2);
                }
                mlx_array_free(decay);
            }
        }

        mlx_array new_p = mlx_array_new();
        if (mlx_subtract(&new_p, *p, update, s) != 0) {
            mlx_array_free(update);
            mlx_array_free(new_p);
            continue;
        }
        /* Evaluate new_p before overwriting p to ensure the lazy computation
         * graph is materialized. Otherwise, new_p might still reference the
         * old p value and get invalidated when we overwrite p. */
        mlx_array_eval(new_p);

        mlx_array_set(p, new_p);

        mlx_array_free(new_p);
        mlx_array_free(update);
    }

    /* Free shared scalar temporaries */
    mlx_array_free(scal_b1);
    mlx_array_free(scal_1mb1);
    mlx_array_free(scal_b2);
    mlx_array_free(scal_1mb2);
    mlx_array_free(eps_arr);
    mlx_array_free(lr_arr);
    mlx_array_free(decay_scale_arr);

    mlx_stream_free(s);
    return 0;
}

void mlx_optimizer_set_global_adam_params(float beta1, float beta2, float eps) {
    g_def_beta1 = beta1;
    g_def_beta2 = beta2;
    g_def_eps = eps;
}

void mlx_optimizer_set_global_adam_bias_correction(int enabled) {
    g_def_bias_correction = enabled ? 1 : 0;
}

void mlx_optimizer_set_global_adam_weight_decay(float weight_decay) {
    g_def_weight_decay = weight_decay;
}

void mlx_optimizer_get_global_adam_params(float *beta1, float *beta2,
        float *eps) {
    if (beta1)
        *beta1 = g_def_beta1;
    if (beta2)
        *beta2 = g_def_beta2;
    if (eps)
        *eps = g_def_eps;
}

int mlx_optimizer_get_global_adam_bias_correction(void) {
    return g_def_bias_correction;
}

float mlx_optimizer_get_global_adam_weight_decay(void) {
    return g_def_weight_decay;
}

void mlx_optimizer_set_lr(MLXOptimizer *opt, float lr) {
    if (!opt)
        return;
    opt->lr = lr;
}

void mlx_optimizer_attach_scheduler(MLXOptimizer *opt, struct MLXScheduler *s) {
    if (!opt)
        return;
    opt->attached_scheduler = s;
}

struct MLXScheduler *mlx_optimizer_get_attached_scheduler(MLXOptimizer *opt) {
    if (!opt)
        return NULL;
    return opt->attached_scheduler;
}

float mlx_optimizer_get_last_used_lr(MLXOptimizer *opt) {
    if (!opt)
        return 0.0f;
    return opt->last_used_lr;
}

float mlx_optimizer_get_lr(MLXOptimizer *opt) {
    if (!opt)
        return 0.0f;
    return opt->lr;
}

void mlx_optimizer_init_from_params(MLXOptimizer *opt, mlx_array **params,
                                    int n) {
    if (!opt || !params || n <= 0)
        return;
    mlx_stream s = mlx_default_gpu_stream_new();
    opt->param_n = n;
    if (mlx_alloc_mlx_array_ptrs(&opt->m, n) != 0) {
        opt->param_n = 0;
        mlx_stream_free(s);
        return;
    }
    if (mlx_alloc_mlx_array_ptrs(&opt->v, n) != 0) {
        mlx_free_mlx_array_ptrs(&opt->m, n);
        opt->param_n = 0;
        mlx_stream_free(s);
        return;
    }
    for (int i = 0; i < n; ++i) {
        opt->m[i] = NULL;
        opt->v[i] = NULL;
        mlx_array tmp = mlx_array_new();
        mlx_array zero = mlx_array_new_float(0.0f);
        if (params[i]) {
            if (mlx_multiply(&tmp, *params[i], zero, s) == 0) {
                opt->m[i] = NULL;
                if (mlx_alloc_pod((void **)&opt->m[i], sizeof(mlx_array), 1) == 0)
                    *opt->m[i] = tmp;
                mlx_array tmp2 = mlx_array_new();
                if (mlx_multiply(&tmp2, *params[i], zero, s) == 0) {
                    opt->v[i] = NULL;
                    if (mlx_alloc_pod((void **)&opt->v[i], sizeof(mlx_array), 1) == 0)
                        *opt->v[i] = tmp2;
                } else {
                    if (opt->m[i]) {
                        mlx_array_free(*opt->m[i]);
                        mlx_free_pod((void **)&opt->m[i]);
                        opt->m[i] = NULL;
                    }
                }
            }
        } else {
            mlx_array_free(tmp);
        }
        mlx_array_free(zero);
    }
    mlx_stream_free(s);
}

int mlx_optimizer_apply_flat(MLXOptimizer *opt, mlx_array **params,
                             mlx_array **grads, int n) {
    return mlx_adam_step(opt, params, grads, n);
}

/* Simple in-memory writer used to capture .npy bytes written by
 * mlx_save_writer. */
typedef struct mlx_mem_stream_ {
    char *data;
    size_t pos;
    size_t size;
    bool err;
    bool free_data;
} mlx_mem_stream;

static bool mem_is_open(void *desc) {
    return true;
}
static bool mem_good(void *desc) {
    mlx_mem_stream *m = desc;
    return !m->err;
}
static size_t mem_tell(void *desc) {
    mlx_mem_stream *m = desc;
    return m->pos;
}
static void mem_seek(void *desc, int64_t off, int whence) {
    mlx_mem_stream *m = desc;
    size_t newpos = m->pos;
    if (whence == SEEK_SET) {
        newpos = (off < 0) ? 0 : (size_t)off;
    } else if (whence == SEEK_CUR) {
        if (off < 0 && (size_t)(-off) > newpos)
            newpos = 0;
        else
            newpos = newpos + off;
    } else if (whence == SEEK_END) {
        if (off < 0 && (size_t)(-off) > m->size)
            newpos = 0;
        else
            newpos = m->size + off;
    }
    if (newpos > m->size)
        m->err = true;
    else
        m->pos = newpos;
}
static void mem_read(void *desc, char *data, size_t n) {
    mlx_mem_stream *m = desc;
    size_t avail = m->size - m->pos;
    size_t toread = n <= avail ? n : avail;
    if (toread > 0)
        memcpy(data, m->data + m->pos, toread);
    if (n > toread)
        memset(data + toread, 0, n - toread);
    m->pos += toread;
}
static void mem_read_at_offset(void *desc, char *data, size_t n, size_t off) {
    mlx_mem_stream *m = desc;
    if (off >= m->size) {
        if (n > 0)
            memset(data, 0, n);
        return;
    }
    size_t avail = m->size - off;
    size_t toread = n <= avail ? n : avail;
    if (toread > 0)
        memcpy(data, m->data + off, toread);
    if (n > toread)
        memset(data + toread, 0, n - toread);
    m->pos = off;
}
static void mem_write(void *desc, const char *data, size_t n) {
    mlx_mem_stream *m = desc;
    if (n + m->pos > m->size) {
        m->err = true;
        return;
    }
    memcpy(m->data + m->pos, data, n);
    m->pos += n;
}
static const char *mem_label(void *desc) {
    return "<mem>";
}
static void mem_free(void *desc) {
    mlx_mem_stream *m = desc;
    if (!m)
        return;
    if (m->free_data && m->data)
        free(m->data);
    mlx_free_pod((void **)&m);
}

static mlx_io_vtable mlx_io_vtable_mlx_mem_stream = {
    &mem_is_open,        &mem_good,  &mem_tell,  &mem_seek, &mem_read,
    &mem_read_at_offset, &mem_write, &mem_label, &mem_free
};

/* Serialize an mlx_array into an in-memory .npy buffer. Caller owns *out_buf.
static int serialize_array_to_npy_bytes(const mlx_array arr, void **out_buf,
                                        size_t *out_size) {
    if (!out_buf || !out_size)
        return -1;
    size_t nbytes = mlx_array_nbytes(arr);
    size_t bufsize = nbytes + 4096; /* generous header space */
    mlx_mem_stream *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(mlx_mem_stream), 1) != 0)
        return -1;
    m->data = (char *)malloc(bufsize);
    if (!m->data) {
        mlx_free_pod((void **)&m);
        return -1;
    }
    m->pos = 0;
    m->size = bufsize;
    m->err = false;
    m->free_data = true;

    mlx_io_writer writer = mlx_io_writer_new(m, mlx_io_vtable_mlx_mem_stream);
    if (mlx_save_writer(writer, arr) != 0) {
        mlx_io_writer_free(writer);
        /* writer free will call mem_free */
        return -1;
    }
    /* capture bytes */
    size_t used = m->pos;
    void *buf = malloc(used);
    if (!buf) {
        mlx_io_writer_free(writer);
        return -1;
    }
    memcpy(buf, m->data, used);
    /* free writer which will free mem_stream */
    mlx_io_writer_free(writer);
    *out_buf = buf;
    *out_size = used;
    return 0;
}

int mlx_optimizer_save_to_npz(MLXOptimizer *opt, const char *npz_path) {
    if (!opt || !npz_path)
        return -1;
    /* collect members: m_i.npy, v_i.npy for each param, plus scalars */
    int n_params = opt->param_n;
    int n_members =
        n_params * 2 +
        7; /* m/v pairs + step,lr,beta1,beta2,eps,weight_decay,bias_correction */
    const char **names = NULL;
    if (mlx_alloc_pod((void **)&names, sizeof(char *), n_members) != 0)
        return -1;
    const void **bufs = NULL;
    if (mlx_alloc_pod((void **)&bufs, sizeof(void *), n_members) != 0) {
        mlx_free_pod((void **)&names);
        return -1;
    }
    size_t *sizes = NULL;
    if (mlx_alloc_pod((void **)&sizes, sizeof(size_t), n_members) != 0) {
        mlx_free_pod((void **)&names);
        mlx_free_pod((void **)&bufs);
        return -1;
    }

    int idx = 0;
    /* per-parameter m/v */
    for (int i = 0; i < n_params; ++i) {
        char *nm = (char *)malloc(64);
        snprintf(nm, 64, "m_%d.npy", i);
        names[idx] = nm;
        void *buf = NULL;
        size_t sz = 0;
        if (opt->m && opt->m[i] &&
                serialize_array_to_npy_bytes(*opt->m[i], &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
            idx++;
        } else {
            /* placeholder empty array */
            const char *empty = "";
            bufs[idx] = empty;
            sizes[idx] = 0;
            idx++;
        }

        char *vn = (char *)malloc(64);
        snprintf(vn, 64, "v_%d.npy", i);
        names[idx] = vn;
        buf = NULL;
        sz = 0;
        if (opt->v && opt->v[i] &&
                serialize_array_to_npy_bytes(*opt->v[i], &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
            idx++;
        } else {
            const char *empty = "";
            bufs[idx] = empty;
            sizes[idx] = 0;
            idx++;
        }
    }

    /* scalar entries: step (int64), lr, beta1, beta2, eps, weight_decay
     * (float32), bias_correction (int32) */
    /* step */
    names[idx] = "step.npy";
    {
        long long s = (long long)opt->step;
        int shape[1] = {1};
        mlx_array a = mlx_array_new_data(&s, shape, 1, MLX_INT64);
        void *buf = NULL;
        size_t sz = 0;
        if (serialize_array_to_npy_bytes(a, &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
        } else {
            bufs[idx] = "";
            sizes[idx] = 0;
        }
        mlx_array_free(a);
    }
    idx++;

    /* lr */
    names[idx] = "lr.npy";
    {
        mlx_array a = mlx_array_new_float(opt->lr);
        void *buf = NULL;
        size_t sz = 0;
        if (serialize_array_to_npy_bytes(a, &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
        } else {
            bufs[idx] = "";
            sizes[idx] = 0;
        }
        mlx_array_free(a);
    }
    idx++;

    /* beta1 */
    names[idx] = "beta1.npy";
    {
        mlx_array a = mlx_array_new_float(opt->beta1);
        void *buf = NULL;
        size_t sz = 0;
        if (serialize_array_to_npy_bytes(a, &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
        } else {
            bufs[idx] = "";
            sizes[idx] = 0;
        }
        mlx_array_free(a);
    }
    idx++;

    /* beta2 */
    names[idx] = "beta2.npy";
    {
        mlx_array a = mlx_array_new_float(opt->beta2);
        void *buf = NULL;
        size_t sz = 0;
        if (serialize_array_to_npy_bytes(a, &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
        } else {
            bufs[idx] = "";
            sizes[idx] = 0;
        }
        mlx_array_free(a);
    }
    idx++;

    /* eps */
    names[idx] = "eps.npy";
    {
        mlx_array a = mlx_array_new_float(opt->eps);
        void *buf = NULL;
        size_t sz = 0;
        if (serialize_array_to_npy_bytes(a, &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
        } else {
            bufs[idx] = "";
            sizes[idx] = 0;
        }
        mlx_array_free(a);
    }
    idx++;

    /* weight_decay */
    names[idx] = "weight_decay.npy";
    {
        mlx_array a = mlx_array_new_float(opt->weight_decay);
        void *buf = NULL;
        size_t sz = 0;
        if (serialize_array_to_npy_bytes(a, &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
        } else {
            bufs[idx] = "";
            sizes[idx] = 0;
        }
        mlx_array_free(a);
    }
    idx++;

    /* bias_correction as int32 */
    names[idx] = "bias_correction.npy";
    {
        int v = opt->bias_correction ? 1 : 0;
        int shape[1] = {1};
        mlx_array a = mlx_array_new_data(&v, shape, 1, MLX_INT32);
        void *buf = NULL;
        size_t sz = 0;
        if (serialize_array_to_npy_bytes(a, &buf, &sz) == 0) {
            bufs[idx] = buf;
            sizes[idx] = sz;
        } else {
            bufs[idx] = "";
            sizes[idx] = 0;
        }
        mlx_array_free(a);
    }
    idx++;

    /* create npz */
    int rc = npz_create_from_memory(npz_path, names, bufs, sizes, idx);

    /* free allocated names and buffers */
    for (int i = 0; i < idx; ++i) {
        /* free names for m_i/v_i created with malloc */
        if (i < n_params * 2)
            mlx_free_pod((void **)&names[i]);
        if (bufs[i] && sizes[i] > 0)
            mlx_free_pod((void **)&bufs[i]);
    }
    mlx_free_pod((void **)&names);
    mlx_free_pod((void **)&bufs);
    mlx_free_pod((void **)&sizes);
    return rc;
}

int mlx_optimizer_load_from_npz(MLXOptimizer *opt, const char *npz_path) {
    if (!opt || !npz_path)
        return -1;
    mlx_stream s = mlx_default_gpu_stream_new();
    /* per-parameter m/v: try m_0.npy/v_0.npy ... if present */
    for (int i = 0; i < opt->param_n; ++i) {
        char nm[64];
        snprintf(nm, sizeof(nm), "m_%d.npy", i);
        mlx_io_reader reader;
        if (npz_extract_member_to_mlx_reader(npz_path, nm, &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                if (opt->m && opt->m[i]) {
                    mlx_array_set(opt->m[i], tmp);
                } else {
                    opt->m[i] = NULL;
                    if (mlx_alloc_pod((void **)&opt->m[i], sizeof(mlx_array), 1) == 0)
                        *opt->m[i] = tmp;
                }
            }
            mlx_io_reader_free(reader);
        }
        snprintf(nm, sizeof(nm), "v_%d.npy", i);
        if (npz_extract_member_to_mlx_reader(npz_path, nm, &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                if (opt->v && opt->v[i]) {
                    mlx_array_set(opt->v[i], tmp);
                } else {
                    opt->v[i] = NULL;
                    if (mlx_alloc_pod((void **)&opt->v[i], sizeof(mlx_array), 1) == 0)
                        *opt->v[i] = tmp;
                }
            }
            mlx_io_reader_free(reader);
        }
    }

    /* scalars: step, lr, beta1, beta2, eps, weight_decay, bias_correction */
    {
        mlx_io_reader reader;
        if (npz_extract_member_to_mlx_reader(npz_path, "step.npy", &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                long long v = 0; /* read as int64 */
                mlx_array_item_int64(&v, tmp);
                opt->step = (int)v;
            }
            mlx_io_reader_free(reader);
        }
        if (npz_extract_member_to_mlx_reader(npz_path, "lr.npy", &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                float v = 0.0f;
                mlx_array_item_float32(&v, tmp);
                opt->lr = v;
            }
            mlx_io_reader_free(reader);
        }
        if (npz_extract_member_to_mlx_reader(npz_path, "beta1.npy", &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                float v = 0.0f;
                mlx_array_item_float32(&v, tmp);
                opt->beta1 = v;
            }
            mlx_io_reader_free(reader);
        }
        if (npz_extract_member_to_mlx_reader(npz_path, "beta2.npy", &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                float v = 0.0f;
                mlx_array_item_float32(&v, tmp);
                opt->beta2 = v;
            }
            mlx_io_reader_free(reader);
        }
        if (npz_extract_member_to_mlx_reader(npz_path, "eps.npy", &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                float v = 0.0f;
                mlx_array_item_float32(&v, tmp);
                opt->eps = v;
            }
            mlx_io_reader_free(reader);
        }
        if (npz_extract_member_to_mlx_reader(npz_path, "weight_decay.npy",
                                             &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                float v = 0.0f;
                mlx_array_item_float32(&v, tmp);
                opt->weight_decay = v;
            }
            mlx_io_reader_free(reader);
        }
        if (npz_extract_member_to_mlx_reader(npz_path, "bias_correction.npy",
                                             &reader) == 0) {
            mlx_array tmp = mlx_array_new();
            if (mlx_load_reader(&tmp, reader, s) == 0) {
                int v = 0;
                mlx_array_item_int32(&v, tmp);
                opt->bias_correction = v ? 1 : 0;
            }
            mlx_io_reader_free(reader);
        }
    }

    mlx_stream_free(s);
    return 0;
}
