#include "scheduler.h"
#include "array_helpers.h"
#include "io/npz_create.h"
#include "io/npz_unzip.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct MLXScheduler {
    int is_multistep;
    /* step-lr fields */
    int step_size;
    /* multistep fields */
    int *milestones;
    int n_milestones;
    float gamma;

    /* parity additions */
    float *base_lrs; /* array of base learning rates */
    int n_base_lrs;
    int last_step;                    /* last step index applied */
    MLXOptimizer *attached_optimizer; /* optional attached optimizer */
    int verbose;
    /* cached last lr values to mirror Python's _last_lr */
    float *_last_lr;
    int _last_lr_n;
};

/* Non-nested helper: find a key in JSON and return pointer after ':' */
static const char *find_key_in(const char *p, const char *key) {
    const char *k = strstr(p, key);
    if (!k)
        return NULL;
    const char *col = strchr(k, ':');
    if (!col)
        return NULL;
    return col + 1;
}

/* After loading state, sync attached optimizer (if any) to current lr. */
static void sync_attached_optimizer_after_load(MLXScheduler *s) {
    if (!s || !s->attached_optimizer)
        return;
    float tmp[16];
    int n = mlx_scheduler_lr_for_step(s, s->last_step < 0 ? 0 : s->last_step, tmp,
                                      sizeof(tmp) / sizeof(tmp[0]));
    if (n > 0)
        mlx_optimizer_set_lr(s->attached_optimizer, tmp[0]);
}

static void update_cached_last_lr(MLXScheduler *s, const float *lrs, int n) {
    if (!s)
        return;
    /* reuse helper to free then copy */
    mlx_free_float_array(&s->_last_lr, &s->_last_lr_n);
    if (n <= 0)
        return;
    mlx_copy_float_array(&s->_last_lr, &s->_last_lr_n, lrs, n);
}

static void print_lr_update(MLXScheduler *s, int step, const float *lrs,
                            int n) {
    if (!s || !s->verbose)
        return;
    /* verbose scheduler logging removed */
}

MLXScheduler *mlx_scheduler_step_lr_create(int step_size, float gamma) {
    MLXScheduler *s = NULL;
    if (mlx_alloc_pod((void **)&s, sizeof(MLXScheduler), 1) != 0)
        return NULL;
    s->is_multistep = 0;
    s->step_size = step_size;
    s->milestones = NULL;
    s->n_milestones = 0;
    s->gamma = gamma;
    s->base_lrs = NULL;
    s->n_base_lrs = 0;
    s->last_step = -1;
    s->attached_optimizer = NULL;
    s->verbose = 0;
    s->_last_lr = NULL;
    s->_last_lr_n = 0;

    return s;
}

MLXScheduler *mlx_scheduler_step_lr_create_with_init(int step_size, float gamma,
        const float *init_lr,
        int n_init_lrs) {
    MLXScheduler *s = mlx_scheduler_step_lr_create(step_size, gamma);
    if (!s)
        return NULL;
    if (init_lr && n_init_lrs > 0)
        mlx_scheduler_set_base_lrs(s, init_lr, n_init_lrs);
    return s;
}

MLXScheduler *mlx_scheduler_multistep_create(const int *milestones,
        int n_milestones, float gamma) {
    if (!milestones || n_milestones <= 0)
        return NULL;
    MLXScheduler *s = NULL;
    if (mlx_alloc_pod((void **)&s, sizeof(MLXScheduler), 1) != 0)
        return NULL;
    s->is_multistep = 1;
    s->step_size = 0;
    s->n_milestones = n_milestones;
    if (mlx_alloc_int_array(&s->milestones, n_milestones) != 0) {
        mlx_free_pod((void **)&s);
        return NULL;
    }
    for (int i = 0; i < n_milestones; ++i)
        s->milestones[i] = milestones[i];
    s->gamma = gamma;
    s->base_lrs = NULL;
    s->n_base_lrs = 0;
    s->last_step = -1;
    s->attached_optimizer = NULL;
    s->verbose = 0;
    s->_last_lr = NULL;
    s->_last_lr_n = 0;

    return s;
}

MLXScheduler *mlx_scheduler_multistep_create_with_init(const int *milestones,
        int n_milestones,
        float gamma,
        const float *init_lr,
        int n_init_lrs) {
    MLXScheduler *s =
        mlx_scheduler_multistep_create(milestones, n_milestones, gamma);
    if (!s)
        return NULL;
    if (init_lr && n_init_lrs > 0)
        mlx_scheduler_set_base_lrs(s, init_lr, n_init_lrs);
    return s;
}

void mlx_scheduler_free(MLXScheduler *s) {
    if (!s)
        return;

    mlx_free_int_array(&s->milestones, &s->n_milestones);
    mlx_free_float_array(&s->base_lrs, &s->n_base_lrs);
    mlx_free_float_array(&s->_last_lr, &s->_last_lr_n);
    mlx_free_pod((void **)&s);
}

static void ensure_base_lrs_from_optimizer(MLXScheduler *s, MLXOptimizer *opt) {
    if (!s)
        return;
    if (s->n_base_lrs > 0)
        return;
    if (!opt)
        return;
    float cur = mlx_optimizer_get_lr(opt);
    if (mlx_alloc_float_buf(&s->base_lrs, 1) != 0)
        return;
    s->base_lrs[0] = cur;
    s->n_base_lrs = 1;
}

int mlx_scheduler_lr_for_step(MLXScheduler *s, int step, float *out_lrs,
                              int max_out) {
    if (!s || !out_lrs || max_out <= 0)
        return 0;

    /* Determine multiplicative factor */
    int count = 0;
    if (s->is_multistep) {
        for (int i = 0; i < s->n_milestones; ++i)
            if (step >= s->milestones[i])
                ++count;
    } else {
        if (s->step_size > 0)
            count = step / s->step_size;
    }
    float mul = powf(s->gamma, (float)count);

    if (s->n_base_lrs > 0) {
        int n = s->n_base_lrs < max_out ? s->n_base_lrs : max_out;
        for (int i = 0; i < n; ++i)
            out_lrs[i] = s->base_lrs[i] * mul;
        return n;
    }

    /* No base_lrs defined: if there is an attached optimizer, use its lr */
    if (s->attached_optimizer) {
        float cur = mlx_optimizer_get_lr(s->attached_optimizer);
        out_lrs[0] = cur * mul;
        return 1;
    }

    /* Nothing to compute */
    return 0;
}

int mlx_scheduler_get_lr(MLXScheduler *s, float *out_lrs, int max_out) {
    if (!s)
        return 0;
    int step = s->last_step;
    if (step < 0)
        step = 0;
    return mlx_scheduler_lr_for_step(s, step, out_lrs, max_out);
}

void mlx_scheduler_set_base_lrs(MLXScheduler *s, const float *base_lrs,
                                int n_base_lrs) {
    if (!s)
        return;
    /* free existing base_lrs then copy new values using helpers */

    mlx_free_float_array(&s->base_lrs, &s->n_base_lrs);
    if (!base_lrs || n_base_lrs <= 0) {
        s->base_lrs = NULL;
        s->n_base_lrs = 0;
        return;
    }
    mlx_copy_float_array(&s->base_lrs, &s->n_base_lrs, base_lrs,
                         n_base_lrs);

}

void mlx_scheduler_set_last_step(MLXScheduler *s, int last_step) {
    if (!s)
        return;
    s->last_step = last_step;
}

int mlx_scheduler_get_last_step(MLXScheduler *s) {
    if (!s)
        return -1;
    return s->last_step;
}

void mlx_scheduler_attach_optimizer(MLXScheduler *s, MLXOptimizer *opt) {
    if (!s)
        return;
    s->attached_optimizer = opt;
}

int mlx_scheduler_serialize_state(MLXScheduler *s, char **out_json) {
    if (!s || !out_json)
        return -1;
    /* Serialize into stable JSON. */
    int est = 256 + (s->n_base_lrs * 32) + (s->n_milestones * 16);
    char *buf = (char *)malloc(est);
    if (!buf)
        return -1;
    int off = 0;
    off += snprintf(buf + off, est - off, "{");
    off += snprintf(buf + off, est - off, "\"last_step\":%d,", s->last_step);
    off +=
        snprintf(buf + off, est - off, "\"is_multistep\":%d,", s->is_multistep);
    off += snprintf(buf + off, est - off, "\"step_size\":%d,", s->step_size);
    off += snprintf(buf + off, est - off, "\"gamma\":%g,", s->gamma);

    /* base_lrs array */
    off += snprintf(buf + off, est - off, "\"base_lrs\":[");
    for (int i = 0; i < s->n_base_lrs; ++i) {
        off += snprintf(buf + off, est - off, "%g", s->base_lrs[i]);
        if (i + 1 < s->n_base_lrs)
            off += snprintf(buf + off, est - off, ",");
    }
    off += snprintf(buf + off, est - off, "],");

    /* _last_lr cache (if present) */
    off += snprintf(buf + off, est - off, "\"_last_lr\":[");
    if (s->_last_lr && s->_last_lr_n > 0) {
        for (int i = 0; i < s->_last_lr_n; ++i) {
            off += snprintf(buf + off, est - off, "%g", s->_last_lr[i]);
            if (i + 1 < s->_last_lr_n)
                off += snprintf(buf + off, est - off, ",");
        }
    }
    off += snprintf(buf + off, est - off, "],");

    /* milestones array */
    off += snprintf(buf + off, est - off, "\"milestones\":[");
    for (int i = 0; i < s->n_milestones; ++i) {
        off += snprintf(buf + off, est - off, "%d", s->milestones[i]);
        if (i + 1 < s->n_milestones)
            off += snprintf(buf + off, est - off, ",");
    }
    off += snprintf(buf + off, est - off, "]");

    off += snprintf(buf + off, est - off, "}");
    *out_json = buf;
    return 0;
}

int mlx_scheduler_load_state_from_json(MLXScheduler *s, const char *json_str) {
    if (!s || !json_str)
        return -1;
    /* Minimal JSON parser tailored for our schema. Not a general JSON parser. */
    const char *p = json_str;
    const char *v;
    /* find_key_in defined below (non-nested helper) */
    v = find_key_in(p, "\"last_step\"");
    if (v)
        s->last_step = atoi(v);
    v = find_key_in(p, "\"is_multistep\"");
    if (v)
        s->is_multistep = atoi(v);
    v = find_key_in(p, "\"step_size\"");
    if (v)
        s->step_size = atoi(v);
    v = find_key_in(p, "\"gamma\"");
    if (v)
        s->gamma = (float)atof(v);

    /* parse base_lrs array */
    const char *bstart = strstr(p, "\"base_lrs\"");
    if (bstart) {
        const char *lb = strchr(bstart, '[');
        const char *rb = lb ? strchr(lb, ']') : NULL;
        if (lb && rb && rb > lb) {
            int n = 0;
            const char *q = lb + 1;
            while (q < rb) {
                while (q < rb && (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t' ||
                                  *q == ','))
                    q++;
                if (q >= rb)
                    break;
                char *endptr = NULL;
                strtod(q, &endptr);
                if (endptr && endptr > q) {
                    n++;
                    q = endptr;
                } else
                    break;
            }
            if (n > 0) {
                if (s->base_lrs)
                    mlx_free_float_array(&s->base_lrs, &s->n_base_lrs);
                if (mlx_alloc_float_buf(&s->base_lrs, n) == 0)
                    s->n_base_lrs = n;
                q = lb + 1;
                for (int i = 0; i < n; ++i) {
                    while (q < rb && (*q == ' ' || *q == '\n' || *q == '\r' ||
                                      *q == '\t' || *q == ','))
                        q++;
                    if (q >= rb)
                        break;
                    char *endptr = NULL;
                    double val = strtod(q, &endptr);
                    s->base_lrs[i] = (float)val;
                    q = endptr;
                }
            }
        }
    }

    /* parse _last_lr array if present */
    const char *lstart = strstr(p, "\"_last_lr\"");
    if (lstart) {
        const char *lb = strchr(lstart, '[');
        const char *rb = lb ? strchr(lb, ']') : NULL;
        if (lb && rb && rb > lb) {
            int n = 0;
            const char *q = lb + 1;
            while (q < rb) {
                while (q < rb && (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t' ||
                                  *q == ','))
                    q++;
                if (q >= rb)
                    break;
                char *endptr = NULL;
                strtod(q, &endptr);
                if (endptr && endptr > q) {
                    n++;
                    q = endptr;
                } else
                    break;
            }
            if (n > 0) {
                if (s->_last_lr)
                    mlx_free_float_array(&s->_last_lr, &s->_last_lr_n);
                if (mlx_alloc_float_buf(&s->_last_lr, n) == 0)
                    s->_last_lr_n = n;
                q = lb + 1;
                for (int i = 0; i < n; ++i) {
                    while (q < rb && (*q == ' ' || *q == '\n' || *q == '\r' ||
                                      *q == '\t' || *q == ','))
                        q++;
                    if (q >= rb)
                        break;
                    char *endptr = NULL;
                    double val = strtod(q, &endptr);
                    s->_last_lr[i] = (float)val;
                    q = endptr;
                }
            }
        }
    }
    /* parse milestones array */
    const char *mstart = strstr(p, "\"milestones\"");
    if (mstart) {
        const char *lb = strchr(mstart, '[');
        const char *rb = lb ? strchr(lb, ']') : NULL;
        if (lb && rb && rb > lb) {
            int n = 0;
            const char *q = lb + 1;
            while (q < rb) {
                while (q < rb && (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t' ||
                                  *q == ','))
                    q++;
                if (q >= rb)
                    break;
                char *endptr = NULL;
                strtol(q, &endptr, 10);
                if (endptr && endptr > q) {
                    n++;
                    q = endptr;
                } else
                    break;
            }
            if (n > 0) {
                if (s->milestones)
                    mlx_free_int_array(&s->milestones, &s->n_milestones);
                if (mlx_alloc_int_array(&s->milestones, n) == 0)
                    s->n_milestones = n;
                q = lb + 1;
                for (int i = 0; i < n; ++i) {
                    while (q < rb && (*q == ' ' || *q == '\n' || *q == '\r' ||
                                      *q == '\t' || *q == ','))
                        q++;
                    if (q >= rb)
                        break;
                    char *endptr = NULL;
                    long val = strtol(q, &endptr, 10);
                    s->milestones[i] = (int)val;
                    q = endptr;
                }
            }
        }
    }

    /* sync optimizer if attached so its learning_rate matches the scheduler */
    sync_attached_optimizer_after_load(s);
    return 0;
}
int mlx_scheduler_save_to_npz(MLXScheduler *s, const char *npz_path) {
    if (!s || !npz_path)
        return -1;
    char *json = NULL;
    if (mlx_scheduler_serialize_state(s, &json) != 0 || !json)
        return -1;
    const char *names[1];
    const void *bufs[1];
    size_t sizes[1];
    names[0] = "state.json";
    bufs[0] = (const void *)json;
    sizes[0] = strlen(json);
    int rc = npz_create_from_memory(npz_path, names, bufs, sizes, 1);
    free(json);
    return rc;
}

int mlx_scheduler_load_from_npz(MLXScheduler *s, const char *npz_path) {
    if (!s || !npz_path)
        return -1;
    void *buf = NULL;
    size_t sz = 0;
    if (npz_extract_member_to_memory(npz_path, "state.json", &buf, &sz) != 0)
        return -1;
    int rc = mlx_scheduler_load_state_from_json(s, (const char *)buf);
    free(buf);
    if (rc == 0)
        sync_attached_optimizer_after_load(s);
    return rc;
}

int mlx_scheduler_step_nullable(MLXScheduler *s, const int *step,
                                float *out_lrs, int max_out) {
    if (!s || !out_lrs || max_out <= 0)
        return 0;

    int target_step;
    if (step == NULL)
        target_step = (s->last_step < 0) ? 0 : s->last_step + 1;
    else
        target_step = *step;

    int n = mlx_scheduler_step_and_get_lr(s, target_step, NULL, out_lrs, max_out);
    return n;
}

int mlx_scheduler_step_and_get_lr(MLXScheduler *s, int step, MLXOptimizer *opt,
                                  float *out_lrs, int max_out) {
    if (!s || !out_lrs || max_out <= 0)
        return 0;

    /* prefer explicit opt param for update, otherwise attached optimizer */
    MLXOptimizer *target = opt ? opt : s->attached_optimizer;

    /* ensure base_lrs from provided optimizer if none set */
    if (s->n_base_lrs == 0 && target)
        ensure_base_lrs_from_optimizer(s, target);

    /* compute lrs for this step */
    float tmp[16];
    int n = mlx_scheduler_lr_for_step(s, step, tmp, sizeof(tmp) / sizeof(tmp[0]));
    if (n <= 0) {
        s->last_step = step;
        return 0;
    }

    /* copy to out_lrs */
    int to_copy = n < max_out ? n : max_out;
    for (int i = 0; i < to_copy; ++i)
        out_lrs[i] = tmp[i];

    /* update optimizer if available (use first lr) */
    if (target)
        mlx_optimizer_set_lr(target, tmp[0]);

    /* update cache and verbose logging */
    update_cached_last_lr(s, tmp, n);
    print_lr_update(s, step, tmp, n);

    s->last_step = step;
    return to_copy;
}

void mlx_scheduler_step_auto(MLXScheduler *s, MLXOptimizer *opt) {
    if (!s)
        return;
    int next = s->last_step < 0 ? 0 : s->last_step + 1;
    mlx_scheduler_step(s, next, opt);
}

int mlx_scheduler_call(MLXScheduler *s, int step, float *out_lrs, int max_out) {
    return mlx_scheduler_lr_for_step(s, step, out_lrs, max_out);
}

void mlx_scheduler_apply_to_optimizers(MLXScheduler *s, MLXOptimizer **opts,
                                       int n_opts) {
    if (!s || !opts || n_opts <= 0)
        return;
    float *tmp = NULL;
    int n = 0;
    int alloced = 0;
    if (s->n_base_lrs > 0) {
        tmp = s->base_lrs;
        n = s->n_base_lrs;
    } else {
        if (mlx_alloc_float_buf(&tmp, 1) != 0)
            return;
        tmp[0] = s->attached_optimizer ? mlx_optimizer_get_lr(s->attached_optimizer)
                 : 0.0f;
        n = 1;
        alloced = 1;
    }

    for (int i = 0; i < n_opts; ++i) {
        float v = tmp[i < n ? i : (n - 1)];
        if (opts[i])
            mlx_optimizer_set_lr(opts[i], v);
    }

    if (alloced)
        mlx_free_float_buf(&tmp, NULL);
}

void mlx_scheduler_step(MLXScheduler *s, int step, MLXOptimizer *opt) {
    if (!s)
        return;

    /* prefer explicit opt param for update, otherwise attached optimizer */
    MLXOptimizer *target = opt ? opt : s->attached_optimizer;

    /* ensure base_lrs from provided optimizer if none set */
    if (s->n_base_lrs == 0 && target)
        ensure_base_lrs_from_optimizer(s, target);

    /* compute lrs for this step */
    float tmp[16];
    int n = mlx_scheduler_lr_for_step(s, step, tmp, sizeof(tmp) / sizeof(tmp[0]));
    if (n <= 0) {
        /* still update last_step */
        s->last_step = step;
        return;
    }

    update_cached_last_lr(s, tmp, n);
    print_lr_update(s, step, tmp, n);
    s->last_step = step;
    if (target)
        mlx_optimizer_set_lr(target, tmp[0]);

    s->last_step = step;
}
