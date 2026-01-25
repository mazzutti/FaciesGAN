#include "dataloader.h"
#include "collate.h"
#include "faciesgan-c/utils.h"
#include "mlx/c/array.h"
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* forward declarations for IPC helpers used across this file */
static ssize_t read_all(int fd, void *buf, size_t count);
static ssize_t write_all(int fd, const void *buf, size_t count);
static int read_vector_array_from_fd(int fd, mlx_vector_array *out_vec);
static int serialize_vec_to_fd(int fd, mlx_vector_array vec);
static int write_vv_fd(int fd, const mlx_vector_vector_array vv,
                       uint64_t n_samples_local);

typedef struct batch_item_s {
  mlx_vector_array facies;
  mlx_vector_array wells;
  mlx_vector_array seismic;
  int error;
} batch_item;

static int write_vv_fd(int fd, const mlx_vector_vector_array vv,
                       uint64_t n_samples_local) {
  size_t n = mlx_vector_vector_array_size(vv);
  if ((uint64_t)n != n_samples_local)
    return 1;
  for (size_t i = 0; i < n; ++i) {
    mlx_vector_array sample = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&sample, vv, i)) {
      mlx_vector_array_free(sample);
      return 1;
    }
    uint32_t nscales = (uint32_t)mlx_vector_array_size(sample);
    if (write_all(fd, &nscales, sizeof(nscales)) <= 0) {
      mlx_vector_array_free(sample);
      return 1;
    }
    for (uint32_t si = 0; si < nscales; ++si) {
      mlx_array arr = mlx_array_new();
      if (mlx_vector_array_get(&arr, sample, si) != 0) {
        uint32_t zero = 0;
        write_all(fd, &zero, sizeof(zero));
        continue;
      }
      float *buf = NULL;
      size_t elems = 0;
      int ndim = 0;
      int *shape = NULL;
      if (mlx_array_to_float_buffer(arr, &buf, &elems, &ndim, &shape) != 0) {
        uint32_t zero = 0;
        write_all(fd, &zero, sizeof(zero));
        mlx_array_free(arr);
        continue;
      }
      uint32_t udim = (uint32_t)ndim;
      if (write_all(fd, &udim, sizeof(udim)) <= 0) {
        free(buf);
        free(shape);
        mlx_array_free(arr);
        return 1;
      }
      for (int d = 0; d < ndim; ++d) {
        int32_t s32 = shape[d];
        if (write_all(fd, &s32, sizeof(s32)) <= 0) {
          free(buf);
          free(shape);
          mlx_array_free(arr);
          return 1;
        }
      }
      uint64_t uelems = (uint64_t)elems;
      if (write_all(fd, &uelems, sizeof(uelems)) <= 0) {
        free(buf);
        free(shape);
        mlx_array_free(arr);
        return 1;
      }
      if (elems > 0) {
        if (write_all(fd, buf, sizeof(float) * elems) <= 0) {
          free(buf);
          free(shape);
          mlx_array_free(arr);
          return 1;
        }
      }
      free(buf);
      free(shape);
      mlx_array_free(arr);
    }
    mlx_vector_array_free(sample);
  }
  return 0;
}

struct MLXDataloader {
  MLXPyramidsDataset *ds;
  size_t batch_size;
  size_t *indices;
  size_t idx; // next index
  size_t n_indices;
  bool shuffle;
  bool drop_last;
  /* Extended options */
  int num_workers;
  int prefetch_factor;
  bool persistent_workers;
  int timeout_ms; /* milliseconds */
  facies_collate_fn collate_cb;
  void *collate_ctx;
  bool pin_memory;
  /* sampler callbacks */
  facies_sampler_next_fn sampler_next;
  void *sampler_ctx;
  facies_sampler_reset_fn sampler_reset;
  facies_batch_sampler_next_fn batch_sampler_next;
  void *batch_sampler_ctx;
  facies_sampler_reset_fn batch_sampler_reset;
  /* worker init callback */
  facies_worker_init_fn worker_init;
  void *worker_init_ctx;
  unsigned int base_seed;
  uint64_t iterator_base_seed;
  uint64_t epoch_counter;
  struct pthread_worker_arg *worker_args;
  /* for spawned worker init: optional shared lib path and symbol name */
  char *worker_init_lib;
  char *worker_init_sym;
  void *worker_init_ctx_data;
  unsigned int worker_init_ctx_len;

  /* Worker infrastructure */
  pthread_t *workers;
  /* batch tasks: each task is an array of indices */
  size_t **tasks;
  int *task_sizes;
  size_t n_tasks;
  size_t next_task;
  pthread_mutex_t task_mutex;

  /* process backend threads */
  pthread_t *proc_readers;
  int *proc_reader_ids;
  pthread_t proc_dispatcher;
  bool process_workers_started;
  struct reader_arg *reader_args;
  /* per-worker init status/message forwarded from spawned workers */
  char **worker_init_err_msg;
  int *worker_init_done;

  /* result queue for precomputed batches */
  batch_item *queue;
  size_t q_head, q_tail, q_count, q_capacity;
  pthread_mutex_t q_mutex;
  pthread_cond_t q_nonempty;
  pthread_cond_t q_nonfull;
  bool workers_started;
  bool finished;
  /* process worker fields */
  pid_t *pids;
  int *task_wfds;   /* parent writes tasks to worker */
  int *result_rfds; /* parent reads results from worker */
  int next_worker;
};

int facies_dataset_new(MLXPyramidsDataset **out,
                       const mlx_vector_vector_array facies_pyramids,
                       const mlx_vector_vector_array wells_pyramids,
                       const mlx_vector_vector_array seismic_pyramids) {
  if (!out)
    return 1;
  MLXPyramidsDataset *ds = (MLXPyramidsDataset *)calloc(1, sizeof(*ds));
  if (!ds)
    return 1;

  ds->facies = mlx_vector_vector_array_new();
  ds->wells = mlx_vector_vector_array_new();
  ds->seismic = mlx_vector_vector_array_new();

  /* copy data references into dataset's vectors */
  size_t nf = mlx_vector_vector_array_size(facies_pyramids);
  for (size_t i = 0; i < nf; ++i) {
    mlx_vector_array tmp = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&tmp, facies_pyramids, i)) {
      mlx_vector_vector_array_free(ds->facies);
      free(ds);
      return 1;
    }
    if (mlx_vector_vector_array_append_value(ds->facies, tmp)) {
      mlx_vector_array_free(tmp);
      mlx_vector_vector_array_free(ds->facies);
      free(ds);
      return 1;
    }
    mlx_vector_array_free(tmp);
  }

  /* wells/seismic may be empty */
  size_t nw = mlx_vector_vector_array_size(wells_pyramids);
  for (size_t i = 0; i < nw; ++i) {
    mlx_vector_array tmp = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&tmp, wells_pyramids, i)) {
      mlx_vector_vector_array_free(ds->facies);
      mlx_vector_vector_array_free(ds->wells);
      free(ds);
      return 1;
    }
    if (mlx_vector_vector_array_append_value(ds->wells, tmp)) {
      mlx_vector_array_free(tmp);
      mlx_vector_vector_array_free(ds->facies);
      mlx_vector_vector_array_free(ds->wells);
      free(ds);
      return 1;
    }
    mlx_vector_array_free(tmp);
  }

  size_t ns = mlx_vector_vector_array_size(seismic_pyramids);
  for (size_t i = 0; i < ns; ++i) {
    mlx_vector_array tmp = mlx_vector_array_new();
    if (mlx_vector_vector_array_get(&tmp, seismic_pyramids, i)) {
      mlx_vector_vector_array_free(ds->facies);
      mlx_vector_vector_array_free(ds->wells);
      mlx_vector_vector_array_free(ds->seismic);
      free(ds);
      return 1;
    }
    if (mlx_vector_vector_array_append_value(ds->seismic, tmp)) {
      mlx_vector_array_free(tmp);
      mlx_vector_vector_array_free(ds->facies);
      mlx_vector_vector_array_free(ds->wells);
      mlx_vector_vector_array_free(ds->seismic);
      free(ds);
      return 1;
    }
    mlx_vector_array_free(tmp);
  }

  ds->n_samples = (int)mlx_vector_vector_array_size(ds->facies);
  ds->batches = NULL;
  ds->n_batches = 0;
  ds->scales = NULL;
  ds->n_scales = 0;

  *out = ds;
  return 0;
}

int facies_dataset_free(MLXPyramidsDataset *ds) {
  if (!ds)
    return 1;
  /* reuse existing dataset free helper */
  mlx_pyramids_dataset_free(ds);
  return 0;
}

static void shuffle_indices(size_t *idxs, size_t n, unsigned int seed) {
  if (!idxs)
    return;
  srand(seed ? seed : (unsigned int)time(NULL));
  for (size_t i = n - 1; i > 0; --i) {
    size_t j = (size_t)(rand() % (i + 1));
    size_t t = idxs[i];
    idxs[i] = idxs[j];
    idxs[j] = t;
  }
}

int facies_dataloader_new(struct MLXDataloader **out, MLXPyramidsDataset *ds,
                          size_t batch_size, bool shuffle, bool drop_last,
                          unsigned int seed) {
  return facies_dataloader_new_ex(
      out, ds, batch_size, shuffle, drop_last, seed, 0, 0, false, 0, NULL, NULL,
      false, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL);
}

static int build_tasks(struct MLXDataloader *dl) {
  if (!dl)
    return 1;
  size_t n = dl->n_indices;
  size_t bs = dl->batch_size;
  if (bs == 0)
    return 1;
  /* If a batch_sampler callback is provided, use it to obtain batches. */
  if (dl->batch_sampler_next) {
    size_t cap = 64;
    size_t t = 0;
    dl->tasks = (size_t **)calloc(cap, sizeof(size_t *));
    dl->task_sizes = (int *)calloc(cap, sizeof(int));
    if (!dl->tasks || !dl->task_sizes)
      return 1;
    size_t *buf = (size_t *)malloc(sizeof(size_t) * bs);
    if (!buf)
      return 1;
    while (1) {
      int out_count = 0;
      int rc = dl->batch_sampler_next(dl->batch_sampler_ctx, buf, (int)bs,
                                      &out_count);
      if (rc == 2)
        break;
      if (rc != 0) {
        free(buf);
        return 1;
      }
      if (out_count <= 0)
        continue;
      if (t >= cap) {
        cap *= 2;
        dl->tasks = (size_t **)realloc(dl->tasks, sizeof(size_t *) * cap);
        dl->task_sizes = (int *)realloc(dl->task_sizes, sizeof(int) * cap);
      }
      dl->tasks[t] = (size_t *)malloc(sizeof(size_t) * out_count);
      if (!dl->tasks[t]) {
        free(buf);
        return 1;
      }
      for (int i = 0; i < out_count; ++i)
        dl->tasks[t][i] = buf[i];
      dl->task_sizes[t] = out_count;
      ++t;
    }
    free(buf);
    dl->n_tasks = t;
    dl->next_task = 0;
    return 0;
  }

  /* If only a sampler_next callback is provided, consume it into batches. */
  if (dl->sampler_next) {
    size_t cap = 64;
    size_t t = 0;
    dl->tasks = (size_t **)calloc(cap, sizeof(size_t *));
    dl->task_sizes = (int *)calloc(cap, sizeof(int));
    if (!dl->tasks || !dl->task_sizes)
      return 1;
    size_t *buf = (size_t *)malloc(sizeof(size_t) * bs);
    if (!buf)
      return 1;
    while (1) {
      int filled = 0;
      for (size_t i = 0; i < bs; ++i) {
        size_t idx = 0;
        int rc = dl->sampler_next(dl->sampler_ctx, &idx);
        if (rc == 2) {
          break;
        }
        if (rc != 0) {
          free(buf);
          return 1;
        }
        buf[filled++] = idx;
      }
      if (filled == 0)
        break;
      if (t >= cap) {
        cap *= 2;
        dl->tasks = (size_t **)realloc(dl->tasks, sizeof(size_t *) * cap);
        dl->task_sizes = (int *)realloc(dl->task_sizes, sizeof(int) * cap);
      }
      dl->tasks[t] = (size_t *)malloc(sizeof(size_t) * filled);
      if (!dl->tasks[t]) {
        free(buf);
        return 1;
      }
      for (int i = 0; i < filled; ++i)
        dl->tasks[t][i] = buf[i];
      dl->task_sizes[t] = filled;
      ++t;
    }
    free(buf);
    dl->n_tasks = t;
    dl->next_task = 0;
    return 0;
  }

  /* Default: partition the internal indices array into contiguous batches */
  size_t max_full = n / bs;
  size_t rem = n % bs;
  size_t ntasks = max_full + (rem && !dl->drop_last ? 1 : 0);
  dl->tasks = (size_t **)calloc(ntasks, sizeof(size_t *));
  dl->task_sizes = (int *)calloc(ntasks, sizeof(int));
  if (!dl->tasks || !dl->task_sizes)
    return 1;
  size_t off = 0;
  size_t t = 0;
  while (off < n && t < ntasks) {
    size_t cur = bs;
    if (off + cur > n)
      cur = n - off;
    dl->tasks[t] = (size_t *)malloc(sizeof(size_t) * cur);
    if (!dl->tasks[t])
      return 1;
    for (size_t i = 0; i < cur; ++i)
      dl->tasks[t][i] = dl->indices[off + i];
    dl->task_sizes[t] = (int)cur;
    off += cur;
    ++t;
  }
  dl->n_tasks = ntasks;
  dl->next_task = 0;
  return 0;
}

static int queue_push(struct MLXDataloader *dl, mlx_vector_array fac,
                      mlx_vector_array wells, mlx_vector_array seismic,
                      int err) {
  pthread_mutex_lock(&dl->q_mutex);
  while (dl->q_count == dl->q_capacity && !dl->finished) {
    pthread_cond_wait(&dl->q_nonfull, &dl->q_mutex);
  }
  if (dl->finished) {
    pthread_mutex_unlock(&dl->q_mutex);
    return 1;
  }
  size_t pos = dl->q_tail;
  dl->queue[pos].facies = fac;
  dl->queue[pos].wells = wells;
  dl->queue[pos].seismic = seismic;
  dl->queue[pos].error = err;
  dl->q_tail = (dl->q_tail + 1) % dl->q_capacity;
  dl->q_count++;
  pthread_cond_signal(&dl->q_nonempty);
  pthread_mutex_unlock(&dl->q_mutex);
  return 0;
}

static int queue_pop_timeout(struct MLXDataloader *dl,
                             mlx_vector_array *out_fac,
                             mlx_vector_array *out_well,
                             mlx_vector_array *out_seis, int *out_err) {
  struct timespec ts;
  int rc = 0;
  pthread_mutex_lock(&dl->q_mutex);
  while (dl->q_count == 0 && !dl->finished) {
    if (dl->timeout_ms <= 0) {
      pthread_cond_wait(&dl->q_nonempty, &dl->q_mutex);
    } else {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      long long ns =
          (long long)tv.tv_sec * 1000LL + tv.tv_usec / 1000LL + dl->timeout_ms;
      ts.tv_sec = ns / 1000LL;
      ts.tv_nsec = (ns % 1000LL) * 1000000LL;
      int wrc = pthread_cond_timedwait(&dl->q_nonempty, &dl->q_mutex, &ts);
      if (wrc == ETIMEDOUT) {
        rc = 1; /* timeout */
        break;
      }
    }
  }
  if (rc == 0 && dl->q_count > 0) {
    size_t pos = dl->q_head;
    /* move ownership out of the queue slot */
    *out_fac = dl->queue[pos].facies;
    *out_well = dl->queue[pos].wells;
    *out_seis = dl->queue[pos].seismic;
    *out_err = dl->queue[pos].error;
    /* replace with empty vectors so destructor won't double-free */
    dl->queue[pos].facies = mlx_vector_array_new();
    dl->queue[pos].wells = mlx_vector_array_new();
    dl->queue[pos].seismic = mlx_vector_array_new();
    dl->queue[pos].error = 0;
    dl->q_head = (dl->q_head + 1) % dl->q_capacity;
    dl->q_count--;
    pthread_cond_signal(&dl->q_nonfull);
    rc = 0;
  }
  pthread_mutex_unlock(&dl->q_mutex);
  return rc;
}

struct pthread_worker_arg {
  struct MLXDataloader *dl;
  int worker_id;
  uint64_t last_epoch;
};

/* Serialize calls into MLX evaluation paths that are not thread-safe
 * (avoid modifying third-party mlx sources). This mutex protects the
 * facies_collate invocation which triggers MLX evaluation internals. */
static pthread_mutex_t mlx_eval_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *worker_thread(void *arg) {
  struct pthread_worker_arg *wa = (struct pthread_worker_arg *)arg;
  struct MLXDataloader *dl = wa->dl;
  int worker_id = wa->worker_id;
  mlx_stream s = mlx_default_cpu_stream_new();
  /* worker-local epoch tracking */
  uint64_t local_epoch = (uint64_t)-1;
  while (1) {
    /* check for epoch change and reseed/call worker_init if needed */
    pthread_mutex_lock(&dl->task_mutex);
    if (local_epoch != dl->epoch_counter) {
      local_epoch = dl->epoch_counter;
      unsigned int seed =
          (unsigned int)(dl->iterator_base_seed + (uint64_t)worker_id);
      srand(seed);
      if (dl->worker_init) {
        dl->worker_init(worker_id, dl->worker_init_ctx);
      }
    }
    if (dl->next_task >= dl->n_tasks) {
      pthread_mutex_unlock(&dl->task_mutex);
      break;
    }
    size_t tid = dl->next_task++;
    pthread_mutex_unlock(&dl->task_mutex);
    /* build batch sample vectors */
    mlx_vector_vector_array batch_fac = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_wells = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_seis = mlx_vector_vector_array_new();
    int err = 0;
    for (int i = 0; i < dl->task_sizes[tid]; ++i) {
      size_t si = dl->tasks[tid][i];
      mlx_vector_array sample_fac = mlx_vector_array_new();
      if (mlx_vector_vector_array_get(&sample_fac, dl->ds->facies, si) ||
          mlx_vector_vector_array_append_value(batch_fac, sample_fac)) {
        mlx_vector_array_free(sample_fac);
        err = 1;
        break;
      }
      mlx_vector_array_free(sample_fac);
      if (mlx_vector_vector_array_size(dl->ds->wells) > 0) {
        mlx_vector_array sample_w = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_w, dl->ds->wells, si) ||
            mlx_vector_vector_array_append_value(batch_wells, sample_w)) {
          mlx_vector_array_free(sample_w);
          err = 1;
          break;
        }
        mlx_vector_array_free(sample_w);
      }
      if (mlx_vector_vector_array_size(dl->ds->seismic) > 0) {
        mlx_vector_array sample_s = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_s, dl->ds->seismic, si) ||
            mlx_vector_vector_array_append_value(batch_seis, sample_s)) {
          mlx_vector_array_free(sample_s);
          err = 1;
          break;
        }
        mlx_vector_array_free(sample_s);
      }
    }

    mlx_vector_array out_fac = mlx_vector_array_new();
    mlx_vector_array out_w = mlx_vector_array_new();
    mlx_vector_array out_s = mlx_vector_array_new();
    if (!err) {
      facies_collate_fn cb =
          dl->collate_cb ? dl->collate_cb : (facies_collate_fn)facies_collate;
      /* serialize MLX eval/collate calls to avoid races inside MLX internals */
      pthread_mutex_lock(&mlx_eval_mutex);
      if (cb(&out_fac, &out_w, &out_s, batch_fac, batch_wells, batch_seis, s,
             dl->collate_ctx)) {
        err = 1;
      }
      pthread_mutex_unlock(&mlx_eval_mutex);
    }

    mlx_vector_vector_array_free(batch_fac);
    mlx_vector_vector_array_free(batch_wells);
    mlx_vector_vector_array_free(batch_seis);

    queue_push(dl, out_fac, out_w, out_s, err);
  }
  /* mark finished when all workers exit; last exiting worker sets finished */
  pthread_mutex_lock(&dl->q_mutex);
  dl->finished = true;
  pthread_cond_broadcast(&dl->q_nonempty);
  pthread_mutex_unlock(&dl->q_mutex);
  mlx_stream_free(s);
  return NULL;
}

/* forward declarations for helpers used by reader/dispatcher threads */
static ssize_t read_all(int fd, void *buf, size_t count);
static ssize_t write_all(int fd, const void *buf, size_t count);
static int read_vector_array_from_fd(int fd, mlx_vector_array *out_vec);
static int serialize_vec_to_fd(int fd, mlx_vector_array vec);

struct reader_arg {
  struct MLXDataloader *dl;
  int worker;
};

static void *proc_reader_thread(void *arg) {
  struct reader_arg *ra = (struct reader_arg *)arg;
  struct MLXDataloader *dl = ra->dl;
  int worker = ra->worker;
  int fd = dl->result_rfds[worker];
  while (1) {
    int32_t status = 0;
    if (read_all(fd, &status, sizeof(status)) <= 0)
      break;
    /* If this is the worker's initial response after startup, the worker
       sends an init-status message followed by an optional text payload.
       The reader treats the first message from each worker as that init
       response and stores any error message for the caller. */
    if (!dl->worker_init_done[worker]) {
      dl->worker_init_done[worker] = 1;
      /* read the message length (may be zero) and optional message */
      uint32_t msglen = 0;
      if (read_all(fd, &msglen, sizeof(msglen)) <= 0) {
        /* couldn't read message length; push generic error */
        mlx_vector_array fac = mlx_vector_array_new();
        mlx_vector_array wells = mlx_vector_array_new();
        mlx_vector_array seis = mlx_vector_array_new();
        queue_push(dl, fac, wells, seis, 1);
      } else {
        if (msglen > 0) {
          char *msg = (char *)malloc(msglen + 1);
          if (msg) {
            if (read_all(fd, msg, msglen) > 0) {
              msg[msglen] = '\0';
              if (status != 0) {
                dl->worker_init_err_msg[worker] = msg;
              } else {
                /* success message, just free */
                free(msg);
              }
            } else {
              free(msg);
              mlx_vector_array fac = mlx_vector_array_new();
              mlx_vector_array wells = mlx_vector_array_new();
              mlx_vector_array seis = mlx_vector_array_new();
              queue_push(dl, fac, wells, seis, 1);
            }
          }
        } else {
          if (status != 0) {
            /* empty message but non-zero status: push generic error */
            mlx_vector_array fac = mlx_vector_array_new();
            mlx_vector_array wells = mlx_vector_array_new();
            mlx_vector_array seis = mlx_vector_array_new();
            queue_push(dl, fac, wells, seis, 1);
          }
        }
      }
      /* After processing init response, continue to next message (which
         may be a normal batch result). */
      continue;
    }
    mlx_vector_array fac = mlx_vector_array_new();
    mlx_vector_array wells = mlx_vector_array_new();
    mlx_vector_array seis = mlx_vector_array_new();
    if (read_vector_array_from_fd(fd, &fac) != 0) {
      mlx_vector_array_free(fac);
      mlx_vector_array_free(wells);
      mlx_vector_array_free(seis);
      queue_push(dl, mlx_vector_array_new(), mlx_vector_array_new(),
                 mlx_vector_array_new(), 1);
      continue;
    }
    if (read_vector_array_from_fd(fd, &wells) != 0) {
      mlx_vector_array_free(fac);
      mlx_vector_array_free(wells);
      mlx_vector_array_free(seis);
      queue_push(dl, mlx_vector_array_new(), mlx_vector_array_new(),
                 mlx_vector_array_new(), 1);
      continue;
    }
    if (read_vector_array_from_fd(fd, &seis) != 0) {
      mlx_vector_array_free(fac);
      mlx_vector_array_free(wells);
      mlx_vector_array_free(seis);
      queue_push(dl, mlx_vector_array_new(), mlx_vector_array_new(),
                 mlx_vector_array_new(), 1);
      continue;
    }
    queue_push(dl, fac, wells, seis, 0);
  }
  return NULL;
}

static void *proc_dispatcher_thread(void *arg) {
  struct MLXDataloader *dl = (struct MLXDataloader *)arg;
  while (1) {
    pthread_mutex_lock(&dl->task_mutex);
    if (dl->next_task >= dl->n_tasks) {
      pthread_mutex_unlock(&dl->task_mutex);
      break;
    }
    size_t tid = dl->next_task++;
    pthread_mutex_unlock(&dl->task_mutex);

    /* wait for queue space */
    pthread_mutex_lock(&dl->q_mutex);
    while (dl->q_count >= dl->q_capacity && !dl->finished) {
      pthread_cond_wait(&dl->q_nonfull, &dl->q_mutex);
    }
    pthread_mutex_unlock(&dl->q_mutex);

    int worker = dl->next_worker % dl->num_workers;
    dl->next_worker = (dl->next_worker + 1) % dl->num_workers;

    uint32_t nidx = (uint32_t)dl->task_sizes[tid];
    if (write_all(dl->task_wfds[worker], &nidx, sizeof(nidx)) <= 0) {
      continue;
    }
    if (nidx > 0) {
      uint64_t *buf = (uint64_t *)malloc(sizeof(uint64_t) * nidx);
      if (!buf)
        continue;
      for (uint32_t i = 0; i < nidx; ++i)
        buf[i] = (uint64_t)dl->tasks[tid][i];
      if (write_all(dl->task_wfds[worker], buf, sizeof(uint64_t) * nidx) <= 0) {
        free(buf);
        continue;
      }
      free(buf);
    }
  }
  return NULL;
}

/* helper: read exactly count bytes or return -1 on error/EOF */
static ssize_t read_all(int fd, void *buf, size_t count) {
  size_t off = 0;
  while (off < count) {
    ssize_t r = read(fd, (char *)buf + off, count - off);
    if (r <= 0)
      return -1;
    off += (size_t)r;
  }
  return (ssize_t)off;
}

/* helper: write exactly count bytes or return -1 on error */
static ssize_t write_all(int fd, const void *buf, size_t count) {
  size_t off = 0;
  while (off < count) {
    ssize_t w = write(fd, (const char *)buf + off, count - off);
    if (w <= 0)
      return -1;
    off += (size_t)w;
  }
  return (ssize_t)off;
}

static int read_vector_array_from_fd(int fd, mlx_vector_array *out_vec) {
  *out_vec = mlx_vector_array_new();
  uint32_t nscales = 0;
  if (read_all(fd, &nscales, sizeof(nscales)) <= 0)
    return 1;
  for (uint32_t si = 0; si < nscales; ++si) {
    uint32_t ndim = 0;
    if (read_all(fd, &ndim, sizeof(ndim)) <= 0)
      return 1;
    int *shape = (int *)malloc(sizeof(int) * ndim);
    if (!shape)
      return 1;
    for (uint32_t d = 0; d < ndim; ++d) {
      int32_t s32 = 0;
      if (read_all(fd, &s32, sizeof(s32)) <= 0) {
        free(shape);
        return 1;
      }
      shape[d] = s32;
    }
    uint64_t uelems = 0;
    if (read_all(fd, &uelems, sizeof(uelems)) <= 0) {
      free(shape);
      return 1;
    }
    size_t elems = (size_t)uelems;
    float *fbuf = NULL;
    if (elems > 0) {
      fbuf = (float *)malloc(sizeof(float) * elems);
      if (!fbuf) {
        free(shape);
        return 1;
      }
      if (read_all(fd, fbuf, sizeof(float) * elems) <= 0) {
        free(shape);
        free(fbuf);
        return 1;
      }
    }
    mlx_array arr = mlx_array_new_data(fbuf, shape, (int)ndim, MLX_FLOAT32);
    if (mlx_vector_array_append_value(*out_vec, arr)) {
      mlx_array_free(arr);
      free(shape);
      if (fbuf)
        free(fbuf);
      return 1;
    }
    mlx_array_free(arr);
    free(shape);
    if (fbuf)
      free(fbuf);
  }
  return 0;
}

static int serialize_vec_to_fd(int fd, mlx_vector_array vec) {
  uint32_t nscales = (uint32_t)mlx_vector_array_size(vec);
  if (write_all(fd, &nscales, sizeof(nscales)) <= 0)
    return 1;
  for (uint32_t si = 0; si < nscales; ++si) {
    mlx_array arr = mlx_array_new();
    if (mlx_vector_array_get(&arr, vec, si) != 0) {
      uint32_t zero = 0;
      write_all(fd, &zero, sizeof(zero));
      continue;
    }
    float *buf = NULL;
    size_t elems = 0;
    int ndim = 0;
    int *shape = NULL;
    if (mlx_array_to_float_buffer(arr, &buf, &elems, &ndim, &shape) != 0) {
      uint32_t zero = 0;
      write_all(fd, &zero, sizeof(zero));
      mlx_array_free(arr);
      continue;
    }
    uint32_t udim = (uint32_t)ndim;
    if (write_all(fd, &udim, sizeof(udim)) <= 0) {
      free(buf);
      free(shape);
      mlx_array_free(arr);
      return 1;
    }
    for (int d = 0; d < ndim; ++d) {
      int32_t s32 = shape[d];
      if (write_all(fd, &s32, sizeof(s32)) <= 0) {
        free(buf);
        free(shape);
        mlx_array_free(arr);
        return 1;
      }
    }
    uint64_t uelems = (uint64_t)elems;
    if (write_all(fd, &uelems, sizeof(uelems)) <= 0) {
      free(buf);
      free(shape);
      mlx_array_free(arr);
      return 1;
    }
    if (elems > 0) {
      if (write_all(fd, buf, sizeof(float) * elems) <= 0) {
        free(buf);
        free(shape);
        mlx_array_free(arr);
        return 1;
      }
    }
    free(buf);
    free(shape);
    mlx_array_free(arr);
  }
  return 0;
}

/* Serialize the entire dataset to fd so a spawned worker can reconstruct it.
 * Format: uint64_t n_samples
 * For each sample i: uint32_t nscales_fac, then for each scale the array
 * (ndim,int32 dims[], uint64 elems, float[elems]) Then similarly for wells
 * (per-sample scales) and seismic.
 */
static int write_dataset_to_fd(int fd, MLXPyramidsDataset *ds) {
  if (!ds)
    return 1;
  uint64_t n_samples = (uint64_t)ds->n_samples;
  if (write_all(fd, &n_samples, sizeof(n_samples)) <= 0)
    return 1;
  if (write_vv_fd(fd, ds->facies, n_samples))
    return 1;
  if (write_vv_fd(fd, ds->wells, n_samples))
    return 1;
  if (write_vv_fd(fd, ds->seismic, n_samples))
    return 1;
  return 0;
}

#ifndef FACIES_RESEED_TOKEN
#define FACIES_RESEED_TOKEN (UINT32_MAX - 1u)
#define FACIES_TERM_TOKEN (UINT32_MAX)
#endif

/* Worker process loop: read task from parent (fd), process, write result to
 * parent (fd) Protocol (parent->worker): uint32_t n_indices; followed by
 * n_indices of uint64_t indices. If n_indices == UINT32_MAX -> exit. Protocol
 * (worker->parent): int32_t status (0=ok,1=error), then uint32_t
 * n_scales_facies, for each scale: uint32_t ndim, int32_t dims[ndim], uint64_t
 * elems, float[elems] After facies, uint32_t n_scales_wells, ... then
 * n_scales_seismic, ...
 */
static int worker_process_loop(int task_fd, int result_fd,
                               MLXPyramidsDataset *ds,
                               facies_collate_fn collate_cb,
                               void *collate_ctx) {
  while (1) {
    uint32_t n_indices = 0;
    ssize_t r = read(task_fd, &n_indices, sizeof(n_indices));
    if (r <= 0)
      return 0;
    if (n_indices == UINT32_MAX) {
      return 0; /* exit */
    }
    size_t ni = (size_t)n_indices;
    uint64_t *idxs = (uint64_t *)malloc(sizeof(uint64_t) * ni);
    ssize_t rr = read(task_fd, idxs, sizeof(uint64_t) * ni);
    if (rr != (ssize_t)(sizeof(uint64_t) * ni)) {
      free(idxs);
      int32_t status = 1;
      write(result_fd, &status, sizeof(status));
      continue;
    }

    /* build batch vectors */
    mlx_vector_vector_array batch_fac = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_wells = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_seis = mlx_vector_vector_array_new();
    int err = 0;
    for (size_t i = 0; i < ni; ++i) {
      size_t si = (size_t)idxs[i];
      mlx_vector_array sample_fac = mlx_vector_array_new();
      if (mlx_vector_vector_array_get(&sample_fac, ds->facies, si) ||
          mlx_vector_vector_array_append_value(batch_fac, sample_fac)) {
        mlx_vector_array_free(sample_fac);
        err = 1;
        break;
      }
      mlx_vector_array_free(sample_fac);
      if (mlx_vector_vector_array_size(ds->wells) > 0) {
        mlx_vector_array sample_w = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_w, ds->wells, si) ||
            mlx_vector_vector_array_append_value(batch_wells, sample_w)) {
          mlx_vector_array_free(sample_w);
          err = 1;
          break;
        }
        mlx_vector_array_free(sample_w);
      }
      if (mlx_vector_vector_array_size(ds->seismic) > 0) {
        mlx_vector_array sample_s = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_s, ds->seismic, si) ||
            mlx_vector_vector_array_append_value(batch_seis, sample_s)) {
          mlx_vector_array_free(sample_s);
          err = 1;
          break;
        }
        mlx_vector_array_free(sample_s);
      }
    }
    free(idxs);

    if (err) {
      int32_t status = 1;
      write(result_fd, &status, sizeof(status));
      mlx_vector_vector_array_free(batch_fac);
      mlx_vector_vector_array_free(batch_wells);
      mlx_vector_vector_array_free(batch_seis);
      continue;
    }

    mlx_vector_array out_fac = mlx_vector_array_new();
    mlx_vector_array out_w = mlx_vector_array_new();
    mlx_vector_array out_s = mlx_vector_array_new();
    facies_collate_fn cb =
        collate_cb ? collate_cb : (facies_collate_fn)facies_collate;
    int rc = cb(&out_fac, &out_w, &out_s, batch_fac, batch_wells, batch_seis,
                mlx_default_cpu_stream_new(), collate_ctx);
    mlx_vector_vector_array_free(batch_fac);
    mlx_vector_vector_array_free(batch_wells);
    mlx_vector_vector_array_free(batch_seis);

    if (rc != 0) {
      int32_t status = 1;
      write(result_fd, &status, sizeof(status));
      mlx_vector_array_free(out_fac);
      mlx_vector_array_free(out_w);
      mlx_vector_array_free(out_s);
      continue;
    }

    int32_t status = 0;
    write(result_fd, &status, sizeof(status));

    /* serialize results back to parent */
    if (serialize_vec_to_fd(result_fd, out_fac) != 0) {
      /* best effort: continue to free */
    }
    if (serialize_vec_to_fd(result_fd, out_w) != 0) {
    }
    if (serialize_vec_to_fd(result_fd, out_s) != 0) {
    }

    mlx_vector_array_free(out_fac);
    mlx_vector_array_free(out_w);
    mlx_vector_array_free(out_s);
  }
  return 0;
}

int facies_dataloader_reset(struct MLXDataloader *dl) {
  if (!dl)
    return 1;
  dl->idx = 0;
  if (dl->shuffle)
    shuffle_indices(dl->indices, dl->n_indices, 0);
  /* rebuild tasks and reset queue */
  if (dl->num_workers > 0) {
    pthread_mutex_lock(&dl->task_mutex);
    /* reset user-provided samplers if available */
    if (dl->sampler_reset && dl->sampler_ctx)
      dl->sampler_reset(dl->sampler_ctx);
    if (dl->batch_sampler_reset && dl->batch_sampler_ctx)
      dl->batch_sampler_reset(dl->batch_sampler_ctx);
    /* free old tasks */
    if (dl->tasks) {
      for (size_t t = 0; t < dl->n_tasks; ++t)
        free(dl->tasks[t]);
      free(dl->tasks);
      free(dl->task_sizes);
      dl->tasks = NULL;
      dl->task_sizes = NULL;
      dl->n_tasks = 0;
    }
    /* rebuild tasks based on (possibly reset) samplers */
    build_tasks(dl);
    dl->next_task = 0;
    pthread_mutex_unlock(&dl->task_mutex);
    pthread_mutex_lock(&dl->q_mutex);
    dl->q_head = dl->q_tail = dl->q_count = 0;
    dl->finished = false;
    pthread_cond_broadcast(&dl->q_nonfull);
    pthread_mutex_unlock(&dl->q_mutex);
    /* produce a new iterator base seed and notify workers */
    if (dl->base_seed != 0) {
      dl->iterator_base_seed =
          (uint64_t)dl->base_seed + (dl->epoch_counter + 1);
    } else {
      dl->iterator_base_seed = ((uint64_t)time(NULL) << 32) ^ (uint64_t)rand();
    }
    dl->epoch_counter++;
    /* notify pthread workers by task_mutex visibility; for process workers
     * send a reseed token with new seed so workers can reseed
     * deterministically. */
    if (dl->pids) {
      for (int i = 0; i < dl->num_workers; ++i) {
        uint32_t token = FACIES_RESEED_TOKEN;
        uint64_t seed = dl->iterator_base_seed + (uint64_t)i;
        write_all(dl->task_wfds[i], &token, sizeof(token));
        write_all(dl->task_wfds[i], &seed, sizeof(seed));
      }
    }
  }
  return 0;
}

int facies_dataloader_next(struct MLXDataloader *dl,
                           mlx_vector_array *out_facies,
                           mlx_vector_array *out_wells,
                           mlx_vector_array *out_seismic, const mlx_stream s) {
  if (!dl)
    return 1;
  if (dl->num_workers <= 0) {
    /* fallback: single-threaded behavior */
    size_t remaining = dl->n_indices - dl->idx;
    if (remaining == 0)
      return 2; // finished
    if (remaining < dl->batch_size && dl->drop_last)
      return 2;
    size_t cur_batch = dl->batch_size;
    if (remaining < dl->batch_size)
      cur_batch = remaining;

    // prepare temporary vector_vector_array to hold per-sample vector_array
    mlx_vector_vector_array batch_fac = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_wells = mlx_vector_vector_array_new();
    mlx_vector_vector_array batch_seismic = mlx_vector_vector_array_new();

    for (size_t i = 0; i < cur_batch; ++i) {
      size_t si = dl->indices[dl->idx + i];
      mlx_vector_array sample_fac = mlx_vector_array_new();
      if (mlx_vector_vector_array_get(&sample_fac, dl->ds->facies, si)) {
        // cleanup
        mlx_vector_vector_array_free(batch_fac);
        mlx_vector_vector_array_free(batch_wells);
        mlx_vector_vector_array_free(batch_seismic);
        return 1;
      }
      if (mlx_vector_vector_array_append_value(batch_fac, sample_fac)) {
        mlx_vector_array_free(sample_fac);
        mlx_vector_vector_array_free(batch_fac);
        mlx_vector_vector_array_free(batch_wells);
        mlx_vector_vector_array_free(batch_seismic);
        return 1;
      }
      mlx_vector_array_free(sample_fac);

      // wells/seismic may be shorter; try to get and append if exists
      if (mlx_vector_vector_array_size(dl->ds->wells) > 0) {
        mlx_vector_array sample_w = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_w, dl->ds->wells, si)) {
          mlx_vector_vector_array_free(batch_fac);
          mlx_vector_vector_array_free(batch_wells);
          mlx_vector_vector_array_free(batch_seismic);
          return 1;
        }
        if (mlx_vector_vector_array_append_value(batch_wells, sample_w)) {
          mlx_vector_array_free(sample_w);
          mlx_vector_vector_array_free(batch_fac);
          mlx_vector_vector_array_free(batch_wells);
          mlx_vector_vector_array_free(batch_seismic);
          return 1;
        }
        mlx_vector_array_free(sample_w);
      }

      if (mlx_vector_vector_array_size(dl->ds->seismic) > 0) {
        mlx_vector_array sample_s = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_s, dl->ds->seismic, si)) {
          mlx_vector_vector_array_free(batch_fac);
          mlx_vector_vector_array_free(batch_wells);
          mlx_vector_vector_array_free(batch_seismic);
          return 1;
        }
        if (mlx_vector_vector_array_append_value(batch_seismic, sample_s)) {
          mlx_vector_array_free(sample_s);
          mlx_vector_vector_array_free(batch_fac);
          mlx_vector_vector_array_free(batch_wells);
          mlx_vector_vector_array_free(batch_seismic);
          return 1;
        }
        mlx_vector_array_free(sample_s);
      }
    }

    /* Use provided collate callback if present */
    facies_collate_fn cb =
        dl->collate_cb ? dl->collate_cb : (facies_collate_fn)facies_collate;
    int rc = cb(out_facies, out_wells, out_seismic, batch_fac, batch_wells,
                batch_seismic, s, dl->collate_ctx);

    dl->idx += cur_batch;

    mlx_vector_vector_array_free(batch_fac);
    mlx_vector_vector_array_free(batch_wells);
    mlx_vector_vector_array_free(batch_seismic);

    return rc == 0 ? 0 : 1;
  }

  /* multithreaded path: pop a precomputed result from queue */
  /* If process worker backend was created, synchronously send next task
   * to a worker and read the serialized result back. This keeps the
   * implementation simple (no background reader thread) while providing
   * correct IPC handling for the fork-based workers. */
  if (dl->pids) {
    /* ensure dispatcher/reader threads are started (created in constructor)
     * and then consume precomputed results from the queue like the
     * threaded backend. */
    int perr = 0;
    int qrc = queue_pop_timeout(dl, out_facies, out_wells, out_seismic, &perr);
    if (qrc != 0) {
      return 1; /* timeout or error */
    }
    if (perr != 0)
      return 1;
    pthread_mutex_lock(&dl->q_mutex);
    bool fin = dl->finished && dl->q_count == 0;
    pthread_mutex_unlock(&dl->q_mutex);
    if (fin)
      return 2;
    return 0;
  }

  /* fallback: threaded queue path */
  int perr = 0;
  int qrc = queue_pop_timeout(dl, out_facies, out_wells, out_seismic, &perr);
  if (qrc != 0) {
    return 1; /* timeout or error */
  }
  if (perr != 0)
    return 1;
  pthread_mutex_lock(&dl->q_mutex);
  bool fin = dl->finished && dl->q_count == 0;
  pthread_mutex_unlock(&dl->q_mutex);
  if (fin)
    return 2;
  return 0;
}

int facies_dataloader_free(struct MLXDataloader *dl) {
  if (!dl)
    return 0;
  /* signal workers to finish and join */
  if (dl->num_workers > 0 && dl->workers) {
    pthread_mutex_lock(&dl->task_mutex);
    dl->next_task = dl->n_tasks; /* no more tasks */
    pthread_mutex_unlock(&dl->task_mutex);
    for (int i = 0; i < dl->num_workers; ++i)
      pthread_join(dl->workers[i], NULL);
  }
  if (dl->workers)
    free(dl->workers);
  /* if process workers were spawned, stop dispatcher, signal termination and
   * reap */
  if (dl->pids) {
    /* stop dispatcher by marking no more tasks */
    pthread_mutex_lock(&dl->task_mutex);
    dl->next_task = dl->n_tasks;
    pthread_mutex_unlock(&dl->task_mutex);
    pthread_cond_broadcast(&dl->q_nonfull);
    /* join dispatcher thread */
    if (dl->process_workers_started) {
      pthread_join(dl->proc_dispatcher, NULL);
    }
    /* send termination token to workers so they exit and readers return */
    for (int i = 0; i < dl->num_workers; ++i) {
      if (dl->task_wfds && dl->task_wfds[i] >= 0) {
        uint32_t term = UINT32_MAX;
        write_all(dl->task_wfds[i], &term, sizeof(term));
        close(dl->task_wfds[i]);
      }
    }
    /* join reader threads */
    if (dl->process_workers_started && dl->proc_readers) {
      for (int i = 0; i < dl->num_workers; ++i) {
        pthread_join(dl->proc_readers[i], NULL);
      }
    }
    /* close result fds and reap children */
    for (int i = 0; i < dl->num_workers; ++i) {
      if (dl->result_rfds && dl->result_rfds[i] >= 0) {
        close(dl->result_rfds[i]);
      }
      if (dl->pids[i] > 0) {
        int st = 0;
        waitpid(dl->pids[i], &st, 0);
      }
    }
    free(dl->pids);
    if (dl->task_wfds)
      free(dl->task_wfds);
    if (dl->result_rfds)
      free(dl->result_rfds);
    if (dl->proc_readers)
      free(dl->proc_readers);
    if (dl->reader_args)
      free(dl->reader_args);
  }
  if (dl->worker_init_err_msg) {
    for (int i = 0; i < dl->num_workers; ++i) {
      if (dl->worker_init_err_msg[i])
        free(dl->worker_init_err_msg[i]);
    }
    free(dl->worker_init_err_msg);
  }
  if (dl->worker_init_done)
    free(dl->worker_init_done);
  if (dl->tasks) {
    for (size_t t = 0; t < dl->n_tasks; ++t)
      free(dl->tasks[t]);
    free(dl->tasks);
    free(dl->task_sizes);
  }
  if (dl->queue) {
    /* free queued arrays */
    for (size_t i = 0; i < dl->q_capacity; ++i) {
      mlx_vector_array_free(dl->queue[i].facies);
      mlx_vector_array_free(dl->queue[i].wells);
      mlx_vector_array_free(dl->queue[i].seismic);
    }
    free(dl->queue);
  }
  if (dl->worker_init_lib)
    free(dl->worker_init_lib);
  if (dl->worker_init_sym)
    free(dl->worker_init_sym);
  pthread_mutex_destroy(&dl->task_mutex);
  pthread_mutex_destroy(&dl->q_mutex);
  pthread_cond_destroy(&dl->q_nonempty);
  pthread_cond_destroy(&dl->q_nonfull);
  if (dl->indices)
    free(dl->indices);
  free(dl);
  return 0;
}

int facies_dataloader_new_ex(
    struct MLXDataloader **out, MLXPyramidsDataset *ds, size_t batch_size,
    bool shuffle, bool drop_last, unsigned int seed, int num_workers,
    int prefetch_factor, bool persistent_workers, int timeout_ms,
    facies_collate_fn collate, void *collate_ctx, bool pin_memory,
    facies_sampler_next_fn sampler_next, void *sampler_ctx,
    facies_sampler_reset_fn sampler_reset,
    facies_batch_sampler_next_fn batch_sampler_next, void *batch_sampler_ctx,
    facies_sampler_reset_fn batch_sampler_reset,
    facies_worker_init_fn worker_init, void *worker_init_ctx,
    unsigned int worker_init_ctx_len, const char *worker_init_lib,
    const char *worker_init_sym) {
  if (!out || !ds)
    return 1;
  struct MLXDataloader *dl =
      (struct MLXDataloader *)calloc(1, sizeof(struct MLXDataloader));
  if (!dl)
    return 1;
  dl->ds = ds;
  dl->batch_size = batch_size;
  dl->n_indices = ds->n_samples;
  dl->indices = (size_t *)malloc(sizeof(size_t) * dl->n_indices);
  if (!dl->indices) {
    free(dl);
    return 1;
  }
  for (size_t i = 0; i < dl->n_indices; ++i)
    dl->indices[i] = i;
  dl->shuffle = shuffle;
  dl->drop_last = drop_last;
  dl->idx = 0;
  if (dl->shuffle)
    shuffle_indices(dl->indices, dl->n_indices, seed);

  /* extended fields */
  dl->num_workers = num_workers > 0 ? num_workers : 0;
  dl->prefetch_factor = prefetch_factor > 0 ? prefetch_factor : 2;
  dl->persistent_workers = persistent_workers;
  dl->timeout_ms = timeout_ms;
  dl->collate_cb = collate;
  dl->collate_ctx = collate_ctx;
  dl->pin_memory = pin_memory;
  dl->sampler_next = sampler_next;
  dl->sampler_ctx = sampler_ctx;
  dl->sampler_reset = sampler_reset;
  dl->batch_sampler_next = batch_sampler_next;
  dl->batch_sampler_ctx = batch_sampler_ctx;
  dl->batch_sampler_reset = batch_sampler_reset;
  dl->worker_init = worker_init;
  dl->worker_init_ctx = worker_init_ctx;
  dl->worker_init_ctx_data = NULL;
  dl->worker_init_ctx_len = 0;
  dl->base_seed = seed;
  dl->worker_init_lib = NULL;
  dl->worker_init_sym = NULL;
  if (worker_init_lib)
    dl->worker_init_lib = strdup(worker_init_lib);
  if (worker_init_sym)
    dl->worker_init_sym = strdup(worker_init_sym);
  if (worker_init_ctx && worker_init_ctx_len > 0) {
    dl->worker_init_ctx_data = malloc(worker_init_ctx_len);
    if (dl->worker_init_ctx_data) {
      memcpy(dl->worker_init_ctx_data, worker_init_ctx, worker_init_ctx_len);
      dl->worker_init_ctx_len = worker_init_ctx_len;
    }
  }
  dl->iterator_base_seed = (uint64_t)seed;
  dl->epoch_counter = 0;

  /* prepare task list */
  if (build_tasks(dl)) {
    free(dl->indices);
    free(dl);
    return 1;
  }

  /* init sync primitives */
  pthread_mutex_init(&dl->task_mutex, NULL);
  pthread_mutex_init(&dl->q_mutex, NULL);
  pthread_cond_init(&dl->q_nonempty, NULL);
  pthread_cond_init(&dl->q_nonfull, NULL);

  /* setup queue capacity based on prefetch_factor and num_workers */
  dl->q_capacity = (size_t)(dl->prefetch_factor *
                            (dl->num_workers > 0 ? dl->num_workers : 1));
  if (dl->q_capacity < 1)
    dl->q_capacity = 1;
  dl->queue = (batch_item *)calloc(dl->q_capacity, sizeof(batch_item));
  dl->q_head = dl->q_tail = dl->q_count = 0;
  dl->finished = false;

  /* initialize queue entries to empty vectors so frees are safe */
  for (size_t i = 0; i < dl->q_capacity; ++i) {
    dl->queue[i].facies = mlx_vector_array_new();
    dl->queue[i].wells = mlx_vector_array_new();
    dl->queue[i].seismic = mlx_vector_array_new();
    dl->queue[i].error = 0;
  }

  if (dl->num_workers > 0) {
    /* ensure MLX runtime globals are initialized in main thread before
     * spawning worker threads/processes to avoid concurrent static init */
    mlx_stream _init_s = mlx_default_cpu_stream_new();
    mlx_stream_free(_init_s);
#ifdef __APPLE__
    /* On macOS, forking after libraries that create GPU/dispatch resources
     * can crash in the child. Use pthread workers as a safe fallback. */
    dl->workers = (pthread_t *)calloc(dl->num_workers, sizeof(pthread_t));
    dl->worker_args =
        calloc(dl->num_workers, sizeof(struct pthread_worker_arg));
    struct pthread_worker_arg *wargs = dl->worker_args;
    for (int i = 0; i < dl->num_workers; ++i) {
      wargs[i].dl = dl;
      wargs[i].worker_id = i;
      wargs[i].last_epoch = (uint64_t)-1;
      pthread_create(&dl->workers[i], NULL, worker_thread, &wargs[i]);
    }
#else
    /* If num_workers > 0 we will spawn processes for worker backend */
    dl->pids = (pid_t *)calloc(dl->num_workers, sizeof(pid_t));
    dl->task_wfds = (int *)calloc(dl->num_workers, sizeof(int));
    dl->result_rfds = (int *)calloc(dl->num_workers, sizeof(int));
    dl->worker_init_err_msg = (char **)calloc(dl->num_workers, sizeof(char *));
    dl->worker_init_done = (int *)calloc(dl->num_workers, sizeof(int));
    for (int i = 0; i < dl->num_workers; ++i) {
      int taskpipe[2];
      int resultpipe[2];
      if (pipe(taskpipe) != 0 || pipe(resultpipe) != 0) {
        /* cleanup */
        continue;
      }
      pid_t pid = fork();
      if (pid == 0) {
        /* child: exec the spawn worker executable which will read the
         * serialized dataset from the task fd and then enter its loop.
         */
        close(taskpipe[1]);
        close(resultpipe[0]);
        char task_fd_str[32];
        char result_fd_str[32];
        snprintf(task_fd_str, sizeof(task_fd_str), "%d", taskpipe[0]);
        snprintf(result_fd_str, sizeof(result_fd_str), "%d", resultpipe[1]);
        setenv("FACIES_WORKER_TASK_FD", task_fd_str, 1);
        setenv("FACIES_WORKER_RESULT_FD", result_fd_str, 1);
        /* exec worker binary located in current working directory */
        const char *exe = "./facies_worker";
        /* resolve exe to absolute path for logging */
        char exe_resolved[PATH_MAX];
        const char *exe_to_report = exe;
        if (realpath(exe, exe_resolved) != NULL) {
          exe_to_report = exe_resolved;
        }
        fprintf(stderr,
                "facies_dataloader: child pid=%d exec '%s' (resolved '%s')\n",
                (int)getpid(), exe, exe_to_report);
        const char *argv_exec[2] = {"facies_worker", NULL};
        execv(exe, (char *const *)argv_exec);
        /* if exec failed, log error and exit child */
        fprintf(stderr, "facies_dataloader: execv('%s') failed: %s\n", exe,
                strerror(errno));
        _exit(1);
      } else if (pid > 0) {
        /* parent */
        close(taskpipe[0]);
        close(resultpipe[1]);
        dl->pids[i] = pid;
        dl->task_wfds[i] = taskpipe[1];
        dl->result_rfds[i] = resultpipe[0];
        /* immediately send an init header (worker id + seed) then optional
         * init lib/sym strings and finally the serialized dataset so the
         * spawned worker can dlopen/dlsym and initialize itself. */
        uint32_t wid = (uint32_t)i;
        uint64_t wseed = (uint64_t)(dl->iterator_base_seed + (uint64_t)i);
        write_all(dl->task_wfds[i], &wid, sizeof(wid));
        write_all(dl->task_wfds[i], &wseed, sizeof(wseed));
        /* write lib path and symbol as uint32_t length-prefixed bytes */
        if (dl->worker_init_lib) {
          uint32_t l = (uint32_t)strlen(dl->worker_init_lib);
          write_all(dl->task_wfds[i], &l, sizeof(l));
          write_all(dl->task_wfds[i], dl->worker_init_lib, l);
        } else {
          uint32_t l = 0;
          write_all(dl->task_wfds[i], &l, sizeof(l));
        }
        if (dl->worker_init_sym) {
          uint32_t l = (uint32_t)strlen(dl->worker_init_sym);
          write_all(dl->task_wfds[i], &l, sizeof(l));
          write_all(dl->task_wfds[i], dl->worker_init_sym, l);
        } else {
          uint32_t l = 0;
          write_all(dl->task_wfds[i], &l, sizeof(l));
        }
        /* send serialized worker_init_ctx */
        if (dl->worker_init_ctx_data && dl->worker_init_ctx_len > 0) {
          uint32_t l = dl->worker_init_ctx_len;
          write_all(dl->task_wfds[i], &l, sizeof(l));
          write_all(dl->task_wfds[i], dl->worker_init_ctx_data, l);
        } else {
          uint32_t l = 0;
          write_all(dl->task_wfds[i], &l, sizeof(l));
        }
        write_dataset_to_fd(dl->task_wfds[i], dl->ds);
      } else {
        /* fork failed */
      }
    }
    dl->next_worker = 0;
    /* start per-worker reader threads and a dispatcher thread */
    dl->proc_readers = (pthread_t *)calloc(dl->num_workers, sizeof(pthread_t));
    dl->reader_args =
        (struct reader_arg *)calloc(dl->num_workers, sizeof(struct reader_arg));
    for (int i = 0; i < dl->num_workers; ++i) {
      dl->reader_args[i].dl = dl;
      dl->reader_args[i].worker = i;
      if (dl->result_rfds[i] >= 0) {
        pthread_create(&dl->proc_readers[i], NULL, proc_reader_thread,
                       &dl->reader_args[i]);
      }
    }
    /* wait for per-worker init responses (proc_reader_thread sets
     * dl->worker_init_done and dl->worker_init_err_msg). If any worker
     * reports an init error, kill children and fail fast. */
    {
      int wait_ms = 5000;
      struct timeval st, now;
      gettimeofday(&st, NULL);
      int all_done = 0;
      while (1) {
        all_done = 1;
        for (int wi = 0; wi < dl->num_workers; ++wi) {
          if (!dl->worker_init_done[wi]) {
            all_done = 0;
            break;
          }
        }
        if (all_done)
          break;
        gettimeofday(&now, NULL);
        long elapsed =
            (now.tv_sec - st.tv_sec) * 1000 + (now.tv_usec - st.tv_usec) / 1000;
        if (elapsed > wait_ms)
          break;
        usleep(50000);
      }
      for (int wi = 0; wi < dl->num_workers; ++wi) {
        if (dl->worker_init_err_msg && dl->worker_init_err_msg[wi]) {
          /* mark global fail-fast so other threads/tools can observe */
          ff_set();
          for (int j = 0; j < dl->num_workers; ++j) {
            if (dl->pids && dl->pids[j] > 0)
              kill(dl->pids[j], SIGKILL);
          }
          facies_dataloader_free(dl);
          return 1;
        }
      }
    }
    pthread_create(&dl->proc_dispatcher, NULL, proc_dispatcher_thread, dl);
    dl->process_workers_started = true;
#endif
  }

  *out = dl;
  return 0;
}
