#include "../datasets/collate.h"
#include "../datasets/dataloader.h"
#include "../utils_extra.h"
#include "mlx/c/array.h"
#include "mlx/c/vector.h"

#ifndef FACIES_RESEED_TOKEN
#define FACIES_RESEED_TOKEN (UINT32_MAX - 1u)
#define FACIES_TERM_TOKEN (UINT32_MAX)
#endif

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static int read_dataset_from_fd(int fd, mlx_vector_vector_array *out_fac,
                                mlx_vector_vector_array *out_well,
                                mlx_vector_vector_array *out_seis) {
  uint64_t n_samples = 0;
  if (read_all(fd, &n_samples, sizeof(n_samples)) <= 0)
    return 1;
  size_t n = (size_t)n_samples;
  *out_fac = mlx_vector_vector_array_new();
  *out_well = mlx_vector_vector_array_new();
  *out_seis = mlx_vector_vector_array_new();
  for (size_t i = 0; i < n; ++i) {
    mlx_vector_array sample = mlx_vector_array_new();
    if (read_vector_array_from_fd(fd, &sample) != 0) {
      mlx_vector_array_free(sample);
      return 1;
    }
    if (mlx_vector_vector_array_append_value(*out_fac, sample)) {
      mlx_vector_array_free(sample);
      return 1;
    }
    mlx_vector_array_free(sample);
  }
  for (size_t i = 0; i < n; ++i) {
    mlx_vector_array sample = mlx_vector_array_new();
    if (read_vector_array_from_fd(fd, &sample) != 0) {
      mlx_vector_array_free(sample);
      return 1;
    }
    if (mlx_vector_vector_array_append_value(*out_well, sample)) {
      mlx_vector_array_free(sample);
      return 1;
    }
    mlx_vector_array_free(sample);
  }
  for (size_t i = 0; i < n; ++i) {
    mlx_vector_array sample = mlx_vector_array_new();
    if (read_vector_array_from_fd(fd, &sample) != 0) {
      mlx_vector_array_free(sample);
      return 1;
    }
    if (mlx_vector_vector_array_append_value(*out_seis, sample)) {
      mlx_vector_array_free(sample);
      return 1;
    }
    mlx_vector_array_free(sample);
  }
  return 0;
}

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

int main(int argc, char **argv) {
  const char *task_fd_env = getenv("FACIES_WORKER_TASK_FD");
  const char *result_fd_env = getenv("FACIES_WORKER_RESULT_FD");
  if (!task_fd_env || !result_fd_env) {
    fprintf(stderr, "facies_worker: missing env fds\n");
    return 1;
  }
  int task_fd = atoi(task_fd_env);
  int result_fd = atoi(result_fd_env);

  /* read serialized dataset */
  /* read init header: worker id and seed */
  uint32_t worker_id = 0;
  uint64_t worker_seed = 0;
  if (read_all(task_fd, &worker_id, sizeof(worker_id)) <= 0) {
    fprintf(stderr, "facies_worker: failed to read init header\n");
    return 1;
  }
  if (read_all(task_fd, &worker_seed, sizeof(worker_seed)) <= 0) {
    fprintf(stderr, "facies_worker: failed to read seed\n");
    return 1;
  }
  /* seed PRNG deterministically */
  srand((unsigned int)worker_seed);

  /* read optional init lib path and symbol */
  uint32_t liblen = 0;
  if (read_all(task_fd, &liblen, sizeof(liblen)) <= 0) {
    fprintf(stderr, "facies_worker: failed to read liblen\n");
    return 1;
  }
  char *libpath = NULL;
  if (liblen > 0) {
    libpath = (char *)malloc(liblen + 1);
    if (!libpath)
      return 1;
    if (read_all(task_fd, libpath, liblen) <= 0) {
      free(libpath);
      return 1;
    }
    libpath[liblen] = '\0';
  }
  uint32_t symlen = 0;
  if (read_all(task_fd, &symlen, sizeof(symlen)) <= 0) {
    if (libpath)
      free(libpath);
    return 1;
  }
  char *sym = NULL;
  if (symlen > 0) {
    sym = (char *)malloc(symlen + 1);
    if (!sym) {
      if (libpath)
        free(libpath);
      return 1;
    }
    if (read_all(task_fd, sym, symlen) <= 0) {
      free(sym);
      if (libpath)
        free(libpath);
      return 1;
    }
    sym[symlen] = '\0';
  }
  /* read worker_init_ctx bytes */
  uint32_t ctx_len = 0;
  if (read_all(task_fd, &ctx_len, sizeof(ctx_len)) != sizeof(ctx_len)) {
    fprintf(stderr, "worker %d failed reading ctx_len\n", worker_id);
    if (libpath)
      free(libpath);
    if (sym)
      free(sym);
    return 1;
  }
  void *worker_init_ctx = NULL;
  if (ctx_len > 0) {
    worker_init_ctx = malloc(ctx_len);
    if (!worker_init_ctx) {
      if (libpath)
        free(libpath);
      if (sym)
        free(sym);
      return 1;
    }
    if (read_all(task_fd, worker_init_ctx, ctx_len) <= 0) {
      free(worker_init_ctx);
      if (libpath)
        free(libpath);
      if (sym)
        free(sym);
      return 1;
    }
  }

  mlx_vector_vector_array facies;
  mlx_vector_vector_array wells;
  mlx_vector_vector_array seismic;
  if (read_dataset_from_fd(task_fd, &facies, &wells, &seismic) != 0) {
    fprintf(stderr, "facies_worker: failed to read dataset\n");
    if (libpath)
      free(libpath);
    if (sym)
      free(sym);
    return 1;
  }

  /* if provided, dlopen the lib and resolve/call the init symbol.
     We support new signature int fn(int, void*) with ctx bytes, and
     fall back to int fn(int, uint64_t). Forward a structured init
     status+message to the parent on result_fd. */
  void *dlh = NULL;
  /* persistent init function pointers for reseed handling */
  typedef int (*init_fn_ctx_t)(int, void *);
  typedef int (*init_fn_seed_t)(int, uint64_t);
  init_fn_ctx_t init_ctx_fn = NULL;
  init_fn_seed_t init_seed_fn = NULL;
  int32_t init_status = 0;
  char *init_msg = NULL;
  if (libpath && sym) {
    dlh = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
    if (!dlh) {
      init_status = 1;
      const char *e = dlerror();
      init_msg = e ? strdup(e) : strdup("dlopen failed");
    } else {
      dlerror();
      /* try (int, void*) */
      init_ctx_fn = (init_fn_ctx_t)dlsym(dlh, sym);
      if (init_ctx_fn) {
        int r = init_ctx_fn((int)worker_id, worker_init_ctx);
        if (r != 0) {
          init_status = 1;
          init_msg = (char *)malloc(64);
          if (init_msg)
            snprintf(init_msg, 64, "user init(ctx) returned %d", r);
        }
      } else {
        /* fallback to (int, uint64_t) */
        dlerror();
        init_seed_fn = (init_fn_seed_t)dlsym(dlh, sym);
        if (init_seed_fn) {
          int r = init_seed_fn((int)worker_id, worker_seed);
          if (r != 0) {
            init_status = 1;
            init_msg = (char *)malloc(64);
            if (init_msg)
              snprintf(init_msg, 64, "user init(seed) returned %d", r);
          }
        } else {
          const char *e = dlerror();
          init_status = 1;
          init_msg = e ? strdup(e) : strdup("dlsym failed");
        }
      }
    }
  }

  /* send init status and optional message to parent */
  if (write_all(result_fd, &init_status, sizeof(init_status)) <= 0) {
    /* cannot report init status; exit */
    if (init_msg)
      free(init_msg);
    if (libpath)
      free(libpath);
    if (sym)
      free(sym);
    if (dlh)
      dlclose(dlh);
    mlx_vector_vector_array_free(facies);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    return 1;
  }
  uint32_t mlen = init_msg ? (uint32_t)strlen(init_msg) : 0;
  if (write_all(result_fd, &mlen, sizeof(mlen)) <= 0) {
    if (init_msg)
      free(init_msg);
    if (libpath)
      free(libpath);
    if (sym)
      free(sym);
    if (dlh)
      dlclose(dlh);
    mlx_vector_vector_array_free(facies);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    return 1;
  }
  if (mlen > 0) {
    if (write_all(result_fd, init_msg, mlen) <= 0) {
      if (init_msg)
        free(init_msg);
      if (libpath)
        free(libpath);
      if (sym)
        free(sym);
      if (dlh)
        dlclose(dlh);
      mlx_vector_vector_array_free(facies);
      mlx_vector_vector_array_free(wells);
      mlx_vector_vector_array_free(seismic);
      return 1;
    }
  }
  if (init_msg)
    free(init_msg);
  if (libpath)
    free(libpath);
  if (sym)
    free(sym);
  /* worker will use local facies/wells/seismic structures directly */

  /* worker loop: read tasks as indices and process */
  while (1) {
    uint32_t n_indices = 0;
    ssize_t r = read_all(task_fd, &n_indices, sizeof(n_indices));
    if (r <= 0)
      break;
    if (n_indices == FACIES_TERM_TOKEN)
      break;
    if (n_indices == FACIES_RESEED_TOKEN) {
      uint64_t seed = 0;
      if (read_all(task_fd, &seed, sizeof(seed)) <= 0)
        break;
      srand((unsigned int)seed);
      /* call user init fn on reseed if present */
      if (init_ctx_fn) {
        init_ctx_fn((int)worker_id, worker_init_ctx);
      } else if (init_seed_fn) {
        init_seed_fn((int)worker_id, seed);
      }
      continue;
    }
    size_t ni = (size_t)n_indices;
    uint64_t *idxs = (uint64_t *)malloc(sizeof(uint64_t) * ni);
    if (!idxs)
      break;
    if (read_all(task_fd, idxs, sizeof(uint64_t) * ni) !=
        (ssize_t)(sizeof(uint64_t) * ni)) {
      free(idxs);
      int32_t status = 1;
      write_all(result_fd, &status, sizeof(status));
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
      if (mlx_vector_vector_array_get(&sample_fac, facies, si) ||
          mlx_vector_vector_array_append_value(batch_fac, sample_fac)) {
        mlx_vector_array_free(sample_fac);
        err = 1;
        break;
      }
      mlx_vector_array_free(sample_fac);
      if (mlx_vector_vector_array_size(wells) > 0) {
        mlx_vector_array sample_w = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_w, wells, si) ||
            mlx_vector_vector_array_append_value(batch_wells, sample_w)) {
          mlx_vector_array_free(sample_w);
          err = 1;
          break;
        }
        mlx_vector_array_free(sample_w);
      }
      if (mlx_vector_vector_array_size(seismic) > 0) {
        mlx_vector_array sample_s = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample_s, seismic, si) ||
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
      write_all(result_fd, &status, sizeof(status));
      mlx_vector_vector_array_free(batch_fac);
      mlx_vector_vector_array_free(batch_wells);
      mlx_vector_vector_array_free(batch_seis);
      continue;
    }

    mlx_vector_array out_fac = mlx_vector_array_new();
    mlx_vector_array out_w = mlx_vector_array_new();
    mlx_vector_array out_s = mlx_vector_array_new();
    int rc = facies_collate(&out_fac, &out_w, &out_s, batch_fac, batch_wells,
                            batch_seis, mlx_default_cpu_stream_new());
    mlx_vector_vector_array_free(batch_fac);
    mlx_vector_vector_array_free(batch_wells);
    mlx_vector_vector_array_free(batch_seis);
    if (rc != 0) {
      int32_t status = 1;
      write_all(result_fd, &status, sizeof(status));
      mlx_vector_array_free(out_fac);
      mlx_vector_array_free(out_w);
      mlx_vector_array_free(out_s);
      continue;
    }

    int32_t status = 0;
    write_all(result_fd, &status, sizeof(status));
    serialize_vec_to_fd(result_fd, out_fac);
    serialize_vec_to_fd(result_fd, out_w);
    serialize_vec_to_fd(result_fd, out_s);

    mlx_vector_array_free(out_fac);
    mlx_vector_array_free(out_w);
    mlx_vector_array_free(out_s);
  }

  mlx_vector_vector_array_free(facies);
  mlx_vector_vector_array_free(wells);
  mlx_vector_vector_array_free(seismic);
  if (worker_init_ctx)
    free(worker_init_ctx);
  if (dlh)
    dlclose(dlh);
  return 0;
}
