#include "collate.h"
#include "dataloader.h"
#include "trainning/array_helpers.h"
#include "utils.h"
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/transforms.h>

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>
#include <time.h>

#ifdef __APPLE__
#include <pthread.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#endif

#ifndef FACIES_RESEED_TOKEN
#define FACIES_RESEED_TOKEN (UINT32_MAX - 1u)
#define FACIES_TERM_TOKEN (UINT32_MAX)
#endif

static int worker_log_enabled(void) {
    static int cached = -1;
    if (cached >= 0)
        return cached;
    const char *log_env = getenv("FACIESGAN_WORKER_LOG");
    cached = (log_env && atoi(log_env) != 0) ? 1 : 0;
    return cached;
}

static int worker_ipc_use_shm(void) {
    static int cached = -1;
    if (cached >= 0)
        return cached;
    const char *env = getenv("FACIESGAN_IPC_SHM");
    cached = (env && atoi(env) != 0) ? 1 : 0;
    return cached;
}

static void worker_apply_affinity(int worker_id) {
    const char *env = getenv("FACIESGAN_PIN_WORKER");
    if (!env || atoi(env) == 0)
        return;
#if defined(__linux__)
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu <= 0)
        return;
    int cpu = (int)((unsigned int)worker_id % (unsigned int)ncpu);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
#elif defined(__APPLE__)
    thread_affinity_policy_data_t policy = { .affinity_tag = worker_id + 1 };
    thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
#endif
}

static void worker_apply_realtime(void) {
    const char *env = getenv("FACIESGAN_RT_SCHED");
    if (!env || atoi(env) == 0)
        return;
#if defined(__linux__)
    const char *prio_env = getenv("FACIESGAN_RT_PRIORITY");
    int prio = prio_env ? atoi(prio_env) : 10;
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = prio;
    sched_setscheduler(0, SCHED_FIFO, &sp);
#elif defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    setpriority(PRIO_PROCESS, 0, -10);
#endif
}

static void worker_advise_hugepages(void *mem, size_t size) {
    const char *env = getenv("FACIESGAN_HUGEPAGE");
    if (!env || atoi(env) == 0)
        return;
#ifdef MADV_HUGEPAGE
    madvise(mem, size, MADV_HUGEPAGE);
#elif defined(MADV_SEQUENTIAL)
    madvise(mem, size, MADV_SEQUENTIAL);
#endif
}

static int worker_use_borrow_batches(void) {
    static int cached = -1;
    if (cached >= 0)
        return cached;
    const char *env = getenv("FACIESGAN_BORROW_BATCHES");
    cached = (env && atoi(env) != 0) ? 1 : 0;
    return cached;
}

static int worker_profile_enabled(void) {
    static int cached = -1;
    if (cached >= 0)
        return cached;
    const char *env = getenv("FACIESGAN_PROFILE");
    cached = (env && atoi(env) != 0) ? 1 : 0;
    return cached;
}

static uint64_t worker_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int array_to_float_buffer_cpu(const mlx_array arr, float **out_buf,
                                     size_t *out_elems, int *out_ndim,
                                     int **out_shape) {
    if (mlx_array_to_float_buffer(arr, out_buf, out_elems, out_ndim, out_shape) ==
            0)
        return 0;

    mlx_stream cpu_s = mlx_default_cpu_stream_new();
    mlx_array src = arr;
    mlx_array cast = mlx_array_new();
    int used_cast = 0;
    if (mlx_array_dtype(arr) != MLX_FLOAT32) {
        if (mlx_astype(&cast, arr, MLX_FLOAT32, cpu_s) == 0) {
            src = cast;
            used_cast = 1;
        }
    }
    mlx_array cpu_arr = mlx_array_new();
    int copy_ok = (mlx_copy(&cpu_arr, src, cpu_s) == 0) ? 1 : 0;
    if (copy_ok) {
        mlx_vector_array eval_vec = mlx_vector_array_new();
        mlx_vector_array_append_value(eval_vec, cpu_arr);
        mlx_eval(eval_vec);
        mlx_vector_array_free(eval_vec);
        mlx_synchronize(cpu_s);
    }
    mlx_stream_free(cpu_s);

    int rc = -1;
    if (copy_ok)
        rc = mlx_array_to_float_buffer(cpu_arr, out_buf, out_elems, out_ndim,
                                       out_shape);
    mlx_array_free(cpu_arr);
    if (used_cast)
        mlx_array_free(cast);
    return rc == 0 ? 0 : -1;
}

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

static ssize_t writev_all(int fd, struct iovec *iov, int iovcnt) {
    int cur = 0;
    while (cur < iovcnt) {
        ssize_t w = writev(fd, &iov[cur], iovcnt - cur);
        if (w <= 0)
            return -1;
        size_t left = (size_t)w;
        while (left > 0 && cur < iovcnt) {
            if (left >= iov[cur].iov_len) {
                left -= iov[cur].iov_len;
                cur++;
            } else {
                iov[cur].iov_base = (char *)iov[cur].iov_base + left;
                iov[cur].iov_len -= left;
                left = 0;
            }
        }
    }
    return 0;
}

static int read_vector_array_from_fd(int fd, mlx_vector_array *out_vec) {
    *out_vec = mlx_vector_array_new();
    uint32_t nscales = 0;
    if (read_all(fd, &nscales, sizeof(nscales)) <= 0)
        return 1;
    if (worker_log_enabled())
        fprintf(stderr, "facies_worker: nscales=%u\n", nscales);
    int rc = 0;
    for (uint32_t si = 0; si < nscales; ++si) {
        uint32_t ndim = 0;
        if (read_all(fd, &ndim, sizeof(ndim)) <= 0)
            return 1;
        if (worker_log_enabled())
            fprintf(stderr, "facies_worker: scale=%u ndim=%u\n", si, ndim);
        int s_ndim = (int)ndim;
        int *shape = NULL;
        if (s_ndim > 0) {
            if (mlx_alloc_int_array(&shape, s_ndim) != 0)
                return 1;
        }
        for (uint32_t d = 0; d < ndim; ++d) {
            int32_t s32 = 0;
            if (read_all(fd, &s32, sizeof(s32)) <= 0) {
                if (shape)
                    mlx_free_int_array(&shape, &s_ndim);
                return 1;
            }
            shape[d] = s32;
        }
        uint64_t uelems = 0;
        if (read_all(fd, &uelems, sizeof(uelems)) <= 0) {
            if (shape)
                mlx_free_int_array(&shape, &s_ndim);
            return 1;
        }
        if (worker_log_enabled())
            fprintf(stderr, "facies_worker: scale=%u elems=%llu\n", si,
                    (unsigned long long)uelems);
        size_t elems = (size_t)uelems;
        float *fbuf = NULL;
        if (elems > 0) {
            if (elems > (size_t)INT_MAX) {
                fbuf = (float *)malloc(sizeof(float) * elems);
            } else {
                if (mlx_alloc_float_buf(&fbuf, (int)elems) != 0)
                    fbuf = NULL;
            }
            if (!fbuf) {
                if (shape)
                    mlx_free_int_array(&shape, &s_ndim);
                return 1;
            }
            if (read_all(fd, fbuf, sizeof(float) * elems) <= 0) {
                if (shape)
                    mlx_free_int_array(&shape, &s_ndim);
                if (elems > (size_t)INT_MAX)
                    free(fbuf);
                else
                    mlx_free_float_buf(&fbuf, NULL);
                return 1;
            }
        }
        mlx_array arr = mlx_array_new_data(fbuf, shape, (int)ndim, MLX_FLOAT32);
        if (mlx_vector_array_append_value(*out_vec, arr)) {
            mlx_array_free(arr);
            if (shape)
                mlx_free_int_array(&shape, &s_ndim);
            if (fbuf) {
                if (elems > (size_t)INT_MAX)
                    free(fbuf);
                else
                    mlx_free_float_buf(&fbuf, NULL);
            }
            return 1;
        }
        mlx_array_free(arr);
        if (shape)
            mlx_free_int_array(&shape, &s_ndim);
        if (fbuf) {
            if (elems > (size_t)INT_MAX)
                free(fbuf);
            else
                mlx_free_float_buf(&fbuf, NULL);
        }
    }
    return 0;
}

static int serialize_vec_to_fd(int fd, mlx_vector_array vec) {
    uint32_t nscales = (uint32_t)mlx_vector_array_size(vec);
    if (write_all(fd, &nscales, sizeof(nscales)) <= 0)
        return 1;

    typedef struct serialize_entry_s {
        int direct_ready;
        int has_error;
        float *buf;
        size_t elems;
        int ndim;
        int *shape;
        mlx_array cpu_arr;
        int has_cpu_arr;
        mlx_array cast_arr;
        int used_cast;
    } serialize_entry;

    serialize_entry *entries = (serialize_entry *)calloc(nscales, sizeof(*entries));
    if (!entries)
        return 1;

    mlx_stream cpu_s = mlx_default_cpu_stream_new();
    mlx_vector_array eval_vec = mlx_vector_array_new();
    int rc = 0;

    for (uint32_t si = 0; si < nscales; ++si) {
        mlx_array arr = mlx_array_new();
        if (mlx_vector_array_get(&arr, vec, si) != 0) {
            entries[si].has_error = 1;
            mlx_array_free(arr);
            continue;
        }

        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(arr, &buf, &elems, &ndim, &shape) == 0) {
            entries[si].direct_ready = 1;
            entries[si].buf = buf;
            entries[si].elems = elems;
            entries[si].ndim = ndim;
            entries[si].shape = shape;
            mlx_array_free(arr);
            continue;
        }

        mlx_array src = arr;
        mlx_array cast = mlx_array_new();
        int used_cast = 0;
        if (mlx_array_dtype(arr) != MLX_FLOAT32) {
            if (mlx_astype(&cast, arr, MLX_FLOAT32, cpu_s) == 0) {
                src = cast;
                used_cast = 1;
            }
        }
        mlx_array cpu_arr = mlx_array_new();
        int copy_ok = (mlx_copy(&cpu_arr, src, cpu_s) == 0) ? 1 : 0;
        if (copy_ok) {
            mlx_vector_array_append_value(eval_vec, cpu_arr);
            entries[si].cpu_arr = cpu_arr;
            entries[si].has_cpu_arr = 1;
            entries[si].cast_arr = cast;
            entries[si].used_cast = used_cast;
        } else {
            entries[si].has_error = 1;
            mlx_array_free(cpu_arr);
            if (used_cast)
                mlx_array_free(cast);
        }
        mlx_array_free(arr);
    }

    if (mlx_vector_array_size(eval_vec) > 0) {
        mlx_eval(eval_vec);
        mlx_synchronize(cpu_s);
    }

    for (uint32_t si = 0; si < nscales; ++si) {
        if (entries[si].has_error) {
            uint32_t zero = 0;
            uint64_t zero64 = 0;
            write_all(fd, &zero, sizeof(zero));
            write_all(fd, &zero64, sizeof(zero64));
            continue;
        }

        if (!entries[si].direct_ready) {
            if (mlx_array_to_float_buffer(entries[si].cpu_arr, &entries[si].buf,
                                          &entries[si].elems, &entries[si].ndim,
                                          &entries[si].shape) != 0) {
                uint32_t zero = 0;
                uint64_t zero64 = 0;
                write_all(fd, &zero, sizeof(zero));
                write_all(fd, &zero64, sizeof(zero64));
                entries[si].has_error = 1;
                continue;
            }
        }

        uint32_t udim = (uint32_t)entries[si].ndim;
        uint64_t uelems = (uint64_t)entries[si].elems;
        struct iovec iov[4];
        int iovcnt = 0;
        iov[iovcnt].iov_base = &udim;
        iov[iovcnt].iov_len = sizeof(udim);
        iovcnt++;
        if (entries[si].ndim > 0) {
            iov[iovcnt].iov_base = entries[si].shape;
            iov[iovcnt].iov_len = sizeof(int32_t) * (size_t)entries[si].ndim;
            iovcnt++;
        }
        iov[iovcnt].iov_base = &uelems;
        iov[iovcnt].iov_len = sizeof(uelems);
        iovcnt++;
        if (entries[si].elems > 0) {
            iov[iovcnt].iov_base = entries[si].buf;
            iov[iovcnt].iov_len = sizeof(float) * entries[si].elems;
            iovcnt++;
        }
        if (writev_all(fd, iov, iovcnt) < 0) {
            entries[si].has_error = 1;
            rc = 1;
            goto cleanup;
        }
    }

cleanup:
    for (uint32_t si = 0; si < nscales; ++si) {
        if (entries[si].buf) {
            if (entries[si].elems > (size_t)INT_MAX)
                free(entries[si].buf);
            else
                mlx_free_float_buf(&entries[si].buf, NULL);
        }
        if (entries[si].shape)
            mlx_free_int_array(&entries[si].shape, &entries[si].ndim);
        if (!entries[si].direct_ready && entries[si].has_cpu_arr) {
            mlx_array_free(entries[si].cpu_arr);
            if (entries[si].used_cast)
                mlx_array_free(entries[si].cast_arr);
        }
    }

    mlx_vector_array_free(eval_vec);
    mlx_stream_free(cpu_s);
    free(entries);
    return rc;
}

static int serialize_vec_to_buffer(mlx_vector_array vec, unsigned char **out_buf,
                                   size_t *out_len) {
    if (!out_buf || !out_len)
        return 1;
    *out_buf = NULL;
    *out_len = 0;

    uint32_t nscales = (uint32_t)mlx_vector_array_size(vec);

    typedef struct serialize_entry_s {
        int direct_ready;
        int has_error;
        float *buf;
        size_t elems;
        int ndim;
        int *shape;
        mlx_array cpu_arr;
        int has_cpu_arr;
        mlx_array cast_arr;
        int used_cast;
    } serialize_entry;

    serialize_entry *entries = (serialize_entry *)calloc(nscales, sizeof(*entries));
    if (!entries)
        return 1;

    mlx_stream cpu_s = mlx_default_cpu_stream_new();
    mlx_vector_array eval_vec = mlx_vector_array_new();

    for (uint32_t si = 0; si < nscales; ++si) {
        mlx_array arr = mlx_array_new();
        if (mlx_vector_array_get(&arr, vec, si) != 0) {
            entries[si].has_error = 1;
            mlx_array_free(arr);
            continue;
        }

        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(arr, &buf, &elems, &ndim, &shape) == 0) {
            entries[si].direct_ready = 1;
            entries[si].buf = buf;
            entries[si].elems = elems;
            entries[si].ndim = ndim;
            entries[si].shape = shape;
            mlx_array_free(arr);
            continue;
        }

        mlx_array src = arr;
        mlx_array cast = mlx_array_new();
        int used_cast = 0;
        if (mlx_array_dtype(arr) != MLX_FLOAT32) {
            if (mlx_astype(&cast, arr, MLX_FLOAT32, cpu_s) == 0) {
                src = cast;
                used_cast = 1;
            }
        }
        mlx_array cpu_arr = mlx_array_new();
        int copy_ok = (mlx_copy(&cpu_arr, src, cpu_s) == 0) ? 1 : 0;
        if (copy_ok) {
            mlx_vector_array_append_value(eval_vec, cpu_arr);
            entries[si].cpu_arr = cpu_arr;
            entries[si].has_cpu_arr = 1;
            entries[si].cast_arr = cast;
            entries[si].used_cast = used_cast;
        } else {
            entries[si].has_error = 1;
            mlx_array_free(cpu_arr);
            if (used_cast)
                mlx_array_free(cast);
        }
        mlx_array_free(arr);
    }

    if (mlx_vector_array_size(eval_vec) > 0) {
        mlx_eval(eval_vec);
        mlx_synchronize(cpu_s);
    }

    size_t total = sizeof(uint32_t);
    for (uint32_t si = 0; si < nscales; ++si) {
        if (entries[si].has_error) {
            total += sizeof(uint32_t) + sizeof(uint64_t);
            continue;
        }
        if (!entries[si].direct_ready) {
            if (mlx_array_to_float_buffer(entries[si].cpu_arr, &entries[si].buf,
                                          &entries[si].elems, &entries[si].ndim,
                                          &entries[si].shape) != 0) {
                entries[si].has_error = 1;
                total += sizeof(uint32_t) + sizeof(uint64_t);
                continue;
            }
        }
        total += sizeof(uint32_t);
        total += sizeof(int32_t) * (size_t)entries[si].ndim;
        total += sizeof(uint64_t);
        total += sizeof(float) * entries[si].elems;
    }

    unsigned char *buf_out = (unsigned char *)malloc(total);
    if (!buf_out) {
        total = 0;
    }

    size_t off = 0;
    if (buf_out) {
        memcpy(buf_out + off, &nscales, sizeof(nscales));
        off += sizeof(nscales);
        for (uint32_t si = 0; si < nscales; ++si) {
            if (entries[si].has_error) {
                uint32_t zero = 0;
                uint64_t zero64 = 0;
                memcpy(buf_out + off, &zero, sizeof(zero));
                off += sizeof(zero);
                memcpy(buf_out + off, &zero64, sizeof(zero64));
                off += sizeof(zero64);
                continue;
            }
            uint32_t udim = (uint32_t)entries[si].ndim;
            uint64_t uelems = (uint64_t)entries[si].elems;
            memcpy(buf_out + off, &udim, sizeof(udim));
            off += sizeof(udim);
            if (entries[si].ndim > 0) {
                memcpy(buf_out + off, entries[si].shape,
                       sizeof(int32_t) * (size_t)entries[si].ndim);
                off += sizeof(int32_t) * (size_t)entries[si].ndim;
            }
            memcpy(buf_out + off, &uelems, sizeof(uelems));
            off += sizeof(uelems);
            if (entries[si].elems > 0) {
                memcpy(buf_out + off, entries[si].buf,
                       sizeof(float) * entries[si].elems);
                off += sizeof(float) * entries[si].elems;
            }
        }
    }

    for (uint32_t si = 0; si < nscales; ++si) {
        if (entries[si].buf) {
            if (entries[si].elems > (size_t)INT_MAX)
                free(entries[si].buf);
            else
                mlx_free_float_buf(&entries[si].buf, NULL);
        }
        if (entries[si].shape)
            mlx_free_int_array(&entries[si].shape, &entries[si].ndim);
        if (!entries[si].direct_ready && entries[si].has_cpu_arr) {
            mlx_array_free(entries[si].cpu_arr);
            if (entries[si].used_cast)
                mlx_array_free(entries[si].cast_arr);
        }
    }

    mlx_vector_array_free(eval_vec);
    mlx_stream_free(cpu_s);
    free(entries);

    if (!buf_out)
        return 1;
    *out_buf = buf_out;
    *out_len = off;
    return 0;
}

static int send_batch_shm(int fd, mlx_vector_array out_fac,
                          mlx_vector_array out_w, mlx_vector_array out_s) {
    unsigned char *buf_fac = NULL;
    unsigned char *buf_w = NULL;
    unsigned char *buf_s = NULL;
    size_t len_fac = 0, len_w = 0, len_s = 0;
    if (serialize_vec_to_buffer(out_fac, &buf_fac, &len_fac) != 0)
        goto fail;
    if (serialize_vec_to_buffer(out_w, &buf_w, &len_w) != 0)
        goto fail;
    if (serialize_vec_to_buffer(out_s, &buf_s, &len_s) != 0)
        goto fail;

    size_t total = len_fac + len_w + len_s;
    if (total == 0)
        goto fail;

    static uint64_t shm_counter = 0;
    char shm_name[128];
    snprintf(shm_name, sizeof(shm_name), "/faciesgan_%d_%llu",
             (int)getpid(), (unsigned long long)++shm_counter);

    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
    if (shm_fd < 0)
        goto fail;
    if (ftruncate(shm_fd, (off_t)total) != 0) {
        close(shm_fd);
        shm_unlink(shm_name);
        goto fail;
    }
    void *mem = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (mem == MAP_FAILED) {
        close(shm_fd);
        shm_unlink(shm_name);
        goto fail;
    }
    worker_advise_hugepages(mem, total);
    unsigned char *dst = (unsigned char *)mem;
    memcpy(dst, buf_fac, len_fac);
    memcpy(dst + len_fac, buf_w, len_w);
    memcpy(dst + len_fac + len_w, buf_s, len_s);
    munmap(mem, total);
    close(shm_fd);

    uint32_t name_len = (uint32_t)strlen(shm_name);
    uint64_t total64 = (uint64_t)total;
    if (write_all(fd, &name_len, sizeof(name_len)) <= 0 ||
            write_all(fd, shm_name, name_len) <= 0 ||
            write_all(fd, &total64, sizeof(total64)) <= 0) {
        shm_unlink(shm_name);
        goto fail;
    }

    free(buf_fac);
    free(buf_w);
    free(buf_s);
    return 0;

fail:
    free(buf_fac);
    free(buf_w);
    free(buf_s);
    return 1;
}

static int read_vv_fd(int fd, mlx_vector_vector_array *out_vv,
                      uint64_t n_samples) {
    *out_vv = mlx_vector_vector_array_new();
    for (uint64_t i = 0; i < n_samples; ++i) {
        mlx_vector_array sample = mlx_vector_array_new();
        if (read_vector_array_from_fd(fd, &sample) != 0) {
            mlx_vector_array_free(sample);
            return 1;
        }
        if (mlx_vector_vector_array_append_value(*out_vv, sample)) {
            mlx_vector_array_free(sample);
            return 1;
        }
        mlx_vector_array_free(sample);
    }
    return 0;
}

static int read_dataset_from_fd(int fd, MLXPyramidsDataset **out_ds) {
    uint64_t n_samples = 0;
    if (read_all(fd, &n_samples, sizeof(n_samples)) <= 0)
        return 1;

    if (worker_log_enabled()) {
        fprintf(stderr, "facies_worker: reading dataset, n_samples=%llu\n",
                (unsigned long long)n_samples);
    }

    mlx_vector_vector_array facies = mlx_vector_vector_array_new();
    mlx_vector_vector_array wells = mlx_vector_vector_array_new();
    mlx_vector_vector_array seismic = mlx_vector_vector_array_new();

    if (read_vv_fd(fd, &facies, n_samples) != 0)
        goto fail;
    if (worker_log_enabled())
        fprintf(stderr, "facies_worker: facies read ok\n");
    if (read_vv_fd(fd, &wells, n_samples) != 0)
        goto fail;
    if (worker_log_enabled())
        fprintf(stderr, "facies_worker: wells read ok\n");
    if (read_vv_fd(fd, &seismic, n_samples) != 0)
        goto fail;
    if (worker_log_enabled())
        fprintf(stderr, "facies_worker: seismic read ok\n");

    if (facies_dataset_new(out_ds, facies, wells, seismic) != 0)
        goto fail;

    mlx_vector_vector_array_free(facies);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    return 0;

fail:
    mlx_vector_vector_array_free(facies);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);
    return 1;
}

static int send_init_status(int fd, int status, const char *msg) {
    int32_t st = status ? 1 : 0;
    if (write_all(fd, &st, sizeof(st)) <= 0)
        return 1;
    uint32_t len = msg ? (uint32_t)strlen(msg) : 0;
    if (write_all(fd, &len, sizeof(len)) <= 0)
        return 1;
    if (len > 0) {
        if (write_all(fd, msg, len) <= 0)
            return 1;
    }
    return 0;
}

static int worker_process_loop(int task_fd, int result_fd,
                               MLXPyramidsDataset *ds,
                               facies_worker_init_fn worker_init,
                               void *worker_init_ctx, int worker_id) {
    uint64_t *idxs = NULL;
    size_t idxs_cap = 0;
    mlx_stream collate_s = mlx_default_cpu_stream_new();
    while (1) {
        uint32_t n_indices = 0;
        if (read_all(task_fd, &n_indices, sizeof(n_indices)) <= 0)
            break;
        if (n_indices == FACIES_TERM_TOKEN)
            break;
        if (n_indices == FACIES_RESEED_TOKEN) {
            uint64_t seed = 0;
            if (read_all(task_fd, &seed, sizeof(seed)) <= 0)
                break;
            srand((unsigned int)seed);
            if (worker_init)
                worker_init(worker_id, worker_init_ctx);
            continue;
        }

        if (worker_log_enabled())
            fprintf(stderr, "facies_worker: received task n_indices=%u\n",
                    n_indices);

        size_t ni = (size_t)n_indices;
        if (ni > idxs_cap) {
            size_t new_cap = idxs_cap == 0 ? ni : idxs_cap;
            while (new_cap < ni)
                new_cap = new_cap * 2;
            uint64_t *new_idxs = (uint64_t *)realloc(idxs, sizeof(uint64_t) * new_cap);
            if (!new_idxs) {
                int32_t status = 1;
                write_all(result_fd, &status, sizeof(status));
                continue;
            }
            idxs = new_idxs;
            idxs_cap = new_cap;
        }
        if (!idxs) {
            int32_t status = 1;
            write_all(result_fd, &status, sizeof(status));
            continue;
        }
        if (read_all(task_fd, idxs, sizeof(uint64_t) * ni) <= 0) {
            int32_t status = 1;
            write_all(result_fd, &status, sizeof(status));
            continue;
        }

        mlx_vector_vector_array batch_fac = mlx_vector_vector_array_new();
        mlx_vector_vector_array batch_wells = mlx_vector_vector_array_new();
        mlx_vector_vector_array batch_seis = mlx_vector_vector_array_new();
        int err = 0;
        int borrow_batches = worker_use_borrow_batches() && ds->batches;
        for (size_t i = 0; i < ni; ++i) {
            size_t si = (size_t)idxs[i];
            if (borrow_batches) {
                MLXBatch *b = &ds->batches[si];
                if (mlx_vector_vector_array_append_value(batch_fac, b->facies)) {
                    err = 1;
                    break;
                }
                if (mlx_vector_vector_array_size(ds->wells) > 0) {
                    if (mlx_vector_vector_array_append_value(batch_wells, b->wells)) {
                        err = 1;
                        break;
                    }
                }
                if (mlx_vector_vector_array_size(ds->seismic) > 0) {
                    if (mlx_vector_vector_array_append_value(batch_seis, b->seismic)) {
                        err = 1;
                        break;
                    }
                }
                continue;
            }

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
        uint64_t t0 = 0, t1 = 0, t2 = 0;
        if (worker_profile_enabled())
            t0 = worker_now_ns();
        facies_collate_fn cb = (facies_collate_fn)facies_collate;
        int rc = cb(&out_fac, &out_w, &out_s, batch_fac, batch_wells, batch_seis,
                    collate_s, NULL);
        if (worker_profile_enabled())
            t1 = worker_now_ns();
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
        if (worker_ipc_use_shm()) {
            if (send_batch_shm(result_fd, out_fac, out_w, out_s) != 0) {
                uint32_t zero = 0;
                uint64_t zero64 = 0;
                write_all(result_fd, &zero, sizeof(zero));
                write_all(result_fd, &zero64, sizeof(zero64));
            }
        } else {
            serialize_vec_to_fd(result_fd, out_fac);
            serialize_vec_to_fd(result_fd, out_w);
            serialize_vec_to_fd(result_fd, out_s);
        }

        if (worker_profile_enabled()) {
            t2 = worker_now_ns();
            fprintf(stderr,
                    "facies_worker: batch timings collate=%.3fms serialize=%.3fms\n",
                    (double)(t1 - t0) / 1e6, (double)(t2 - t1) / 1e6);
        }

        if (worker_log_enabled())
            fprintf(stderr, "facies_worker: sent batch result\n");

        mlx_vector_array_free(out_fac);
        mlx_vector_array_free(out_w);
        mlx_vector_array_free(out_s);
    }
    free(idxs);
    return 0;
}

int main(void) {
    const char *task_fd_env = getenv("FACIES_WORKER_TASK_FD");
    const char *result_fd_env = getenv("FACIES_WORKER_RESULT_FD");
    if (!task_fd_env || !result_fd_env) {
        fprintf(stderr, "facies_worker: missing FACIES_WORKER_TASK_FD/RESULT_FD\n");
        return 1;
    }
    int task_fd = atoi(task_fd_env);
    int result_fd = atoi(result_fd_env);

    if (worker_log_enabled())
        fprintf(stderr, "facies_worker: pid=%d starting\n", (int)getpid());

    uint32_t worker_id = 0;
    uint64_t seed = 0;
    if (read_all(task_fd, &worker_id, sizeof(worker_id)) <= 0)
        return 1;
    if (read_all(task_fd, &seed, sizeof(seed)) <= 0)
        return 1;

    worker_apply_affinity((int)worker_id);
    worker_apply_realtime();

    uint32_t lib_len = 0;
    uint32_t sym_len = 0;
    uint32_t ctx_len = 0;
    char *lib_path = NULL;
    char *sym_name = NULL;
    void *ctx_data = NULL;

    if (read_all(task_fd, &lib_len, sizeof(lib_len)) <= 0)
        return 1;
    if (lib_len > 0) {
        lib_path = (char *)malloc(lib_len + 1);
        if (!lib_path)
            return 1;
        if (read_all(task_fd, lib_path, lib_len) <= 0)
            return 1;
        lib_path[lib_len] = '\0';
    }
    if (read_all(task_fd, &sym_len, sizeof(sym_len)) <= 0)
        return 1;
    if (sym_len > 0) {
        sym_name = (char *)malloc(sym_len + 1);
        if (!sym_name)
            return 1;
        if (read_all(task_fd, sym_name, sym_len) <= 0)
            return 1;
        sym_name[sym_len] = '\0';
    }
    if (read_all(task_fd, &ctx_len, sizeof(ctx_len)) <= 0)
        return 1;
    if (ctx_len > 0) {
        ctx_data = malloc(ctx_len);
        if (!ctx_data)
            return 1;
        if (read_all(task_fd, ctx_data, ctx_len) <= 0)
            return 1;
    }

    MLXPyramidsDataset *ds = NULL;
    if (read_dataset_from_fd(task_fd, &ds) != 0) {
        send_init_status(result_fd, 1, "failed to read dataset");
        return 1;
    }

    if (worker_log_enabled())
        fprintf(stderr, "facies_worker: dataset loaded, n_samples=%d\n", ds->n_samples);

    facies_worker_init_fn worker_init = NULL;
    void *worker_init_ctx = ctx_data;
    void *lib_handle = NULL;
    char err_buf[256];
    if (lib_path && sym_name) {
        lib_handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
        if (!lib_handle) {
            snprintf(err_buf, sizeof(err_buf), "dlopen failed: %s", dlerror());
            send_init_status(result_fd, 1, err_buf);
            facies_dataset_free(ds);
            return 1;
        }
        worker_init = (facies_worker_init_fn)dlsym(lib_handle, sym_name);
        if (!worker_init) {
            snprintf(err_buf, sizeof(err_buf), "dlsym failed: %s", dlerror());
            send_init_status(result_fd, 1, err_buf);
            facies_dataset_free(ds);
            return 1;
        }
    }

    srand((unsigned int)seed);
    if (worker_init)
        worker_init((int)worker_id, worker_init_ctx);

    send_init_status(result_fd, 0, NULL);
    if (worker_log_enabled())
        fprintf(stderr, "facies_worker: init status sent\n");

    int rc = worker_process_loop(task_fd, result_fd, ds, worker_init,
                                 worker_init_ctx, (int)worker_id);

    if (lib_handle)
        dlclose(lib_handle);
    facies_dataset_free(ds);
    free(lib_path);
    free(sym_name);
    free(ctx_data);
    return rc == 0 ? 0 : 1;
}
