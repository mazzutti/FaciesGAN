#include "array_helpers.h"
#include <string.h>
#include <pthread.h>

#include <stdio.h>

/* Global MLX mutex for thread safety - using recursive mutex to allow
 * nested locking from the same thread (e.g., generator_forward inside
 * init_rec_noise_and_amp) */
static pthread_mutex_t g_mlx_mutex;
static pthread_once_t g_mlx_mutex_once = PTHREAD_ONCE_INIT;
static _Atomic int g_lock_depth = 0;

static void init_mlx_mutex(void) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_mlx_mutex, &attr);
    pthread_mutexattr_destroy(&attr);
}

void mlx_global_lock(void) {
    pthread_once(&g_mlx_mutex_once, init_mlx_mutex);
    pthread_mutex_lock(&g_mlx_mutex);
    g_lock_depth++;
}

void mlx_global_unlock(void) {
    g_lock_depth--;
    pthread_mutex_unlock(&g_mlx_mutex);
}

int mlx_copy_float_array(float **out, int *out_n, const float *src, int n) {
    if (!out || !out_n)
        return -1;
    *out = NULL;
    *out_n = 0;
    if (!src || n <= 0)
        return 0;
    float *buf = (float *)malloc(sizeof(float) * n);
    if (!buf)
        return -1;
    memcpy(buf, src, sizeof(float) * n);
    *out = buf;
    *out_n = n;
    return 0;
}

void mlx_free_float_array(float **arr, int *n) {
    if (!arr)
        return;
    if (*arr) {
        free(*arr);
        *arr = NULL;
    }
    if (n)
        *n = 0;
}

int mlx_copy_int_array(int **out, int *out_n, const int *src, int n) {
    if (!out || !out_n)
        return -1;
    *out = NULL;
    *out_n = 0;
    if (!src || n <= 0)
        return 0;
    int *buf = (int *)malloc(sizeof(int) * n);
    if (!buf)
        return -1;
    memcpy(buf, src, sizeof(int) * n);
    *out = buf;
    *out_n = n;
    return 0;
}

void mlx_free_int_array(int **arr, int *n) {
    if (!arr)
        return;
    if (*arr) {
        free(*arr);
        *arr = NULL;
    }
    if (n)
        *n = 0;
}

int mlx_alloc_int_array(int **out, int n) {
    if (!out)
        return -1;
    *out = NULL;
    if (n <= 0)
        return 0;
    int *buf = (int *)malloc(sizeof(int) * (size_t)n);
    if (!buf)
        return -1;
    memset(buf, 0, sizeof(int) * (size_t)n);
    *out = buf;
    return 0;
}

int mlx_alloc_ptr_array(void ***out, int n) {
    if (!out)
        return -1;
    *out = NULL;
    if (n <= 0)
        return 0;
    void **buf = (void **)malloc(sizeof(void *) * (size_t)n);
    if (!buf)
        return -1;
    for (int i = 0; i < n; ++i)
        buf[i] = NULL;
    *out = buf;
    return 0;
}

void mlx_free_ptr_array(void ***out, int n) {
    if (!out)
        return;
    if (*out) {
        free(*out);
        *out = NULL;
    }
    /* `n` is for API parity; not needed for freeing in this implementation */
}

int mlx_alloc_float_buf(float **out, int n) {
    if (!out)
        return -1;
    *out = NULL;
    if (n <= 0)
        return 0;
    float *buf = (float *)malloc(sizeof(float) * (size_t)n);
    if (!buf)
        return -1;
    memset(buf, 0, sizeof(float) * (size_t)n);
    *out = buf;
    return 0;
}

int mlx_realloc_float_buf(float **out, int old_n, int new_n) {
    if (!out)
        return -1;
    if (new_n <= 0) {
        if (*out) {
            free(*out);
            *out = NULL;
        }
        return 0;
    }
    if (!*out) {
        return mlx_alloc_float_buf(out, new_n);
    }
    float *nb = (float *)realloc(*out, sizeof(float) * (size_t)new_n);
    if (!nb)
        return -1;
    /* zero the new tail if expanded */
    if (new_n > old_n) {
        memset(nb + old_n, 0, sizeof(float) * (size_t)(new_n - old_n));
    }
    *out = nb;
    return 0;
}

void mlx_free_float_buf(float **out, int *n) {
    if (!out)
        return;
    if (*out) {
        free(*out);
        *out = NULL;
    }
    if (n)
        *n = 0;
}

int mlx_alloc_mlx_array_vals(mlx_array **out, int n) {
    if (!out)
        return -1;
    *out = NULL;
    if (n <= 0)
        return 0;
    mlx_array *buf = (mlx_array *)malloc(sizeof(mlx_array) * (size_t)n);
    if (!buf)
        return -1;
    /* initialize to empty arrays */
    for (int i = 0; i < n; ++i)
        buf[i] = mlx_array_new();
    *out = buf;
    return 0;
}

void mlx_free_mlx_array_vals(mlx_array **out, int n) {
    if (!out || !*out || n <= 0)
        return;
    mlx_array *buf = *out;
    for (int i = 0; i < n; ++i) {
        mlx_array_free(buf[i]);
    }
    free(buf);
    *out = NULL;
}

int mlx_alloc_mlx_array_raw(mlx_array **out, int n) {
    if (!out)
        return -1;
    *out = NULL;
    if (n <= 0)
        return 0;
    mlx_array *buf = (mlx_array *)malloc(sizeof(mlx_array) * (size_t)n);
    if (!buf)
        return -1;
    /* Do not initialize elements; caller will populate/assign them. */
    *out = buf;
    return 0;
}

void mlx_free_mlx_array_raw(mlx_array **out, int n) {
    /* `n` unused: caller is responsible for element cleanup when applicable */
    if (!out)
        return;
    if (*out) {
        free(*out);
        *out = NULL;
    }
}

int mlx_alloc_pod(void **out, size_t elem_size, int n) {
    if (!out)
        return -1;
    *out = NULL;
    if (n <= 0)
        return 0;
    if (elem_size == 0)
        return -1;
    void *buf = malloc(elem_size * (size_t)n);
    if (!buf)
        return -1;
    memset(buf, 0, elem_size * (size_t)n);
    *out = buf;
    return 0;
}

void mlx_free_pod(void **out) {
    if (!out)
        return;
    if (*out) {
        free(*out);
        *out = NULL;
    }
}

int mlx_alloc_mlx_array_ptrs(mlx_array ***out, int n) {
    if (!out || n <= 0)
        return -1;
    mlx_array **a = (mlx_array **)malloc(sizeof(mlx_array *) * n);
    if (!a)
        return -1;
    for (int i = 0; i < n; ++i)
        a[i] = NULL;
    *out = a;
    return 0;
}

void mlx_free_mlx_array_ptrs(mlx_array ***out, int n) {
    if (!out || !*out || n <= 0)
        return;
    mlx_array **a = *out;
    for (int i = 0; i < n; ++i) {
        if (a[i]) {
            mlx_array_free(*a[i]);
            free(a[i]);
            a[i] = NULL;
        }
    }
    free(a);
    *out = NULL;
}
