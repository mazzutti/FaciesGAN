#include "scalar_pool.h"
#include <pthread.h>
#include <stdatomic.h>

/* ------------------------------------------------------------------ */
/*  Lazy singleton scalars — initialised on first use, never freed    */
/*  until mlx_scalar_pool_destroy().  Thread-safe via pthread_once.   */
/* ------------------------------------------------------------------ */

static pthread_once_t g_scalar_once = PTHREAD_ONCE_INIT;

static mlx_array g_zero;
static mlx_array g_one;
static mlx_array g_neg_one;
static mlx_array g_eps;
static mlx_array g_neg_ten;
static mlx_array g_half;
static atomic_int  g_pool_alive = 0;

static void init_pool(void) {
    g_zero    = mlx_array_new_float(0.0f);
    g_one     = mlx_array_new_float(1.0f);
    g_neg_one = mlx_array_new_float(-1.0f);
    g_eps     = mlx_array_new_float(1e-5f);
    g_neg_ten = mlx_array_new_float(-10.0f);
    g_half    = mlx_array_new_float(0.5f);
    atomic_store(&g_pool_alive, 1);
}

static inline void ensure_pool(void) {
    pthread_once(&g_scalar_once, init_pool);
}

mlx_array mlx_scalar_zero(void)    { ensure_pool(); return g_zero;    }
mlx_array mlx_scalar_one(void)     { ensure_pool(); return g_one;     }
mlx_array mlx_scalar_neg_one(void) { ensure_pool(); return g_neg_one; }
mlx_array mlx_scalar_eps(void)     { ensure_pool(); return g_eps;     }
mlx_array mlx_scalar_neg_ten(void) { ensure_pool(); return g_neg_ten; }
mlx_array mlx_scalar_half(void)    { ensure_pool(); return g_half;    }

void mlx_scalar_pool_destroy(void) {
    if (!atomic_load(&g_pool_alive))
        return;
    mlx_array_free(g_zero);
    mlx_array_free(g_one);
    mlx_array_free(g_neg_one);
    mlx_array_free(g_eps);
    mlx_array_free(g_neg_ten);
    mlx_array_free(g_half);
    atomic_store(&g_pool_alive, 0);
}
