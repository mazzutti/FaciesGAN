#define MLX_MEM_NO_WRAP 1
#include "mem_debug.h"
#include "mlx_compat.h"
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * MLX Array Memory Leak Tracker Implementation
 *
 * Uses a simple hash table keyed by mlx_array.ctx pointer to track
 * allocations. Thread-safe via pthread mutex.
 */

/* Hash table for O(1) lookup by ctx pointer */
#define HASH_SIZE 8192
#define HASH_MASK (HASH_SIZE - 1)

typedef struct hash_entry {
    mlx_mem_record_t record;
    struct hash_entry *next;
} hash_entry_t;

static struct {
    hash_entry_t *buckets[HASH_SIZE];
    size_t count;              /* Current tracked allocations */
    size_t total_allocs;       /* Total allocations since init */
    size_t total_frees;        /* Total frees since init */
    size_t peak_count;         /* Peak concurrent allocations */
    size_t next_alloc_id;      /* Sequential allocation ID */
    bool initialized;
    bool enabled;
    int verbosity;             /* 0=quiet, 1=leaks, 2=verbose */
    pthread_mutex_t lock;
} g_tracker = {
    .initialized = false,
    .enabled = false,
    .verbosity = 1,
};

static int g_crash_handler_installed = 0;
static bool g_warn_untracked_frees = false;

static void mlx_mem_crash_handler(int sig) {
    fprintf(stderr, "\n[mem_debug] Caught signal %d; dumping leak report.\n", sig);
    mlx_mem_print_stats();
    mlx_mem_print_leaks();
    signal(sig, SIG_DFL);
    raise(sig);
}

static void mlx_mem_maybe_install_crash_handler(void) {
    if (g_crash_handler_installed) return;
    const char *env = getenv("MLX_MEM_ON_CRASH");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        signal(SIGABRT, mlx_mem_crash_handler);
        signal(SIGSEGV, mlx_mem_crash_handler);
        signal(SIGBUS, mlx_mem_crash_handler);
        signal(SIGILL, mlx_mem_crash_handler);
        g_crash_handler_installed = 1;
    }
}

/* Simple hash function for pointers */
static inline size_t ptr_hash(void *ptr) {
    size_t h = (size_t)ptr;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h & HASH_MASK;
}

void mlx_mem_init(void) {
    if (g_tracker.initialized) return;

    pthread_mutex_init(&g_tracker.lock, NULL);
    memset(g_tracker.buckets, 0, sizeof(g_tracker.buckets));
    g_tracker.count = 0;
    g_tracker.total_allocs = 0;
    g_tracker.total_frees = 0;
    g_tracker.peak_count = 0;
    g_tracker.next_alloc_id = 1;
    g_tracker.initialized = true;

    /* Check environment variable for runtime enable */
    const char *env = getenv("MLX_MEM_DEBUG");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        g_tracker.enabled = true;
    }

    /* Check verbosity level */
    const char *verb = getenv("MLX_MEM_VERBOSITY");
    if (verb) {
        g_tracker.verbosity = atoi(verb);
    }

    /* Optional warnings for untracked frees */
    const char *warn_untracked = getenv("MLX_MEM_WARN_UNTRACKED");
    if (warn_untracked && (strcmp(warn_untracked, "1") == 0 ||
                           strcmp(warn_untracked, "true") == 0)) {
        g_warn_untracked_frees = true;
    }

#ifdef MLX_MEM_DEBUG
    g_tracker.enabled = true;
#endif

    if (g_tracker.enabled && g_tracker.verbosity >= 1) {
        fprintf(stderr, "[mem_debug] MLX memory tracking enabled\n");
    }

    mlx_mem_maybe_install_crash_handler();
}

void mlx_mem_shutdown(bool print_leaks) {
    if (!g_tracker.initialized) return;

    pthread_mutex_lock(&g_tracker.lock);

    if (print_leaks && g_tracker.count > 0) {
        pthread_mutex_unlock(&g_tracker.lock);
        mlx_mem_print_leaks();
        pthread_mutex_lock(&g_tracker.lock);
    }

    /* Free all hash entries */
    for (size_t i = 0; i < HASH_SIZE; i++) {
        hash_entry_t *entry = g_tracker.buckets[i];
        while (entry) {
            hash_entry_t *next = entry->next;
            free(entry);
            entry = next;
        }
        g_tracker.buckets[i] = NULL;
    }

    g_tracker.count = 0;
    g_tracker.initialized = false;
    pthread_mutex_unlock(&g_tracker.lock);
    pthread_mutex_destroy(&g_tracker.lock);
}

bool mlx_mem_is_enabled(void) {
    if (!g_tracker.initialized) {
        mlx_mem_init();
    }
    return g_tracker.enabled;
}

void mlx_mem_set_enabled(bool enabled) {
    if (!g_tracker.initialized) {
        mlx_mem_init();
    }
    g_tracker.enabled = enabled;
}

void mlx_mem_set_verbosity(int level) {
    if (!g_tracker.initialized) {
        mlx_mem_init();
    }
    g_tracker.verbosity = level;
}

void mlx_mem_register(mlx_array arr, const char *file, int line,
                      const char *func, const char *varname) {
    if (!g_tracker.initialized) {
        mlx_mem_init();
    }
    if (!g_tracker.enabled || !arr.ctx) return;

    pthread_mutex_lock(&g_tracker.lock);

    size_t bucket = ptr_hash(arr.ctx);
    hash_entry_t *entry = malloc(sizeof(hash_entry_t));
    if (!entry) {
        pthread_mutex_unlock(&g_tracker.lock);
        return;
    }

    entry->record.ctx = arr.ctx;
    entry->record.file = file;
    entry->record.line = line;
    entry->record.func = func;
    entry->record.varname = varname;
    entry->record.alloc_id = g_tracker.next_alloc_id++;
    entry->record.ndim = 0;
    for (int i = 0; i < 4; i++) {
        entry->record.shape[i] = 0;
    }

    entry->next = g_tracker.buckets[bucket];
    g_tracker.buckets[bucket] = entry;
    g_tracker.count++;
    g_tracker.total_allocs++;

    if (g_tracker.count > g_tracker.peak_count) {
        g_tracker.peak_count = g_tracker.count;
    }

    if (g_tracker.verbosity >= 2) {
        fprintf(stderr, "[mem_debug] ALLOC #%zu at %s:%d (%s) var=%s ctx=%p\n",
                entry->record.alloc_id, file, line, func,
                varname ? varname : "<anon>", arr.ctx);
    }

    pthread_mutex_unlock(&g_tracker.lock);
}

void mlx_mem_unregister(mlx_array arr, const char *file, int line,
                        const char *func) {
    if (!g_tracker.initialized || !g_tracker.enabled || !arr.ctx) return;

    pthread_mutex_lock(&g_tracker.lock);

    size_t bucket = ptr_hash(arr.ctx);
    hash_entry_t *prev = NULL;
    hash_entry_t *entry = g_tracker.buckets[bucket];

    while (entry) {
        if (entry->record.ctx == arr.ctx) {
            if (g_tracker.verbosity >= 2) {
                fprintf(stderr,
                        "[mem_debug] FREE #%zu at %s:%d (%s) "
                        "(allocated at %s:%d)\n",
                        entry->record.alloc_id, file, line, func,
                        entry->record.file, entry->record.line);
            }

            if (prev) {
                prev->next = entry->next;
            } else {
                g_tracker.buckets[bucket] = entry->next;
            }
            free(entry);
            g_tracker.count--;
            g_tracker.total_frees++;
            pthread_mutex_unlock(&g_tracker.lock);
            return;
        }
        prev = entry;
        entry = entry->next;
    }

    /* Not found - might be untracked allocation or double-free attempt */
    if (g_warn_untracked_frees || g_tracker.verbosity >= 2) {
        fprintf(stderr,
                "[mem_debug] WARNING: FREE of untracked array at %s:%d (%s) "
                "ctx=%p\n",
                file, line, func, arr.ctx);
    }

    pthread_mutex_unlock(&g_tracker.lock);
}

void mlx_mem_print_leaks(void) {
    if (!g_tracker.initialized) return;

    pthread_mutex_lock(&g_tracker.lock);

    if (g_tracker.count == 0) {
        fprintf(stderr, "[mem_debug] No memory leaks detected.\n");
        pthread_mutex_unlock(&g_tracker.lock);
        return;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║             MLX ARRAY MEMORY LEAK REPORT                     ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ %zu leaked allocation(s) detected                             \n",
            g_tracker.count);
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");

    size_t leak_num = 0;
    for (size_t i = 0; i < HASH_SIZE; i++) {
        hash_entry_t *entry = g_tracker.buckets[i];
        while (entry) {
            leak_num++;
            mlx_mem_record_t *r = &entry->record;

            fprintf(stderr, "║ Leak #%zu (alloc #%zu):\n", leak_num, r->alloc_id);
            fprintf(stderr, "║   Location: %s:%d\n", r->file, r->line);
            fprintf(stderr, "║   Function: %s\n", r->func);
            if (r->varname) {
                fprintf(stderr, "║   Variable: %s\n", r->varname);
            }
            fprintf(stderr, "║   Shape: [");
            for (int d = 0; d < r->ndim; d++) {
                fprintf(stderr, "%d%s", r->shape[d], d < r->ndim - 1 ? ", " : "");
            }
            fprintf(stderr, "] (ndim=%d)\n", r->ndim);
            fprintf(stderr, "║   Context: %p\n", r->ctx);
            fprintf(stderr, "╟──────────────────────────────────────────────────────────────╢\n");

            entry = entry->next;
        }
    }

    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n\n");

    pthread_mutex_unlock(&g_tracker.lock);
}

size_t mlx_mem_get_count(void) {
    if (!g_tracker.initialized) return 0;
    pthread_mutex_lock(&g_tracker.lock);
    size_t count = g_tracker.count;
    pthread_mutex_unlock(&g_tracker.lock);
    return count;
}

size_t mlx_mem_get_total_allocs(void) {
    if (!g_tracker.initialized) return 0;
    return g_tracker.total_allocs;
}

size_t mlx_mem_get_total_frees(void) {
    if (!g_tracker.initialized) return 0;
    return g_tracker.total_frees;
}

void mlx_mem_print_stats(void) {
    if (!g_tracker.initialized) {
        fprintf(stderr, "[mem_debug] Tracker not initialized.\n");
        return;
    }

    pthread_mutex_lock(&g_tracker.lock);
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║             MLX ARRAY MEMORY STATISTICS                      ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Total allocations:     %10zu                            ║\n",
            g_tracker.total_allocs);
    fprintf(stderr, "║ Total frees:           %10zu                            ║\n",
            g_tracker.total_frees);
    fprintf(stderr, "║ Current tracked:       %10zu                            ║\n",
            g_tracker.count);
    fprintf(stderr, "║ Peak concurrent:       %10zu                            ║\n",
            g_tracker.peak_count);
    fprintf(stderr, "║ Leaked (if any):       %10zu                            ║\n",
            g_tracker.total_allocs - g_tracker.total_frees);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n\n");
    pthread_mutex_unlock(&g_tracker.lock);
}

/* Helper functions used by macros */

mlx_array mlx_mem_tracked_new(const char *file, int line, const char *func,
                              const char *varname) {
    mlx_array arr = mlx_array_new();
    mlx_mem_register(arr, file, line, func, varname);
    return arr;
}

mlx_array mlx_mem_tracked_new_float(float v, const char *file, int line,
                                    const char *func) {
    mlx_array arr = mlx_array_new_float(v);
    mlx_mem_register(arr, file, line, func, "float");
    return arr;
}

mlx_array mlx_mem_tracked_new_int(int v, const char *file, int line,
                                  const char *func) {
    mlx_array arr = mlx_array_new_int(v);
    mlx_mem_register(arr, file, line, func, "int");
    return arr;
}

mlx_array mlx_mem_tracked_new_bool(bool v, const char *file, int line,
                                   const char *func) {
    mlx_array arr = mlx_array_new_bool(v);
    mlx_mem_register(arr, file, line, func, "bool");
    return arr;
}

mlx_array mlx_mem_tracked_new_data(void *buf, const int *shape, int ndim,
                                   mlx_dtype dtype, const char *file, int line,
                                   const char *func) {
    mlx_array arr = mlx_array_new_data(buf, shape, ndim, dtype);
    mlx_mem_register(arr, file, line, func, "data");
    return arr;
}

void mlx_mem_tracked_free(mlx_array arr, const char *file, int line,
                          const char *func) {
    mlx_mem_unregister(arr, file, line, func);
    mlx_array_free(arr);
}

void mlx_mem_tracked_detach_free(mlx_array arr, const char *file, int line,
                                 const char *func) {
    mlx_mem_unregister(arr, file, line, func);
    detach_and_free(arr);
}

/* Scope-based leak checking */

size_t mlx_mem_scope_begin(void) {
    if (!g_tracker.initialized) {
        mlx_mem_init();
    }
    return g_tracker.count;
}

void mlx_mem_scope_end(size_t start_count, const char *scope_name) {
    if (!g_tracker.initialized) return;

    size_t current = g_tracker.count;
    if (current > start_count) {
        fprintf(stderr,
                "[mem_debug] SCOPE LEAK in '%s': %zu new allocation(s) not "
                "freed\n",
                scope_name, current - start_count);
    } else if (g_tracker.verbosity >= 2) {
        fprintf(stderr, "[mem_debug] SCOPE '%s': clean (no leaks)\n",
                scope_name);
    }
}

/* Automatic shutdown registration */
__attribute__((destructor)) static void mlx_mem_auto_shutdown(void) {
    if (g_tracker.initialized && g_tracker.enabled) {
        mlx_mem_print_stats();
        mlx_mem_shutdown(true);
    }
}
