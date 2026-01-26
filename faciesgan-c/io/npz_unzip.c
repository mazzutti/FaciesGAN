#include "npz_unzip.h"
#include "mlx/c/io_types.h"
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "miniz.h"
#include "trainning/array_helpers.h"

/* In-memory descriptor and callbacks for MLX io reader */
struct npz_mem_desc {
    unsigned char *data;
    size_t size;
    size_t pos;
    char *label;
};

static bool npz_mem_is_open(void *ctx) {
    (void)ctx;
    return true;
}
static bool npz_mem_good(void *ctx) {
    struct npz_mem_desc *m = ctx;
    return m->pos < m->size;
}
static size_t npz_mem_tell(void *ctx) {
    struct npz_mem_desc *m = ctx;
    return m->pos;
}
static void npz_mem_seek(void *ctx, int64_t off, int whence) {
    struct npz_mem_desc *m = ctx;
    size_t newpos = m->pos;
    if (whence == SEEK_SET) {
        if (off < 0)
            newpos = 0;
        else
            newpos = (size_t)off;
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
        newpos = m->size;
    m->pos = newpos;
}
static void npz_mem_read(void *ctx, char *data, size_t n) {
    struct npz_mem_desc *m = ctx;
    size_t avail = m->size - m->pos;
    size_t toread = n <= avail ? n : avail;
    if (toread > 0)
        memcpy(data, m->data + m->pos, toread);
    if (n > toread)
        memset(data + toread, 0, n - toread);
    m->pos += toread;
}
static void npz_mem_read_at_offset(void *ctx, char *data, size_t n,
                                   size_t off) {
    struct npz_mem_desc *m = ctx;
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
}
static void npz_mem_write(void *ctx, const char *data, size_t n) {
    /* write not supported for in-memory descriptor */
}
static const char *npz_mem_label(void *ctx) {
    struct npz_mem_desc *m = ctx;
    return m->label ? m->label : "<mem>";
}
static void npz_mem_free(void *ctx) {
    struct npz_mem_desc *m = ctx;
    if (m) {
        free(m->data);
        free(m->label);
        mlx_free_pod((void **)&m);
    }
}

int npz_extract_member_to_temp(const char *npz_path, const char *member_name,
                               char **out_temp_path) {
    if (!npz_path || !member_name || !out_temp_path)
        return -1;

    mz_zip_archive za;
    memset(&za, 0, sizeof(za));
    if (!mz_zip_reader_init_file(&za, npz_path, 0))
        return -1;
    int idx = mz_zip_reader_locate_file(&za, member_name, NULL, 0);
    if (idx < 0) {
        mz_zip_reader_end(&za);
        return -1;
    }
    // Create temp file for extraction
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/facies_npz_%ld_XXXXXX", (long)time(NULL));
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        mz_zip_reader_end(&za);
        return -1;
    }
    close(fd);
    if (!mz_zip_reader_extract_to_file(&za, idx, tmpl, 0)) {
        mz_zip_reader_end(&za);
        unlink(tmpl);
        return -1;
    }
    mz_zip_reader_end(&za);
    *out_temp_path = strdup(tmpl);
    if (!*out_temp_path) {
        unlink(tmpl);
        return -1;
    }
    return 0;
}

int npz_extract_member_to_memory(const char *npz_path, const char *member_name,
                                 void **out_buf, size_t *out_size) {
    if (!npz_path || !member_name || !out_buf || !out_size)
        return -1;

    mz_zip_archive za;
    memset(&za, 0, sizeof(za));
    if (!mz_zip_reader_init_file(&za, npz_path, 0))
        return -1;
    int idx = mz_zip_reader_locate_file(&za, member_name, NULL, 0);
    if (idx < 0) {
        mz_zip_reader_end(&za);
        return -1;
    }
    size_t sz = 0;
    void *heap = mz_zip_reader_extract_to_heap(&za, idx, &sz, 0);
    if (!heap) {
        mz_zip_reader_end(&za);
        return -1;
    }
    void *buf = malloc(sz);
    if (!buf) {
        mz_free(heap);
        mz_zip_reader_end(&za);
        return -1;
    }
    memcpy(buf, heap, sz);
    mz_free(heap);
    mz_zip_reader_end(&za);
    *out_buf = buf;
    *out_size = sz;
    return 0;
}

/* Create an MLX IO reader that wraps an in-memory buffer extracted from an npz
 * member. */
int npz_extract_member_to_mlx_reader(const char *npz_path,
                                     const char *member_name,
                                     mlx_io_reader *out_reader) {
    if (!npz_path || !member_name || !out_reader)
        return -1;
    void *buf = NULL;
    size_t buf_size = 0;
    if (npz_extract_member_to_memory(npz_path, member_name, &buf, &buf_size) != 0)
        return -1;

    struct npz_mem_desc *d = NULL;
    if (mlx_alloc_pod((void **)&d, sizeof(*d), 1) != 0) {
        free(buf);
        return -1;
    }
    d->data = (unsigned char *)buf;
    d->size = buf_size;
    d->pos = 0;
    d->label = strdup(member_name);
    mlx_io_vtable vt = {
        .is_open = npz_mem_is_open,
        .good = npz_mem_good,
        .tell = npz_mem_tell,
        .seek = npz_mem_seek,
        .read = npz_mem_read,
        .read_at_offset = npz_mem_read_at_offset,
        .write = npz_mem_write,
        .label = npz_mem_label,
        .free = npz_mem_free,
    };

    *out_reader = mlx_io_reader_new(d, vt);
    return 0;
}
