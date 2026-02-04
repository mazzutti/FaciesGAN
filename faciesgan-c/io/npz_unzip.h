#ifndef FACIESGAN_IO_NPZ_UNZIP_H
#define FACIESGAN_IO_NPZ_UNZIP_H

#include <stddef.h>
#include <stdint.h>
#include "mlx/c/io_types.h"

/* Extract a member from a .npz (zip) archive using the vendored miniz or
 * libzip backend. Writes the member contents into a temporary file and
 * returns its path in `out_temp_path` (caller must free()). Returns 0 on
 * success.
int npz_extract_member_to_temp(const char *npz_path, const char *member_name, char **out_temp_path);

/* Extract a member into a memory buffer. The buffer is allocated with
 * `malloc()` and must be freed by the caller using `free()`.
 * Returns 0 on success.
int npz_extract_member_to_memory(const char *npz_path, const char *member_name, void **out_buf, size_t *out_size);

/* Extract a member and create an MLX IO reader that reads from the in-memory
 * buffer. The returned `mlx_io_reader` should be freed with
 * `mlx_io_reader_free()` when no longer needed. Returns 0 on success.
int npz_extract_member_to_mlx_reader(const char *npz_path, const char *member_name, mlx_io_reader *out_reader);

#endif
