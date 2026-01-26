#ifndef MLX_ARRAY_HELPERS_H
#define MLX_ARRAY_HELPERS_H

#include <mlx/c/mlx.h>
#include <stdlib.h>

/* Copy a float array into a newly allocated buffer stored in *out and set
 * *out_n. Returns 0 on success, -1 on allocation failure. If src is NULL or
 * n<=0, out is set to NULL and out_n to 0.
 */
int mlx_copy_float_array(float **out, int *out_n, const float *src, int n);

/* Free a float array allocated by mlx_copy_float_array. */
void mlx_free_float_array(float **arr, int *n);

/* Copy an int array into a newly allocated buffer stored in *out and set
 * *out_n. Returns 0 on success, -1 on allocation failure.
 */
int mlx_copy_int_array(int **out, int *out_n, const int *src, int n);

/* Free an int array allocated by mlx_copy_int_array. */
void mlx_free_int_array(int **arr, int *n);

/* Allocate an int array of length n and zero-initialize it. Returns 0 on
 * success, -1 on failure. If n<=0, *out is set to NULL and 0 returned.
 */
int mlx_alloc_int_array(int **out, int n);

/* Allocate an array of mlx_array* of length n and zero-initialize entries.
 * On success *out will point to the allocated array and 0 returned.
 * On failure *out is left NULL and -1 returned.
 */
int mlx_alloc_mlx_array_ptrs(mlx_array ***out, int n);

/* Free an array of mlx_array* of length n. Each non-NULL pointed mlx_array
 * will be freed with mlx_array_free and the pointer freed. The array itself
 * is then freed and *out set to NULL.
 */
void mlx_free_mlx_array_ptrs(mlx_array ***out, int n);

/* Allocate a generic pointer array (void**) of length n and zero-initialize.
 * On success *out points to the allocated array and 0 returned.
 */
int mlx_alloc_ptr_array(void ***out, int n);

/* Free a generic pointer array previously allocated with mlx_alloc_ptr_array.
 * The function frees the array but not the pointed-to elements.
 */
void mlx_free_ptr_array(void ***out, int n);

/* Allocate a float buffer of length n and store pointer in *out.
 * Returns 0 on success, -1 on failure. If n<=0, *out is set to NULL and 0
 * returned.
 */
int mlx_alloc_float_buf(float **out, int n);

/* Reallocate an existing float buffer to new size n, preserving contents up to
 * min(old,new). If *out is NULL behaves like mlx_alloc_float_buf.
 */
int mlx_realloc_float_buf(float **out, int old_n, int new_n);

/* Free a float buffer allocated by the helpers (or plain malloc). */
void mlx_free_float_buf(float **out, int *n);

/* Allocate a contiguous buffer of `mlx_array` values (not pointers).
 * On success *out will point to the allocated block and 0 returned.
 */
int mlx_alloc_mlx_array_vals(mlx_array **out, int n);

/* Free a contiguous `mlx_array` buffer allocated with mlx_alloc_mlx_array_vals.
 */
void mlx_free_mlx_array_vals(mlx_array **out, int n);

/* Allocate an uninitialized contiguous buffer of `mlx_array` values (not
 * pointers). On success *out will point to the allocated block and 0 returned.
 * Unlike `mlx_alloc_mlx_array_vals` this does NOT initialize elements; callers
 * are expected to initialize or overwrite each element. This is useful when the
 * caller will move/assign existing mlx_array values into the buffer and does
 * not want helper-initialized arrays to be allocated.
 */
int mlx_alloc_mlx_array_raw(mlx_array **out, int n);

/* Free a contiguous `mlx_array` buffer previously allocated with
 * `mlx_alloc_mlx_array_raw`. This only frees the container; it does NOT free
 * individual `mlx_array` elements. Use `mlx_free_mlx_array_vals` if you want
 * elements freed as well.
 */
void mlx_free_mlx_array_raw(mlx_array **out, int n);

/* Allocate a plain-old-data (POD) array of `n` elements each of size
 * `elem_size`. On success *out will point to the allocated block and 0
 * returned. Use `mlx_free_pod` to free the allocation. Passing n==0 sets *out
 * to NULL and returns 0.
 */
int mlx_alloc_pod(void **out, size_t elem_size, int n);

/* Free a POD array allocated with `mlx_alloc_pod`. Sets *out to NULL. */
void mlx_free_pod(void **out);

#endif
