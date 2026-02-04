#ifndef FACIES_DATASETS_WELLS_H
#define FACIES_DATASETS_WELLS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Extract columns.npy and counts.npy from the first .npz mapping file
 * found under data_root/subdir. Returns 0 on success and allocates
 * temp paths for `out_columns_tmp` and `out_counts_tmp` which the caller
 * must free. Also returns an array of image file paths (caller frees each
 * string and the array) and the number of images.
 */
int datasets_extract_mapping_columns_counts(const char *data_root, const char *subdir,
                                            char **out_columns_tmp, char **out_counts_tmp,
                                            char ***out_image_files, int *out_image_count);

/* Load integer mapping arrays from the mapping .npz into C buffers.
 * The returned `out_columns` and `out_counts` are heap-allocated and must
 * be freed by the caller. `out_n` is the number of entries.
 */
int datasets_load_wells_mapping(const char *data_root, const char *subdir,
                               int32_t **out_columns, int32_t **out_counts, int *out_n,
                               char ***out_image_files, int *out_image_count);

#ifdef __cplusplus
}
#endif

#endif /* FACIES_DATASETS_WELLS_H */
