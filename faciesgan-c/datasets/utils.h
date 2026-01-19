#ifndef FACIESGAN_DATASETS_UTILS_H
#define FACIESGAN_DATASETS_UTILS_H

#include <stddef.h>
#include "options.h"

typedef struct DatasetScale
{
    int batch;
    int channels;
    int height;
    int width;
} DatasetScale;

/* Generate scales from TrainningOptions. Returns 0 on success and fills
 * `out` with a malloc'd array of DatasetScale; caller must free(*out).
 */
int datasets_generate_scales(const TrainningOptions *opts, int channels_last, DatasetScale **out, int *out_count);

/* List image files in a data subdirectory (e.g., "facies", "wells", "seismic").
 * Returns 0 on success and fills `files` with a malloc'd array of C strings
 * (each strdup'd). Caller must free each string and then free(files).
 */
int datasets_list_image_files(const char *data_root, const char *subdir, char ***files, int *count);

/* List model files (e.g., '*.pt') under a subdir. Similar ownership as above. */
int datasets_list_model_files(const char *data_root, const char *subdir, char ***files, int *count);

#endif
