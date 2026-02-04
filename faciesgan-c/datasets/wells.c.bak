#include "wells.h"
#include "io/npz_unzip.h"
#include "trainning/array_helpers.h"
#include "utils.h"
#include <dirent.h>
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/io.h>
#include <mlx/c/stream.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* comparator for qsort: compare strings pointed to by array elements */
static int cmp_strptr(const void *a, const void *b) {
  const char *const *pa = (const char *const *)a;
  const char *const *pb = (const char *const *)b;
  const unsigned char *sa = (const unsigned char *)*pa;
  const unsigned char *sb = (const unsigned char *)*pb;
  while (*sa && (*sa == *sb)) {
    sa++;
    sb++;
  }
  return (int)(*sa) - (int)(*sb);
}

/* Helper: find first .npz file in data_root/subdir (sorted lexicographically)
 */
static int find_first_npz(const char *data_root, const char *subdir,
                          char *out_path, size_t out_len) {
  if (!data_root || !subdir || !out_path)
    return -1;
  char path[PATH_MAX];
  snprintf(path, sizeof(path), "%s/%s", data_root, subdir);
  DIR *d = opendir(path);
  if (!d)
    return -1;
  struct dirent *ent;
  char **list = NULL;
  int n = 0;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_type == DT_DIR)
      continue;
    const char *ext = strrchr(ent->d_name, '.');
    if (!ext)
      continue;
    if (strcmp(ext, ".npz") != 0)
      continue;
    char full[PATH_MAX];
    snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
    char *s = strdup(full);
    if (!s)
      continue;
    char **tmp = realloc(list, sizeof(char *) * (n + 1));
    if (!tmp) {
      free(s);
      break;
    }
    list = tmp;
    list[n++] = s;
  }
  closedir(d);
  if (n == 0)
    return -1;
  if (n > 1)
    qsort(list, n, sizeof(char *), cmp_strptr);
  strncpy(out_path, list[0], out_len - 1);
  out_path[out_len - 1] = '\0';
  for (int i = 0; i < n; ++i)
    free(list[i]);
  free(list);
  return 0;
}

/* Public: extract columns.npy and counts.npy from mapping .npz, and list images
 */
int datasets_extract_mapping_columns_counts(
    const char *data_root, const char *subdir, char **out_columns_tmp,
    char **out_counts_tmp, char ***out_image_files, int *out_image_count) {
  if (!data_root || !subdir || !out_columns_tmp || !out_counts_tmp ||
      !out_image_files || !out_image_count)
    return -1;
  char npz_path[PATH_MAX];
  if (find_first_npz(data_root, subdir, npz_path, sizeof(npz_path)) != 0)
    return -1;

  /* Try member names with extension (typical for .npz): columns.npy and
   * counts.npy */
  char *cols_tmp = NULL;
  char *counts_tmp = NULL;
  mlx_io_reader cols_reader = {0};
  mlx_io_reader counts_reader = {0};
  /* Prefer in-memory extraction into MLX readers. Fall back to temp-file
   * extraction if needed. */
  if (npz_extract_member_to_mlx_reader(npz_path, "columns.npy", &cols_reader) !=
      0) {
    if (npz_extract_member_to_mlx_reader(npz_path, "columns", &cols_reader) !=
        0)
      return -1;
  }
  if (npz_extract_member_to_mlx_reader(npz_path, "counts.npy",
                                       &counts_reader) != 0) {
    if (npz_extract_member_to_mlx_reader(npz_path, "counts", &counts_reader) !=
        0) {
      mlx_io_reader_free(cols_reader);
      return -1;
    }
  }

  /* List image files in the same directory (sorted) */
  char **image_list = NULL;
  int img_count = 0;
  if (datasets_list_image_files(data_root, subdir, &image_list, &img_count) !=
      0) {
    free(cols_tmp);
    free(counts_tmp);
    return -1;
  }

  *out_columns_tmp = cols_tmp;
  *out_counts_tmp = counts_tmp;
  *out_image_files = image_list;
  *out_image_count = img_count;
  return 0;
}

int datasets_load_wells_mapping(const char *data_root, const char *subdir,
                                int32_t **out_columns, int32_t **out_counts,
                                int *out_n, char ***out_image_files,
                                int *out_image_count) {
  if (!data_root || !subdir || !out_columns || !out_counts || !out_n ||
      !out_image_files || !out_image_count)
    return -1;

  char *cols_tmp = NULL;
  char *counts_tmp = NULL;
  char **image_list = NULL;
  int img_count = 0;
  mlx_io_reader cols_reader = {0};
  mlx_io_reader counts_reader = {0};

  /* Try in-memory extraction first by locating the .npz and extracting into MLX
   * readers. */
  char npz_path[PATH_MAX];
  if (find_first_npz(data_root, subdir, npz_path, sizeof(npz_path)) == 0) {
    if (npz_extract_member_to_mlx_reader(npz_path, "columns.npy",
                                         &cols_reader) != 0) {
      /* try without extension */
      npz_extract_member_to_mlx_reader(npz_path, "columns", &cols_reader);
    }
    if (npz_extract_member_to_mlx_reader(npz_path, "counts.npy",
                                         &counts_reader) != 0) {
      npz_extract_member_to_mlx_reader(npz_path, "counts", &counts_reader);
    }
  }

  if (!cols_reader.ctx || !counts_reader.ctx) {
    /* If in-memory readers are not present, abort — we require the
     * vendored miniz or libzip backend. */
    if (cols_reader.ctx)
      mlx_io_reader_free(cols_reader);
    if (counts_reader.ctx)
      mlx_io_reader_free(counts_reader);
    return -1;
  }
  /* List image files in the directory. */
  if (datasets_list_image_files(data_root, subdir, &image_list, &img_count) !=
      0) {
    if (cols_reader.ctx)
      mlx_io_reader_free(cols_reader);
    if (counts_reader.ctx)
      mlx_io_reader_free(counts_reader);
    return -1;
  }

  mlx_stream s = mlx_default_gpu_stream_new();
  mlx_array col_arr = mlx_array_new();
  mlx_array cnt_arr = mlx_array_new();
  int rc = 0;
  if (cols_reader.ctx) {
    if (mlx_load_reader(&col_arr, cols_reader, s) != 0)
      rc = -1;
  } else {
    if (mlx_load(&col_arr, cols_tmp, s) != 0)
      rc = -1;
  }
  if (rc == 0) {
    if (counts_reader.ctx) {
      if (mlx_load_reader(&cnt_arr, counts_reader, s) != 0)
        rc = -1;
    } else {
      if (mlx_load(&cnt_arr, counts_tmp, s) != 0)
        rc = -1;
    }
  }

  if (rc != 0) {
    if (cols_tmp)
      unlink(cols_tmp);
    if (counts_tmp)
      unlink(counts_tmp);
    free(cols_tmp);
    free(counts_tmp);
    if (cols_reader.ctx)
      mlx_io_reader_free(cols_reader);
    if (counts_reader.ctx)
      mlx_io_reader_free(counts_reader);
    mlx_array_free(col_arr);
    mlx_array_free(cnt_arr);
    mlx_stream_free(s);
    /* free image list */
    for (int i = 0; i < img_count; ++i)
      free(image_list[i]);
    free(image_list);
    return -1;
  }

  size_t ncols = mlx_array_size(col_arr);
  if (ncols == 0) {
    rc = -1;
  }

  int32_t *cols = NULL;
  int32_t *counts = NULL;
  if (rc == 0) {
    const int32_t *cdata = mlx_array_data_int32(col_arr);
    const int32_t *kdata = mlx_array_data_int32(cnt_arr);
    if (!cdata || !kdata)
      rc = -1;
    else {
      cols = NULL;
      counts = NULL;
      if (ncols > (size_t)INT_MAX) {
        cols = (int32_t *)malloc(sizeof(int32_t) * ncols);
        counts = (int32_t *)malloc(sizeof(int32_t) * ncols);
        if (!cols || !counts)
          rc = -1;
      } else {
        if (mlx_alloc_pod((void **)&cols, sizeof(int32_t), (int)ncols) != 0 ||
            mlx_alloc_pod((void **)&counts, sizeof(int32_t), (int)ncols) != 0)
          rc = -1;
      }
      if (rc == 0) {
        for (size_t i = 0; i < ncols; ++i) {
          cols[i] = cdata[i];
          counts[i] = kdata[i];
        }
      }
    }
  }

  /* cleanup temp files and mlx objects */
  mlx_array_free(col_arr);
  mlx_array_free(cnt_arr);
  mlx_stream_free(s);
  if (cols_reader.ctx)
    mlx_io_reader_free(cols_reader);
  if (counts_reader.ctx)
    mlx_io_reader_free(counts_reader);
  if (cols_tmp) {
    unlink(cols_tmp);
    free(cols_tmp);
  }
  if (counts_tmp) {
    unlink(counts_tmp);
    free(counts_tmp);
  }

  if (rc != 0) {
    if (cols) {
      if (ncols > (size_t)INT_MAX)
        free(cols);
      else
        mlx_free_pod((void **)&cols);
    }
    if (counts) {
      if (ncols > (size_t)INT_MAX)
        free(counts);
      else
        mlx_free_pod((void **)&counts);
    }
    for (int i = 0; i < img_count; ++i)
      free(image_list[i]);
    free(image_list);
    return -1;
  }

  *out_columns = cols;
  *out_counts = counts;
  *out_n = (int)ncols;
  *out_image_files = image_list;
  *out_image_count = img_count;
  return 0;
}
