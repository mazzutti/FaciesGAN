#include "utils.h"
#include "func_cache.h"
#include "io/npz_unzip.h"
#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* MLX headers needed for array/stream helpers */
#include "trainning/array_helpers.h"
#include <mlx/c/array.h>
#include <mlx/c/io.h>
#include <mlx/c/ops.h>
#include <mlx/c/stream.h>

static int has_image_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext)
        return 0;
    ++ext;
    char low[16];
    size_t i = 0;
    while (ext[i] && i + 1 < sizeof(low)) {
        low[i] = (char)tolower((unsigned char)ext[i]);
        i++;
    }
    low[i] = '\0';
    return (strcmp(low, "png") == 0 || strcmp(low, "jpg") == 0 ||
            strcmp(low, "jpeg") == 0 || strcmp(low, "bmp") == 0);
}

static int has_model_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext)
        return 0;
    ++ext;
    return (strcmp(ext, "pt") == 0 || strcmp(ext, "pth") == 0);
}

static int cmp_str_ptr(const void *a, const void *b) {
    const char *const *pa = (const char *const *)a;
    const char *const *pb = (const char *const *)b;
    return strcmp(*pa, *pb);
}

int datasets_list_image_files(const char *data_root, const char *subdir,
                              char ***files, int *count) {
    if (!data_root || !subdir || !files || !count)
        return -1;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", data_root, subdir);
    DIR *d = opendir(path);
    if (!d) {
        *files = NULL;
        *count = 0;
        return 0;
    }
    struct dirent *ent;
    char **list = NULL;
    int n = 0;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_type == DT_DIR)
            continue;
        if (!has_image_ext(ent->d_name))
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
    /* sort */
    if (n > 1) {
        qsort(list, n, sizeof(char *), cmp_str_ptr);
    }
    *files = list;
    *count = n;
    return 0;
    if (n > 1) {
        qsort(list, n, sizeof(char *), cmp_str_ptr);
    }
    *files = list;
    *count = n;
    return 0;
}

int datasets_list_model_files(const char *data_root, const char *subdir,
                              char ***files, int *count) {
    if (!data_root || !subdir || !files || !count)
        return -1;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", data_root, subdir);
    DIR *d = opendir(path);
    if (!d) {
        *files = NULL;
        *count = 0;
        return 0;
    }
    struct dirent *ent;
    char **list = NULL;
    int n = 0;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_type == DT_DIR)
            continue;
        if (!has_model_ext(ent->d_name))
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
    if (n > 1)
        qsort(list, n, sizeof(char *), cmp_str_ptr);
    *files = list;
    *count = n;
    return 0;
}

int dataset_generate_scales(const TrainningOptions *opts, int channels_last,
                            DatasetScale **out, int *out_count) {
    if (!opts || !out || !out_count)
        return -1;
    int stop_scale = opts->stop_scale;
    int crop_size = opts->crop_size;
    int max_size = opts->max_size > 0 ? opts->max_size : crop_size;
    int min_size = opts->min_size > 0 ? opts->min_size : 1;
    int batch = opts->batch_size > 0 ? opts->batch_size : 1;
    int channels = opts->num_img_channels > 0 ? opts->num_img_channels : 1;

    int nscales = stop_scale + 1;
    DatasetScale *arr = NULL;
    if ((size_t)nscales > (size_t)INT_MAX) {
        arr = malloc(sizeof(DatasetScale) * (size_t)nscales);
        if (!arr)
            return -1;
    } else {
        if (mlx_alloc_pod((void **)&arr, sizeof(DatasetScale), nscales) != 0)
            return -1;
    }

    double scale_factor = 1.0;
    if (stop_scale > 0) {
        scale_factor =
            pow((double)min_size /
                (double)((max_size < crop_size ? max_size : crop_size)),
                1.0 / (double)stop_scale);
    }

    for (int i = 0; i < nscales; ++i) {
        double s = pow(scale_factor, (double)(stop_scale - i));
        double base = (double)((max_size < crop_size ? max_size : crop_size)) * s;
        int out_wh = (int)(round(base));
        if (out_wh % 2 != 0)
            out_wh += 1;
        arr[i].batch = batch;
        arr[i].height = out_wh;
        arr[i].width = out_wh;
        arr[i].channels = channels;
    }

    *out = arr;
    *out_count = nscales;
    return 0;
}

int to_facies_pyramids(const TrainningOptions *opts, int channels_last,
                       DatasetScale *scales, int n_scales, mlx_array **out,
                       int *out_count) {
    if (!opts || !out || !out_count)
        return -1;
    /* Ensure function cache exists. If caller didn't provide scales, generate
     * them locally so the cache key includes the full scale list (matches
     * Python behavior). Only free scales we allocate here. */
    char cache_npz[PATH_MAX] = {0};
    int actual_samples = 0;
    int desired = opts->num_train_pyramids > 0 ? opts->num_train_pyramids : 1024;
    int use_wells = opts->use_wells ? 1 : 0;
    int use_seismic = opts->use_seismic ? 1 : 0;
    DatasetScale *local_scales = scales;
    int local_n = n_scales;
    int local_alloc = 0;
    if (!local_scales || local_n <= 0) {
        if (dataset_generate_scales(opts, channels_last, &local_scales, &local_n) !=
                0)
            return -1;
        local_alloc = 1;
    }
    int rc = ensure_function_cache(
                 opts->input_path ? opts->input_path : ".",
                 opts->output_path ? opts->output_path : ".", desired, local_scales,
                 local_n, opts->num_img_channels, use_wells, use_seismic,
                 opts->manual_seed, cache_npz, sizeof(cache_npz), &actual_samples);
    if (rc != 0) {
        return -1;
    }

    mlx_array *fac = NULL;
    if (mlx_alloc_mlx_array_vals(&fac, local_n) != 0) {
        return -1;
    }

    for (int sidx = 0; sidx < local_n; ++sidx) {
        char member[512];
        mlx_io_reader r = {0};
        int found = 0;

        snprintf(member, sizeof(member), "facies_%d.npy", sidx);
        if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0) {
            found = 1;
        } else {
            /* try alternate names */
            snprintf(member, sizeof(member), "sample_0/facies_%d.npy", sidx);
            if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0)
                found = 1;
            else {
                snprintf(member, sizeof(member), "facies_%d_scale_%d.npy", sidx, 0);
                if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0)
                    found = 1;
            }
        }

        mlx_stream s = mlx_default_cpu_stream_new();
        mlx_array a = mlx_array_new();
        if (found) {
            if (mlx_load_reader(&a, r, s) != 0) {
                /* fallthrough to assemble from per-sample members */
                mlx_io_reader_free(r);
                mlx_array_free(a);
                found = 0;
            } else {
                mlx_io_reader_free(r);
                /* If the loaded array is 3D (H,W,C), reshape to 4D (1,H,W,C)
                 * so downstream code can reliably slice with a leading sample axis. */
                int ndim = (int)mlx_array_ndim(a);
                if (ndim == 3) {
                    const int *sh = mlx_array_shape(a);
                    int new_shape[4] = {1, sh[0], sh[1], sh[2]};
                    mlx_array tmp = mlx_array_new();
                    if (mlx_reshape(&tmp, a, new_shape, 4, s) == 0) {
                        mlx_array_free(a);
                        a = tmp;
                    } else {
                        mlx_array_free(tmp);
                    }
                }
                fac[sidx] = a;
                mlx_stream_free(s);
                continue;
            }
        }

        /* assemble by reading sample_%d/facies_%d.npy for each sample */
        mlx_vector_array scale_vec = mlx_vector_array_new();
        for (int si = 0; si < actual_samples; ++si) {
            char mem2[512];
            snprintf(mem2, sizeof(mem2), "sample_%d/facies_%d.npy", si, sidx);
            mlx_io_reader r2 = {0};
            if (npz_extract_member_to_mlx_reader(cache_npz, mem2, &r2) != 0)
                break;
            mlx_array ai = mlx_array_new();
            if (mlx_load_reader(&ai, r2, s) != 0) {
                mlx_io_reader_free(r2);
                mlx_array_free(ai);
                break;
            }
            mlx_io_reader_free(r2);
            if (mlx_vector_array_append_value(scale_vec, ai) != 0) {
                mlx_array_free(ai);
                mlx_vector_array_free(scale_vec);
                mlx_stream_free(s);
                free(fac);
                return -1;
            }
            mlx_array_free(ai);
        }

        /* stack the per-sample arrays into a single array */
        mlx_array stacked = mlx_array_new();
        size_t vec_n = mlx_vector_array_size(scale_vec);
        if (vec_n == 0) {
            mlx_array_free(stacked);
            int shape0[4] = {0, local_scales[sidx].height, local_scales[sidx].width,
                             local_scales[sidx].channels
                            };
            {
                mlx_stream zst = mlx_default_cpu_stream_new();
                if (mlx_zeros(&stacked, shape0, 4, MLX_FLOAT32, zst) != 0) {
                    mlx_stream_free(zst);
                    free(fac);
                    mlx_stream_free(s);
                    return -1;
                }
                mlx_stream_free(zst);
            }
        } else if (mlx_stack(&stacked, scale_vec, s) != 0) {
            mlx_array_free(stacked);
            /* create empty zeros array as fallback */
            int shape0[4] = {0, local_scales[sidx].height, local_scales[sidx].width,
                             local_scales[sidx].channels
                            };
            {
                mlx_stream zst = mlx_default_cpu_stream_new();
                if (mlx_zeros(&stacked, shape0, 4, MLX_FLOAT32, zst) != 0) {
                    mlx_stream_free(zst);
                    free(fac);
                    mlx_stream_free(s);
                    return -1;
                }
                mlx_stream_free(zst);
            }
        }
        mlx_vector_array_free(scale_vec);
        fac[sidx] = stacked;
        mlx_stream_free(s);
    }
    *out = fac;
    *out_count = local_n;
    return 0;
}

int to_seismic_pyramids(const TrainningOptions *opts, int channels_last,
                        DatasetScale *scales, int n_scales, mlx_array **out,
                        int *out_count) {
    if (!opts || !out || !out_count)
        return -1;
    /* Ensure function cache exists. Generate scales if none provided. */
    char cache_npz[PATH_MAX] = {0};
    int actual_samples = 0;
    int desired = opts->num_train_pyramids > 0 ? opts->num_train_pyramids : 1024;
    int use_wells = opts->use_wells ? 1 : 0;
    int use_seismic = opts->use_seismic ? 1 : 0;
    DatasetScale *local_scales = scales;
    int local_n = n_scales;
    int local_alloc = 0;
    if (!local_scales || local_n <= 0) {
        if (dataset_generate_scales(opts, channels_last, &local_scales, &local_n) !=
                0)
            return -1;
        local_alloc = 1;
    }
    int rc = ensure_function_cache(
                 opts->input_path ? opts->input_path : ".",
                 opts->output_path ? opts->output_path : ".", desired, local_scales,
                 local_n, opts->num_img_channels, use_wells, use_seismic,
                 opts->manual_seed, cache_npz, sizeof(cache_npz), &actual_samples);
    if (rc != 0) {
        if (local_alloc)
            free(local_scales);
        return -1;
    }

    mlx_array *seis = NULL;
    if (mlx_alloc_mlx_array_vals(&seis, local_n) != 0) {
        return -1;
    }

    for (int sidx = 0; sidx < n_scales; ++sidx) {
        char member[512];
        mlx_io_reader r = {0};
        int found = 0;

        snprintf(member, sizeof(member), "seismic_%d.npy", sidx);
        if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0) {
            found = 1;
        } else {
            snprintf(member, sizeof(member), "sample_0/seismic_%d.npy", sidx);
            if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0)
                found = 1;
            else {
                snprintf(member, sizeof(member), "seismic_%d_scale_%d.npy", sidx, 0);
                if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0)
                    found = 1;
            }
        }

        mlx_stream s = mlx_default_cpu_stream_new();
        mlx_array a = mlx_array_new();
        if (found) {
            if (mlx_load_reader(&a, r, s) != 0) {
                mlx_io_reader_free(r);
                mlx_array_free(a);
                found = 0;
            } else {
                mlx_io_reader_free(r);
                /* Normalize shape to (N,H,W,C) if a 3D array was saved */
                int ndim = (int)mlx_array_ndim(a);
                if (ndim == 3) {
                    const int *sh = mlx_array_shape(a);
                    int new_shape[4] = {1, sh[0], sh[1], sh[2]};
                    mlx_array tmp = mlx_array_new();
                    if (mlx_reshape(&tmp, a, new_shape, 4, s) == 0) {
                        mlx_array_free(a);
                        a = tmp;
                    } else {
                        mlx_array_free(tmp);
                    }
                }
                seis[sidx] = a;
                mlx_stream_free(s);
                continue;
            }
        }

        mlx_vector_array scale_vec = mlx_vector_array_new();
        for (int si = 0; si < actual_samples; ++si) {
            char mem2[512];
            snprintf(mem2, sizeof(mem2), "sample_%d/seismic_%d.npy", si, sidx);
            mlx_io_reader r2 = {0};
            if (npz_extract_member_to_mlx_reader(cache_npz, mem2, &r2) != 0)
                break;
            mlx_array ai = mlx_array_new();
            if (mlx_load_reader(&ai, r2, s) != 0) {
                mlx_io_reader_free(r2);
                mlx_array_free(ai);
                break;
            }
            mlx_io_reader_free(r2);
            if (mlx_vector_array_append_value(scale_vec, ai) != 0) {
                mlx_array_free(ai);
                mlx_vector_array_free(scale_vec);
                mlx_stream_free(s);
                free(seis);
                return -1;
            }
            mlx_array_free(ai);
        }

        mlx_array stacked = mlx_array_new();
        size_t vec_n = mlx_vector_array_size(scale_vec);
        if (vec_n == 0) {
            mlx_array_free(stacked);
            int shape0[4] = {0, scales[sidx].height, scales[sidx].width,
                             scales[sidx].channels
                            };
            {
                mlx_stream zst = mlx_default_cpu_stream_new();
                if (mlx_zeros(&stacked, shape0, 4, MLX_FLOAT32, zst) != 0) {
                    mlx_stream_free(zst);
                    free(seis);
                    mlx_stream_free(s);
                    return -1;
                }
                mlx_stream_free(zst);
            }
        } else if (mlx_stack(&stacked, scale_vec, s) != 0) {
            mlx_array_free(stacked);
            int shape0[4] = {0, scales[sidx].height, scales[sidx].width,
                             scales[sidx].channels
                            };
            {
                mlx_stream zst = mlx_default_cpu_stream_new();
                if (mlx_zeros(&stacked, shape0, 4, MLX_FLOAT32, zst) != 0) {
                    mlx_stream_free(zst);
                    free(seis);
                    mlx_stream_free(s);
                    return -1;
                }
                mlx_stream_free(zst);
            }
        }
        mlx_vector_array_free(scale_vec);
        seis[sidx] = stacked;
        mlx_stream_free(s);
    }
    *out = seis;
    *out_count = local_n;
    return 0;
}

int to_wells_pyramids(const TrainningOptions *opts, int channels_last,
                      DatasetScale *scales, int n_scales, mlx_array **out,
                      int *out_count) {
    if (!opts || !out || !out_count)
        return -1;

    char cache_npz[PATH_MAX] = {0};
    int actual_samples = 0;
    int desired = opts->num_train_pyramids > 0 ? opts->num_train_pyramids : 1024;
    int use_wells = opts->use_wells ? 1 : 0;
    int use_seismic = opts->use_seismic ? 1 : 0;
    DatasetScale *local_scales = scales;
    int local_n = n_scales;
    int local_alloc = 0;
    if (!local_scales || local_n <= 0) {
        if (dataset_generate_scales(opts, channels_last, &local_scales, &local_n) !=
                0)
            return -1;
        local_alloc = 1;
    }
    int rc = ensure_function_cache(
                 opts->input_path ? opts->input_path : ".",
                 opts->output_path ? opts->output_path : ".", desired, local_scales,
                 local_n, opts->num_img_channels, use_wells, use_seismic,
                 opts->manual_seed, cache_npz, sizeof(cache_npz), &actual_samples);
    if (rc != 0) {
        if (local_alloc)
            free(local_scales);
        return -1;
    }

    mlx_array *wells = NULL;
    if (mlx_alloc_mlx_array_vals(&wells, local_n) != 0) {
        return -1;
    }

    for (int sidx = 0; sidx < n_scales; ++sidx) {
        char member[512];
        mlx_io_reader r = {0};
        int found = 0;

        snprintf(member, sizeof(member), "wells_%d.npy", sidx);
        if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0) {
            found = 1;
        } else {
            snprintf(member, sizeof(member), "sample_0/wells_%d.npy", sidx);
            if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0)
                found = 1;
            else {
                snprintf(member, sizeof(member), "wells_%d_scale_%d.npy", sidx, 0);
                if (npz_extract_member_to_mlx_reader(cache_npz, member, &r) == 0)
                    found = 1;
            }
        }

        mlx_stream s = mlx_default_cpu_stream_new();
        mlx_array a = mlx_array_new();
        if (found) {
            if (mlx_load_reader(&a, r, s) != 0) {
                mlx_io_reader_free(r);
                mlx_array_free(a);
                found = 0;
            } else {
                mlx_io_reader_free(r);
                /* Normalize to (N,H,W,C) if a 3D array was saved */
                int ndim = (int)mlx_array_ndim(a);
                if (ndim == 3) {
                    const int *sh = mlx_array_shape(a);
                    int new_shape[4] = {1, sh[0], sh[1], sh[2]};
                    mlx_array tmp = mlx_array_new();
                    if (mlx_reshape(&tmp, a, new_shape, 4, s) == 0) {
                        mlx_array_free(a);
                        a = tmp;
                    } else {
                        mlx_array_free(tmp);
                    }
                }
                wells[sidx] = a;
                mlx_stream_free(s);
                continue;
            }
        }

        mlx_vector_array scale_vec = mlx_vector_array_new();
        for (int si = 0; si < actual_samples; ++si) {
            char mem2[512];
            snprintf(mem2, sizeof(mem2), "sample_%d/wells_%d.npy", si, sidx);
            mlx_io_reader r2 = {0};
            if (npz_extract_member_to_mlx_reader(cache_npz, mem2, &r2) != 0)
                break;
            mlx_array ai = mlx_array_new();
            if (mlx_load_reader(&ai, r2, s) != 0) {
                mlx_io_reader_free(r2);
                mlx_array_free(ai);
                break;
            }
            mlx_io_reader_free(r2);
            if (mlx_vector_array_append_value(scale_vec, ai) != 0) {
                mlx_array_free(ai);
                mlx_vector_array_free(scale_vec);
                mlx_stream_free(s);
                free(wells);
                return -1;
            }
            mlx_array_free(ai);
        }

        mlx_array stacked = mlx_array_new();
        size_t vec_n = mlx_vector_array_size(scale_vec);
        if (vec_n == 0) {
            mlx_array_free(stacked);
            int shape0[4] = {0, scales[sidx].height, scales[sidx].width,
                             scales[sidx].channels
                            };
            {
                mlx_stream zst = mlx_default_cpu_stream_new();
                if (mlx_zeros(&stacked, shape0, 4, MLX_FLOAT32, zst) != 0) {
                    mlx_stream_free(zst);
                    free(wells);
                    mlx_stream_free(s);
                    return -1;
                }
                mlx_stream_free(zst);
            }
        } else if (mlx_stack(&stacked, scale_vec, s) != 0) {
            mlx_vector_array_free(scale_vec);
            mlx_array_free(stacked);
            int shape0[4] = {0, scales[sidx].height, scales[sidx].width,
                             scales[sidx].channels
                            };
            {
                mlx_stream zst = mlx_default_cpu_stream_new();
                if (mlx_zeros(&stacked, shape0, 4, MLX_FLOAT32, zst) != 0) {
                    mlx_stream_free(zst);
                    free(wells);
                    mlx_stream_free(s);
                    return -1;
                }
                mlx_stream_free(zst);
            }
        }
        mlx_vector_array_free(scale_vec);
        wells[sidx] = stacked;
        mlx_stream_free(s);
    }

    *out = wells;
    *out_count = local_n;
    return 0;
}
