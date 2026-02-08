#include "mlx_dataset.h"
#include "datasets/utils.h"
#include "func_cache.h"
#include "io/npz_create.h"
#include "io/npz_unzip.h"
#include "trainning/array_helpers.h"
#include <dirent.h>
#include <limits.h>
#include <mlx/c/io.h>
#include <mlx/c/ops.h>
#include <mlx/c/stream.h>
#include <mlx/c/vector.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* in-file mlx_mem_stream vtable and helpers (copied pattern from optimizer.c)
 */
typedef struct mlx_mem_stream_ {
    char *data;
    size_t pos;
    size_t size;
    bool err;
    bool free_data;
} mlx_mem_stream;

static bool mem_is_open(void *desc) {
    return true;
}
static bool mem_good(void *desc) {
    mlx_mem_stream *m = desc;
    return !m->err;
}
static size_t mem_tell(void *desc) {
    mlx_mem_stream *m = desc;
    return m->pos;
}
static void mem_seek(void *desc, int64_t off, int whence) {
    mlx_mem_stream *m = desc;
    size_t newpos = m->pos;
    if (whence == SEEK_SET) {
        newpos = (off < 0) ? 0 : (size_t)off;
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
        m->err = true;
    else
        m->pos = newpos;
}
static void mem_read(void *desc, char *data, size_t n) {
    mlx_mem_stream *m = desc;
    size_t avail = m->size - m->pos;
    size_t toread = n <= avail ? n : avail;
    if (toread > 0)
        memcpy(data, m->data + m->pos, toread);
    if (n > toread)
        memset(data + toread, 0, n - toread);
    m->pos += toread;
}
static void mem_read_at_offset(void *desc, char *data, size_t n, size_t off) {
    mlx_mem_stream *m = desc;
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
    m->pos = off;
}
static void mem_write(void *desc, const char *data, size_t n) {
    mlx_mem_stream *m = desc;
    if (n + m->pos > m->size) {
        m->err = true;
        return;
    }
    memcpy(m->data + m->pos, data, n);
    m->pos += n;
}
static const char *mem_label(void *desc) {
    return "<mem>";
}
static void mem_free(void *desc) {
    mlx_mem_stream *m = desc;
    if (!m)
        return;
    if (m->free_data && m->data)
        free(m->data);
    free(m);
}

static mlx_io_vtable mlx_io_vtable_mlx_mem_stream = {
    &mem_is_open,        &mem_good,  &mem_tell,  &mem_seek, &mem_read,
    &mem_read_at_offset, &mem_write, &mem_label, &mem_free
};

int mlx_generate_scales_flat(MLXPyramidsDataset *ds, int channels_last) {
    if (!ds || !ds->options)
        return -1;

    const TrainningOptions *opts = ds->options;
    DatasetScale *arr = NULL;
    int n = 0;
    if (dataset_generate_scales(opts, channels_last, &arr, &n) != 0)
        return -1;

    int *flat = NULL;
    size_t flat_elems = 4 * (size_t)n;
    if (flat_elems > (size_t)INT_MAX) {
        flat = (int *)malloc(sizeof(int) * flat_elems);
        if (!flat) {
            mlx_free_pod((void **)&arr);
            return -1;
        }
    } else {
        if (mlx_alloc_pod((void **)&flat, sizeof(int), (int)flat_elems) != 0) {
            mlx_free_pod((void **)&arr);
            return -1;
        }
    }
    for (int i = 0; i < n; ++i) {
        /* DatasetScale has fields: batch, channels, height, width. We store
         * NHWC as: batch, height, width, channels to match trainer convention. */
        flat[i * 4 + 0] = arr[i].batch;
        flat[i * 4 + 1] = arr[i].height;
        flat[i * 4 + 2] = arr[i].width;
        flat[i * 4 + 3] = arr[i].channels;
    }
    mlx_free_pod((void **)&arr);

    /* store into dataset fields; caller should not pass separate out pointers
     * anymore. Free previous values if present. */
    if (ds->scales)
        mlx_free_int_array(&ds->scales, &ds->n_scales);
    ds->scales = flat;
    ds->n_scales = n;
    return 0;
}

/*
 * Populate dataset samples from an existing function-cache NPZ.
 * Encapsulates the NPZ-reading loop used by the constructor. Returns 0
 * on success (and sets ds->n_samples) or -1 on error.
 */
static int generate_pyramids(MLXPyramidsDataset *ds, const char *npz_path,

                             int sample_count, int channels_last) {
    if (!ds || !npz_path)
        return -1;
    mlx_array *fac_scales = NULL;
    mlx_array *wells_scales = NULL;
    mlx_array *seis_scales = NULL;
    int n_fac_scales = 0;
    int n_wells_scales = 0;
    int n_seis_scales = 0;

    DatasetScale *tmp_scales = NULL;
    int tmp_n = 0;
    if (dataset_generate_scales(ds->options, channels_last, &tmp_scales,
                                &tmp_n) != 0) {
        return -1;
    }
    if (to_facies_pyramids(ds->options, channels_last, tmp_scales, tmp_n,
                           &fac_scales, &n_fac_scales) != 0) {
        mlx_free_pod((void **)&tmp_scales);
        return -1;
    }

    if (to_wells_pyramids(ds->options, channels_last, tmp_scales, tmp_n,
                          &wells_scales, &n_wells_scales) != 0) {
        /* treat as no wells available */
        wells_scales = NULL;
        n_wells_scales = 0;
    }
    if (to_seismic_pyramids(ds->options, channels_last, tmp_scales, tmp_n,
                            &seis_scales, &n_seis_scales) != 0) {
        seis_scales = NULL;
        n_seis_scales = 0;
    }

    mlx_free_pod((void **)&tmp_scales);

    /* Determine number of samples: prefer the provided sample_count but
     * fall back to the first facies scale's leading dim if available. */
    int N = sample_count > 0 ? sample_count : 0;
    if (n_fac_scales > 0 && fac_scales) {
        if (mlx_array_ndim(fac_scales[0]) > 0) {
            const int *sh = mlx_array_shape(fac_scales[0]);
            if (sh && sh[0] > N)
                N = sh[0];
        }
    }

    int found_samples = 0;
    int gen_rc = 0; /* track error for goto-based cleanup */

    /* Populate facies per-sample vectors */
    for (int si = 0; si < N; ++si) {
        mlx_vector_array fac_sample = mlx_vector_array_new();
        for (int sidx = 0; sidx < n_fac_scales; ++sidx) {
            mlx_stream s = mlx_default_gpu_stream_new();
            mlx_array elem = mlx_array_new();
            int appended = 0;
            if (fac_scales && mlx_array_ndim(fac_scales[sidx]) > 0) {
                int fnd = (int)mlx_array_ndim(fac_scales[sidx]);
                const int *fsh = mlx_array_shape(fac_scales[sidx]);
                if (fsh && fsh[0] > si) {
                    int start[4] = {si, 0, 0, 0};
                    int stop[4] = {si + 1, fsh[1], fsh[2], fsh[3]};
                    /* Validate slice bounds before calling into mlx_slice to avoid
                      kernel exceptions when shapes or dims are unexpected. */
                    int valid_slice = 0;
                    const int *fsh = mlx_array_shape(fac_scales[sidx]);
                    if (fsh) {
                        valid_slice =
                            (start[0] >= 0 && stop[0] <= fsh[0] && start[0] < stop[0] &&
                             start[1] >= 0 && stop[1] <= fsh[1] && start[2] >= 0 &&
                             stop[2] <= fsh[2] && start[3] >= 0 && stop[3] <= fsh[3]);
                    }
                    if (valid_slice) {
                        /* Fast path: if the scale array has a single leading sample and we
                           are extracting sample 0, avoid mlx_slice and reshape directly. */
                        if (fsh && fsh[0] == 1 && si == 0) {
                            int new_shape[3] = {fsh[1], fsh[2], fsh[3]};
                            if (mlx_reshape(&elem, fac_scales[sidx], new_shape, 3, s) == 0) {
                                appended = 1;
                            } else {
                                mlx_array_free(elem);
                            }
                        } else if (mlx_slice(&elem, fac_scales[sidx], start, 4, stop, 4,
                                             NULL, 0, s) == 0) {
                            /* reshape (1,H,W,C) -> (H,W,C) */
                            int new_shape[3] = {fsh[1], fsh[2], fsh[3]};
                            mlx_array tmp = mlx_array_new();
                            if (mlx_reshape(&tmp, elem, new_shape, 3, s) == 0) {
                                mlx_array_free(elem);
                                elem = tmp;
                            } else {
                                mlx_array_free(tmp);
                            }
                            appended = 1;
                        } else {
                            mlx_array_free(elem);
                        }
                    }
                }
                if (!appended) {
                    /* fallback zeros with shape matching scales if available */
                    if (fac_scales && mlx_array_ndim(fac_scales[0]) > 0) {
                        const int *fsh0 = mlx_array_shape(fac_scales[0]);
                        int zshape[3] = {fsh0[1], fsh0[2], fsh0[3]};
                        mlx_array_free(elem);
                        mlx_array z = mlx_array_new();
                        if (mlx_zeros(&z, zshape, 3, MLX_FLOAT32, s) == 0)
                            elem = z;
                    }
                }
                if (mlx_array_ndim(elem) > 0) {
                    if (mlx_vector_array_append_value(fac_sample, elem) != 0) {
                        mlx_array_free(elem);
                        mlx_vector_array_free(fac_sample);
                        mlx_stream_free(s);
                        gen_rc = -1;
                        goto gen_cleanup;
                    }
                    mlx_array_free(elem);
                }
                mlx_stream_free(s);
            }

            /* end per-scale loop for facies */
        }

        /* append completed facies sample */
        if (mlx_vector_vector_array_append_value(ds->facies, fac_sample) != 0) {
            mlx_vector_array_free(fac_sample);
            gen_rc = -1;
            goto gen_cleanup;
        }
        mlx_vector_array_free(fac_sample);
        ++found_samples;

        /* Populate wells and masks if available */
        if (n_wells_scales > 0 && wells_scales) {
            mlx_vector_array well_sample = mlx_vector_array_new();
            mlx_vector_array mask_sample = mlx_vector_array_new();
            for (int sidx = 0; sidx < n_wells_scales; ++sidx) {
                mlx_stream s = mlx_default_gpu_stream_new();
                mlx_array elem = mlx_array_new();
                int appended = 0;
                if (wells_scales && mlx_array_ndim(wells_scales[sidx]) > 0) {
                    const int *wsh = mlx_array_shape(wells_scales[sidx]);
                    if (wsh && wsh[0] > si) {
                        int start[4] = {si, 0, 0, 0};
                        int stop[4] = {si + 1, wsh[1], wsh[2], wsh[3]};

                        int valid_wslice = 0;
                        if (wsh) {
                            valid_wslice =
                                (start[0] >= 0 && stop[0] <= wsh[0] && start[0] < stop[0] &&
                                 start[1] >= 0 && stop[1] <= wsh[1] && start[2] >= 0 &&
                                 stop[2] <= wsh[2] && start[3] >= 0 && stop[3] <= wsh[3]);
                        }
                        if (valid_wslice) {
                            if (wsh && wsh[0] == 1 && si == 0) {
                                int new_shape[3] = {wsh[1], wsh[2], wsh[3]};
                                if (mlx_reshape(&elem, wells_scales[sidx], new_shape, 3, s) ==
                                        0) {
                                    appended = 1;
                                } else {
                                    mlx_array_free(elem);
                                }
                            } else if (mlx_slice(&elem, wells_scales[sidx], start, 4, stop, 4,
                                                 NULL, 0, s) == 0) {
                                int new_shape[3] = {wsh[1], wsh[2], wsh[3]};
                                mlx_array tmp = mlx_array_new();
                                if (mlx_reshape(&tmp, elem, new_shape, 3, s) == 0) {
                                    mlx_array_free(elem);
                                    elem = tmp;
                                } else {
                                    mlx_array_free(tmp);
                                }
                                appended = 1;
                            } else {
                                mlx_array_free(elem);
                            }
                        }
                    }
                }
                if (!appended) {
                    if (wells_scales && mlx_array_ndim(wells_scales[0]) > 0) {
                        const int *wsh0 = mlx_array_shape(wells_scales[0]);
                        int zshape[3] = {wsh0[1], wsh0[2], wsh0[3]};
                        mlx_array_free(elem);
                        mlx_array z = mlx_array_new();
                        if (mlx_zeros(&z, zshape, 3, MLX_FLOAT32, s) == 0)
                            elem = z;
                    }
                }

                if (mlx_array_ndim(elem) > 0) {
                    if (mlx_vector_array_append_value(well_sample, elem) != 0) {
                        mlx_array_free(elem);
                        mlx_vector_array_free(well_sample);
                        mlx_vector_array_free(mask_sample);
                        mlx_stream_free(s);
                        gen_rc = -1;
                        goto gen_cleanup;
                    }

                    /* attempt to load an explicit mask from the NPZ; fall back to
                     * computed mask if missing */
                    char mmember[512];
                    snprintf(mmember, sizeof(mmember), "sample_%d/masks_%d.npy", si,
                             sidx);
                    mlx_io_reader mr = {0};
                    int mfound = 0;
                    if (npz_extract_member_to_mlx_reader(npz_path, mmember, &mr) == 0) {
                        mlx_array m = mlx_array_new();
                        mlx_stream load_s = mlx_default_cpu_stream_new();  /* Load needs CPU */
                        if (mlx_load_reader(&m, mr, load_s) == 0) {
                            mlx_stream_free(load_s);
                            mlx_io_reader_free(mr);
                            if (mlx_vector_array_append_value(mask_sample, m) != 0) {
                                mlx_array_free(m);
                                mlx_vector_array_free(well_sample);
                                mlx_vector_array_free(mask_sample);
                                mlx_stream_free(s);
                                gen_rc = -1;
                                goto gen_cleanup;
                            }
                            mlx_array_free(m);
                            mfound = 1;
                        } else {
                            mlx_stream_free(load_s);
                            mlx_io_reader_free(mr);
                        }
                    }

                    if (!mfound) {
                        /* compute mask from absolute values */
                        mlx_array abs_arr = mlx_array_new();
                        if (mlx_abs(&abs_arr, elem, s) == 0) {
                            int abs_ndim = (int)mlx_array_ndim(abs_arr);
                            if (abs_ndim > 0) {
                                int axis = 3;
                                if (axis >= abs_ndim)
                                    axis = abs_ndim - 1;
                                mlx_array sum_arr = mlx_array_new();
                                if (mlx_sum_axis(&sum_arr, abs_arr, axis, 1, s) == 0) {
                                    mlx_array zero = mlx_array_new();
                                    if (mlx_zeros_like(&zero, sum_arr, s) == 0) {
                                        mlx_array mask_arr = mlx_array_new();
                                        if (mlx_greater(&mask_arr, sum_arr, zero, s) == 0) {
                                            if (mlx_vector_array_append_value(mask_sample,
                                                                              mask_arr) != 0) {
                                                mlx_array_free(mask_arr);
                                                mlx_array_free(zero);
                                                mlx_array_free(sum_arr);
                                                mlx_array_free(abs_arr);
                                                mlx_vector_array_free(well_sample);
                                                mlx_vector_array_free(mask_sample);
                                                mlx_stream_free(s);
                                                gen_rc = -1;
                                                goto gen_cleanup;
                                            }
                                        }
                                        mlx_array_free(mask_arr);
                                    }
                                    mlx_array_free(zero);
                                }
                                mlx_array_free(sum_arr);
                            }
                        }
                        mlx_array_free(abs_arr);
                    }

                    mlx_array_free(elem);
                }
                mlx_stream_free(s);
            }

            if (mlx_vector_vector_array_append_value(ds->wells, well_sample) != 0) {
                mlx_vector_array_free(well_sample);
                mlx_vector_array_free(mask_sample);
                gen_rc = -1;
                goto gen_cleanup;
            }
            mlx_vector_array_free(well_sample);

            if (mlx_vector_vector_array_append_value(ds->masks, mask_sample) != 0) {
                mlx_vector_array_free(mask_sample);
                gen_rc = -1;
                goto gen_cleanup;
            }
            mlx_vector_array_free(mask_sample);
        }

        /* Populate seismic if available */
        if (n_seis_scales > 0 && seis_scales) {
            mlx_vector_array seis_sample = mlx_vector_array_new();
            for (int sidx = 0; sidx < n_seis_scales; ++sidx) {
                mlx_stream s = mlx_default_gpu_stream_new();
                mlx_array elem = mlx_array_new();
                int appended = 0;
                if (seis_scales && mlx_array_ndim(seis_scales[sidx]) > 0) {
                    const int *ssh = mlx_array_shape(seis_scales[sidx]);
                    if (ssh && ssh[0] > si) {
                        int start[4] = {si, 0, 0, 0};
                        int stop[4] = {si + 1, ssh[1], ssh[2], ssh[3]};

                        int valid_sslice = 0;
                        if (ssh) {
                            valid_sslice =
                                (start[0] >= 0 && stop[0] <= ssh[0] && start[0] < stop[0] &&
                                 start[1] >= 0 && stop[1] <= ssh[1] && start[2] >= 0 &&
                                 stop[2] <= ssh[2] && start[3] >= 0 && stop[3] <= ssh[3]);
                        }
                        if (valid_sslice) {
                            if (ssh && ssh[0] == 1 && si == 0) {
                                int new_shape[3] = {ssh[1], ssh[2], ssh[3]};
                                if (mlx_reshape(&elem, seis_scales[sidx], new_shape, 3, s) ==
                                        0) {
                                    appended = 1;
                                } else {
                                    mlx_array_free(elem);
                                }
                            } else if (mlx_slice(&elem, seis_scales[sidx], start, 4, stop, 4,
                                                 NULL, 0, s) == 0) {
                                int new_shape[3] = {ssh[1], ssh[2], ssh[3]};
                                mlx_array tmp = mlx_array_new();
                                if (mlx_reshape(&tmp, elem, new_shape, 3, s) == 0) {
                                    mlx_array_free(elem);
                                    elem = tmp;
                                } else {
                                    mlx_array_free(tmp);
                                }
                                appended = 1;
                            } else {
                                mlx_array_free(elem);
                            }
                        }
                    }
                }
                if (!appended) {
                    if (seis_scales && mlx_array_ndim(seis_scales[0]) > 0) {
                        const int *ssh0 = mlx_array_shape(seis_scales[0]);
                        int zshape[3] = {ssh0[1], ssh0[2], ssh0[3]};
                        mlx_array_free(elem);
                        mlx_array z = mlx_array_new();
                        if (mlx_zeros(&z, zshape, 3, MLX_FLOAT32, s) == 0)
                            elem = z;
                    }
                }
                if (mlx_array_ndim(elem) > 0) {
                    if (mlx_vector_array_append_value(seis_sample, elem) != 0) {
                        mlx_array_free(elem);
                        mlx_vector_array_free(seis_sample);
                        mlx_stream_free(s);
                        gen_rc = -1;
                        goto gen_cleanup;
                    }
                    mlx_array_free(elem);
                }
                mlx_stream_free(s);
            }
            if (mlx_vector_vector_array_append_value(ds->seismic, seis_sample) != 0) {
                mlx_vector_array_free(seis_sample);
                gen_rc = -1;
                goto gen_cleanup;
            }
            mlx_vector_array_free(seis_sample);
        }
    }

    ds->n_samples = found_samples;

    /* Respect explicit flags: if wells/seismic are disabled in options,
     * treat their pyramids as empty even if files exist on disk. */
    if (gen_rc == 0) {
        if (ds->options && !ds->options->use_wells) {
            mlx_vector_vector_array_free(ds->wells);
            ds->wells = mlx_vector_vector_array_new();
        }
        if (ds->options && !ds->options->use_seismic) {
            mlx_vector_vector_array_free(ds->seismic);
            ds->seismic = mlx_vector_vector_array_new();
        }
    }

    /* free scale arrays allocated by to_{facies,wells,seismic}_pyramids */
gen_cleanup:
    if (fac_scales)
        mlx_free_mlx_array_vals(&fac_scales, n_fac_scales);
    if (wells_scales)
        mlx_free_mlx_array_vals(&wells_scales, n_wells_scales);
    if (seis_scales)
        mlx_free_mlx_array_vals(&seis_scales, n_seis_scales);

    return gen_rc;
}

/* close generate_pyramids (missing brace fixed) */

/* static helpers are defined below */
/* forward declaration placed before usage */
static int mlx_pyramids_dataset_populate_batches(MLXPyramidsDataset *ds);

int mlx_pyramids_dataset_new(MLXPyramidsDataset **out,
                             const TrainningOptions *options, int shuffle,
                             int regenerate, int channels_last) {
    if (!out || !options)
        return -1;
    *out = NULL;
    char npz_path[PATH_MAX] = {0};
    int sample_count = 0;

    const char *input_path = options->input_path ? options->input_path : ".";
    /* Use a hidden `.cache` directory (literal) as the cache directory. */
    const char *cache_dir = options->output_path ? options->output_path : ".";
    int desired_num =
        options->num_train_pyramids > 0 ? options->num_train_pyramids : 1024;
    int stop_scale = options->stop_scale;
    int crop_size = options->crop_size;
    int num_img_channels = options->num_img_channels;
    int use_wells = options->use_wells ? 1 : 0;
    int use_seismic = options->use_seismic ? 1 : 0;
    int manual_seed = options->manual_seed;

    // if (regenerate) {
    /* remove any existing function cache to force regeneration */
    mlx_pyramids_dataset_clean_cache(cache_dir);
    // }

    MLXPyramidsDataset *ds = (MLXPyramidsDataset *)calloc(1, sizeof(*ds));
    if (!ds) {
        return -1;
    }

    ds->options = options;
    ds->data_dir = input_path;

    ds->facies = mlx_vector_vector_array_new();
    ds->wells = mlx_vector_vector_array_new();
    ds->masks = mlx_vector_vector_array_new();
    ds->seismic = mlx_vector_vector_array_new();

    /* synthesize scales first so we can pass explicit scale descriptors to the
     * function-cache generator (avoids implicit stop_scale/crop_size derivation)
     */
    DatasetScale *tmp_scales = NULL;
    int tmp_n = 0;
    if (dataset_generate_scales(options, channels_last ? 1 : 0, &tmp_scales,
                                &tmp_n) != 0) {
        mlx_pyramids_dataset_free(ds);
        return -1;
    }

    int rc = ensure_function_cache(input_path, cache_dir, desired_num, tmp_scales,
                                   tmp_n, num_img_channels, use_wells,
                                   use_seismic, manual_seed, npz_path,
                                   sizeof(npz_path), &sample_count);
    mlx_free_pod((void **)&tmp_scales);
    /* Debug: report cache results */
    fprintf(stderr,
            "mlx_pyramids_dataset_new: ensure_function_cache -> npz_path=%s "
            "sample_count=%d rc=%d\n",
            npz_path, sample_count, rc);
    if (rc != 0) {
        mlx_pyramids_dataset_free(ds);
        return rc;
    }

    /* synthesize scales using the provided options and channels_last flag */
    mlx_generate_scales_flat(ds, channels_last ? 1 : 0);

    /* Populate dataset samples by reading the function-cache NPZ. The helper
     * will set ds->n_samples and handle optional wells/seismic loading. */
    int rc2 =
        generate_pyramids(ds, npz_path, sample_count, channels_last ? 1 : 0);
    fprintf(stderr,
            "mlx_pyramids_dataset_new: generate_pyramids returned rc2=%d "
            "ds->n_samples=%d facies_count=%zu\n",
            rc2, ds->n_samples, mlx_vector_vector_array_size(ds->facies));
    if (rc2 != 0) {
        mlx_pyramids_dataset_free(ds);
        return -1;
    }
    int found_samples = ds->n_samples;
    /* Safety: clamp found_samples to actual loaded facies count to avoid
     * out-of-bounds access if the NPZ reported more samples than were
     * successfully loaded into `ds->facies`. */
    size_t actual_facies = mlx_vector_vector_array_size(ds->facies);
    if ((size_t)found_samples > actual_facies) {
        found_samples = (int)actual_facies;
    }

    /* initialize batches and populate them safely (avoid whole-vector copies) */
    if (mlx_pyramids_dataset_populate_batches(ds) != 0) {
        mlx_pyramids_dataset_free(ds);
        return -1;
    }

    if (shuffle && ds->n_samples > 1) {
        mlx_pyramids_dataset_shuffle(ds, (unsigned int)manual_seed);
    }

    *out = ds;
    return 0;
}

void mlx_pyramids_dataset_free(MLXPyramidsDataset *ds) {
    if (!ds)
        return;
    mlx_vector_vector_array_free(ds->facies);
    mlx_vector_vector_array_free(ds->wells);
    mlx_vector_vector_array_free(ds->masks);
    mlx_vector_vector_array_free(ds->seismic);
    /* free Batch structs and their owned vector_arrays */
    if (ds->batches) {
        for (int i = 0; i < ds->n_batches; ++i) {
            MLXBatch *b = &ds->batches[i];
            mlx_vector_array_free(b->facies);
            mlx_vector_array_free(b->wells);
            mlx_vector_array_free(b->seismic);
        }
        mlx_free_pod((void **)&ds->batches);
        ds->batches = NULL;
        ds->n_batches = 0;
    }
    if (ds->scales)
        mlx_free_int_array(&ds->scales, &ds->n_scales);
    free(ds);
}

/* -------------------------------------------------------------------------- */
/* Static helpers (used by public API below) */

static void swap_sample(mlx_vector_vector_array *v, int i, int j) {
    size_t n = mlx_vector_vector_array_size(*v);
    if (i < 0 || j < 0 || (size_t)i >= n || (size_t)j >= n)
        return;
    mlx_vector_array *arr = NULL;
    if (n > (size_t)INT_MAX) {
        arr = (mlx_vector_array *)malloc(sizeof(mlx_vector_array) * n);
        if (!arr)
            return;
    } else {
        if (mlx_alloc_pod((void **)&arr, sizeof(mlx_vector_array), (int)n) != 0)
            return;
    }
    for (size_t k = 0; k < n; ++k) {
        mlx_vector_array tmp;
        mlx_vector_vector_array_get(&tmp, *v, k);
        arr[k] = tmp;
    }
    mlx_vector_array tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
    mlx_vector_vector_array tmpvec = mlx_vector_vector_array_new_data(arr, n);
    mlx_vector_vector_array_set(v, tmpvec);
    mlx_vector_vector_array_free(tmpvec);
    if (n > (size_t)INT_MAX)
        free(arr);
    else
        mlx_free_pod((void **)&arr);
}

static void shuffle_indices(int *idx, int n, unsigned int seed) {
    if (seed == 0)
        seed = (unsigned int)time(NULL);
    for (int i = n - 1; i > 0; --i) {
        unsigned int r =
            (unsigned int)((seed * 1664525u + 1013904223u) & 0xffffffffu);
        seed = r;
        int j = r % (i + 1);
        int t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}

int mlx_pyramids_dataset_shuffle(MLXPyramidsDataset *ds, unsigned int seed) {
    if (!ds)
        return -1;
    int n = ds->n_samples;
    if (n <= 1)
        return 0;
    int *idx = NULL;
    if (mlx_alloc_int_array(&idx, n) != 0)
        return -1;
    for (int i = 0; i < n; ++i)
        idx[i] = i;
    shuffle_indices(idx, n, seed);

    mlx_vector_vector_array fac_new = mlx_vector_vector_array_new();
    mlx_vector_vector_array wells_new = mlx_vector_vector_array_new();
    mlx_vector_vector_array masks_new = mlx_vector_vector_array_new();
    mlx_vector_vector_array seis_new = mlx_vector_vector_array_new();
    for (int i = 0; i < n; ++i) {
        mlx_vector_array tmp = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&tmp, ds->facies, idx[i])) {
            mlx_vector_array_free(tmp);
            mlx_vector_vector_array_free(fac_new);
            mlx_free_int_array(&idx, &n);
            return -1;
        }
        if (mlx_vector_vector_array_append_value(fac_new, tmp)) {
            mlx_vector_array_free(tmp);
            mlx_vector_vector_array_free(fac_new);
            mlx_free_int_array(&idx, &n);
            return -1;
        }
        mlx_vector_array_free(tmp);
    }
    if (mlx_vector_vector_array_size(ds->wells) > 0) {
        for (int i = 0; i < n; ++i) {
            mlx_vector_array tmp = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&tmp, ds->wells, idx[i])) {
                mlx_vector_array_free(tmp);
            } else {
                mlx_vector_vector_array_append_value(wells_new, tmp);
                mlx_vector_array_free(tmp);
            }
        }
    }
    if (mlx_vector_vector_array_size(ds->seismic) > 0) {
        for (int i = 0; i < n; ++i) {
            mlx_vector_array tmp = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&tmp, ds->seismic, idx[i])) {
                mlx_vector_array_free(tmp);
            } else {
                mlx_vector_vector_array_append_value(seis_new, tmp);
                mlx_vector_array_free(tmp);
            }
        }
    }
    if (mlx_vector_vector_array_size(ds->masks) > 0) {
        for (int i = 0; i < n; ++i) {
            mlx_vector_array tmp = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&tmp, ds->masks, idx[i])) {
                mlx_vector_array_free(tmp);
            } else {
                mlx_vector_vector_array_append_value(masks_new, tmp);
                mlx_vector_array_free(tmp);
            }
        }
    }

    mlx_vector_vector_array_free(ds->facies);
    ds->facies = fac_new;
    if (mlx_vector_vector_array_size(wells_new) > 0) {
        mlx_vector_vector_array_free(ds->wells);
        ds->wells = wells_new;
    } else {
        mlx_vector_vector_array_free(wells_new);
    }
    if (mlx_vector_vector_array_size(seis_new) > 0) {
        mlx_vector_vector_array_free(ds->seismic);
        ds->seismic = seis_new;
    } else {
        mlx_vector_vector_array_free(seis_new);
    }
    if (mlx_vector_vector_array_size(masks_new) > 0) {
        mlx_vector_vector_array_free(ds->masks);
        ds->masks = masks_new;
    } else {
        mlx_vector_vector_array_free(masks_new);
    }

    mlx_free_int_array(&idx, &n);
    return 0;
}

int mlx_pyramids_dataset_clean_cache(const char *cache_dir) {
    if (!cache_dir)
        return -1;
    DIR *d = opendir(cache_dir);
    if (!d)
        return -1;
    struct dirent *entry;
    int rc = 0;
    while ((entry = readdir(d)) != NULL) {
        if (strncmp(entry->d_name, "func_cache_", 11) == 0) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", cache_dir, entry->d_name);
            if (remove(path) != 0)
                rc = -1;
        }
    }
    closedir(d);
    return rc;
}

int mlx_pyramids_dataset_get_scale_stack(MLXPyramidsDataset *ds, int scale,
        mlx_array *out) {
    if (!ds || !out)
        return -1;

    /* Ensure function cache exists before attempting to load per-scale data.
     * Mirrors Python's @memory.cache behavior used by the torch utils helpers.
     */
    if (ds->options) {
        char cache_npz[PATH_MAX] = {0};
        int actual_samples = 0;
        int desired = ds->options->num_train_pyramids > 0
                      ? ds->options->num_train_pyramids
                      : 1024;
        int use_wells = ds->options->use_wells ? 1 : 0;
        int use_seismic = ds->options->use_seismic ? 1 : 0;
        DatasetScale *tmp_scales = NULL;
        int tmp_n = 0;
        if (dataset_generate_scales(ds->options, 1, &tmp_scales, &tmp_n) != 0) {
            fprintf(stderr, "failed to generate scales for cache lookup\n");
            return -1;
        }
        int rc = ensure_function_cache(
                     ds->options->input_path ? ds->options->input_path : ".",
                     ds->options->output_path ? ds->options->output_path : ".", desired,
                     tmp_scales, tmp_n, ds->options->num_img_channels, use_wells,
                     use_seismic, ds->options->manual_seed, cache_npz, sizeof(cache_npz),
                     &actual_samples);
        mlx_free_pod((void **)&tmp_scales);
        if (rc != 0) {
            fprintf(stderr, "ensure_function_cache failed in get_scale_stack\n");
            return -1;
        }
    }
    int N = ds->n_samples;
    if (N <= 0)
        return -1;

    mlx_vector_array sample0;
    if (mlx_vector_vector_array_get(&sample0, ds->facies, 0) != 0)
        return -1;
    if (scale >= (int)mlx_vector_array_size(sample0)) {
        mlx_vector_array_free(sample0);
        return -1;
    }
    mlx_array first = mlx_array_new();
    if (mlx_vector_array_get(&first, sample0, scale) != 0) {
        mlx_array_free(first);
        mlx_vector_array_free(sample0);
        return -1;
    }
    const int *shape = mlx_array_shape(first);
    int h = shape[0];
    int w = shape[1];
    int c = shape[2];
    mlx_array_free(first);
    mlx_vector_array_free(sample0);

    mlx_vector_array scale_vec = mlx_vector_array_new();
    for (int i = 0; i < N; ++i) {
        mlx_vector_array sample = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample, ds->facies, i)) {
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        mlx_array elem = mlx_array_new();
        if (mlx_vector_array_get(&elem, sample, scale)) {
            mlx_array_free(elem);
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        if (mlx_vector_array_append_value(scale_vec, elem)) {
            mlx_array_free(elem);
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        mlx_array_free(elem);
        mlx_vector_array_free(sample);
    }

    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array stacked = mlx_array_new();
    size_t vec_n = mlx_vector_array_size(scale_vec);
    if (vec_n == 0) {
        mlx_vector_array_free(scale_vec);
        mlx_stream_free(s);
        mlx_array_free(stacked);
        int shape0[4] = {0, h, w, c};
        mlx_stream s2 = mlx_default_gpu_stream_new();
        if (mlx_zeros(&stacked, shape0, 4, MLX_FLOAT32, s2) != 0) {
            mlx_stream_free(s2);
            return -1;
        }
        mlx_stream_free(s2);
    } else {
        int rc = mlx_stack(&stacked, scale_vec, s);
        mlx_vector_array_free(scale_vec);
        mlx_stream_free(s);
        if (rc != 0) {
            mlx_array_free(stacked);
            return -1;
        }
    }
    *out = stacked;
    return 0;
}

static int build_stack_from_source(mlx_vector_vector_array src,
                                   MLXPyramidsDataset *ds, int scale,
                                   mlx_array *out) {
    if (!ds || !out)
        return -1;

    /* Ensure function cache exists for the dataset options so sources can be
     * loaded/generated consistently. This mirrors Python cached helpers. */
    if (ds->options) {
        char cache_npz[PATH_MAX] = {0};
        int actual_samples = 0;
        int desired = ds->options->num_train_pyramids > 0
                      ? ds->options->num_train_pyramids
                      : 1024;
        int use_wells = ds->options->use_wells ? 1 : 0;
        int use_seismic = ds->options->use_seismic ? 1 : 0;
        DatasetScale *tmp_scales = NULL;
        int tmp_n = 0;
        if (dataset_generate_scales(ds->options, 1, &tmp_scales, &tmp_n) != 0) {
            fprintf(stderr, "failed to generate scales for cache lookup\n");
            return -1;
        }
        int rc = ensure_function_cache(
                     ds->options->input_path ? ds->options->input_path : ".",
                     ds->options->output_path ? ds->options->output_path : ".", desired,
                     tmp_scales, tmp_n, ds->options->num_img_channels, use_wells,
                     use_seismic, ds->options->manual_seed, cache_npz, sizeof(cache_npz),
                     &actual_samples);
        mlx_free_pod((void **)&tmp_scales);
        if (rc != 0) {
            fprintf(stderr,
                    "ensure_function_cache failed in build_stack_from_source\n");
            return -1;
        }
    }
    int N = ds->n_samples;
    if (mlx_vector_vector_array_size(src) == 0) {
        mlx_vector_array sample0;
        if (mlx_vector_vector_array_get(&sample0, ds->facies, 0) != 0)
            return -1;
        mlx_array first = mlx_array_new();
        if (mlx_vector_array_get(&first, sample0, scale) != 0) {
            mlx_array_free(first);
            mlx_vector_array_free(sample0);
            return -1;
        }
        const int *shape = mlx_array_shape(first);
        int h = shape[0];
        int w = shape[1];
        int c = shape[2];
        int zeros_shape[4] = {0, h, w, c};
        int dtype = (int)mlx_array_dtype(first);
        mlx_array_free(first);
        mlx_vector_array_free(sample0);
        mlx_stream s = mlx_default_gpu_stream_new();
        mlx_array z = mlx_array_new();
        int rc = mlx_zeros(&z, zeros_shape, 4, dtype, s);
        mlx_stream_free(s);
        if (rc != 0) {
            mlx_array_free(z);
            return -1;
        }
        *out = z;
        return 0;
    }

    mlx_vector_array scale_vec = mlx_vector_array_new();
    for (int i = 0; i < N; ++i) {
        mlx_vector_array sample = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample, src, i)) {
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        mlx_array elem = mlx_array_new();
        if (mlx_vector_array_get(&elem, sample, scale)) {
            mlx_array_free(elem);
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        if (mlx_vector_array_append_value(scale_vec, elem)) {
            mlx_array_free(elem);
            mlx_vector_array_free(scale_vec);
            mlx_vector_array_free(sample);
            return -1;
        }
        mlx_array_free(elem);
        mlx_vector_array_free(sample);
    }

    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_array stacked = mlx_array_new();
    size_t vec_n = mlx_vector_array_size(scale_vec);
    if (vec_n == 0) {
        /* derive fallback shape from ds->facies first sample */
        mlx_vector_array_free(scale_vec);
        mlx_stream_free(s);
        mlx_array_free(stacked);
        mlx_vector_array sample0;
        if (mlx_vector_vector_array_get(&sample0, ds->facies, 0) != 0)
            return -1;
        mlx_array first = mlx_array_new();
        if (mlx_vector_array_get(&first, sample0, scale) != 0) {
            mlx_array_free(first);
            mlx_vector_array_free(sample0);
            return -1;
        }
        const int *shape = mlx_array_shape(first);
        int h = shape[0];
        int w = shape[1];
        int c = shape[2];
        int zeros_shape[4] = {0, h, w, c};
        int dtype = (int)mlx_array_dtype(first);
        mlx_array_free(first);
        mlx_vector_array_free(sample0);
        mlx_array z = mlx_array_new();
        mlx_stream s2 = mlx_default_gpu_stream_new();
        if (mlx_zeros(&z, zeros_shape, 4, dtype, s2) != 0) {
            mlx_stream_free(s2);
            return -1;
        }
        mlx_stream_free(s2);
        *out = z;
        return 0;
    } else {
        int rc = mlx_stack(&stacked, scale_vec, s);
        mlx_vector_array_free(scale_vec);
        mlx_stream_free(s);
        if (rc != 0) {
            mlx_array_free(stacked);
            return -1;
        }
    }
    *out = stacked;
    return 0;
}

/* (static helpers were moved earlier in the file for clarity) */

int mlx_pyramids_dataset_get_wells_stack(MLXPyramidsDataset *ds, int scale,
        mlx_array *out) {
    return build_stack_from_source(ds->wells, ds, scale, out);
}

int mlx_pyramids_dataset_get_masks_stack(MLXPyramidsDataset *ds, int scale,
        mlx_array *out) {
    return build_stack_from_source(ds->masks, ds, scale, out);
}

int mlx_pyramids_dataset_get_seismic_stack(MLXPyramidsDataset *ds, int scale,
        mlx_array *out) {
    return build_stack_from_source(ds->seismic, ds, scale, out);
}

int mlx_pyramids_dataset_generate_pyramids(MLXPyramidsDataset *ds,
        mlx_array **out_facies,
        mlx_array **out_wells,
        mlx_array **out_seismic) {
    if (!ds || !out_facies || !out_wells || !out_seismic)
        return -1;

    /* Ensure on-disk function cache exists for the dataset options so the
     * generators (interpolators) have their cached inputs available. This
     * mirrors the Python `@memory.cache` behavior by forcing generation of
     * the underlying function cache files before stacking per-scale tensors.
     */
    if (ds->options) {
        char cache_npz[PATH_MAX] = {0};
        int actual_samples = 0;
        int desired = ds->options->num_train_pyramids > 0
                      ? ds->options->num_train_pyramids
                      : 1024;
        int use_wells = ds->options->use_wells ? 1 : 0;
        int use_seismic = ds->options->use_seismic ? 1 : 0;
        DatasetScale *tmp_scales = NULL;
        int tmp_n = 0;
        if (dataset_generate_scales(ds->options, 1, &tmp_scales, &tmp_n) != 0) {
            fprintf(stderr, "failed to generate scales for cache lookup\n");
            return -1;
        }
        int rc = ensure_function_cache(
                     ds->options->input_path ? ds->options->input_path : ".",
                     ds->options->output_path ? ds->options->output_path : ".", desired,
                     tmp_scales, tmp_n, ds->options->num_img_channels, use_wells,
                     use_seismic, ds->options->manual_seed, cache_npz, sizeof(cache_npz),
                     &actual_samples);
        mlx_free_pod((void **)&tmp_scales);
        if (rc != 0) {
            fprintf(stderr, "ensure_function_cache failed in generate_pyramids\n");
            return -1;
        }
    }

    int n = ds->n_scales;
    if (n <= 0) {
        *out_facies = NULL;
        *out_wells = NULL;
        *out_seismic = NULL;
        return 0;
    }

    mlx_array *fac = NULL;
    mlx_array *wells = NULL;
    mlx_array *seis = NULL;
    if (mlx_alloc_mlx_array_vals(&fac, n) != 0 ||
            mlx_alloc_mlx_array_vals(&wells, n) != 0 ||
            mlx_alloc_mlx_array_vals(&seis, n) != 0) {
        if (fac)
            mlx_free_mlx_array_vals(&fac, n);
        if (wells)
            mlx_free_mlx_array_vals(&wells, n);
        if (seis)
            mlx_free_mlx_array_vals(&seis, n);
        return -1;
    }

    for (int i = 0; i < n; ++i) {
        if (mlx_pyramids_dataset_get_scale_stack(ds, i, &fac[i]) != 0) {
            fac[i] = mlx_array_new();
        }
        if (mlx_pyramids_dataset_get_wells_stack(ds, i, &wells[i]) != 0) {
            wells[i] = mlx_array_new();
        }
        if (mlx_pyramids_dataset_get_seismic_stack(ds, i, &seis[i]) != 0) {
            seis[i] = mlx_array_new();
        }
    }

    *out_facies = fac;
    *out_wells = wells;
    *out_seismic = seis;
    return 0;
}

int mlx_pyramids_dataset_get_batch(MLXPyramidsDataset *ds, int index,
                                   mlx_vector_array *out_facies,
                                   mlx_vector_array *out_wells,
                                   mlx_vector_array *out_seismic) {
    if (!ds || !out_facies || !out_wells || !out_seismic)
        return -1;
    int n = ds->n_samples;
    if (index < 0 || index >= n)
        return -1;

    /* facies is required */
    if (mlx_vector_vector_array_get(out_facies, ds->facies, index) != 0)
        return -1;

    /* wells/seismic may be empty; return an empty vector_array in that case */
    if (mlx_vector_vector_array_size(ds->wells) > 0) {
        if (mlx_vector_vector_array_get(out_wells, ds->wells, index) != 0) {
            mlx_vector_array_free(*out_facies);
            return -1;
        }
    } else {
        *out_wells = mlx_vector_array_new();
    }

    if (mlx_vector_vector_array_size(ds->seismic) > 0) {
        if (mlx_vector_vector_array_get(out_seismic, ds->seismic, index) != 0) {
            mlx_vector_array_free(*out_facies);
            mlx_vector_array_free(*out_wells);
            return -1;
        }
    } else {
        *out_seismic = mlx_vector_array_new();
    }

    return 0;
}

int mlx_pyramids_dataset_get_batch_struct(MLXPyramidsDataset *ds, int index,
        MLXBatch *out) {
    if (!ds || !out)
        return -1;
    if (ds->batches) {
        if (index < 0 || index >= ds->n_batches)
            return -1;

        MLXBatch *src = &ds->batches[index];

        /* initialize destination arrays */
        out->facies = mlx_vector_array_new();
        out->wells = mlx_vector_array_new();
        out->seismic = mlx_vector_array_new();

        /* copy facies */
        size_t nf = mlx_vector_array_size(src->facies);
        for (size_t i = 0; i < nf; ++i) {
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, src->facies, (int)i) == 0) {
                if (mlx_vector_array_append_value(out->facies, elem) != 0) {
                    mlx_array_free(elem);
                    goto err;
                }
                mlx_array_free(elem);
            } else {
                mlx_array_free(elem);
            }
        }

        /* copy wells */
        size_t nw = mlx_vector_array_size(src->wells);
        for (size_t i = 0; i < nw; ++i) {
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, src->wells, (int)i) == 0) {
                if (mlx_vector_array_append_value(out->wells, elem) != 0) {
                    mlx_array_free(elem);
                    goto err;
                }
                mlx_array_free(elem);
            } else {
                mlx_array_free(elem);
            }
        }

        /* copy seismic */
        size_t ns = mlx_vector_array_size(src->seismic);
        for (size_t i = 0; i < ns; ++i) {
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, src->seismic, (int)i) == 0) {
                if (mlx_vector_array_append_value(out->seismic, elem) != 0) {
                    mlx_array_free(elem);
                    goto err;
                }
                mlx_array_free(elem);
            } else {
                mlx_array_free(elem);
            }
        }

        return 0;

err:
        mlx_vector_array_free(out->facies);
        mlx_vector_array_free(out->wells);
        mlx_vector_array_free(out->seismic);
        return -1;
    }

    /* If batches were not pre-populated (to avoid eager deep copies that
     * caused shared_ptr crashes), return the per-sample vectors on demand. */
    if (index < 0 || index >= ds->n_samples)
        return -1;
    return mlx_pyramids_dataset_get_batch(ds, index, &out->facies, &out->wells,
                                          &out->seismic);
}

/* serialize mlx_array into in-memory .npy bytes (caller frees *out_buf) */
static int _serialize_array_to_npy_bytes_local(const mlx_array arr,
        void **out_buf,
        size_t *out_size) {
    if (!out_buf || !out_size)
        return -1;

    /* Use CPU stream for I/O - mlx_save_writer doesn't support GPU eval */
    mlx_stream cpu_s = mlx_default_cpu_stream_new();

    /* Synchronize to ensure any pending GPU ops are complete */
    mlx_synchronize(cpu_s);

    size_t nbytes = mlx_array_nbytes(arr);
    size_t bufsize = nbytes + 4096;
    void *data = malloc(bufsize);
    if (!data) {
        mlx_stream_free(cpu_s);
        return -1;
    }
    mlx_mem_stream *m = NULL;
    if (mlx_alloc_pod((void **)&m, sizeof(mlx_mem_stream), 1) != 0) {
        free(data);
        mlx_stream_free(cpu_s);
        return -1;
    }
    m->data = (char *)data;
    m->pos = 0;
    m->size = bufsize;
    m->err = false;
    m->free_data = true;

    mlx_io_writer writer = mlx_io_writer_new(m, mlx_io_vtable_mlx_mem_stream);
    int save_rc = mlx_save_writer(writer, arr);
    if (save_rc != 0) {
        mlx_io_writer_free(writer);
        mlx_stream_free(cpu_s);
        return -1;
    }
    size_t used = m->pos;
    void *buf = malloc(used);
    if (!buf) {
        mlx_io_writer_free(writer);
        mlx_stream_free(cpu_s);
        return -1;
    }
    memcpy(buf, m->data, used);
    mlx_io_writer_free(writer);
    mlx_stream_free(cpu_s);
    *out_buf = buf;
    *out_size = used;
    return 0;
}

int mlx_pyramids_dataset_dump_batches_npz(MLXPyramidsDataset *ds,
        const char *npz_path) {
    if (!ds || !npz_path)
        return -1;

    int n_samples = ds->n_samples;
    if (n_samples <= 0)
        return -1;

    const char **names = NULL;
    const void **bufs = NULL;
    size_t *sizes = NULL;
    int n_members = 0;

    for (int si = 0; si < n_samples; ++si) {
        /* facies */
        mlx_vector_array fac = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&fac, ds->facies, si) == 0) {
            size_t nf = mlx_vector_array_size(fac);
            for (size_t s = 0; s < nf; ++s) {
                /* write levels from smallest->largest by mapping output index `s`
                 * to source index `src = nf-1-s` because internal storage is
                 * large->small. Member names remain facies_0..facies_{nf-1}. */
                size_t src = (nf > 0) ? (nf - 1 - s) : 0;
                mlx_array a = mlx_array_new();
                if (mlx_vector_array_get(&a, fac, (int)src) == 0) {
                    void *buf = NULL;
                    size_t sz = 0;
                    if (_serialize_array_to_npy_bytes_local(a, &buf, &sz) == 0) {
                        char *nm = (char *)malloc(64 + 32);
                        snprintf(nm, 64 + 32, "sample_%d/facies_%zu.npy", si, s);
                        const char **tmpn =
                            realloc((void *)names, sizeof(char *) * (n_members + 1));
                        const void **tmpb =
                            realloc((void *)bufs, sizeof(void *) * (n_members + 1));
                        size_t *tmps = realloc(sizes, sizeof(size_t) * (n_members + 1));
                        if (!tmpn || !tmpb || !tmps) {
                            free(nm);
                            mlx_free_pod((void **)&buf);
                            free(names);
                            free((void *)bufs);
                            free(sizes);
                            mlx_array_free(a);
                            mlx_vector_array_free(fac);
                            return -1;
                        }
                        names = tmpn;
                        bufs = tmpb;
                        sizes = tmps;
                        names[n_members] = nm;
                        bufs[n_members] = buf;
                        sizes[n_members] = sz;
                        n_members++;
                    }
                }
                mlx_array_free(a);
            }
            mlx_vector_array_free(fac);
        }

        /* wells */
        if (mlx_vector_vector_array_size(ds->wells) > 0) {
            mlx_vector_array wv = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&wv, ds->wells, si) == 0) {
                size_t nw = mlx_vector_array_size(wv);
                for (size_t s = 0; s < nw; ++s) {
                    /* map output index to source index to emit smallest->largest */
                    size_t src = (nw > 0) ? (nw - 1 - s) : 0;
                    mlx_array a = mlx_array_new();
                    if (mlx_vector_array_get(&a, wv, (int)src) == 0) {
                        void *buf = NULL;
                        size_t sz = 0;
                        if (_serialize_array_to_npy_bytes_local(a, &buf, &sz) == 0) {
                            char *nm = (char *)malloc(64 + 32);
                            snprintf(nm, 64 + 32, "sample_%d/wells_%zu.npy", si, s);
                            const char **tmpn =
                                realloc((void *)names, sizeof(char *) * (n_members + 1));
                            const void **tmpb =
                                realloc((void *)bufs, sizeof(void *) * (n_members + 1));
                            size_t *tmps = realloc(sizes, sizeof(size_t) * (n_members + 1));
                            if (!tmpn || !tmpb || !tmps) {
                                free(nm);
                                mlx_free_pod((void **)&buf);
                                free(names);
                                free((void *)bufs);
                                free(sizes);
                                mlx_array_free(a);
                                mlx_vector_array_free(wv);
                                return -1;
                            }
                            names = tmpn;
                            bufs = tmpb;
                            sizes = tmps;
                            names[n_members] = nm;
                            bufs[n_members] = buf;
                            sizes[n_members] = sz;
                            n_members++;
                        }
                    }
                    mlx_array_free(a);
                }
                mlx_vector_array_free(wv);
            }
        }

        /* seismic */
        if (mlx_vector_vector_array_size(ds->seismic) > 0) {
            mlx_vector_array sv = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sv, ds->seismic, si) == 0) {
                size_t ns = mlx_vector_array_size(sv);
                for (size_t s = 0; s < ns; ++s) {
                    /* map output index to source index to emit smallest->largest */
                    size_t src = (ns > 0) ? (ns - 1 - s) : 0;
                    mlx_array a = mlx_array_new();
                    if (mlx_vector_array_get(&a, sv, (int)src) == 0) {
                        void *buf = NULL;
                        size_t sz = 0;
                        if (_serialize_array_to_npy_bytes_local(a, &buf, &sz) == 0) {
                            char *nm = (char *)malloc(64 + 32);
                            snprintf(nm, 64 + 32, "sample_%d/seismic_%zu.npy", si, s);
                            const char **tmpn =
                                realloc((void *)names, sizeof(char *) * (n_members + 1));
                            const void **tmpb =
                                realloc((void *)bufs, sizeof(void *) * (n_members + 1));
                            size_t *tmps = realloc(sizes, sizeof(size_t) * (n_members + 1));
                            if (!tmpn || !tmpb || !tmps) {
                                free(nm);
                                mlx_free_pod((void **)&buf);
                                free(names);
                                free((void *)bufs);
                                free(sizes);
                                mlx_array_free(a);
                                mlx_vector_array_free(sv);
                                return -1;
                            }
                            names = tmpn;
                            bufs = tmpb;
                            sizes = tmps;
                            names[n_members] = nm;
                            bufs[n_members] = buf;
                            sizes[n_members] = sz;
                            n_members++;
                        }
                    }
                    mlx_array_free(a);
                }
                mlx_vector_array_free(sv);
            }
        }
    }

    if (n_members == 0) {
        /* nothing to write */
        return -1;
    }

    int rc = npz_create_from_memory(npz_path, names, bufs, sizes, n_members);

    /* cleanup allocated names and buffers */
    for (int i = 0; i < n_members; ++i) {
        if (names[i])
            free((void *)names[i]);
        if (bufs[i])
            free((void *)bufs[i]);
    }
    free(names);
    free((void *)bufs);
    free(sizes);
    return rc;
}

/* forward declaration (removed above) */

/* Safely populate `ds->batches` by creating new per-sample `MLXBatch`
 * objects and appending each element one-by-one into fresh `mlx_vector_array`
 * instances. This avoids copying whole C++ vectors of arrays (which previously
 * triggered shared_ptr/ArrayDesc crashes) by ensuring each appended `mlx_array`
 * is host-materialized with `mlx_array_eval()` before being copied into the
 * destination vectors.
 */
static int mlx_pyramids_dataset_populate_batches(MLXPyramidsDataset *ds) {
    if (!ds)
        return -1;
    int n = ds->n_samples;
    if (n <= 0)
        return 0;

    MLXBatch *batches = (MLXBatch *)calloc((size_t)n, sizeof(MLXBatch));
    if (!batches)
        return -1;

    for (int i = 0; i < n; ++i) {
        /* Initialize empty vectors for this batch */
        batches[i].facies = mlx_vector_array_new();
        batches[i].wells = mlx_vector_array_new();
        batches[i].seismic = mlx_vector_array_new();

        /* copy facies sample elements one-by-one */
        mlx_vector_array sample = mlx_vector_array_new();
        if (mlx_vector_vector_array_get(&sample, ds->facies, i) != 0) {
            mlx_vector_array_free(sample);
            goto err;
        }
        size_t nf = mlx_vector_array_size(sample);
        for (size_t j = 0; j < nf; ++j) {
            mlx_array elem = mlx_array_new();
            if (mlx_vector_array_get(&elem, sample, (int)j) != 0) {
                mlx_array_free(elem);
                goto err;
            }
            /* materialize host memory to avoid device-backed ephemeral refs */
            if (mlx_vector_array_append_value(batches[i].facies, elem) != 0) {
                mlx_array_free(elem);
                goto err;
            }
            mlx_array_free(elem);
        }
        mlx_vector_array_free(sample);

        /* wells: copy if available, otherwise keep empty vector */
        if (mlx_vector_vector_array_size(ds->wells) > 0) {
            mlx_vector_array sample_w = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_w, ds->wells, i) != 0) {
                mlx_vector_array_free(sample_w);
                goto err;
            }
            size_t nw = mlx_vector_array_size(sample_w);
            for (size_t j = 0; j < nw; ++j) {
                mlx_array elem = mlx_array_new();
                if (mlx_vector_array_get(&elem, sample_w, (int)j) != 0) {
                    mlx_array_free(elem);
                    goto err;
                }

                if (mlx_vector_array_append_value(batches[i].wells, elem) != 0) {
                    mlx_array_free(elem);
                    goto err;
                }
                mlx_array_free(elem);
            }
            mlx_vector_array_free(sample_w);
        }

        /* seismic: copy if available, otherwise keep empty vector */
        if (mlx_vector_vector_array_size(ds->seismic) > 0) {
            mlx_vector_array sample_s = mlx_vector_array_new();
            if (mlx_vector_vector_array_get(&sample_s, ds->seismic, i) != 0) {
                mlx_vector_array_free(sample_s);
                goto err;
            }
            size_t ns = mlx_vector_array_size(sample_s);
            for (size_t j = 0; j < ns; ++j) {
                mlx_array elem = mlx_array_new();
                if (mlx_vector_array_get(&elem, sample_s, (int)j) != 0) {
                    mlx_array_free(elem);
                    goto err;
                }

                if (mlx_vector_array_append_value(batches[i].seismic, elem) != 0) {
                    mlx_array_free(elem);
                    goto err;
                }
                mlx_array_free(elem);
            }
            mlx_vector_array_free(sample_s);
        }
    }

    ds->batches = batches;
    ds->n_batches = n;
    return 0;

err:
    /* cleanup partial state */
    for (int k = 0; k < n; ++k) {
        mlx_vector_array_free(batches[k].facies);
        mlx_vector_array_free(batches[k].wells);
        mlx_vector_array_free(batches[k].seismic);
    }
    free(batches);
    ds->batches = NULL;
    ds->n_batches = 0;
    return -1;
}
