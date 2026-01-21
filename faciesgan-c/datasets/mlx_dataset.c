#include "mlx_dataset.h"
#include "func_cache.h"
#include "io/npz_unzip.h"
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/io.h>
#include <mlx/c/ops.h>
#include <mlx/c/stream.h>
#include <mlx/c/vector.h>
#include <stdio.h>
#include <string.h>

int mlx_pyramids_dataset_load(
    const char *input_path, const char *cache_dir, int desired_num,
    int stop_scale, int crop_size, int num_img_channels, int use_wells,
    int use_seismic, int manual_seed, mlx_vector_vector_array *out_facies,
    mlx_vector_vector_array *out_wells, mlx_vector_vector_array *out_masks,
    mlx_vector_vector_array *out_seismic, int *out_num_samples) {
  if (!input_path || !cache_dir || !out_facies || !out_num_samples)
    return -1;

  char cache_npz[PATH_MAX] = {0};
  int actual_samples = 0;
  if (ensure_function_cache(input_path, cache_dir, desired_num, stop_scale,
                            crop_size, num_img_channels, use_wells, use_seismic,
                            manual_seed, cache_npz, sizeof(cache_npz),
                            &actual_samples) != 0) {
    fprintf(stderr, "ensure_function_cache failed for %s\n", input_path);
    return -1;
  }

  int num_samples = actual_samples > 0 ? actual_samples : desired_num;
  mlx_stream s = mlx_default_cpu_stream_new();

  /* initialize output vectors */
  *out_facies = mlx_vector_vector_array_new();
  if (out_wells)
    *out_wells = mlx_vector_vector_array_new();
  if (out_masks)
    *out_masks = mlx_vector_vector_array_new();
  if (out_seismic)
    *out_seismic = mlx_vector_vector_array_new();

  for (int si = 0; si < num_samples; ++si) {
    mlx_vector_array fac_sample = mlx_vector_array_new();
    mlx_vector_array well_sample = mlx_vector_array_new();
    mlx_vector_array mask_sample = mlx_vector_array_new();
    mlx_vector_array seis_sample = mlx_vector_array_new();

    for (int sc = 0; sc < stop_scale + 1; ++sc) {
      char member[64];
      snprintf(member, sizeof(member), "sample_%d/facies_%d.npy", si, sc);
      mlx_io_reader reader;
      if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0) {
        mlx_array a = mlx_array_new();
        if (mlx_load_reader(&a, reader, s) != 0) {
          int shape[3] = {crop_size, crop_size, num_img_channels};
          mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
        }
        mlx_io_reader_free(reader);
        mlx_vector_array_append_value(fac_sample, a);
        mlx_array_free(a);
      } else {
        int shape[3] = {crop_size, crop_size, num_img_channels};
        mlx_array a = mlx_array_new();
        mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
        mlx_vector_array_append_value(fac_sample, a);
        mlx_array_free(a);
      }

      /* wells/seismic: present in archive if use_wells/use_seismic true */
      if (out_wells) {
        snprintf(member, sizeof(member), "sample_%d/wells_%d.npy", si, sc);
        if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0) {
          mlx_array a = mlx_array_new();
          if (mlx_load_reader(&a, reader, s) != 0) {
            int shape[3] = {crop_size, crop_size, 1};
            mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
          }
          mlx_io_reader_free(reader);
          /* append well array */
          mlx_vector_array_append_value(well_sample, a);
          /* compute mask for this scale if caller requested masks */
          if (out_masks) {
            /* try to load explicit mask member first */
            char mmember[64];
            mlx_io_reader mreader;
            int mfound = 0;
            snprintf(mmember, sizeof(mmember), "sample_%d/masks_%d.npy", si,
                     sc);
            if (npz_extract_member_to_mlx_reader(cache_npz, mmember,
                                                 &mreader) == 0) {
              mlx_array m = mlx_array_new();
              if (mlx_load_reader(&m, mreader, s) != 0) {
                mlx_array_free(m);
                mlx_io_reader_free(mreader);
                /* fallback: compute from wells below */
              } else {
                mlx_io_reader_free(mreader);
                mlx_vector_array_append_value(mask_sample, m);
                mlx_array_free(m);
                mfound = 1;
              }
            }

            if (!mfound) {
              /* compute mask = greater(sum(abs(a), axis=3, keepdims=true), 0)
               */
              mlx_array abs_arr = mlx_array_new();
              if (mlx_abs(&abs_arr, a, s) == 0) {
                mlx_array sum_arr = mlx_array_new();
                if (mlx_sum_axis(&sum_arr, abs_arr, 3, 1, s) == 0) {
                  mlx_array zero = mlx_array_new();
                  if (mlx_zeros_like(&zero, sum_arr, s) == 0) {
                    mlx_array mask_arr = mlx_array_new();
                    if (mlx_greater(&mask_arr, sum_arr, zero, s) == 0) {
                      mlx_vector_array_append_value(mask_sample, mask_arr);
                    }
                    mlx_array_free(mask_arr);
                  }
                  mlx_array_free(zero);
                }
                mlx_array_free(sum_arr);
              }
              mlx_array_free(abs_arr);
            }
          }
          mlx_array_free(a);
        } else {
          int shape[3] = {crop_size, crop_size, 1};
          mlx_array a = mlx_array_new();
          mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
          mlx_vector_array_append_value(well_sample, a);
          if (out_masks) {
            /* compute mask from zeroed well -> zero mask */
            mlx_array zero = mlx_array_new();
            if (mlx_zeros_like(&zero, a, s) == 0) {
              mlx_array mask_arr = mlx_array_new();
              if (mlx_greater(&mask_arr, zero, zero, s) == 0) {
                mlx_vector_array_append_value(mask_sample, mask_arr);
              }
              mlx_array_free(mask_arr);
            }
            mlx_array_free(zero);
          }
          mlx_array_free(a);
        }
      }

      if (out_seismic) {
        snprintf(member, sizeof(member), "sample_%d/seismic_%d.npy", si, sc);
        if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0) {
          mlx_array a = mlx_array_new();
          if (mlx_load_reader(&a, reader, s) != 0) {
            int shape[3] = {crop_size, crop_size, 1};
            mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
          }
          mlx_io_reader_free(reader);
          mlx_vector_array_append_value(seis_sample, a);
          mlx_array_free(a);
        } else {
          int shape[3] = {crop_size, crop_size, 1};
          mlx_array a = mlx_array_new();
          mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
          mlx_vector_array_append_value(seis_sample, a);
          mlx_array_free(a);
        }
      }
    }

    mlx_vector_vector_array_append_value(*out_facies, fac_sample);
    mlx_vector_array_free(fac_sample);
    if (out_wells) {
      mlx_vector_vector_array_append_value(*out_wells, well_sample);
      mlx_vector_array_free(well_sample);
    }
    if (out_masks) {
      mlx_vector_vector_array_append_value(*out_masks, mask_sample);
      mlx_vector_array_free(mask_sample);
    } else {
      mlx_vector_array_free(mask_sample);
    }
    if (out_seismic) {
      mlx_vector_vector_array_append_value(*out_seismic, seis_sample);
      mlx_vector_array_free(seis_sample);
    }
  }

  mlx_stream_free(s);
  *out_num_samples = num_samples;
  return 0;
}

/* -------------------------------------------------------------------------- */
/* MLXPyramidsDataset implementation (merged from mlx_pyramids_dataset.c) */

#include <dirent.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

struct MLXPyramidsDataset {
  mlx_vector_vector_array facies; /* per-sample per-scale mlx_array */
  mlx_vector_vector_array wells;
  mlx_vector_vector_array masks;
  mlx_vector_vector_array seismic;
  int n_samples;
};

static void swap_sample(mlx_vector_vector_array *v, int i, int j) {
  size_t n = mlx_vector_vector_array_size(*v);
  if (i < 0 || j < 0 || (size_t)i >= n || (size_t)j >= n)
    return;
  mlx_vector_array *arr =
      (mlx_vector_array *)malloc(sizeof(mlx_vector_array) * n);
  if (!arr)
    return;
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
  free(arr);
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

int mlx_pyramids_dataset_new(MLXPyramidsDataset **out, const char *input_path,
                             const char *cache_dir, int desired_num,
                             int stop_scale, int crop_size,
                             int num_img_channels, int use_wells,
                             int use_seismic, int manual_seed, int shuffle) {
  if (!out)
    return -1;
  *out = NULL;
  char npz_path[PATH_MAX] = {0};
  int sample_count = 0;
  int rc = ensure_function_cache(input_path, cache_dir, desired_num, stop_scale,
                                 crop_size, num_img_channels, use_wells,
                                 use_seismic, manual_seed, npz_path,
                                 sizeof(npz_path), &sample_count);
  if (rc != 0) {
    return rc;
  }

  MLXPyramidsDataset *ds = (MLXPyramidsDataset *)calloc(1, sizeof(*ds));
  if (!ds) {
    return -1;
  }

  ds->facies = mlx_vector_vector_array_new();
  ds->wells = mlx_vector_vector_array_new();
  ds->masks = mlx_vector_vector_array_new();
  ds->seismic = mlx_vector_vector_array_new();

  int found_samples = 0;
  for (int si = 0; si < sample_count; ++si) {
    mlx_vector_array fac_sample = mlx_vector_array_new();
    int scale = 0;
    while (1) {
      char member[512];
      mlx_io_reader r = {0};
      int found = 0;

      snprintf(member, sizeof(member), "sample_%d/facies_%d.npy", si, scale);
      if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
        found = 1;
      else {
        snprintf(member, sizeof(member), "facies_%d.npy", scale);
        if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
          found = 1;
        else {
          snprintf(member, sizeof(member), "facies_%d_scale_%d.npy", si, scale);
          if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
            found = 1;
        }
      }

      if (!found) {
        if (scale == 0) {
          mlx_vector_array_free(fac_sample);
          goto sample_end;
        }
        break;
      }

      mlx_stream s = mlx_default_cpu_stream_new();
      mlx_array arr = mlx_array_new();
      if (mlx_load_reader(&arr, r, s) != 0) {
        mlx_io_reader_free(r);
        mlx_vector_array_free(fac_sample);
        mlx_pyramids_dataset_free(ds);
        mlx_stream_free(s);
        return -1;
      }
      mlx_io_reader_free(r);
      if (mlx_vector_array_append_value(fac_sample, arr) != 0) {
        mlx_array_free(arr);
        mlx_vector_array_free(fac_sample);
        mlx_pyramids_dataset_free(ds);
        mlx_stream_free(s);
        return -1;
      }
      mlx_array_free(arr);
      mlx_stream_free(s);
      ++scale;
    }
    if (mlx_vector_vector_array_append_value(ds->facies, fac_sample) != 0) {
      mlx_vector_array_free(fac_sample);
      mlx_pyramids_dataset_free(ds);
      return -1;
    }
    mlx_vector_array_free(fac_sample);
    ++found_samples;

    if (use_wells) {
      char probe[512];
      mlx_io_reader pr = {0};
      int has_wells = 0;
      snprintf(probe, sizeof(probe), "sample_%d/wells_0.npy", si);
      if (npz_extract_member_to_mlx_reader(npz_path, probe, &pr) == 0) {
        has_wells = 1;
        mlx_io_reader_free(pr);
      } else {
        snprintf(probe, sizeof(probe), "wells_0.npy");
        if (npz_extract_member_to_mlx_reader(npz_path, probe, &pr) == 0) {
          has_wells = 1;
          mlx_io_reader_free(pr);
        } else {
          snprintf(probe, sizeof(probe), "wells_0_scale_%d.npy", si);
          if (npz_extract_member_to_mlx_reader(npz_path, probe, &pr) == 0) {
            has_wells = 1;
            mlx_io_reader_free(pr);
          }
        }
      }

      if (has_wells) {
        mlx_vector_array well_sample = mlx_vector_array_new();
        mlx_vector_array mask_sample = mlx_vector_array_new();

        for (int sc2 = 0; sc2 < scale; ++sc2) {
          char member[512];
          mlx_io_reader r = {0};
          int found = 0;
          snprintf(member, sizeof(member), "sample_%d/wells_%d.npy", si, sc2);
          if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
            found = 1;
          else {
            snprintf(member, sizeof(member), "wells_%d.npy", sc2);
            if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
              found = 1;
            else {
              snprintf(member, sizeof(member), "wells_%d_scale_%d.npy", si,
                       sc2);
              if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
                found = 1;
            }
          }

          mlx_stream s = mlx_default_cpu_stream_new();
          if (found) {
            mlx_array a = mlx_array_new();
            if (mlx_load_reader(&a, r, s) != 0) {
              int shape[3] = {crop_size, crop_size, 1};
              mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
            }
            mlx_io_reader_free(r);

            if (mlx_vector_array_append_value(well_sample, a) != 0) {
              mlx_array_free(a);
              mlx_vector_array_free(well_sample);
              mlx_vector_array_free(mask_sample);
              mlx_pyramids_dataset_free(ds);
              mlx_stream_free(s);
              return -1;
            }

            char mmember[512];
            mlx_io_reader mr = {0};
            int mfound = 0;
            snprintf(mmember, sizeof(mmember), "sample_%d/masks_%d.npy", si,
                     sc2);
            if (npz_extract_member_to_mlx_reader(npz_path, mmember, &mr) == 0) {
              mlx_array m = mlx_array_new();
              if (mlx_load_reader(&m, mr, s) == 0) {
                mlx_io_reader_free(mr);
                mlx_vector_array_append_value(mask_sample, m);
                mlx_array_free(m);
                mfound = 1;
              } else {
                mlx_array_free(m);
                mlx_io_reader_free(mr);
              }
            }

            if (!mfound) {
              mlx_array abs_arr = mlx_array_new();
              if (mlx_abs(&abs_arr, a, s) == 0) {
                mlx_array sum_arr = mlx_array_new();
                if (mlx_sum_axis(&sum_arr, abs_arr, 3, 1, s) == 0) {
                  mlx_array zero = mlx_array_new();
                  if (mlx_zeros_like(&zero, sum_arr, s) == 0) {
                    mlx_array mask_arr = mlx_array_new();
                    if (mlx_greater(&mask_arr, sum_arr, zero, s) == 0) {
                      mlx_vector_array_append_value(mask_sample, mask_arr);
                    }
                    mlx_array_free(mask_arr);
                  }
                  mlx_array_free(zero);
                }
                mlx_array_free(sum_arr);
              }
              mlx_array_free(abs_arr);
            }

            mlx_array_free(a);
          } else {
            int shape[3] = {crop_size, crop_size, 1};
            mlx_array a = mlx_array_new();
            mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
            mlx_vector_array_append_value(well_sample, a);

            mlx_array zero = mlx_array_new();
            if (mlx_zeros_like(&zero, a, s) == 0) {
              mlx_array mask_arr = mlx_array_new();
              if (mlx_greater(&mask_arr, zero, zero, s) == 0) {
                mlx_vector_array_append_value(mask_sample, mask_arr);
              }
              mlx_array_free(mask_arr);
            }
            mlx_array_free(zero);
            mlx_array_free(a);
          }
          mlx_stream_free(s);
        }

        if (mlx_vector_vector_array_append_value(ds->wells, well_sample) != 0) {
          mlx_vector_array_free(well_sample);
          mlx_vector_array_free(mask_sample);
          mlx_pyramids_dataset_free(ds);
          return -1;
        }
        mlx_vector_array_free(well_sample);

        if (mlx_vector_vector_array_append_value(ds->masks, mask_sample) != 0) {
          mlx_vector_array_free(mask_sample);
          mlx_pyramids_dataset_free(ds);
          return -1;
        }
        mlx_vector_array_free(mask_sample);
      }
    }

    if (use_seismic) {
      char probe[512];
      mlx_io_reader pr = {0};
      int has_seis = 0;
      snprintf(probe, sizeof(probe), "sample_%d/seismic_0.npy", si);
      if (npz_extract_member_to_mlx_reader(npz_path, probe, &pr) == 0) {
        has_seis = 1;
        mlx_io_reader_free(pr);
      } else {
        snprintf(probe, sizeof(probe), "seismic_0.npy");
        if (npz_extract_member_to_mlx_reader(npz_path, probe, &pr) == 0) {
          has_seis = 1;
          mlx_io_reader_free(pr);
        }
      }

      if (has_seis) {
        mlx_vector_array seis_sample = mlx_vector_array_new();
        for (int sc2 = 0; sc2 < scale; ++sc2) {
          char member[512];
          mlx_io_reader r = {0};
          int found = 0;
          snprintf(member, sizeof(member), "sample_%d/seismic_%d.npy", si, sc2);
          if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
            found = 1;
          else {
            snprintf(member, sizeof(member), "seismic_%d.npy", sc2);
            if (npz_extract_member_to_mlx_reader(npz_path, member, &r) == 0)
              found = 1;
          }

          mlx_stream s = mlx_default_cpu_stream_new();
          if (found) {
            mlx_array a = mlx_array_new();
            if (mlx_load_reader(&a, r, s) != 0) {
              int shape[3] = {crop_size, crop_size, 1};
              mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
            }
            mlx_io_reader_free(r);
            mlx_vector_array_append_value(seis_sample, a);
            mlx_array_free(a);
          } else {
            int shape[3] = {crop_size, crop_size, 1};
            mlx_array a = mlx_array_new();
            mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
            mlx_vector_array_append_value(seis_sample, a);
            mlx_array_free(a);
          }
          mlx_stream_free(s);
        }
        if (mlx_vector_vector_array_append_value(ds->seismic, seis_sample) !=
            0) {
          mlx_vector_array_free(seis_sample);
          mlx_pyramids_dataset_free(ds);
          return -1;
        }
        mlx_vector_array_free(seis_sample);
      }
    }

  sample_end:;
  }

  ds->n_samples = found_samples;

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
  free(ds);
}

int mlx_pyramids_dataset_shuffle(MLXPyramidsDataset *ds, unsigned int seed) {
  if (!ds)
    return -1;
  int n = ds->n_samples;
  if (n <= 1)
    return 0;
  int *idx = (int *)malloc(sizeof(int) * n);
  if (!idx)
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
      free(idx);
      return -1;
    }
    if (mlx_vector_vector_array_append_value(fac_new, tmp)) {
      mlx_vector_array_free(tmp);
      mlx_vector_vector_array_free(fac_new);
      free(idx);
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

  free(idx);
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
  int N = ds->n_samples;
  if (N <= 0)
    return -1;

  mlx_vector_array sample0;
  if (mlx_vector_vector_array_get(&sample0, ds->facies, 0) != 0)
    return -1;
  if (scale >= (int)mlx_vector_array_size(sample0))
    return -1;
  mlx_array first = mlx_array_new();
  if (mlx_vector_array_get(&first, sample0, scale) != 0)
    return -1;
  const int *shape = mlx_array_shape(first);
  int h = shape[0];
  int w = shape[1];
  int c = shape[2];

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

  mlx_stream s = mlx_default_cpu_stream_new();
  mlx_array stacked = mlx_array_new();
  int rc = mlx_stack(&stacked, scale_vec, s);
  mlx_vector_array_free(scale_vec);
  mlx_stream_free(s);
  if (rc != 0) {
    mlx_array_free(stacked);
    return -1;
  }
  *out = stacked;
  return 0;
}

static int build_stack_from_source(mlx_vector_vector_array src,
                                   MLXPyramidsDataset *ds, int scale,
                                   mlx_array *out) {
  if (!ds || !out)
    return -1;
  int N = ds->n_samples;
  if (mlx_vector_vector_array_size(src) == 0) {
    mlx_vector_array sample0;
    if (mlx_vector_vector_array_get(&sample0, ds->facies, 0) != 0)
      return -1;
    mlx_array first = mlx_array_new();
    if (mlx_vector_array_get(&first, sample0, scale) != 0) {
      mlx_array_free(first);
      return -1;
    }
    const int *shape = mlx_array_shape(first);
    int h = shape[0];
    int w = shape[1];
    int c = shape[2];
    int zeros_shape[4] = {0, h, w, c};
    int dtype = (int)mlx_array_dtype(first);
    mlx_array_free(first);
    mlx_stream s = mlx_default_cpu_stream_new();
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

  mlx_stream s = mlx_default_cpu_stream_new();
  mlx_array stacked = mlx_array_new();
  int rc = mlx_stack(&stacked, scale_vec, s);
  mlx_vector_array_free(scale_vec);
  mlx_stream_free(s);
  if (rc != 0) {
    mlx_array_free(stacked);
    return -1;
  }
  *out = stacked;
  return 0;
}

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
