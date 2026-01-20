#include "io/npz_unzip.h"
#include <mlx/c/array.h>
#include <mlx/c/io.h>
#include <mlx/c/ops.h>
#include <mlx/c/stream.h>
#include <mlx/c/vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int build_stack_from_members(const char *npz_path,
                                    const char *member_fmt, int n_samples,
                                    int scale, mlx_array *out) {
  mlx_vector_array vec = mlx_vector_array_new();
  mlx_stream s = mlx_default_cpu_stream_new();
  for (int i = 0; i < n_samples; ++i) {
    char member[512];
    snprintf(member, sizeof(member), member_fmt, i, scale);
    mlx_io_reader reader;
    if (npz_extract_member_to_mlx_reader(npz_path, member, &reader) != 0) {
      mlx_vector_array_free(vec);
      mlx_stream_free(s);
      return -1;
    }
    mlx_array a = mlx_array_new();
    if (mlx_load_reader(&a, reader, s) != 0) {
      mlx_io_reader_free(reader);
      mlx_array_free(a);
      mlx_vector_array_free(vec);
      mlx_stream_free(s);
      return -1;
    }
    mlx_io_reader_free(reader);
    mlx_vector_array_append_value(vec, a);
    mlx_array_free(a);
  }

  mlx_array stacked = mlx_array_new();
  int rc = mlx_stack(&stacked, vec, s);
  mlx_vector_array_free(vec);
  mlx_stream_free(s);
  if (rc != 0) {
    mlx_array_free(stacked);
    return -1;
  }
  *out = stacked;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <npz_path> <n_samples> <scale> <out_prefix>\n",
            argv[0]);
    return 2;
  }
  const char *npz = argv[1];
  int n_samples = atoi(argv[2]);
  int scale = atoi(argv[3]);
  const char *outp = argv[4];

  /* facies */
  mlx_array fac = mlx_array_new();
  char fac_fmt[] = "sample_%d/facies_%d.npy";
  if (build_stack_from_members(npz, fac_fmt, n_samples, scale, &fac) != 0) {
    fprintf(stderr, "failed to build facies stack from %s\n", npz);
    return 1;
  }
  char fname[1024];
  snprintf(fname, sizeof(fname), "%s_facies.npy", outp);
  if (mlx_save(fname, fac) != 0) {
    fprintf(stderr, "failed to save %s\n", fname);
  }
  mlx_array_free(fac);

  /* wells (optional) */
  mlx_array wells = mlx_array_new();
  char well_fmt[] = "sample_%d/wells_%d.npy";
  if (build_stack_from_members(npz, well_fmt, n_samples, scale, &wells) == 0) {
    snprintf(fname, sizeof(fname), "%s_wells.npy", outp);
    if (mlx_save(fname, wells) != 0) {
      fprintf(stderr, "failed to save %s\n", fname);
    }
    mlx_array_free(wells);
  } else {
    /* no wells present; write nothing */
    mlx_array_free(wells);
  }

  /* masks: try explicit members, otherwise compute from wells and save */
  /* try explicit masks stack */
  mlx_array masks = mlx_array_new();
  char mask_fmt[] = "sample_%d/masks_%d.npy";
  int masks_rc =
      build_stack_from_members(npz, mask_fmt, n_samples, scale, &masks);
  if (masks_rc == 0) {
    snprintf(fname, sizeof(fname), "%s_masks.npy", outp);
    if (mlx_save(fname, masks) != 0) {
      fprintf(stderr, "failed to save %s\n", fname);
    }
    mlx_array_free(masks);
  } else {
    /* compute masks from wells if wells exist */
    mlx_array wells_stack = mlx_array_new();
    if (build_stack_from_members(npz, well_fmt, n_samples, scale,
                                 &wells_stack) == 0) {
      mlx_stream s = mlx_default_cpu_stream_new();
      mlx_array abs_arr = mlx_array_new();
      if (mlx_abs(&abs_arr, wells_stack, s) == 0) {
        mlx_array sum_arr = mlx_array_new();
        if (mlx_sum_axis(&sum_arr, abs_arr, 3, 1, s) == 0) {
          mlx_array zero = mlx_array_new();
          if (mlx_zeros_like(&zero, sum_arr, s) == 0) {
            mlx_array mask_arr = mlx_array_new();
            if (mlx_greater(&mask_arr, sum_arr, zero, s) == 0) {
              snprintf(fname, sizeof(fname), "%s_masks.npy", outp);
              if (mlx_save(fname, mask_arr) != 0) {
                fprintf(stderr, "failed to save %s\n", fname);
              }
              mlx_array_free(mask_arr);
            }
            mlx_array_free(zero);
          }
          mlx_array_free(sum_arr);
        }
        mlx_array_free(abs_arr);
      }
      mlx_stream_free(s);
      mlx_array_free(wells_stack);
    }
  }

  /* seismic (optional) */
  mlx_array seis = mlx_array_new();
  char seis_fmt[] = "sample_%d/seismic_%d.npy";
  if (build_stack_from_members(npz, seis_fmt, n_samples, scale, &seis) == 0) {
    snprintf(fname, sizeof(fname), "%s_seismic.npy", outp);
    if (mlx_save(fname, seis) != 0) {
      fprintf(stderr, "failed to save %s\n", fname);
    }
    mlx_array_free(seis);
  } else {
    mlx_array_free(seis);
  }

  return 0;
}
