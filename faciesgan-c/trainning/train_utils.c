#include "train_utils.h"
#include "models/custom_layer.h"
#include "models/facies_gan.h"
#include <execinfo.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int mlx_compute_rec_input(int scale, const int *indexes, int n_indexes,
                          mlx_array **facies_pyramid, mlx_array **out) {
  if (!facies_pyramid || !out)
    return -1;
  if (scale < 0)
    return -1;

  /* Writer canary: emit backtrace when this hot function runs */
  fprintf(stderr, "[writer_canary] func=mlx_compute_rec_input tid=%lu\n",
          (unsigned long)pthread_self());
  {
    void *bt[64];
    int bt_size = backtrace(bt, 64);
    backtrace_symbols_fd(bt, bt_size, fileno(stderr));
  }

  /* create a default CPU stream for MLX ops and ensure it's freed */
  mlx_stream s = mlx_default_cpu_stream_new();

  /* scale 0 -> zeros_like(real) */
  if (scale == 0) {
    if (!facies_pyramid[0]) {
      mlx_stream_free(s);
      return -1;
    }
    mlx_array z = mlx_array_new();
    if (mlx_zeros_like(&z, *facies_pyramid[0], s) != 0) {
      mlx_stream_free(s);
      return -1;
    }
    mlx_array *res = (mlx_array *)malloc(sizeof(mlx_array));
    if (!res) {
      mlx_array_free(z);
      mlx_stream_free(s);
      return -1;
    }
    *res = z;
    *out = res;
    mlx_stream_free(s);
    return 0;
  }

  /* gather previous-scale entries by indexes */
  if (!facies_pyramid[scale - 1] || !facies_pyramid[scale])
    return -1;
  if (!indexes || n_indexes <= 0)
    return -1;

  int idx_shape[1] = {n_indexes};
  mlx_array idx = mlx_array_new_data(indexes, idx_shape, 1, MLX_INT32);
  mlx_array sel = mlx_array_new();
  if (mlx_gather_single(&sel, *facies_pyramid[scale - 1], idx, 0, NULL, 0, s) !=
      0) {
    mlx_array_free(idx);
    mlx_stream_free(s);
    return -1;
  }
  mlx_array_free(idx);

  /* upsample to match target spatial dims */
  const int *shape = mlx_array_shape(*facies_pyramid[scale]);
  if (!shape) {
    mlx_array_free(sel);
    return -1;
  }
  int target_h = shape[1];
  int target_w = shape[2];

  MLXUpsample *u = mlx_upsample_create(target_h, target_w, "linear", 1);
  if (!u) {
    mlx_array_free(sel);
    mlx_stream_free(s);
    return -1;
  }
  mlx_array up = mlx_upsample_forward(u, sel);
  mlx_upsample_free(u);

  /* free intermediate selection if different from result */
  if (sel.ctx)
    mlx_array_free(sel);

  mlx_array *res = (mlx_array *)malloc(sizeof(mlx_array));
  if (!res) {
    mlx_array_free(up);
    mlx_stream_free(s);
    return -1;
  }
  *res = up;
  *out = res;
  mlx_stream_free(s);
  return 0;
}

int mlx_init_rec_noise_and_amp(MLXFaciesGAN *m, int scale, const int *indexes,
                               int n_indexes, const mlx_array *real,
                               mlx_array **wells_pyramid,
                               mlx_array **seismic_pyramid) {
  if (!m || !real)
    return -1;
  if (scale < 0)
    return -1;

  /* if noise amps already set up to this scale, nothing to do */
  float *amps = NULL;
  int n_amps = 0;
  if (mlx_faciesgan_get_noise_amps(m, &amps, &n_amps) == 0) {
    if (amps) {
      if (n_amps > scale) {
        free(amps);
        return 0;
      }
      free(amps);
    }
  }

  /* Defaults (match Python defaults in options.c) */
  const double default_scale0_noise_amp = 1.0;
  const double default_noise_amp = 0.1;
  const double default_min_noise_amp = 0.1;

  /* get pyramid noises */
  mlx_array **noises = NULL;
  int n_noises = 0;
  if (mlx_faciesgan_get_pyramid_noise(m, scale, indexes, n_indexes, &noises,
                                      &n_noises, wells_pyramid, seismic_pyramid,
                                      0) != 0)
    return -1;

  /* obtain amplitude list for generation (will be freed) */
  float *use_amps = NULL;
  int use_n = 0;
  if (mlx_faciesgan_get_noise_amplitude(m, scale, &use_amps, &use_n) != 0) {
    /* fallback to ones */
    use_amps = (float *)malloc(sizeof(float) * (scale + 1));
    if (!use_amps) {
      for (int i = 0; i < n_noises; ++i) {
        if (noises[i]) {
          mlx_array_free(*noises[i]);
          free(noises[i]);
        }
      }
      free(noises);
      return -1;
    }
    for (int i = 0; i < scale + 1; ++i)
      use_amps[i] = 1.0f;
    use_n = scale + 1;
  }

  /* generate fake for this scale (numeric forward) */
  /* Convert returned array-of-pointers into a contiguous array of mlx_array
     values required by mlx_faciesgan_generate_fake. */
  mlx_array *zvals = NULL;
  if (n_noises > 0) {
    zvals = (mlx_array *)malloc(sizeof(mlx_array) * (size_t)n_noises);
    if (!zvals) {
      for (int i = 0; i < n_noises; ++i) {
        if (noises[i]) {
          mlx_array_free(*noises[i]);
          free(noises[i]);
        }
      }
      free(noises);
      if (use_amps)
        free(use_amps);
      return -1;
    }
    for (int i = 0; i < n_noises; ++i)
      zvals[i] = *noises[i];
    /* Ensure none of the zvals are empty -- replace empty entries with zeros */
    for (int i = 0; i < n_noises; ++i) {
      if (mlx_array_ndim(zvals[i]) == 0) {
        mlx_stream _s = mlx_default_cpu_stream_new();
        int shape0[4] = {1, 32, 32, 1};
        mlx_array tmp = mlx_array_new();
        if (mlx_zeros(&tmp, shape0, 4, MLX_FLOAT32, _s) == 0) {
          zvals[i] = tmp;
        }
        mlx_stream_free(_s);
      }
    }
  }

  /* pick an initial in_noise: prefer a zero array sized from the first noise
     minus the full_zero_padding so the generator starts with the expected
     spatial dims. If generator info isn't available, fall back to empty. */
  mlx_array in_noise = mlx_array_new();
  if (n_noises > 0) {
    /* Prefer deriving initial in_noise spatial size from the provided `real`
       image so fake and real match for RMSE. Fall back to leaving in_noise
       empty if real shape is unavailable. */
    if (real) {
      const int *rshape = mlx_array_shape(*real);
      if (rshape) {
        int osh[4] = {rshape[0], rshape[1], rshape[2], rshape[3]};
        mlx_stream _s = mlx_default_cpu_stream_new();
        mlx_array tmp = mlx_array_new();
        if (mlx_zeros(&tmp, osh, 4, MLX_FLOAT32, _s) == 0) {
          in_noise = tmp;
        }
        mlx_stream_free(_s);
      }
    }
  }

  for (int i = 0; i < n_noises; ++i) {
    fprintf(stdout, "[debug] noise[%d] ndim=%zu\n", i,
            mlx_array_ndim(zvals[i]));
  }
  /* Print shapes for real and pyramids to help debug broadcasting issues */
  if (real) {
    int real_ndim = (int)mlx_array_ndim(*real);
    const int *rshape = mlx_array_shape(*real);
    fprintf(stdout, "[debug] real ndim=%d", real_ndim);
    if (rshape) {
      fprintf(stdout, " shape=(");
      for (int a = 0; a < real_ndim; ++a) {
        fprintf(stdout, "%d%s", rshape[a], (a + 1 < real_ndim) ? "," : "");
      }
      fprintf(stdout, ")\n");
    } else {
      fprintf(stdout, " (no shape)\n");
    }
  }
  if (wells_pyramid) {
    for (int wi = 0; wi <= scale && wells_pyramid[wi]; ++wi) {
      int w_ndim = (int)mlx_array_ndim(*wells_pyramid[wi]);
      const int *wshape = mlx_array_shape(*wells_pyramid[wi]);
      fprintf(stdout, "[debug] wells[%d] ndim=%d", wi, w_ndim);
      if (wshape) {
        fprintf(stdout, " shape=(");
        for (int a = 0; a < w_ndim; ++a)
          fprintf(stdout, "%d%s", wshape[a], (a + 1 < w_ndim) ? "," : "");
        fprintf(stdout, ")\n");
      } else
        fprintf(stdout, " (no shape)\n");
    }
  }
  if (seismic_pyramid) {
    for (int si = 0; si <= scale && seismic_pyramid[si]; ++si) {
      int s_ndim = (int)mlx_array_ndim(*seismic_pyramid[si]);
      const int *sshape = mlx_array_shape(*seismic_pyramid[si]);
      fprintf(stdout, "[debug] seismic[%d] ndim=%d", si, s_ndim);
      if (sshape) {
        fprintf(stdout, " shape=(");
        for (int a = 0; a < s_ndim; ++a)
          fprintf(stdout, "%d%s", sshape[a], (a + 1 < s_ndim) ? "," : "");
        fprintf(stdout, ")\n");
      } else
        fprintf(stdout, " (no shape)\n");
    }
  }
  for (int i = 0; i < n_noises; ++i) {
    int ndim = (int)mlx_array_ndim(zvals[i]);
    const int *sh = mlx_array_shape(zvals[i]);
    fprintf(stdout, "[debug] zvals[%d] ndim=%d", i, ndim);
    if (sh) {
      fprintf(stdout, " shape=(");
      for (int a = 0; a < ndim; ++a)
        fprintf(stdout, "%d%s", sh[a], (a + 1 < ndim) ? "," : "");
      fprintf(stdout, ")\n");
    } else
      fprintf(stdout, " (no shape)\n");
  }
  int in_ndim = (int)mlx_array_ndim(in_noise);
  const int *in_sh = mlx_array_shape(in_noise);
  fprintf(stdout, "[debug] in_noise ndim=%d", in_ndim);
  if (in_sh) {
    fprintf(stdout, " shape=(");
    for (int a = 0; a < in_ndim; ++a)
      fprintf(stdout, "%d%s", in_sh[a], (a + 1 < in_ndim) ? "," : "");
    fprintf(stdout, ")\n");
  } else
    fprintf(stdout, " (no shape)\n");
  fprintf(stdout, "[debug] calling generate_fake: n_noises=%d scale=%d\n",
          n_noises, scale);
  mlx_array_t fake = mlx_faciesgan_generate_fake(m, zvals, n_noises, use_amps,
                                                 use_n, in_noise, scale, scale);

  /* free noises (we free underlying arrays and the pointer container) */
  for (int i = 0; i < n_noises; ++i) {
    if (noises[i]) {
      mlx_array_free(*noises[i]);
      free(noises[i]);
    }
  }
  free(noises);
  if (zvals)
    free(zvals);
  if (use_amps)
    free(use_amps);

  /* compute rmse = sqrt(mean((fake - real)^2)) */
  mlx_stream s = mlx_default_cpu_stream_new();
  /* Ensure `real` matches `fake` spatial dims by padding `real` symmetrically
     when needed. Use the padded copy for subtraction to compute RMSE. */
  const int *fshape = mlx_array_shape(fake);
  const int *rshape = mlx_array_shape(*real);
  mlx_array real_to_use = *real;
  int created_real_pad = 0;
  if (fshape && rshape && (fshape[1] != rshape[1] || fshape[2] != rshape[2])) {
    int dh = fshape[1] - rshape[1];
    int dw = fshape[2] - rshape[2];
    int low_h = dh > 0 ? dh / 2 : 0;
    int high_h = dh > 0 ? dh - low_h : 0;
    int low_w = dw > 0 ? dw / 2 : 0;
    int high_w = dw > 0 ? dw - low_w : 0;
    int axes[2] = {1, 2};
    int low_pad[2] = {low_h, low_w};
    int high_pad[2] = {high_h, high_w};
    mlx_array pad_zero = mlx_array_new_float(0.0f);
    mlx_array padded_real = mlx_array_new();
    if (mlx_pad(&padded_real, *real, axes, 2, low_pad, 2, high_pad, 2, pad_zero,
                "constant", s) == 0) {
      real_to_use = padded_real;
      created_real_pad = 1;
    } else {
      mlx_array_free(padded_real);
    }
    mlx_array_free(pad_zero);
  }
  mlx_array diff = mlx_array_new();
  if (mlx_subtract(&diff, fake, real_to_use, s) != 0) {
    if (created_real_pad)
      mlx_array_free(real_to_use);
    mlx_stream_free(s);
    mlx_array_free(fake);
    mlx_array_free(in_noise);
    return -1;
  }
  if (created_real_pad)
    mlx_array_free(real_to_use);
  mlx_array sq = mlx_array_new();
  if (mlx_square(&sq, diff, s) != 0) {
    mlx_array_free(diff);
    mlx_stream_free(s);
    mlx_array_free(fake);
    mlx_array_free(in_noise);
    return -1;
  }
  mlx_array_free(diff);

  mlx_array mean = mlx_array_new();
  if (mlx_mean(&mean, sq, false, s) != 0) {
    mlx_array_free(sq);
    mlx_stream_free(s);
    mlx_array_free(fake);
    mlx_array_free(in_noise);
    return -1;
  }
  mlx_array_free(sq);

  mlx_array root = mlx_array_new();
  if (mlx_sqrt(&root, mean, s) != 0) {
    mlx_array_free(mean);
    mlx_stream_free(s);
    mlx_array_free(fake);
    mlx_array_free(in_noise);
    return -1;
  }
  mlx_array_free(mean);

  /* extract scalar value (skip host eval when disabled via env var to avoid
     invoking device schedulers in smoke runs) */
  double rmse = 0.0;
  /* Prefer reading host value only when available to avoid forcing device eval
   */
  bool ok_root = false;
  if (_mlx_array_is_available(&ok_root, root) == 0 && ok_root) {
    const float *pdata = mlx_array_data_float32(root);
    if (pdata)
      rmse = (double)pdata[0];
  } else {
    rmse = 0.0;
  }

  /* compute new amp */
  double amp = default_noise_amp * rmse;
  if (scale == 0)
    amp = default_scale0_noise_amp * rmse;
  if (amp < default_min_noise_amp)
    amp = default_min_noise_amp;

  /* Build new amp array (scale+1) by copying existing amps if present */
  float *new_amps = (float *)malloc(sizeof(float) * (scale + 1));
  if (!new_amps) {
    mlx_array_free(root);
    mlx_stream_free(s);
    mlx_array_free(fake);
    mlx_array_free(in_noise);
    return -1;
  }
  /* try to read existing amps to preserve previous values */
  if (mlx_faciesgan_get_noise_amps(m, &amps, &n_amps) == 0 && amps &&
      n_amps > 0) {
    for (int i = 0; i < scale + 1; ++i)
      new_amps[i] = (i < n_amps) ? amps[i] : 1.0f;
    free(amps);
  } else {
    for (int i = 0; i < scale + 1; ++i)
      new_amps[i] = 1.0f;
  }
  new_amps[scale] = (float)amp;

  /* set on model */
  int res = mlx_faciesgan_set_noise_amps(m, new_amps, scale + 1);

  free(new_amps);
  mlx_array_free(root);
  mlx_stream_free(s);
  mlx_array_free(fake);
  mlx_array_free(in_noise);
  return res;
}
