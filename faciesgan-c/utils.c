#include "utils.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* MLX headers for array helpers */
#include "options.h"
#include <errno.h>
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <math.h>

int mlx_create_dirs(const char *path) {
  if (!path || path[0] == '\0')
    return -1;
  char tmp[PATH_MAX];
  size_t len = strlen(path);
  if (len >= sizeof(tmp))
    return -1;
  strcpy(tmp, path);
  /* Remove trailing slashes */
  while (len > 1 && tmp[len - 1] == '/') {
    tmp[len - 1] = '\0';
    --len;
  }

  for (char *p = tmp + 1; *p; ++p) {
    if (*p == '/') {
      *p = '\0';
      if (mkdir(tmp, 0755) != 0) {
        if (errno != EEXIST)
          return -1;
      }
      *p = '/';
    }
  }
  if (mkdir(tmp, 0755) != 0) {
    if (errno != EEXIST)
      return -1;
  }
  return 0;
}

void ensure_dir(const char *path) {
  if (!path || !*path)
    return;

  char tmp[PATH_BUFSZ];
  strncpy(tmp, path, sizeof(tmp) - 1);
  tmp[sizeof(tmp) - 1] = '\0';

  /* Remove trailing slashes (but keep single leading '/' for absolute) */
  size_t len = strlen(tmp);
  while (len > 1 && tmp[len - 1] == '/') {
    tmp[--len] = '\0';
  }

  /* Iterate prefixes and create directories as needed (mkdir -p behavior) */
  for (char *p = tmp + 1; *p; ++p) {
    if (*p == '/') {
      *p = '\0';
      struct stat st;
      if (stat(tmp, &st) == -1) {
        if (mkdir(tmp, 0755) == -1 && errno != EEXIST) {
          fprintf(stderr, "error: mkdir '%s' failed: %s\n", tmp,
                  strerror(errno));
          *p = '/';
          return;
        }
      } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr,
                "error: path component '%s' exists and is not a directory\n",
                tmp);
        *p = '/';
        return;
      }
      *p = '/';
    }
  }

  /* Create the final directory */
  struct stat st;
  if (stat(tmp, &st) == -1) {
    if (mkdir(tmp, 0755) == -1 && errno != EEXIST) {
      fprintf(stderr, "error: mkdir '%s' failed: %s\n", tmp, strerror(errno));
      return;
    }
  } else if (!S_ISDIR(st.st_mode)) {
    fprintf(stderr, "error: path '%s' exists and is not a directory\n", tmp);
    return;
  }
}

void write_options_json(const TrainningOptions *topt,
                        const int *wells_mask_columns,
                        size_t wells_mask_count) {
  if (!topt || !topt->output_path)
    return;
  char optpath[PATH_BUFSZ];
  join_path(optpath, sizeof(optpath), topt->output_path, OPT_FILE);
  FILE *of = fopen(optpath, "w");
  if (!of)
    return;

  /* Follow the Python writer order and keys closely to maximize parity. */
  fprintf(of, "{\n");

  /* local helper function-like macro */
#define PRINT_DOUBLE(FP)                                                       \
  do {                                                                         \
    double __v = (double)(FP);                                                 \
    if (isfinite(__v) && fabs(__v - round(__v)) < 1e-9)                        \
      fprintf(of, "%.1f", __v);                                                \
    else                                                                       \
      fprintf(of, "%g", __v);                                                  \
  } while (0)

  fprintf(of, "    \"alpha\": %d,\n", topt->alpha);
  fprintf(of, "    \"batch_size\": %d,\n", topt->batch_size);
  fprintf(of, "    \"beta1\": ");
  PRINT_DOUBLE(topt->beta1);
  fprintf(of, ",\n");
  fprintf(of, "    \"crop_size\": %d,\n", topt->crop_size);
  fprintf(of, "    \"discriminator_steps\": %d,\n", topt->discriminator_steps);
  fprintf(of, "    \"num_img_channels\": %d,\n", topt->num_img_channels);
  fprintf(of, "    \"gamma\": ");
  PRINT_DOUBLE(topt->gamma);
  fprintf(of, ",\n");
  fprintf(of, "    \"generator_steps\": %d,\n", topt->generator_steps);
  fprintf(of, "    \"gpu_device\": %d,\n", topt->gpu_device);
  /* Print arrays in multi-line form to match Python's json.dump(indent=4) */
  fprintf(of, "    \"img_color_range\": [\n");
  fprintf(of, "        %d,\n", topt->img_color_min);
  fprintf(of, "        %d\n", topt->img_color_max);
  fprintf(of, "    ],\n");
  fprintf(of, "    \"input_path\": \"%s\",\n",
          topt->input_path ? topt->input_path : "");
  fprintf(of, "    \"kernel_size\": %d,\n", topt->kernel_size);
  fprintf(of, "    \"lambda_grad\": ");
  PRINT_DOUBLE(topt->lambda_grad);
  fprintf(of, ",\n");
  fprintf(of, "    \"lr_d\": ");
  PRINT_DOUBLE(topt->lr_d);
  fprintf(of, ",\n");
  fprintf(of, "    \"lr_decay\": %d,\n", topt->lr_decay);
  fprintf(of, "    \"lr_g\": ");
  PRINT_DOUBLE(topt->lr_g);
  fprintf(of, ",\n");
  if (topt->manual_seed < 0)
    fprintf(of, "    \"manual_seed\": null,\n");
  else
    fprintf(of, "    \"manual_seed\": %d,\n", topt->manual_seed);
  fprintf(of, "    \"max_size\": %d,\n", topt->max_size);
  fprintf(of, "    \"min_num_feature\": %d,\n", topt->min_num_feature);
  fprintf(of, "    \"min_size\": %d,\n", topt->min_size);
  fprintf(of, "    \"noise_amp\": ");
  PRINT_DOUBLE(topt->noise_amp);
  fprintf(of, ",\n");
  fprintf(of, "    \"min_noise_amp\": ");
  PRINT_DOUBLE(topt->min_noise_amp);
  fprintf(of, ",\n");
  fprintf(of, "    \"scale0_noise_amp\": ");
  PRINT_DOUBLE(topt->scale0_noise_amp);
  fprintf(of, ",\n");
  fprintf(of, "    \"well_loss_penalty\": ");
  PRINT_DOUBLE(topt->well_loss_penalty);
  fprintf(of, ",\n");
  fprintf(of, "    \"lambda_diversity\": ");
  PRINT_DOUBLE(topt->lambda_diversity);
  fprintf(of, ",\n");
  fprintf(of, "    \"num_diversity_samples\": %d,\n",
          topt->num_diversity_samples);
  fprintf(of, "    \"num_feature\": %d,\n", topt->num_feature);
  fprintf(of, "    \"num_generated_per_real\": %d,\n",
          topt->num_generated_per_real);
  fprintf(of, "    \"num_iter\": %d,\n", topt->num_iter);
  fprintf(of, "    \"num_layer\": %d,\n", topt->num_layer);
  fprintf(of, "    \"num_real_facies\": %d,\n", topt->num_real_facies);
  fprintf(of, "    \"num_train_pyramids\": %d,\n", topt->num_train_pyramids);
  fprintf(of, "    \"num_parallel_scales\": %d,\n", topt->num_parallel_scales);
  fprintf(of, "    \"noise_channels\": %d,\n", topt->noise_channels);
  fprintf(of, "    \"num_workers\": %d,\n", topt->num_workers);
  fprintf(of, "    \"output_path\": \"%s\",\n",
          topt->output_path ? topt->output_path : "");
  fprintf(of, "    \"padding_size\": %d,\n", topt->padding_size);
  fprintf(of, "    \"regen_npy_gz\": %s,\n", bool_str(topt->regen_npy_gz));
  fprintf(of, "    \"save_interval\": %d,\n", topt->save_interval);
  fprintf(of, "    \"start_scale\": %d,\n", topt->start_scale);
  fprintf(of, "    \"stride\": %d,\n", topt->stride);
  fprintf(of, "    \"stop_scale\": %d,\n", topt->stop_scale);
  fprintf(of, "    \"use_cpu\": %s,\n", bool_str(topt->use_cpu));
  fprintf(of, "    \"use_mlx\": %s,\n", bool_str(topt->use_mlx));
  fprintf(of, "    \"use_wells\": %s,\n", bool_str(topt->use_wells));
  fprintf(of, "    \"use_seismic\": %s,\n", bool_str(topt->use_seismic));
  if (wells_mask_count == 0) {
    fprintf(of, "    \"wells_mask_columns\": [],\n");
  } else {
    fprintf(of, "    \"wells_mask_columns\": [\n");
    for (size_t wi = 0; wi < wells_mask_count; ++wi) {
      fprintf(of, "        %d", wells_mask_columns[wi]);
      if (wi + 1 < wells_mask_count)
        fprintf(of, ",\n");
      else
        fprintf(of, "\n");
    }
    fprintf(of, "    ],\n");
  }
  fprintf(of, "    \"enable_tensorboard\": %s,\n",
          bool_str(topt->enable_tensorboard));
  fprintf(of, "    \"enable_plot_facies\": %s,\n",
          bool_str(topt->enable_plot_facies));
  /* Make compile_backend the final key to match Python's ordering (no aliases).
   */
  fprintf(of, "    \"compile_backend\": %s\n", bool_str(topt->compile_backend));
  /* Match Python json.dump which does not append a trailing newline */
  fprintf(of, "}");
  fclose(of);
}

void format_timestamp(char *buf, size_t bufsz) {
  if (!buf || bufsz == 0)
    return;
  time_t t = time(NULL);
  struct tm tm;
  localtime_r(&t, &tm);
  strftime(buf, bufsz, "%Y_%m_%d_%H_%M_%S", &tm);
}

const char *bool_str(int v) { return v ? "true" : "false"; }

void join_path(char *dst, size_t dstsz, const char *a, const char *b) {
  if (!dst || dstsz == 0)
    return;
  if (!a)
    a = "";
  if (!b)
    b = "";
  size_t al = strlen(a);
  if (al > 0 && a[al - 1] == '/')
    snprintf(dst, dstsz, "%s%s", a, b);
  else
    snprintf(dst, dstsz, "%s/%s", a, b);
}

void mlx_set_seed(int seed) {
  /* Seed the C library RNG. */
  srand((unsigned int)seed);
  /* Seed POSIX drand48 family as well */
  srand48((long)seed);
}

int mlx_clamp(mlx_array *res, const mlx_array a, float min_val, float max_val,
              const mlx_stream s) {
  if (!res)
    return -1;

  /* Create scalar arrays for min and max */
  mlx_array min_scalar = mlx_array_new_float(min_val);
  mlx_array max_scalar = mlx_array_new_float(max_val);

  int rc = 0;
  mlx_array min_arr = mlx_array_new();
  mlx_array max_arr = mlx_array_new();
  mlx_array tmp = mlx_array_new();

  /* Fill arrays with scalar values matching `a` shape */
  rc = mlx_full_like(&min_arr, a, min_scalar, mlx_array_dtype(a), s);
  if (rc != 0)
    goto cleanup;
  rc = mlx_full_like(&max_arr, a, max_scalar, mlx_array_dtype(a), s);
  if (rc != 0)
    goto cleanup;

  /* tmp = maximum(a, min_arr) */
  rc = mlx_maximum(&tmp, a, min_arr, s);
  if (rc != 0)
    goto cleanup;

  /* res = minimum(tmp, max_arr) */
  rc = mlx_minimum(res, tmp, max_arr, s);

cleanup:
  /* Free temporaries and scalars */
  mlx_array_free(min_scalar);
  mlx_array_free(max_scalar);
  /* mlx_array_free returns int but we ignore errors during cleanup */
  mlx_array_free(min_arr);
  mlx_array_free(max_arr);
  mlx_array_free(tmp);
  return rc;
}

/* Begin moved from utils_extra.c */

int mlx_array_to_float_buffer(const mlx_array a, float **out_buf,
                              size_t *out_elems, int *out_ndim,
                              int **out_shape) {
  if (!out_buf || !out_elems)
    return -1;

  bool ok_avail = false;
  if (_mlx_array_is_available(&ok_avail, a) != 0 || !ok_avail)
    return -1;
  if (mlx_array_dtype(a) != MLX_FLOAT32)
    return -1;
  size_t elems = (size_t)mlx_array_size(a);
  const float *data = mlx_array_data_float32(a);
  if (!data)
    return -1;
  float *buf = (float *)malloc(sizeof(float) * elems);
  if (!buf)
    return -1;
  memcpy(buf, data, sizeof(float) * elems);
  *out_buf = buf;
  *out_elems = elems;
  if (out_ndim)
    *out_ndim = mlx_array_ndim(a);
  if (out_shape) {
    int ndim = mlx_array_ndim(a);
    int *shape = NULL;
    if (ndim > 0) {
      shape = (int *)malloc(sizeof(int) * ndim);
      if (!shape) {
        free(buf);
        return -1;
      }
      const int *s = mlx_array_shape(a);
      for (int i = 0; i < ndim; ++i)
        shape[i] = s[i];
    }
    *out_shape = shape;
  }
  return 0;
}

int mlx_array_from_float_buffer(mlx_array *out, const float *buf,
                                const int *shape, int ndim) {
  if (!out || !buf)
    return -1;
  size_t elems = 1;
  for (int i = 0; i < ndim; ++i)
    elems *= (size_t)shape[i];
  float *tmp = (float *)malloc(sizeof(float) * elems);
  if (!tmp)
    return -1;
  memcpy(tmp, buf, sizeof(float) * elems);
  int rc = mlx_array_set_data(out, tmp, shape, ndim, MLX_FLOAT32);
  free(tmp);
  return rc == 0 ? 0 : -1;
}

void quantize_pixels_float(const float *in_pixels, float *out_pixels,
                           size_t npixels, int c, const float *palette,
                           int ncolors) {
  if (!in_pixels || !out_pixels || !palette)
    return;
  for (size_t i = 0; i < npixels; ++i) {
    const float *px = in_pixels + (size_t)i * c;
    int best = 0;
    float best_dist = FLT_MAX;
    for (int p = 0; p < ncolors; ++p) {
      const float *col = palette + (size_t)p * c;
      float d = 0.0f;
      for (int k = 0; k < c; ++k) {
        float diff = px[k] - col[k];
        d += diff * diff;
      }
      if (d < best_dist) {
        best_dist = d;
        best = p;
      }
    }
    const float *sel = palette + (size_t)best * c;
    for (int k = 0; k < c; ++k)
      out_pixels[i * c + k] = sel[k];
  }
}

void apply_well_mask_cpu(const float *facies, float *out, int h, int w, int c,
                         const unsigned char *mask, const float *well, int wc) {
  if (!facies || !out || !mask || !well)
    return;
  size_t npixels = (size_t)h * (size_t)w;
  for (size_t i = 0; i < npixels * (size_t)c; ++i)
    out[i] = facies[i];

  unsigned char *col_has = (unsigned char *)malloc((size_t)w);
  if (!col_has)
    return;
  memset(col_has, 0, (size_t)w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (mask[y * w + x])
        col_has[x] = 1;
    }
  }

  for (int x = 0; x < w; ++x) {
    if (col_has[x]) {
      for (int y = 0; y < h; ++y) {
        for (int ch = 0; ch < c; ++ch) {
          out[(y * w + x) * c + ch] = 1.0f;
        }
      }
    }
  }

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int idx = y * w + x;
      if (!mask[idx])
        continue;
      float brightness = 0.0f;
      if (wc == c) {
        for (int ch = 0; ch < c; ++ch)
          brightness += fabsf(well[idx * wc + ch]);
        brightness = brightness / (float)c;
      } else if (wc == 1) {
        brightness = fabsf(well[idx]);
      }
      if (brightness < 0.3f)
        continue;
      if (wc == c) {
        for (int ch = 0; ch < c; ++ch)
          out[idx * c + ch] = well[idx * wc + ch];
      } else if (wc == 1) {
        for (int ch = 0; ch < c; ++ch)
          out[idx * c + ch] = well[idx];
      }
    }
  }

  free(col_has);
}

int mlx_denorm_array(const mlx_array in, mlx_array *out, int ceiling) {
  if (!out)
    return -1;
  bool ok_avail = false;
  if (_mlx_array_is_available(&ok_avail, in) != 0 || !ok_avail)
    return -1;
  if (mlx_array_dtype(in) != MLX_FLOAT32)
    return -1;
  size_t elems = mlx_array_size(in);
  const float *data = mlx_array_data_float32(in);
  if (!data)
    return -1;
  float *buf = (float *)malloc(sizeof(float) * elems);
  if (!buf)
    return -1;
  for (size_t i = 0; i < elems; ++i) {
    float v = (data[i] + 1.0f) * 0.5f;
    if (v < 0.0f)
      v = 0.0f;
    if (v > 1.0f)
      v = 1.0f;
    if (ceiling && v > 0.0f)
      v = 1.0f;
    buf[i] = v;
  }
  int ndim = mlx_array_ndim(in);
  const int *shape = mlx_array_shape(in);
  mlx_array out_arr = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
  *out = out_arr;
  free(buf);
  return 0;
}

int mlx_quantize_array(const mlx_array in, mlx_array *out,
                       const mlx_array palette) {
  if (!out)
    return -1;
  float *in_buf = NULL;
  size_t in_elems = 0;
  int in_ndim = 0;
  int *in_shape = NULL;
  if (mlx_array_to_float_buffer(in, &in_buf, &in_elems, &in_ndim, &in_shape) !=
      0)
    return -1;
  float *pal_buf = NULL;
  size_t pal_elems = 0;
  int pal_ndim = 0;
  int *pal_shape = NULL;
  if (mlx_array_to_float_buffer(palette, &pal_buf, &pal_elems, &pal_ndim,
                                &pal_shape) != 0) {
    free(in_buf);
    if (in_shape)
      free(in_shape);
    return -1;
  }
  int c = 1;
  int h = 1, w = 1;
  if (in_ndim == 3) {
    h = in_shape[0];
    w = in_shape[1];
    c = in_shape[2];
  } else if (in_ndim == 2) {
    h = in_shape[0];
    w = in_shape[1];
    c = 1;
  } else {
    free(in_buf);
    free(pal_buf);
    if (in_shape)
      free(in_shape);
    if (pal_shape)
      free(pal_shape);
    return -1;
  }
  size_t npixels = (size_t)h * (size_t)w;
  int ncolors = (int)(pal_elems / (size_t)c);
  float *out_buf = (float *)malloc(sizeof(float) * npixels * (size_t)c);
  if (!out_buf) {
    free(in_buf);
    free(pal_buf);
    if (in_shape)
      free(in_shape);
    if (pal_shape)
      free(pal_shape);
    return -1;
  }
  quantize_pixels_float(in_buf, out_buf, npixels, c, pal_buf, ncolors);
  int shape_out[3];
  int ndim_out = 0;
  if (c == 1) {
    shape_out[0] = h;
    shape_out[1] = w;
    ndim_out = 2;
  } else {
    shape_out[0] = h;
    shape_out[1] = w;
    shape_out[2] = c;
    ndim_out = 3;
  }
  mlx_array out_arr =
      mlx_array_new_data(out_buf, shape_out, ndim_out, MLX_FLOAT32);
  *out = out_arr;
  free(in_buf);
  free(pal_buf);
  if (in_shape)
    free(in_shape);
  if (pal_shape)
    free(pal_shape);
  free(out_buf);
  return 0;
}

int mlx_apply_well_mask_array(const mlx_array facies, mlx_array *out,
                              const mlx_array mask, const mlx_array well) {
  if (!out)
    return -1;
  float *fac_buf = NULL;
  size_t fac_elems = 0;
  int fac_ndim = 0;
  int *fac_shape = NULL;
  if (mlx_array_to_float_buffer(facies, &fac_buf, &fac_elems, &fac_ndim,
                                &fac_shape) != 0)
    return -1;
  bool ok_mask = false;
  if (_mlx_array_is_available(&ok_mask, mask) != 0 || !ok_mask) {
    return -1;
  }
  int m_dtype = mlx_array_dtype(mask);
  size_t mask_elems = mlx_array_size(mask);
  unsigned char *mask_buf = (unsigned char *)malloc(mask_elems);
  if (!mask_buf) {
    free(fac_buf);
    if (fac_shape)
      free(fac_shape);
    return -1;
  }
  if (m_dtype == MLX_BOOL) {
    const bool *mb = mlx_array_data_bool(mask);
    for (size_t i = 0; i < mask_elems; ++i)
      mask_buf[i] = mb[i] ? 1 : 0;
  } else if (m_dtype == MLX_UINT8) {
    const uint8_t *mb = mlx_array_data_uint8(mask);
    for (size_t i = 0; i < mask_elems; ++i)
      mask_buf[i] = mb[i];
  } else {
    free(fac_buf);
    if (fac_shape)
      free(fac_shape);
    free(mask_buf);
    return -1;
  }
  float *well_buf = NULL;
  size_t well_elems = 0;
  int well_ndim = 0;
  int *well_shape = NULL;
  if (mlx_array_to_float_buffer(well, &well_buf, &well_elems, &well_ndim,
                                &well_shape) != 0) {
    free(fac_buf);
    if (fac_shape)
      free(fac_shape);
    free(mask_buf);
    return -1;
  }
  int h = 1, w = 1, c = 1, wc = 1;
  if (fac_ndim == 3) {
    h = fac_shape[0];
    w = fac_shape[1];
    c = fac_shape[2];
  } else if (fac_ndim == 2) {
    h = fac_shape[0];
    w = fac_shape[1];
    c = 1;
  }
  if (well_ndim == 3) {
    wc = well_shape[2];
  } else if (well_ndim == 2) {
    wc = 1;
  }
  float *out_buf = (float *)malloc(sizeof(float) * fac_elems);
  if (!out_buf) {
    free(fac_buf);
    if (fac_shape)
      free(fac_shape);
    free(mask_buf);
    free(well_buf);
    if (well_shape)
      free(well_shape);
    return -1;
  }
  apply_well_mask_cpu(fac_buf, out_buf, h, w, c, mask_buf, well_buf, wc);
  int ndim_out = fac_ndim;
  const int *shape_in = mlx_array_shape(facies);
  mlx_array out_arr =
      mlx_array_new_data(out_buf, shape_in, ndim_out, MLX_FLOAT32);
  *out = out_arr;
  free(fac_buf);
  if (fac_shape)
    free(fac_shape);
  free(mask_buf);
  free(well_buf);
  if (well_shape)
    free(well_shape);
  free(out_buf);
  return 0;
}

/* End moved from utils_extra.c */
