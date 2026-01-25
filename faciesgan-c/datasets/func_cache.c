#include "func_cache.h"
#include "io/npz_create.h"
#include "io/npz_unzip.h"
#include "utils.h"
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <math.h>
#include <mlx/c/array.h>
#include <mlx/c/device.h>
#include <mlx/c/io.h>
#include <mlx/c/stream.h>

static uint64_t fnv1a_hash(const char *s) {
  uint64_t h = 14695981039346656037ULL;
  const unsigned char *p = (const unsigned char *)s;
  while (*p) {
    h ^= (uint64_t)(*p++);
    h *= 1099511628211ULL;
  }
  return h;
}

int ensure_function_cache(const char *input_path, const char *cache_dir,
                          int desired_num, const struct DatasetScale *scales,
                          int n_scales, int num_img_channels, int use_wells,
                          int use_seismic, int manual_seed, char *out_cache_npz,
                          size_t out_len, int *out_num_samples) {
  if (!input_path || !cache_dir || !out_cache_npz || !out_num_samples)
    return -1;

  char keybuf[1024];
  /* include explicit scales description in cache key so callers that pass
   * non-standard scale lists produce distinct caches */
  int off = 0;
  off += snprintf(keybuf + off, sizeof(keybuf) - off,
                  "%s|nscales=%d|ch=%d|w=%d|s=%d|seed=%d", input_path, n_scales,
                  num_img_channels, use_wells, use_seismic, manual_seed);
  for (int i = 0; i < n_scales; ++i) {
    off += snprintf(keybuf + off, sizeof(keybuf) - off, "|h%d=%d", i,
                    scales ? scales[i].height : 0);
    if (off >= (int)sizeof(keybuf) - 64)
      break;
  }
  uint64_t key = fnv1a_hash(keybuf);
  char cache_npz[PATH_MAX];
  snprintf(cache_npz, PATH_MAX, "%s/func_cache_%016llx.npz", cache_dir,
           (unsigned long long)key);

  /* If cache exists, probe members to count available samples */
  if (access(cache_npz, F_OK) == 0) {
    int available = 0;
    for (int i = 0; i < desired_num; ++i) {
      char member[64];
      snprintf(member, sizeof(member), "sample_%d/facies_0.npy", i);
      mlx_io_reader reader;
      if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0) {
        available++;
        mlx_io_reader_free(reader);
      } else
        break;
    }
    *out_num_samples =
        available > 0 ? (available < desired_num ? available : desired_num) : 0;
    strncpy(out_cache_npz, cache_npz, out_len);
    return 0;
  }

  /* Otherwise generate using native C generator into temporary sample_* dirs */
  int num_samples = desired_num > 0 ? desired_num : 1;

  /* First try external gen_cache tool if available (built separately).
   * Command: gen_cache <out_dir> <num_samples> <num_scales> <crop_size>
   * <channels> If it fails, fall back to in-tree `generate_pyramids_cache`
   * implementation. */
  int num_scales = n_scales > 0 ? n_scales : 1;
  /* debug: log incoming scales to help diagnose generation issues */
  fprintf(stdout, "ensure_function_cache: num_scales=%d\n", num_scales);
  for (int _i = 0; _i < num_scales && _i < 8; ++_i) {
    int h = scales ? scales[_i].height : 0;
    fprintf(stdout, "  scale[%d].height=%d\n", _i, h);
  }
  /* Generate using the full in-tree C generator (deterministic, matches
   * Python behavior). */
  /* Try Python/MLX cache generator first to match Python outputs exactly. */
  int python_ok = 0;
  {
    char pycmd[PATH_MAX * 2];
    /* Prefer repository virtualenv python if present, otherwise fall back to
     * system python3. The tools/ script is located relative to repo root. */
    const char *venv_py = ".venv/bin/python";
    if (access(venv_py, X_OK) == 0) {
      snprintf(pycmd, sizeof(pycmd),
               "%s tools/gen_mlx_cache.py --output '%s' --num-pyramids %d",
               venv_py, cache_npz, num_samples);
    } else {
      snprintf(pycmd, sizeof(pycmd),
               "python3 tools/gen_mlx_cache.py --output '%s' --num-pyramids %d",
               cache_npz, num_samples);
    }
    int rc = system(pycmd);
    if (rc == 0 && access(cache_npz, F_OK) == 0) {
      python_ok = 1;
      fprintf(stdout,
              "ensure_function_cache: Python cache generator produced %s\n",
              cache_npz);
    } else {
      fprintf(stdout,
              "ensure_function_cache: Python cache generator failed or not "
              "available (rc=%d), falling back to C generator\n",
              rc);
    }
  }

  if (!python_ok) {
    /* derive stop_scale and crop_size for the legacy generator helper */
    int derived_stop = num_scales - 1;
    int derived_crop =
        (scales && num_scales > 0) ? scales[num_scales - 1].height : 0;
    int grc = generate_pyramids_cache(
        input_path, cache_dir, num_samples, derived_stop, derived_crop,
        num_img_channels, use_wells ? 1 : 0, use_seismic ? 1 : 0, manual_seed);
    if (grc != 0) {
      fprintf(stderr, "generate_pyramids_cache failed rc=%d\n", grc);
      return grc;
    }
  }

  /* If Python generator produced the NPZ, return early (no filesystem
   * packaging required). */
  if (python_ok) {
    *out_num_samples = num_samples;
    strncpy(out_cache_npz, cache_npz, out_len);
    return 0;
  }

  /* Package generated files into .npz */
  int idx = 0;
  int per_sample = num_scales;
  if (use_wells)
    per_sample += num_scales;
  if (use_seismic)
    per_sample += num_scales;
  int n_members = num_samples * per_sample + 1;
  char **names = (char **)malloc(sizeof(char *) * n_members);
  void **bufs = (void **)malloc(sizeof(void *) * n_members);
  size_t *sizes = (size_t *)malloc(sizeof(size_t) * n_members);
  if (!names || !bufs || !sizes) {
    fprintf(stderr, "Out of memory creating packaging arrays\n");
    return -1;
  }

  for (int si = 0; si < num_samples; ++si) {
    char sample_dir[PATH_MAX];
    snprintf(sample_dir, PATH_MAX, "%s/sample_%d", cache_dir, si);
    for (int sc = 0; sc < num_scales; ++sc) {
      char fname[PATH_MAX];
      /* facies */
      snprintf(fname, PATH_MAX, "%s/facies_%d.npy", sample_dir, sc);
      FILE *f = fopen(fname, "rb");
      if (!f) {
        fprintf(stderr, "Failed to open %s\n", fname);
        for (int j = 0; j < idx; ++j) {
          free(names[j]);
          free(bufs[j]);
        }
        free(names);
        free(bufs);
        free(sizes);
        return -1;
      }
      fseek(f, 0, SEEK_END);
      long fsize = ftell(f);
      fseek(f, 0, SEEK_SET);
      void *buf = malloc((size_t)fsize);
      if (!buf) {
        fclose(f);
        fprintf(stderr, "Out of memory reading %s\n", fname);
        for (int j = 0; j < idx; ++j) {
          free(names[j]);
          free(bufs[j]);
        }
        free(names);
        free(bufs);
        free(sizes);
        return -1;
      }
      if (fread(buf, 1, (size_t)fsize, f) != (size_t)fsize) {
        fclose(f);
        free(buf);
        fprintf(stderr, "Failed to read %s\n", fname);
        for (int j = 0; j < idx; ++j) {
          free(names[j]);
          free(bufs[j]);
        }
        free(names);
        free(bufs);
        free(sizes);
        return -1;
      }
      fclose(f);
      char *member_name = (char *)malloc(64);
      snprintf(member_name, 64, "sample_%d/facies_%d.npy", si, sc);
      names[idx] = member_name;
      bufs[idx] = buf;
      sizes[idx] = (size_t)fsize;
      idx++;

      /* wells */
      if (use_wells) {
        snprintf(fname, PATH_MAX, "%s/wells_%d.npy", sample_dir, sc);
        FILE *fw = fopen(fname, "rb");
        if (fw) {
          fseek(fw, 0, SEEK_END);
          long fsizew = ftell(fw);
          fseek(fw, 0, SEEK_SET);
          void *bufw = malloc((size_t)fsizew);
          if (bufw) {
            if (fread(bufw, 1, (size_t)fsizew, fw) == (size_t)fsizew) {
              char *mname = (char *)malloc(64);
              snprintf(mname, 64, "sample_%d/wells_%d.npy", si, sc);
              names[idx] = mname;
              bufs[idx] = bufw;
              sizes[idx] = (size_t)fsizew;
              idx++;
            } else {
              free(bufw);
            }
          }
          fclose(fw);
        }
      }

      /* seismic */
      if (use_seismic) {
        snprintf(fname, PATH_MAX, "%s/seismic_%d.npy", sample_dir, sc);
        FILE *fs = fopen(fname, "rb");
        if (fs) {
          fseek(fs, 0, SEEK_END);
          long fsizes = ftell(fs);
          fseek(fs, 0, SEEK_SET);
          void *bufs_ = malloc((size_t)fsizes);
          if (bufs_) {
            if (fread(bufs_, 1, (size_t)fsizes, fs) == (size_t)fsizes) {
              char *mname = (char *)malloc(64);
              snprintf(mname, 64, "sample_%d/seismic_%d.npy", si, sc);
              names[idx] = mname;
              bufs[idx] = bufs_;
              sizes[idx] = (size_t)fsizes;
              idx++;
            } else {
              free(bufs_);
            }
          }
          fclose(fs);
        }
      }
    }
  }

  names[idx] = strdup("meta.json");
  bufs[idx] = strdup("{}");
  sizes[idx] = strlen((const char *)bufs[idx]);

  if (npz_create_from_memory(cache_npz, (const char **)names,
                             (const void **)bufs, sizes, idx + 1) != 0) {
    fprintf(stderr, "npz_create_from_memory failed\n");
    for (int j = 0; j <= idx; ++j) {
      free(names[j]);
      free(bufs[j]);
    }
    free(names);
    free(bufs);
    free(sizes);
    return -1;
  }

  for (int j = 0; j <= idx; ++j) {
    free(names[j]);
    free(bufs[j]);
  }
  free(names);
  free(bufs);
  free(sizes);

  /* remove temporary sample_* dirs */
  for (int si = 0; si < num_samples; ++si) {
    for (int sc = 0; sc < num_scales; ++sc) {
      char fname[PATH_MAX];
      snprintf(fname, PATH_MAX, "%s/sample_%d/facies_%d.npy", cache_dir, si,
               sc);
      remove(fname);
    }
    char sample_dir[PATH_MAX];
    snprintf(sample_dir, PATH_MAX, "%s/sample_%d", cache_dir, si);
    rmdir(sample_dir);
  }

  *out_num_samples = num_samples;
  strncpy(out_cache_npz, cache_npz, out_len);
  return 0;
}

// Helper to ensure saving uses the CPU backend to avoid triggering GPU/Metal
// evaluation during cache generation.
static int save_on_cpu(const char *fname, mlx_array a) {
  mlx_device old_dev;
  if (mlx_get_default_device(&old_dev) != 0) {
    return mlx_save(fname, a);
  }
  mlx_device cpu = mlx_device_new_type(MLX_CPU, 0);
  mlx_set_default_device(cpu);
  int rc = mlx_save(fname, a);
  mlx_set_default_device(old_dev);
  mlx_device_free(cpu);
  return rc;
}

int generate_pyramids_cache(const char *input_path, const char *cache_dir,
                            int num_samples, int stop_scale, int crop_size,
                            int num_img_channels, int use_wells,
                            int use_seismic, int seed) {
  // List facies image files
  char **files = NULL;
  int count = 0;
  if (datasets_list_image_files(input_path, "facies", &files, &count) != 0 ||
      count == 0) {
    if (files) {
      for (int i = 0; i < count; ++i)
        free(files[i]);
      free(files);
    }
    return -1;
  }

  // Ensure cache dir exists
  struct stat st = {0};
  if (stat(cache_dir, &st) != 0) {
    if (mkdir(cache_dir, 0755) != 0 && errno != EEXIST)
      return -1;
  }

  /* Precompute scaling parameters used for all generators so seismic/wells
   * sections can reuse the same logic. Default min_size matches Python
   * defaults (12). */
  int min_size = 12;
  double scale_factor = 1.0;
  if (stop_scale > 0) {
    scale_factor =
        pow((double)min_size / (double)crop_size, 1.0 / (double)stop_scale);
  }

  // For each sample up to num_samples, load image, create scales, save .npy per
  // scale
  for (int si = 0; si < num_samples && si < count; ++si) {
    const char *img_path = files[si];
    // Build sample dir
    char sample_dir[PATH_MAX];
    snprintf(sample_dir, PATH_MAX, "%s/sample_%d", cache_dir, si);
    if (stat(sample_dir, &st) != 0) {
      if (mkdir(sample_dir, 0755) != 0 && errno != EEXIST)
        return -1;
    }

    int w, h, comp;
    unsigned char *data = stbi_load(img_path, &w, &h, &comp, num_img_channels);
    if (!data) {
      fprintf(stderr, "stbi_load failed for %s\n", img_path);
      return -1;
    }

    /* Create scales using exponential scaling to match Python's
     * `generate_scales` behavior (min_size default = 12). */
    for (int scale = 0; scale <= stop_scale; ++scale) {
      double scale_s = pow(scale_factor, (double)(stop_scale - scale));
      double base = (double)crop_size * scale_s;
      int target = (int)round(base);
      if (target % 2 != 0)
        target += 1;
      if (target <= 0)
        target = min_size;
      unsigned char *resized =
          malloc((size_t)target * target * num_img_channels);
      if (!resized) {
        stbi_image_free(data);
        return -1;
      }
      // simple bilinear resize (avoids dependency on stb_image_resize2.h)
      {
        int iw = w, ih = h, ow = target, oh = target, c = num_img_channels;
        for (int oy = 0; oy < oh; ++oy) {
          float sy = (oy + 0.5f) * ((float)ih / (float)oh) - 0.5f;
          int y0 = (int)floorf(sy);
          int y1 = y0 + 1;
          float wy = sy - y0;
          if (y0 < 0)
            y0 = 0;
          if (y1 < 0)
            y1 = 0;
          if (y0 >= ih)
            y0 = ih - 1;
          if (y1 >= ih)
            y1 = ih - 1;
          for (int ox = 0; ox < ow; ++ox) {
            float sx = (ox + 0.5f) * ((float)iw / (float)ow) - 0.5f;
            int x0 = (int)floorf(sx);
            int x1 = x0 + 1;
            float wx = sx - x0;
            if (x0 < 0)
              x0 = 0;
            if (x1 < 0)
              x1 = 0;
            if (x0 >= iw)
              x0 = iw - 1;
            if (x1 >= iw)
              x1 = iw - 1;
            for (int ch = 0; ch < c; ++ch) {
              unsigned char p00 = data[(y0 * iw + x0) * c + ch];
              unsigned char p01 = data[(y0 * iw + x1) * c + ch];
              unsigned char p10 = data[(y1 * iw + x0) * c + ch];
              unsigned char p11 = data[(y1 * iw + x1) * c + ch];
              float v0 = p00 * (1.0f - wx) + p01 * wx;
              float v1 = p10 * (1.0f - wx) + p11 * wx;
              float v = v0 * (1.0f - wy) + v1 * wy;
              int vi = (int)(v + 0.5f);
              if (vi < 0)
                vi = 0;
              if (vi > 255)
                vi = 255;
              resized[(oy * ow + ox) * c + ch] = (unsigned char)vi;
            }
          }
        }
      }

      // Convert to float32 in [-1,1]
      size_t nelem = (size_t)target * target * num_img_channels;
      float *fdata = malloc(nelem * sizeof(float));
      if (!fdata) {
        free(resized);
        stbi_image_free(data);
        return -1;
      }
      for (size_t i = 0; i < nelem; ++i)
        fdata[i] = ((float)resized[i] / 127.5f) - 1.0f;

      // Create mlx_array and save
      int shape[3] = {target, target, num_img_channels};
      mlx_array a = mlx_array_new();
      mlx_stream s = mlx_default_cpu_stream_new();
      if (mlx_array_set_data(&a, fdata, shape, 3, MLX_FLOAT32) != 0) {
        mlx_stream_free(s);
        free(fdata);
        free(resized);
        stbi_image_free(data);
        return -1;
      }

      char fname[PATH_MAX];
      /* Write files so that facies_0 is the smallest (coarsest) level
       * to match Python naming: out_idx = stop_scale - scale */
      snprintf(fname, PATH_MAX, "%s/facies_%d.npy", sample_dir, scale);
      if (save_on_cpu(fname, a) != 0) {
        mlx_array_free(a);
        mlx_stream_free(s);
        free(fdata);
        free(resized);
        stbi_image_free(data);
        return -1;
      }

      mlx_array_free(a);
      mlx_stream_free(s);
      free(fdata);
      free(resized);
      (void)0;
    }

    stbi_image_free(data);
  }

  for (int i = 0; i < count; ++i)
    free(files[i]);
  free(files);
  // Optionally generate seismic pyramids using the same resizer logic
  if (use_seismic) {
    char **sfiles = NULL;
    int scount = 0;
    if (datasets_list_image_files(input_path, "seismic", &sfiles, &scount) ==
            0 &&
        scount > 0) {
      for (int si = 0; si < num_samples && si < scount; ++si) {
        const char *img_path = sfiles[si];
        char sample_dir[PATH_MAX];
        snprintf(sample_dir, PATH_MAX, "%s/sample_%d", cache_dir, si);
        // reuse sample_dir (already created above)

        int w, h, comp;
        unsigned char *data =
            stbi_load(img_path, &w, &h, &comp, num_img_channels);
        if (!data) {
          fprintf(stderr, "stbi_load failed for seismic %s\n", img_path);
          // continue to next sample rather than abort
          continue;
        }

        /* Generate seismic scales using same exponential scaling as facies */
        for (int scale = 0; scale <= stop_scale; ++scale) {
          double scale_s = pow(scale_factor, (double)(stop_scale - scale));
          double base = (double)crop_size * scale_s;
          int target = (int)round(base);
          if (target % 2 != 0)
            target += 1;
          if (target <= 0)
            target = min_size;
          unsigned char *resized =
              malloc((size_t)target * target * num_img_channels);
          if (!resized) {
            stbi_image_free(data);
            return -1;
          }
          int iw = w, ih = h, ow = target, oh = target, c = num_img_channels;
          for (int oy = 0; oy < oh; ++oy) {
            float sy = (oy + 0.5f) * ((float)ih / (float)oh) - 0.5f;
            int y0 = (int)floorf(sy);
            int y1 = y0 + 1;
            float wy = sy - y0;
            if (y0 < 0)
              y0 = 0;
            if (y1 < 0)
              y1 = 0;
            if (y0 >= ih)
              y0 = ih - 1;
            if (y1 >= ih)
              y1 = ih - 1;
            for (int ox = 0; ox < ow; ++ox) {
              float sx = (ox + 0.5f) * ((float)iw / (float)ow) - 0.5f;
              int x0 = (int)floorf(sx);
              int x1 = x0 + 1;
              float wx = sx - x0;
              if (x0 < 0)
                x0 = 0;
              if (x1 < 0)
                x1 = 0;
              if (x0 >= iw)
                x0 = iw - 1;
              if (x1 >= iw)
                x1 = iw - 1;
              for (int ch = 0; ch < c; ++ch) {
                unsigned char p00 = data[(y0 * iw + x0) * c + ch];
                unsigned char p01 = data[(y0 * iw + x1) * c + ch];
                unsigned char p10 = data[(y1 * iw + x0) * c + ch];
                unsigned char p11 = data[(y1 * iw + x1) * c + ch];
                float v0 = p00 * (1.0f - wx) + p01 * wx;
                float v1 = p10 * (1.0f - wx) + p11 * wx;
                float v = v0 * (1.0f - wy) + v1 * wy;
                int vi = (int)(v + 0.5f);
                if (vi < 0)
                  vi = 0;
                if (vi > 255)
                  vi = 255;
                resized[(oy * ow + ox) * c + ch] = (unsigned char)vi;
              }
            }
          }

          size_t nelem = (size_t)target * target * num_img_channels;
          float *fdata = malloc(nelem * sizeof(float));
          if (!fdata) {
            free(resized);
            stbi_image_free(data);
            return -1;
          }
          for (size_t i = 0; i < nelem; ++i)
            fdata[i] = ((float)resized[i] / 127.5f) - 1.0f;

          int shape[3] = {target, target, num_img_channels};
          mlx_array a = mlx_array_new();
          mlx_stream s = mlx_default_cpu_stream_new();
          if (mlx_array_set_data(&a, fdata, shape, 3, MLX_FLOAT32) != 0) {
            mlx_stream_free(s);
            free(fdata);
            free(resized);
            stbi_image_free(data);
            return -1;
          }

          char fname[PATH_MAX];
          snprintf(fname, PATH_MAX, "%s/seismic_%d.npy", sample_dir, scale);
          if (save_on_cpu(fname, a) != 0) {
            mlx_array_free(a);
            mlx_stream_free(s);
            free(fdata);
            free(resized);
            stbi_image_free(data);
            return -1;
          }

          mlx_array_free(a);
          mlx_stream_free(s);
          free(fdata);
          free(resized);
          (void)0;
        }

        stbi_image_free(data);
      }

      for (int i = 0; i < scount; ++i)
        free(sfiles[i]);
      free(sfiles);
    }
  }

  // Optionally generate wells pyramids
  if (use_wells) {
    char **wfiles = NULL;
    int wcount = 0;
    if (datasets_list_image_files(input_path, "wells", &wfiles, &wcount) == 0 &&
        wcount > 0) {
      for (int si = 0; si < num_samples && si < wcount; ++si) {
        const char *img_path = wfiles[si];
        char sample_dir[PATH_MAX];
        snprintf(sample_dir, PATH_MAX, "%s/sample_%d", cache_dir, si);

        int w, h, comp;
        /* Load wells with the configured number of image channels so C
         * generator matches Python behavior when wells are RGB (3-ch). */
        unsigned char *data =
            stbi_load(img_path, &w, &h, &comp, num_img_channels);
        if (!data) {
          fprintf(stderr, "stbi_load failed for wells %s\n", img_path);
          continue;
        }

        /* Generate wells scales using same exponential scaling as facies */
        for (int scale = 0; scale <= stop_scale; ++scale) {
          double scale_s = pow(scale_factor, (double)(stop_scale - scale));
          double base = (double)crop_size * scale_s;
          int target = (int)round(base);
          if (target % 2 != 0)
            target += 1;
          if (target <= 0)
            target = min_size;
          unsigned char *resized =
              malloc((size_t)target * target * num_img_channels);
          if (!resized) {
            stbi_image_free(data);
            return -1;
          }
          int iw = w, ih = h, ow = target, oh = target, c = num_img_channels;
          for (int oy = 0; oy < oh; ++oy) {
            float sy = (oy + 0.5f) * ((float)ih / (float)oh) - 0.5f;
            int y0 = (int)floorf(sy);
            int y1 = y0 + 1;
            float wy = sy - y0;
            if (y0 < 0)
              y0 = 0;
            if (y1 < 0)
              y1 = 0;
            if (y0 >= ih)
              y0 = ih - 1;
            if (y1 >= ih)
              y1 = ih - 1;
            for (int ox = 0; ox < ow; ++ox) {
              float sx = (ox + 0.5f) * ((float)iw / (float)ow) - 0.5f;
              int x0 = (int)floorf(sx);
              int x1 = x0 + 1;
              float wx = sx - x0;
              if (x0 < 0)
                x0 = 0;
              if (x1 < 0)
                x1 = 0;
              if (x0 >= iw)
                x0 = iw - 1;
              if (x1 >= iw)
                x1 = iw - 1;
              for (int ch = 0; ch < c; ++ch) {
                unsigned char p00 = data[(y0 * iw + x0) * c + ch];
                unsigned char p01 = data[(y0 * iw + x1) * c + ch];
                unsigned char p10 = data[(y1 * iw + x0) * c + ch];
                unsigned char p11 = data[(y1 * iw + x1) * c + ch];
                float v0 = p00 * (1.0f - wx) + p01 * wx;
                float v1 = p10 * (1.0f - wx) + p11 * wx;
                float v = v0 * (1.0f - wy) + v1 * wy;
                int vi = (int)(v + 0.5f);
                if (vi < 0)
                  vi = 0;
                if (vi > 255)
                  vi = 255;
                resized[(oy * ow + ox) * c + ch] = (unsigned char)vi;
              }
            }
          }

          size_t nelem = (size_t)target * target * num_img_channels;
          float *fdata = malloc(nelem * sizeof(float));
          if (!fdata) {
            free(resized);
            stbi_image_free(data);
            return -1;
          }
          for (size_t i = 0; i < nelem; ++i)
            fdata[i] = ((float)resized[i] / 127.5f) - 1.0f;

          int shape[3] = {target, target, num_img_channels};
          mlx_array a = mlx_array_new();
          mlx_stream s = mlx_default_cpu_stream_new();
          if (mlx_array_set_data(&a, fdata, shape, 3, MLX_FLOAT32) != 0) {
            mlx_stream_free(s);
            free(fdata);
            free(resized);
            stbi_image_free(data);
            return -1;
          }

          char fname[PATH_MAX];
          snprintf(fname, PATH_MAX, "%s/wells_%d.npy", sample_dir, scale);
          if (save_on_cpu(fname, a) != 0) {
            mlx_array_free(a);
            mlx_stream_free(s);
            free(fdata);
            free(resized);
            stbi_image_free(data);
            return -1;
          }

          mlx_array_free(a);
          mlx_stream_free(s);
          free(fdata);
          free(resized);
          (void)0;
        }

        stbi_image_free(data);
      }

      for (int i = 0; i < wcount; ++i)
        free(wfiles[i]);
      free(wfiles);
    }
  }

  return 0;
}
