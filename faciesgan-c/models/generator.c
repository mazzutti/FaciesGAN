#include "generator.h"
#include "../trainning/mlx_compat.h"
#include "custom_layer.h"
#include <mlx/c/vector.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Safe free helper: frees internal MLX storage if present and NULLs the ctx
 * field on the caller-side struct so subsequent callers don't double-free.
 */
static inline void mlx_array_safe_free(mlx_array *a) {
  if (!a)
    return;
  if (a->ctx) {
    mlx_array_free(*a);
    a->ctx = NULL;
  }
}

/* Helper: perform elementwise add with defensive central padding of the
 * smaller operand to match spatial dims (axes 1 and 2) before calling
 * `mlx_add`. Returns 0 on success, non-zero on failure. */
static int safe_add(mlx_array *out, mlx_array a, mlx_array b, mlx_stream s) {
  const int *ash = mlx_array_shape(a);
  const int *bsh = mlx_array_shape(b);
  mlx_array a_copy = a;
  mlx_array b_copy = b;
  int a_padded = 0;
  int b_padded = 0;
  /* If shapes available and spatial dims differ, pad the smaller one */
  if (ash && bsh && (ash[1] != bsh[1] || ash[2] != bsh[2])) {
    int dh = bsh[1] - ash[1];
    int dw = bsh[2] - ash[2];
    if (dh >= 0 && dw >= 0) {
      int low_h = dh / 2;
      int high_h = dh - low_h;
      int low_w = dw / 2;
      int high_w = dw - low_w;
      int axes_pad[2] = {1, 2};
      int low_pad[2] = {low_h, low_w};
      int high_pad[2] = {high_h, high_w};
      mlx_array zero = mlx_array_new_float(0.0f);
      mlx_array padded = mlx_array_new();
      if (mlx_pad(&padded, a, axes_pad, 2, low_pad, 2, high_pad, 2, zero,
                  "constant", s) == 0) {
        a_copy = padded;
        a_padded = 1;
      } else {
        mlx_array_safe_free(&padded);
      }
      mlx_array_safe_free(&zero);
    } else if (dh <= 0 && dw <= 0) {
      int pdh = -dh;
      int pdw = -dw;
      int low_h = pdh / 2;
      int high_h = pdh - low_h;
      int low_w = pdw / 2;
      int high_w = pdw - low_w;
      int axes_pad[2] = {1, 2};
      int low_pad[2] = {low_h, low_w};
      int high_pad[2] = {high_h, high_w};
      mlx_array zero = mlx_array_new_float(0.0f);
      mlx_array padded = mlx_array_new();
      if (mlx_pad(&padded, b, axes_pad, 2, low_pad, 2, high_pad, 2, zero,
                  "constant", s) == 0) {
        b_copy = padded;
        b_padded = 1;
      } else {
        mlx_array_safe_free(&padded);
      }
      mlx_array_safe_free(&zero);
    }
  }
  /* To avoid aliasing/in-place updates from the backend which can lead to
   * callers freeing the same underlying storage twice, make local copies of
   * the operands (after any padding) and perform the add on those copies. */
  mlx_array a_tmp = mlx_array_new();
  mlx_array b_tmp = mlx_array_new();
  int r1 = mlx_copy(&a_tmp, a_copy, s);
  int r2 = mlx_copy(&b_tmp, b_copy, s);
  if (r1 != 0 || r2 != 0) {
    if (a_tmp.ctx)
      mlx_array_safe_free(&a_tmp);
    if (b_tmp.ctx)
      mlx_array_safe_free(&b_tmp);
    if (a_padded)
      mlx_array_safe_free(&a_copy);
    if (b_padded)
      mlx_array_safe_free(&b_copy);
    return -1;
  }

  int rc = mlx_add(out, a_tmp, b_tmp, s);

  /* ensure any async work on the provided stream completed before freeing
     temporaries which may be referenced by backend kernels */
  if (s.ctx)
    mlx_synchronize(s);

  /* free the temporary copies */
  if (a_tmp.ctx)
    mlx_array_safe_free(&a_tmp);
  if (b_tmp.ctx)
    mlx_array_safe_free(&b_tmp);

  if (a_padded)
    mlx_array_safe_free(&a_copy);
  if (b_padded)
    mlx_array_safe_free(&b_copy);

  return rc;
}

#include <string.h>

/* Helper: detect if an mlx_array likely originates from the caller-owned list
 * (z_list) or the provided in_noise. We compare the internal ctx pointer which
 * is used elsewhere as a proxy for ownership. If true, avoid freeing it here.
 */
static int mlx_array_is_in_list(mlx_array a, const mlx_array *list, int n,
                                mlx_array in_noise) {
  if (!a.ctx)
    return 0;
  if (in_noise.ctx && a.ctx == in_noise.ctx)
    return 1;
  for (int i = 0; i < n; ++i) {
    if (list && list[i].ctx && a.ctx == list[i].ctx)
      return 1;
  }
  return 0;
}

struct ScaleModule {
  int is_spade;
  MLXSPADEGenerator *spade;
  MLXConvBlock *head;
  MLXConvBlock **body;
  int n_body;
  mlx_array *tail_conv; /* pointer to mlx_array weights */
};

struct MLXGenerator {
  int num_layer;
  int kernel_size;
  int padding_size;
  int input_channels;
  int output_channels;
  MLXScaleModule **gens; /* array of ScaleModule* (stored as MLXScaleModule*) */
  int n_gens;
  int zero_padding;
  int full_zero_padding;
  int cond_channels;
  int has_cond_channels;
  MLXColorQuantization *color_quant;
  int eval_mode;
  mlx_vector_int spade_scales;
};

MLXGenerator *mlx_generator_create(int num_layer, int kernel_size,
                                   int padding_size, int input_channels,
                                   int output_channels) {
  MLXGenerator *m = (MLXGenerator *)malloc(sizeof(MLXGenerator));
  if (!m)
    return NULL;
  m->num_layer = num_layer;
  m->kernel_size = kernel_size;
  m->padding_size = padding_size;
  m->input_channels = input_channels;
  m->output_channels = output_channels;
  m->gens = NULL;
  m->n_gens = 0;
  m->cond_channels = input_channels - output_channels;
  m->has_cond_channels = m->cond_channels > 0 ? 1 : 0;
  m->zero_padding = num_layer * (kernel_size / 2);
  m->full_zero_padding = 2 * m->zero_padding;
  m->color_quant = mlx_colorquant_create(0.1f);
  m->eval_mode = 0;
  m->spade_scales = mlx_vector_int_new();
  return m;
}

void mlx_generator_free(MLXGenerator *m) {
  if (!m)
    return;
  if (m->spade_scales.ctx)
    mlx_vector_int_free(m->spade_scales);
  if (m->gens) {
    for (int i = 0; i < m->n_gens; ++i) {
      struct ScaleModule *s = (struct ScaleModule *)m->gens[i];
      if (!s)
        continue;
      if (s->is_spade) {
        if (s->spade)
          mlx_spadegen_free(s->spade);
      } else {
        if (s->head)
          mlx_convblock_free(s->head);
        if (s->body) {
          for (int j = 0; j < s->n_body; ++j)
            if (s->body[j])
              mlx_convblock_free(s->body[j]);
          free(s->body);
        }
        if (s->tail_conv) {
          mlx_array_safe_free(s->tail_conv);
          free(s->tail_conv);
        }
      }
      free(s);
    }
    free(m->gens);
  }
  if (m->color_quant)
    mlx_colorquant_free(m->color_quant);
  free(m);
}

int mlx_generator_create_scale(MLXGenerator *m, int scale, int num_features,
                               int min_num_features) {
  if (!m)
    return -1;
  struct ScaleModule *s =
      (struct ScaleModule *)calloc(1, sizeof(struct ScaleModule));
  if (!s)
    return -1;
  if (scale == 0) {
    s->is_spade = 1;
    /* Pass conditioning channel count (cond_channels) when available so
     * SPADE internals allocate weights matching the actual conditioning
     * tensor channels. Fall back to m->input_channels if cond not used. */
    int spade_in_ch =
        m->has_cond_channels ? m->cond_channels : m->input_channels;
    s->spade = mlx_spadegen_create(
        m->num_layer, m->kernel_size, m->padding_size, num_features,
        min_num_features, m->output_channels, spade_in_ch);
    s->head = NULL;
    s->body = NULL;
    s->n_body = 0;
    s->tail_conv = NULL;
  } else {
    s->is_spade = 0;
    s->spade = NULL;
    s->head = mlx_convblock_create(m->input_channels, num_features,
                                   m->kernel_size, m->padding_size, 1, 1);
    int body_count = (m->num_layer - 2) > 0 ? (m->num_layer - 2) : 0;
    s->n_body = body_count;
    if (body_count > 0) {
      s->body = (MLXConvBlock **)calloc(body_count, sizeof(MLXConvBlock *));
      int current_features = num_features;
      for (int i = 0; i < body_count; ++i) {
        int denom = (1 << (i + 1));
        int block_features = num_features / denom;
        if (block_features < min_num_features)
          block_features = min_num_features;
        int in_ch = current_features;
        int out_ch = block_features;
        s->body[i] = mlx_convblock_create(in_ch, out_ch, m->kernel_size,
                                          m->padding_size, 1, 1);
        current_features = out_ch;
      }
      /* tail conv weights */
      size_t tail_count = (size_t)m->output_channels * m->kernel_size *
                          m->kernel_size * (size_t)current_features;
      float *tail_buf = (float *)calloc(tail_count, sizeof(float));
      if (tail_buf) {
        /* tail_conv weights in canonical MLX layout:
         * (out_ch=m->output_channels, KH, KW, in_ch=current_features) */
        int tshape[4] = {m->output_channels, m->kernel_size, m->kernel_size,
                         current_features};
        mlx_array tw = mlx_array_new_data(tail_buf, tshape, 4, MLX_FLOAT32);
        free(tail_buf);
        /* keep tail_conv weights in MLX layout: [C_in, KH, KW, C_out] */
        mlx_array *twptr = (mlx_array *)malloc(sizeof(mlx_array));
        if (twptr) {
          *twptr = tw;
          s->tail_conv = twptr;
        } else {
          mlx_array_free(tw);
        }
      }
    }
  }

  MLXScaleModule **tmp = (MLXScaleModule **)realloc(
      m->gens, (size_t)(m->n_gens + 1) * sizeof(MLXScaleModule *));
  if (!tmp) {
    /* cleanup */
    if (s->is_spade && s->spade)
      mlx_spadegen_free(s->spade);
    if (!s->is_spade) {
      if (s->head)
        mlx_convblock_free(s->head);
      if (s->body) {
        for (int j = 0; j < s->n_body; ++j)
          if (s->body[j])
            mlx_convblock_free(s->body[j]);
        free(s->body);
      }
      if (s->tail_conv) {
        mlx_array_free(*s->tail_conv);
        free(s->tail_conv);
      }
    }
    free(s);
    return -1;
  }
  m->gens = tmp;
  m->gens[m->n_gens] = (MLXScaleModule *)s;
  m->n_gens += 1;
  /* If scale was SPADE, record it */
  if (scale == 0) {
    mlx_vector_int_append_value(m->spade_scales, 0);
  }
  return 0;
}

/* SPADE scale helpers */
void mlx_generator_add_spade_scale(MLXGenerator *m, int scale) {
  if (!m)
    return;
  mlx_vector_int_append_value(m->spade_scales, scale);
}

int mlx_generator_has_spade_scale(MLXGenerator *m, int scale) {
  if (!m)
    return 0;
  size_t n = mlx_vector_int_size(m->spade_scales);
  for (size_t i = 0; i < n; ++i) {
    int val = 0;
    mlx_vector_int_get(&val, m->spade_scales, i);
    if (val == scale)
      return 1;
  }
  return 0;
}

size_t mlx_generator_spade_scales_count(MLXGenerator *m) {
  if (!m)
    return 0;
  return mlx_vector_int_size(m->spade_scales);
}

int mlx_generator_spade_scale_at(MLXGenerator *m, size_t idx) {
  if (!m)
    return -1;
  int val = -1;
  mlx_vector_int_get(&val, m->spade_scales, idx);
  return val;
}

/* Gens introspection */
int mlx_generator_get_n_gens(MLXGenerator *m) {
  if (!m)
    return 0;
  return m->n_gens;
}

MLXScaleModule *mlx_generator_get_gen_ptr(MLXGenerator *m, int index) {
  if (!m)
    return NULL;
  if (index < 0 || index >= m->n_gens)
    return NULL;
  return m->gens[index];
}

void mlx_generator_clear_spade_scales(MLXGenerator *m) {
  if (!m)
    return;
  if (m->spade_scales.ctx) {
    mlx_vector_int_free(m->spade_scales);
    m->spade_scales = mlx_vector_int_new();
  }
}

/* Accessor implementations */
int mlx_scale_is_spade(MLXGenerator *m, int index) {
  if (!m)
    return 0;
  if (index < 0 || index >= m->n_gens)
    return 0;
  MLXScaleModule *s = m->gens[index];
  if (!s)
    return 0;
  /* s is our internal ScaleModule casted to MLXScaleModule*; check the flag
   * by probing the spade pointer (non-NULL => spade module)
   */
  /* We can't dereference MLXScaleModule fields safely from outside, but
   * within this compilation unit we know layout: treat as struct ScaleModule.
   */
  struct ScaleModule *sm = (struct ScaleModule *)s;
  return sm->is_spade ? 1 : 0;
}

MLXSPADEGenerator *mlx_scale_get_spade(MLXGenerator *m, int index) {
  if (!m)
    return NULL;
  if (index < 0 || index >= m->n_gens)
    return NULL;
  struct ScaleModule *sm = (struct ScaleModule *)m->gens[index];
  if (!sm)
    return NULL;
  return sm->spade;
}

MLXConvBlock *mlx_scale_get_head(MLXGenerator *m, int index) {
  if (!m)
    return NULL;
  if (index < 0 || index >= m->n_gens)
    return NULL;
  struct ScaleModule *sm = (struct ScaleModule *)m->gens[index];
  if (!sm)
    return NULL;
  return sm->head;
}

int mlx_scale_get_body_count(MLXGenerator *m, int index) {
  if (!m)
    return 0;
  if (index < 0 || index >= m->n_gens)
    return 0;
  struct ScaleModule *sm = (struct ScaleModule *)m->gens[index];
  if (!sm)
    return 0;
  return sm->n_body;
}

MLXConvBlock *mlx_scale_get_body_at(MLXGenerator *m, int index,
                                    int body_index) {
  if (!m)
    return NULL;
  if (index < 0 || index >= m->n_gens)
    return NULL;
  struct ScaleModule *sm = (struct ScaleModule *)m->gens[index];
  if (!sm)
    return NULL;
  if (body_index < 0 || body_index >= sm->n_body)
    return NULL;
  return sm->body ? sm->body[body_index] : NULL;
}

int mlx_scale_has_tail_conv(MLXGenerator *m, int index) {
  if (!m)
    return 0;
  if (index < 0 || index >= m->n_gens)
    return 0;
  struct ScaleModule *sm = (struct ScaleModule *)m->gens[index];
  if (!sm)
    return 0;
  return sm->tail_conv ? 1 : 0;
}

mlx_array *mlx_scale_get_tail_conv(MLXGenerator *m, int index) {
  if (!m)
    return NULL;
  if (index < 0 || index >= m->n_gens)
    return NULL;
  struct ScaleModule *sm = (struct ScaleModule *)m->gens[index];
  if (!sm)
    return NULL;
  return sm->tail_conv;
}

/* Parameter collection implementation */
mlx_array **mlx_generator_get_parameters(MLXGenerator *m, int *out_count) {
  if (!m || !out_count)
    return NULL;
  int total = 0;
  for (int i = 0; i < m->n_gens; ++i) {
    struct ScaleModule *sm = (struct ScaleModule *)m->gens[i];
    if (!sm)
      continue;
    if (sm->is_spade && sm->spade) {
      int t = 0;
      mlx_array **tmp = mlx_spadegen_get_parameters(sm->spade, &t);
      if (tmp) {
        total += t;
        mlx_spadegen_free_parameters_list(tmp);
      }
    } else {
      mlx_array *hw = mlx_convblock_get_conv_weight(sm->head);
      if (hw)
        total++;
      if (sm->body) {
        for (int b = 0; b < sm->n_body; ++b) {
          mlx_array *bw = mlx_convblock_get_conv_weight(sm->body[b]);
          if (bw)
            total++;
        }
      }
      if (sm->tail_conv)
        total++;
    }
  }
  if (total == 0) {
    *out_count = 0;
    return NULL;
  }
  mlx_array **list = (mlx_array **)malloc(sizeof(mlx_array *) * total);
  if (!list) {
    *out_count = 0;
    return NULL;
  }
  /* initialize to NULL to avoid returning uninitialized pointers */
  memset(list, 0, sizeof(mlx_array *) * total);
  int idx = 0;
  for (int i = 0; i < m->n_gens; ++i) {
    struct ScaleModule *sm = (struct ScaleModule *)m->gens[i];
    if (!sm)
      continue;
    if (sm->is_spade && sm->spade) {
      int t = 0;
      mlx_array **tmp = mlx_spadegen_get_parameters(sm->spade, &t);
      if (tmp) {
        for (int j = 0; j < t; ++j) {
          list[idx] = tmp[j];
          idx++;
        }
        mlx_spadegen_free_parameters_list(tmp);
      }
    } else {
      mlx_array *hw = mlx_convblock_get_conv_weight(sm->head);
      if (hw) {
        list[idx] = hw;
        idx++;
      }
      if (sm->body) {
        for (int b = 0; b < sm->n_body; ++b) {
          mlx_array *bw = mlx_convblock_get_conv_weight(sm->body[b]);
          if (bw) {
            list[idx] = bw;
            idx++;
          }
        }
      }
      if (sm->tail_conv) {
        list[idx] = (mlx_array *)sm->tail_conv;
        idx++;
      }
    }
  }
  *out_count = idx;
  if (idx != total) {
    fprintf(stderr,
            "mlx_generator_get_parameters: warning, counted %d but filled %d\n",
            total, idx);
    /* defensive: ensure remaining slots are NULL */
    for (int z = idx; z < total; ++z)
      list[z] = NULL;
  }
  return list;
}

void mlx_generator_free_parameters_list(mlx_array **list) {
  if (list)
    free(list);
}

/* Full forward implementation ported from Python `models/mlx/generator.py`.
 * Implements progressive multi-scale synthesis:
 *  - Upsample current facie to match target z spatial dims
 *  - Build z_in (split noise/cond, scale noise, concat)
 *  - Pad tiled facie and add into conditioning
 *  - Execute per-scale module (SPADE generator for scale 0, conv-block pyramid
 * otherwise)
 *  - Add module output to facie and continue
 */
mlx_array_t mlx_generator_forward(MLXGenerator *m, const mlx_array *z_list,
                                  int z_count, const float *amp, int amp_count,
                                  mlx_array_t in_noise, int start_scale,
                                  int stop_scale) {
  if (!m)
    return in_noise;
  if (z_count <= 0)
    return in_noise;

  if (start_scale < 0)
    start_scale = 0;
  if (stop_scale < 0)
    stop_scale = m->n_gens > 0 ? (m->n_gens - 1) : (z_count - 1);
  if (stop_scale >= m->n_gens)
    stop_scale = m->n_gens - 1;

  mlx_stream s = mlx_default_cpu_stream_new();

  /* Initialize out_facie */
  mlx_array_t out_facie = in_noise;
  if (mlx_array_ndim(in_noise) == 0) {
    /* create zeros with shape derived from z[start_scale] minus padding */
    const int *zshape = mlx_array_shape(z_list[start_scale]);
    int batch = zshape[0];
    int height = zshape[1] - m->full_zero_padding;
    int width = zshape[2] - m->full_zero_padding;
    int channels = m->output_channels;
    int osh[4] = {batch, height, width, channels};
    mlx_array out0 = mlx_array_new();
    if (mlx_zeros(&out0, osh, 4, MLX_FLOAT32, s) == 0) {
      out_facie = out0;
    } else {
      /* fallback: return first z */
      out_facie = z_list[start_scale];
    }
  }

  /* Main progressive loop */
  for (int index = start_scale; index <= stop_scale; ++index) {
    /* determine desired upsample target from z[index] */
    const int *zshape = mlx_array_shape(z_list[index]);
    int target_h = zshape[1] - m->full_zero_padding;
    int target_w = zshape[2] - m->full_zero_padding;

    /* Upsample out_facie to (target_h, target_w) */
    mlx_array_t upsampled = out_facie;
    if (mlx_array_ndim(out_facie) != 0) {
      MLXUpsample *u = mlx_upsample_create(target_h, target_w, "linear", 1);
      if (u) {
        mlx_array_t tmp = mlx_upsample_forward(u, out_facie);
        mlx_upsample_free(u);
        if (tmp.ctx != out_facie.ctx) {
          if (out_facie.ctx &&
              !mlx_array_is_in_list(out_facie, z_list, z_count, in_noise))
            mlx_array_free(out_facie);
        }
        upsampled = tmp;
      }
    }
    /* report upsampled shape removed (debug prints cleared) */

    /* Prepare z_in (possibly scale noise and concat cond) */
    mlx_array_t z_in = z_list[index];
    if (m->has_cond_channels) {
      /* split z_in into noise and cond along last axis */
      const int *shape = mlx_array_shape(z_in);
      int dims = mlx_array_ndim(z_in);
      int last = shape[dims - 1];
      int out_ch = m->output_channels;
      /* noise = z_in[..., :out_ch] */
      int start_noise[4] = {0, 0, 0, 0};
      int stop_noise[4] = {shape[0], shape[1], shape[2], out_ch};
      int znd = mlx_array_ndim(z_in);
      const int *zsh = mlx_array_shape(z_in);
      (void)znd;
      (void)zsh;
      mlx_array noise = mlx_array_new();
      if (mlx_slice(&noise, z_in, start_noise, 4, stop_noise, 4, NULL, 0, s) !=
          0) {
        noise = z_in;
      }
      /* cond = z_in[..., out_ch:] */
      int start_cond[4] = {0, 0, 0, out_ch};
      int stop_cond[4] = {shape[0], shape[1], shape[2], last};
      mlx_array cond = mlx_array_new();
      if (mlx_slice(&cond, z_in, start_cond, 4, stop_cond, 4, NULL, 0, s) !=
          0) {
        cond = z_in;
      }

      /* scale noise by amp[index] */
      if (amp && amp_count > index) {
        mlx_array scale = mlx_array_new_float(amp[index]);
        mlx_array scaled = mlx_array_new();
        if (mlx_multiply(&scaled, noise, scale, s) == 0) {
          if (noise.ctx)
            mlx_array_free(noise);
          noise = scaled;
        }
        mlx_array_free(scale);
      }

      /* pad upsampled facie */
      int p = m->zero_padding;
      mlx_array pad_val = mlx_array_new_float(0.0f);
      int axes[2] = {1, 2};
      int low_pad[2] = {p, p};
      int high_pad[2] = {p, p};
      mlx_array padded = mlx_array_new();
      if (mlx_pad(&padded, upsampled, axes, 2, low_pad, 2, high_pad, 2, pad_val,
                  "constant", s) != 0) {
        padded = upsampled;
      }
      mlx_array_free(pad_val);

      /* tile if needed to match cond channels */
      int num_repeats = 1;
      if (m->cond_channels > 0 && m->output_channels > 0)
        num_repeats = m->cond_channels / m->output_channels;
      if (num_repeats > 1) {
        int reps[4] = {1, 1, 1, num_repeats};
        mlx_array tiled = mlx_array_new();
        if (mlx_tile(&tiled, padded, reps, 4, s) == 0) {
          if (padded.ctx)
            mlx_array_free(padded);
          padded = tiled;
        }
      }

      /* cond = cond + padded */
      /* Ensure spatial shapes match before adding: pad the smaller one */
      {
        const int *csh = mlx_array_shape(cond);
        const int *psh = mlx_array_shape(padded);
        mlx_array cond_check = cond;
        if (csh && psh && (csh[1] != psh[1] || csh[2] != psh[2])) {
          int dh = psh[1] - csh[1];
          int dw = psh[2] - csh[2];
          int low_h = dh > 0 ? dh / 2 : 0;
          int high_h = dh > 0 ? dh - low_h : 0;
          int low_w = dw > 0 ? dw / 2 : 0;
          int high_w = dw > 0 ? dw - low_w : 0;
          int axes_pad[2] = {1, 2};
          int low_pad[2] = {low_h, low_w};
          int high_pad[2] = {high_h, high_w};
          mlx_array pad_zero = mlx_array_new_float(0.0f);
          mlx_array cond_padded = mlx_array_new();
          if (mlx_pad(&cond_padded, cond, axes_pad, 2, low_pad, 2, high_pad, 2,
                      pad_zero, "constant", s) == 0) {
            cond_check = cond_padded;
            if (cond.ctx)
              mlx_array_free(cond);
            cond = cond_check;
          }
          mlx_array_free(pad_zero);
        }
      }
      mlx_array cond_plus = mlx_array_new();
      if (safe_add(&cond_plus, cond, padded, s) == 0) {
        if (cond.ctx)
          mlx_array_free(cond);
        cond = cond_plus;
      }

      /* z_in = concat([noise, cond], axis=-1) */
      mlx_vector_array vec =
          mlx_vector_array_new_data((const mlx_array[]){noise, cond}, 2);
      mlx_array znew = mlx_array_new();
      if (mlx_concatenate_axis(&znew, vec, 3, s) == 0) {
        /* free constituents */
        if (noise.ctx)
          mlx_array_free(noise);
        if (cond.ctx)
          mlx_array_free(cond);
        z_in = znew;
      } else {
        /* fallback to original z_in */
        mlx_array_free(znew);
      }
      mlx_vector_array_free(vec);

      /* free padded/up sample if different */
      if (upsampled.ctx && upsampled.ctx != padded.ctx) {
        mlx_array_free(upsampled);
      }
      upsampled = padded; /* keep for potential reuse */

      /* Debug: report z_in/cond/noise shapes after concat/pad */
      {
        int z_ndim = (int)mlx_array_ndim(z_in);
        const int *zsh = mlx_array_shape(z_in);
        if (zsh) {
          fprintf(stderr, " shape=(");
          for (int _a = 0; _a < z_ndim; ++_a)
            fprintf(stderr, "%d%s", zsh[_a], (_a + 1 < z_ndim) ? "," : "");
          fprintf(stderr, ")\n");
        } else
          fprintf(stderr, " (no shape)\n");
      }
    } else {
      /* no conditioning channels: scale z_in directly by amp and pad */
      if (amp && amp_count > index) {
        mlx_array scale = mlx_array_new_float(amp[index]);
        mlx_array scaled = mlx_array_new();
        if (mlx_multiply(&scaled, z_in, scale, s) == 0) {
          z_in = scaled;
        }
        mlx_array_free(scale);
      }
      int p = m->zero_padding;
      mlx_array pad_val = mlx_array_new_float(0.0f);
      int axes[2] = {1, 2};
      int low_pad[2] = {p, p};
      int high_pad[2] = {p, p};
      mlx_array padded = mlx_array_new();
      if (mlx_pad(&padded, upsampled, axes, 2, low_pad, 2, high_pad, 2, pad_val,
                  "constant", s) == 0) {
        if (upsampled.ctx &&
            !mlx_array_is_in_list(upsampled, z_list, z_count, in_noise))
          mlx_array_free(upsampled);
        upsampled = padded;
      }
      mlx_array_free(pad_val);

      /* debug: report shapes before add */
      {
        int zin_ndim = (int)mlx_array_ndim(z_in);
        const int *zin_sh = mlx_array_shape(z_in);
        int up_ndim2 = (int)mlx_array_ndim(upsampled);
        const int *up_sh2 = mlx_array_shape(upsampled);
        if (zin_sh) {
          fprintf(stderr, " shape=(%d,%d,%d,%d)", zin_sh[0], zin_sh[1],
                  zin_sh[2], zin_sh[3]);
        }
        fprintf(stderr, " vs upsampled ndim=%d", up_ndim2);
        if (up_sh2) {
          fprintf(stderr, " shape=(%d,%d,%d,%d)\n", up_sh2[0], up_sh2[1],
                  up_sh2[2], up_sh2[3]);
        } else
          fprintf(stderr, " (no shape)\n");
      }

      /* Ensure spatial shapes match before adding: pad the smaller one */
      {
        const int *zin_sh = mlx_array_shape(z_in);
        const int *up_sh2 = mlx_array_shape(upsampled);
        if (zin_sh && up_sh2 &&
            (zin_sh[1] != up_sh2[1] || zin_sh[2] != up_sh2[2])) {
          int dh = up_sh2[1] - zin_sh[1];
          int dw = up_sh2[2] - zin_sh[2];
          int low_h = dh > 0 ? dh / 2 : 0;
          int high_h = dh > 0 ? dh - low_h : 0;
          int low_w = dw > 0 ? dw / 2 : 0;
          int high_w = dw > 0 ? dw - low_w : 0;
          int axes_pad[2] = {1, 2};
          int low_pad[2] = {low_h, low_w};
          int high_pad[2] = {high_h, high_w};
          mlx_array pad_zero = mlx_array_new_float(0.0f);
          mlx_array zin_padded = mlx_array_new();
          if (mlx_pad(&zin_padded, z_in, axes_pad, 2, low_pad, 2, high_pad, 2,
                      pad_zero, "constant", s) == 0) {
            if (z_in.ctx) {
              mlx_array_free(z_in);
            }
            z_in = zin_padded;
          }
          mlx_array_free(pad_zero);
        }
      }
      /* z_in = z_in + padded */
      mlx_array sum = mlx_array_new();
      if (safe_add(&sum, z_in, upsampled, s) == 0) {
        if (z_in.ctx) {
          mlx_array_free(z_in);
        }
        z_in = sum;
      }
    }

    /* Execute per-scale module */
    struct ScaleModule *smod = (struct ScaleModule *)m->gens[index];
    /* Debug: report module and input shapes */
    {
      int zin_ndim = (int)mlx_array_ndim(z_in);
      const int *zin_sh = mlx_array_shape(z_in);
      fprintf(stderr,
              "[gen_debug] module index=%d is_spade=%d has_head=%d "
              "body_count=%d has_tail=%d z_in ndim=%d",
              index, smod ? smod->is_spade : -1, smod && smod->head ? 1 : 0,
              smod ? smod->n_body : 0, smod && smod->tail_conv ? 1 : 0,
              zin_ndim);
      (void)zin_sh;
    }
    mlx_array out_mod = mlx_array_new();
    if (smod) {
      if (smod->is_spade && smod->spade) {
        fprintf(stderr,
                "[gen_debug] calling spadegen_forward smod=%p spade=%p "
                "z_in.ctx=%p\n",
                (void *)smod, (void *)smod->spade, (void *)z_in.ctx);
        fflush(stderr);
        out_mod = mlx_spadegen_forward(smod->spade, z_in);
      } else {
        /* head */
        mlx_array cur = z_in;
        int cur_is_alias = 1; /* cur initially aliases z_in; track ownership to
                                 avoid double-free */
        if (smod->head) {
          mlx_array nx = mlx_convblock_forward(smod->head, cur);
          if (cur.ctx != nx.ctx) {
            if (cur_is_alias) {
              if (z_in.ctx) {
                mlx_array_free(z_in);
                z_in = mlx_array_new();
              }
              cur_is_alias = 0;
            } else {
              mlx_array_free(cur);
            }
          }
          cur = nx;
          cur_is_alias = 0;
          /* Debug: head output shape */
          {
            int c_ndim = (int)mlx_array_ndim(cur);
            const int *csh = mlx_array_shape(cur);
            (void)csh;
          }
        }
        /* body */
        if (smod->body) {
          for (int bi = 0; bi < smod->n_body; ++bi) {
            if (!smod->body[bi])
              continue;
            mlx_array nx = mlx_convblock_forward(smod->body[bi], cur);
            if (cur.ctx != nx.ctx) {
              if (cur_is_alias) {
                if (z_in.ctx) {
                  mlx_array_free(z_in);
                  z_in = mlx_array_new();
                }
                cur_is_alias = 0;
              } else {
                mlx_array_free(cur);
              }
            }
            cur = nx;
            cur_is_alias = 0;
            /* Debug: body output shape after body[%d] */
            {
              int b_ndim = (int)mlx_array_ndim(cur);
              const int *bsh = mlx_array_shape(cur);
              (void)bsh;
            }
          }
        }
        /* tail conv + tanh if present */
        if (smod->tail_conv) {
          mlx_array outc = mlx_array_new();
          if (safe_mlx_conv2d(&outc, cur, *smod->tail_conv, 1, 1,
                              m->padding_size, m->padding_size, 1, 1, 1,
                              s) == 0) {
            mlx_array t = mlx_array_new();
            if (mlx_tanh(&t, outc, s) == 0) {
              out_mod = t;
            } else {
              out_mod = outc;
            }
          } else {
            out_mod = cur;
          }
          /* Debug: tail/out_mod shape */
          {
            int o_ndim = (int)mlx_array_ndim(out_mod);
            const int *osh = mlx_array_shape(out_mod);
            (void)osh;
          }
          if (cur.ctx && cur.ctx != out_mod.ctx) {
            if (cur_is_alias) {
              if (z_in.ctx) {
                mlx_array_free(z_in);
                z_in = mlx_array_new();
              }
              cur_is_alias = 0;
            } else {
              mlx_array_free(cur);
            }
          }
        } else {
          out_mod = cur;
          /* Debug: out_mod (no tail) shape */
          {
            int o_ndim = (int)mlx_array_ndim(out_mod);
            const int *osh = mlx_array_shape(out_mod);
            (void)osh;
          }
        }
      }
    }

    /* Debug: shapes before adding module output to current facie */
    {
      int mo_ndim = (int)mlx_array_ndim(out_mod);
      const int *mos = mlx_array_shape(out_mod);
      (void)mos;
      int up_ndim = (int)mlx_array_ndim(upsampled);
      const int *ups = mlx_array_shape(upsampled);
      (void)ups;
    }

    /* Ensure spatial shapes match between out_mod and upsampled before adding.
     * If one is smaller, pad it with zeros centrally. */
    const int *mos_sh = mlx_array_shape(out_mod);
    const int *ups_sh = mlx_array_shape(upsampled);
    if (mos_sh && ups_sh &&
        (mos_sh[1] != ups_sh[1] || mos_sh[2] != ups_sh[2])) {
      int dh = ups_sh[1] - mos_sh[1];
      int dw = ups_sh[2] - mos_sh[2];
      if (dh == 0 && dw == 0) {
        /* shapes match already */
      } else if (dh >= 0 && dw >= 0) {
        /* pad out_mod to match upsampled */
        int low_h = dh / 2;
        int high_h = dh - low_h;
        int low_w = dw / 2;
        int high_w = dw - low_w;
        int axes_pad[2] = {1, 2};
        int low_pad[2] = {low_h, low_w};
        int high_pad[2] = {high_h, high_w};
        mlx_array pad_zero = mlx_array_new_float(0.0f);
        mlx_array out_padded = mlx_array_new();
        if (mlx_pad(&out_padded, out_mod, axes_pad, 2, low_pad, 2, high_pad, 2,
                    pad_zero, "constant", s) == 0) {
          if (out_mod.ctx)
            mlx_array_free(out_mod);
          out_mod = out_padded;
        } else {
          mlx_array_free(out_padded);
        }
        mlx_array_free(pad_zero);
      } else {
        /* pad upsampled to match out_mod (dh,dw negative) */
        int pdh = -dh;
        int pdw = -dw;
        int low_h = pdh / 2;
        int high_h = pdh - low_h;
        int low_w = pdw / 2;
        int high_w = pdw - low_w;
        int axes_pad[2] = {1, 2};
        int low_pad[2] = {low_h, low_w};
        int high_pad[2] = {high_h, high_w};
        mlx_array pad_zero = mlx_array_new_float(0.0f);
        mlx_array up_padded = mlx_array_new();
        if (mlx_pad(&up_padded, upsampled, axes_pad, 2, low_pad, 2, high_pad, 2,
                    pad_zero, "constant", s) == 0) {
          if (upsampled.ctx)
            mlx_array_free(upsampled);
          upsampled = up_padded;
        } else {
          mlx_array_free(up_padded);
        }
        mlx_array_free(pad_zero);
      }
    }

    /* Add module output to current facie (use safe_add to pad operands if
     * needed) */
    mlx_array new_out = mlx_array_new();
    if (safe_add(&new_out, out_mod, upsampled, s) == 0) {
      if (out_mod.ctx &&
          !mlx_array_is_in_list(out_mod, z_list, z_count, in_noise)) {
        mlx_array_safe_free(&out_mod);
      }
      if (upsampled.ctx &&
          !mlx_array_is_in_list(upsampled, z_list, z_count, in_noise)) {
        mlx_array_safe_free(&upsampled);
      }
      out_facie = new_out;
    } else {
      /* fallback: keep upsampled */
      if (out_mod.ctx && out_mod.ctx != upsampled.ctx &&
          !mlx_array_is_in_list(out_mod, z_list, z_count, in_noise)) {
        mlx_array_safe_free(&out_mod);
        mlx_array_free(out_mod);
      }
      out_facie = upsampled;
    }

    /* if we created a new z_in (concat/slices), free it */
    if (z_in.ctx && z_in.ctx != z_list[index].ctx)
      mlx_array_free(z_in);
  }

  /* color quantization (can be disabled via FACIESGAN_DISABLE_COLOR_QUANT) */
  if (m->color_quant && getenv("FACIESGAN_DISABLE_COLOR_QUANT") == NULL) {
    mlx_array_t q = mlx_colorquant_forward(m->color_quant, out_facie, 0);
    if (out_facie.ctx && out_facie.ctx != q.ctx &&
        !mlx_array_is_in_list(out_facie, z_list, z_count, in_noise))
      mlx_array_free(out_facie);
    out_facie = q;
  }

  mlx_stream_free(s);
  return out_facie;
}

mlx_array_t mlx_generator_call(MLXGenerator *m, const mlx_array *z_list,
                               int z_count, const float *amp, int amp_count,
                               mlx_array_t in_noise, int start_scale,
                               int stop_scale) {
  return mlx_generator_forward(m, z_list, z_count, amp, amp_count, in_noise,
                               start_scale, stop_scale);
}

void mlx_generator_eval(MLXGenerator *m, int enable) {
  if (!m)
    return;
  m->eval_mode = enable ? 1 : 0;
}
