/* Stub for example_train_from_ag.c — minimal entry so CMake can build examples
 * during parity checks */
#include <stdio.h>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  printf("example_train_from_ag stub running\n");
  return 0;
}
#include "autodiff.h"
#include "base_manager.h"
#include "checkpoint.h"
#include "optimizer.h"
#include "options.h"
#include "train_step.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void copy_host_data_of_array(const mlx_array *a, float **out_buf,
                                    size_t *out_count) {
  if (!a) {
    *out_buf = NULL;
    *out_count = 0;
    return;
  }
  /* Ensure array is materialized on host before accessing data */
  mlx_array_eval(*a);
  int ndim = (int)mlx_array_ndim(*a);
  const int *shape = mlx_array_shape(*a);
  size_t count = 1;
  for (int i = 0; i < ndim; ++i)
    count *= (size_t)shape[i];
  const float *pdata = mlx_array_data_float32(*a);
  if (!pdata) {
    *out_buf = NULL;
    *out_count = 0;
    return;
  }
  float *buf = (float *)malloc(sizeof(float) * count);
  if (!buf) {
    *out_buf = NULL;
    *out_count = 0;
    return;
  }
  memcpy(buf, pdata, sizeof(float) * count);
  *out_buf = buf;
  *out_count = count;
}

int main(void) {
  printf("Smoke: AG-backed params -> one train step\n");

  TrainningOptions *t = mlx_options_new_trainning_defaults();
  if (!t)
    return 2;
  /* overrides for this smoke test */
  t->num_feature = 16;
  t->min_num_feature = 8;
  t->num_layer = 1;
  t->kernel_size = 3;
  t->padding_size = 1;
  t->num_img_channels = 1;
  t->discriminator_steps = 1;
  t->generator_steps = 1;

  MLXBaseManager *mgr = mlx_base_manager_create_with_faciesgan(t);
  if (!mgr) {
    fprintf(stderr, "failed to create base manager\n");
    mlx_options_free_trainning(t);
    return 2;
  }

  /* initialize scales: allow overriding via NUM_SCALES env var (default 7) */
  const char *env_nsc = getenv("NUM_SCALES");
  int nscales = env_nsc ? atoi(env_nsc) : 7;
  if (nscales <= 0)
    nscales = 1;
  printf("initializing %d scales\n", nscales);
  mlx_base_manager_init_scales(mgr, 0, nscales);

  MLXFaciesGAN *fg = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
  if (!fg) {
    fprintf(stderr, "no faciesgan ctx\n");
    mlx_base_manager_free(mgr);
    return 3;
  }

  MLXGenerator *g = mlx_faciesgan_build_generator(fg);
  if (!g) {
    fprintf(stderr, "failed to build generator\n");
    mlx_base_manager_free(mgr);
    return 4;
  }

  int param_n = 0;
  mlx_array **params = mlx_generator_get_parameters(g, &param_n);
  if (!params || param_n == 0) {
    fprintf(stderr, "no params\n");
    mlx_generator_free_parameters_list(params);
    mlx_base_manager_free(mgr);
    return 5;
  }

  printf("param_n = %d\n", param_n);
  for (int pi = 0; pi < param_n && pi < 5; ++pi) {
    printf(" param[%d] ptr=%p\n", pi, (void *)params[pi]);
    if (params[pi]) {
      int ndim = (int)mlx_array_ndim(*params[pi]);
      printf("  ndim=%d\n", ndim);
    }
  }

  /* snapshot first param for comparison */
  float *before_buf = NULL;
  size_t before_count = 0;
  copy_host_data_of_array(params[0], &before_buf, &before_count);

  /* wrap all params as AGValue requiring grad */
  AGValue **ag_params = (AGValue **)malloc(sizeof(AGValue *) * param_n);
  for (int i = 0; i < param_n; ++i) {
    ag_params[i] = ag_value_from_array(params[i], 1);
  }

  /* Run multiple train steps: build loss, backprop, apply SGD, repeat */
  const char *env_steps = getenv("NUM_STEPS");
  int num_steps = env_steps ? atoi(env_steps) : 100;
  if (num_steps <= 0)
    num_steps = 100;

  float gb1 = 0.0f, gb2 = 0.0f, geps = 0.0f;
  mlx_optimizer_get_global_adam_params(&gb1, &gb2, &geps);
  int gbc = mlx_optimizer_get_global_adam_bias_correction();
  float gwd = mlx_optimizer_get_global_adam_weight_decay();
  MLXOptimizer *opt_g = mlx_adam_create_ext(0.01f, gb1, gb2, geps, gbc, gwd);

  for (int step = 0; step < num_steps; ++step) {
    if ((step % 10) == 0)
      printf("step %d/%d\n", step, num_steps);
    /* clear grads from previous step */
    ag_zero_grad_all();

    /* Build a simple loss: sum(square(p)) over all params -> scalar */
    AGValue *total = NULL;
    for (int i = 0; i < param_n; ++i) {
      AGValue *sq = ag_square(ag_params[i]);
      /* reduce to scalar by summing along all axes */
      mlx_array *arr = ag_value_array(sq);
      int ndim = (int)mlx_array_ndim(*arr);
      AGValue *s = sq;
      for (int ax = 0; ax < ndim; ++ax)
        s = ag_sum_axis(s, 0, 0);
      if (!total)
        total = s;
      else
        total = ag_add(total, s);
    }

    if (!total) {
      fprintf(stderr, "failed to construct loss\n");
      break;
    }

    /* Backpropagate */
    if (ag_backward(total) != 0) {
      fprintf(stderr, "ag_backward failed\n");
    }

    /* Apply train step via AG bridge */
    int r = mlx_base_train_step_from_ag(mgr, opt_g, ag_params, param_n, NULL,
                                        NULL, 0);
    if (r != 0)
      fprintf(stderr, "train_step_from_ag returned %d\n", r);

    /* release tape / temporaries created while building the loss */
    ag_reset_tape();
  }

  printf("about to snapshot after params\n");
  fflush(stdout);

  /* snapshot after */
  float *after_buf = NULL;
  size_t after_count = 0;
  copy_host_data_of_array(params[0], &after_buf, &after_count);

  if (before_buf && after_buf && before_count == after_count) {
    printf("first-param before[0]=%f after[0]=%f\n", before_buf[0],
           after_buf[0]);
  }

  /* Try to save checkpoint and per-scale generator.npz into OUTPUT_PATH if
   * provided. */
  const char *outenv = getenv("OUTPUT_PATH");
  const char *out_dir = outenv ? outenv : NULL;
  if (out_dir) {
    char ckpath[1024];
    snprintf(ckpath, sizeof(ckpath), "%s/checkpoint.bin", out_dir);
    if (fg) {
      (void)mlx_checkpoint_save(ckpath, fg);
    }

    if (fg) {
      MLXGenerator *g2 = mlx_faciesgan_build_generator(fg);
      if (g2) {
        int nscales = mlx_generator_get_n_gens(g2);
        for (int si = 0; si < nscales; ++si) {
          char scale_dir[1024];
          snprintf(scale_dir, sizeof(scale_dir), "%s/%d", out_dir, si);
          /* ensure directory exists */
          (void)mlx_create_dirs(scale_dir);

          if (mlx_generator_has_spade_scale(g2, si)) {
            MLXSPADEGenerator *sp = mlx_scale_get_spade(g2, si);
            if (sp) {
              int pcount = 0;
              mlx_array **plist = mlx_spadegen_get_parameters(sp, &pcount);
              for (int pi = 0; pi < pcount; ++pi) {
                if (!plist[pi])
                  continue;
                char fname[1024];
                if (pi == 0)
                  snprintf(fname, sizeof(fname), "%s/init_conv.weight.npy",
                           scale_dir);
                else if (pi == pcount - 1)
                  snprintf(fname, sizeof(fname), "%s/tail_conv.weight.npy",
                           scale_dir);
                else
                  snprintf(fname, sizeof(fname),
                           "%s/spade_blocks.layers.%d.conv.weight.npy",
                           scale_dir, pi - 1);
                (void)mlx_save(fname, *plist[pi]);
              }
              mlx_spadegen_free_parameters_list(plist);
            }
          } else {
            MLXConvBlock *head = mlx_scale_get_head(g2, si);
            if (head) {
              mlx_array *hw = mlx_convblock_get_conv_weight(head);
              if (hw) {
                char fname[1024];
                snprintf(fname, sizeof(fname), "%s/head.conv.weight.npy",
                         scale_dir);
                (void)mlx_save(fname, *hw);
              }
            }
            int bc = mlx_scale_get_body_count(g2, si);
            for (int bi = 0; bi < bc; ++bi) {
              MLXConvBlock *b = mlx_scale_get_body_at(g2, si, bi);
              if (!b)
                continue;
              mlx_array *bw = mlx_convblock_get_conv_weight(b);
              if (bw) {
                char fname[1024];
                snprintf(fname, sizeof(fname),
                         "%s/body.layers.%d.conv.weight.npy", scale_dir, bi);
                (void)mlx_save(fname, *bw);
              }
            }
            if (mlx_scale_has_tail_conv(g2, si)) {
              mlx_array *tw = mlx_scale_get_tail_conv(g2, si);
              if (tw) {
                char fname[1024];
                snprintf(fname, sizeof(fname), "%s/tail.layers.0.weight.npy",
                         scale_dir);
                (void)mlx_save(fname, *tw);
              }
            }
          }

          /* pack into generator.npz using helper script; optionally provide
           * Python template NPZ and scale index */
          const char *py_tpl = getenv("PYTHON_GOLD_NPZ");
          char pack_cmd[4096];
          if (py_tpl && py_tpl[0]) {
            snprintf(
                pack_cmd, sizeof(pack_cmd),
                "python3 '%s/trainning/mlx-c/pack_scale_npz.py' '%s' '%s' %d",
                "/Users/mazzutti/POSDOC/Experimentos/FaciesGAN", scale_dir,
                py_tpl, si);
          } else {
            snprintf(pack_cmd, sizeof(pack_cmd),
                     "python3 '%s/trainning/mlx-c/pack_scale_npz.py' '%s'",
                     "/Users/mazzutti/POSDOC/Experimentos/FaciesGAN",
                     scale_dir);
          }
          (void)system(pack_cmd);
        }
      }
    }
  }

  printf("cleanup: freeing before/after buffers\n");

  /* cleanup */
  if (before_buf)
    free(before_buf);
  if (after_buf)
    free(after_buf);

  printf("cleanup: freeing AGValue wrappers (%d)\n", param_n);
  for (int i = 0; i < param_n; ++i) {
    if (i % 10 == 0)
      printf(" freeing ag param %d\n", i);
    ag_value_free(ag_params[i]);
  }
  free(ag_params);

  mlx_adam_free(opt_g);
  mlx_generator_free_parameters_list(params);
  /* Explicitly free the faciesgan user context (if present) before
   * freeing the manager to ensure model-owned mlx_array weights are
   * released. */
  if (fg) {
    mlx_faciesgan_free(fg);
    mlx_base_manager_set_user_ctx(mgr, NULL);
    fg = NULL;
  }
  mlx_base_manager_free(mgr);

  printf("done\n");
  return 0;
}
