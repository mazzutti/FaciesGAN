#include "train_manager.h"
#include "array_helpers.h"
#include "checkpoint.h"
#include "train_step.h"
#include "utils.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "io/npz_create.h"

int mlx_faciesgan_train_manager(MLXBaseManager *mgr, MLXOptimizer *opt_g,
                                MLXOptimizer *opt_d, MLXTrainStepCallback cb,
                                void *cb_ctx, int epochs, int steps_per_epoch,
                                MLXLogger *logger, MLXScheduler *sched,
                                const char *checkpoint_path,
                                int checkpoint_every) {
    if (!mgr || !cb)
        return -1;
    for (int e = 0; e < epochs; ++e) {
        for (int s = 0; s < steps_per_epoch; ++s) {
            int step_idx = e * steps_per_epoch + s;
            /* scheduler step: use step-and-get-lr then apply to available optimizers
             */
            if (sched) {
                float tmp_lr[16];
                mlx_scheduler_step_and_get_lr(sched, step_idx, opt_g, tmp_lr,
                                              sizeof(tmp_lr) / sizeof(tmp_lr[0]));
                MLXOptimizer *opts_arr[2] = {opt_g, opt_d};
                mlx_scheduler_apply_to_optimizers(sched, opts_arr, 2);
            }

            /* call user callback to produce grads and loss */
            mlx_array **gen_grads = NULL;
            int gen_n = 0;
            mlx_array **disc_grads = NULL;
            int disc_n = 0;
            float loss = 0.0f;
            int rc = cb(mgr, step_idx, cb_ctx, &gen_grads, &gen_n, &disc_grads,
                        &disc_n, &loss);
            if (rc != 0)
                return -1;

            /* apply grads */
            int r1 = 0, r2 = 0;
            if (opt_g && gen_n > 0)
                r1 = mlx_base_apply_sgd_to_generator(mgr, opt_g, gen_grads, gen_n);
            if (opt_d && disc_n > 0)
                r2 =
                    mlx_base_apply_sgd_to_discriminator(mgr, opt_d, disc_grads, disc_n);

            /* logging */
            if (logger) {
                char buf[256];
                snprintf(buf, sizeof(buf), "epoch=%d step=%d loss=%f rg=%d rd=%d", e, s,
                         loss, r1, r2);
                mlx_logger_log(logger, buf);
            }

            /* free grads arrays created by callback */
            if (gen_grads) {
                mlx_free_mlx_array_ptrs(&gen_grads, gen_n);
            }
            if (disc_grads) {
                mlx_free_mlx_array_ptrs(&disc_grads, disc_n);
            }

            /* checkpointing */
            if (checkpoint_path && checkpoint_every > 0 &&
                    ((step_idx + 1) % checkpoint_every == 0)) {
                /* checkpoint_save currently expects MLXFaciesGAN; retrieve and pass */
                MLXFaciesGAN *fg = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
                if (fg) {
                    /* save binary checkpoint */
                    mlx_checkpoint_save(checkpoint_path, fg);

                    /* Also emit per-scale `generator.npz` files for tooling:
                     * - Create per-scale directories under the checkpoint's parent
                     * directory
                     * - Save per-scale parameter arrays as temporary .npy files
                     * - Invoke Python helper to pack them into `generator.npz`
                     */
                    /* derive base run dir from checkpoint_path (strip file name) */
                    char base_dir[PATH_MAX];
                    strncpy(base_dir, checkpoint_path, PATH_MAX - 1);
                    base_dir[PATH_MAX - 1] = '\0';
                    char *p = strrchr(base_dir, '/');
                    if (p)
                        *p = '\0';

                    /* build generator pointer */
                    MLXGenerator *g = mlx_faciesgan_build_generator(fg);
                    if (g) {
                        int nscales = mlx_generator_get_n_gens(g);
                        for (int si = 0; si < nscales; ++si) {
                            char scale_dir[PATH_MAX];
                            snprintf(scale_dir, PATH_MAX, "%s/%d", base_dir, si);
                            /* ensure directory exists (create parents) */
                            if (mlx_create_dirs(scale_dir) != 0) {
                                /* ignore and continue */
                            }

                            /* collect and save arrays for this scale as .npy files */
                            if (mlx_generator_has_spade_scale(g, si)) {
                                MLXSPADEGenerator *sp = mlx_scale_get_spade(g, si);
                                if (sp) {
                                    int pcount = 0;
                                    mlx_array **plist = mlx_spadegen_get_parameters(sp, &pcount);
                                    for (int pi = 0; pi < pcount; ++pi) {
                                        if (!plist[pi])
                                            continue;
                                        char fname[PATH_MAX];
                                        snprintf(fname, PATH_MAX, "%s/param_spade_%d.npy",
                                                 scale_dir, pi);
                                        /* mlx_save expects mlx_array (not pointer) */
                                        mlx_save(fname, *plist[pi]);
                                    }
                                    mlx_spadegen_free_parameters_list(plist);
                                }
                            } else {
                                /* head conv */
                                MLXConvBlock *head = mlx_scale_get_head(g, si);
                                if (head) {
                                    mlx_array *hw = mlx_convblock_get_conv_weight(head);
                                    if (hw) {
                                        char fname[PATH_MAX];
                                        snprintf(fname, PATH_MAX, "%s/param_head.npy", scale_dir);
                                        mlx_save(fname, *hw);
                                    }
                                }
                                /* body convs */
                                int bc = mlx_scale_get_body_count(g, si);
                                for (int bi = 0; bi < bc; ++bi) {
                                    MLXConvBlock *b = mlx_scale_get_body_at(g, si, bi);
                                    if (!b)
                                        continue;
                                    mlx_array *bw = mlx_convblock_get_conv_weight(b);
                                    if (bw) {
                                        char fname[PATH_MAX];
                                        snprintf(fname, PATH_MAX, "%s/param_body_%d.npy", scale_dir,
                                                 bi);
                                        mlx_save(fname, *bw);
                                    }
                                }
                                /* tail conv */
                                if (mlx_scale_has_tail_conv(g, si)) {
                                    mlx_array *tw = mlx_scale_get_tail_conv(g, si);
                                    if (tw) {
                                        char fname[PATH_MAX];
                                        snprintf(fname, PATH_MAX, "%s/param_tail.npy", scale_dir);
                                        mlx_save(fname, *tw);
                                    }
                                }
                            }

                            /* Pack .npy files into generator.npz directly in C
                             * (replaces previous system() call to Python helper) */
                            npz_pack_npy_dir(scale_dir, "generator.npz");
                        }
                    }
                }
            }
        }
    }
    return 0;
}
