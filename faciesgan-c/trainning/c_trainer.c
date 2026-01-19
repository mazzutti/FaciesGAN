#include "datasets/c_generator.h"
#include "datasets/dataloader.h"
#include "datasets/wells.h"
#include "io/npz_unzip.h"
#include "models/base_manager.h"
#include "options.h"
#include "trainning/pybridge.h"
#include "trainning/train_step.h"
#include "datasets/func_cache.h"
#include <errno.h>
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/io.h>
#include <mlx/c/ops.h>
#include <mlx/c/random.h>
#include <mlx/c/stream.h>
#include <mlx/c/vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <stdint.h>

/*
 * Simple C-native trainer that demonstrates creating synthetic per-sample
 * pyramids, building a facies_dataset, iterating with facies_dataloader and
 * using the facies_collate path via facies_dataloader_next.
 */

int c_trainer_run(int num_samples, int num_scales, int channels, int height,
                  int width, int batch_size)
{
    if (num_samples <= 0 || num_scales <= 0 || batch_size <= 0)
    {
        fprintf(stderr, "invalid trainer args\n");
        return 1;
    }

    mlx_vector_vector_array facies_pyramids = mlx_vector_vector_array_new();
    // Create per-sample vector_array entries
    for (int si = 0; si < num_samples; ++si)
    {
        mlx_vector_array sample = mlx_vector_array_new();
        for (int sc = 0; sc < num_scales; ++sc)
        {
            // Create synthetic array for this sample/scale with shape (H,W,C)
            int shape[3] = {height, width, channels};
            mlx_array a = mlx_array_new();
            // MLX random_normal supports shape with ndim; provide 3-d shape
            mlx_stream s = mlx_default_cpu_stream_new();
            if (mlx_random_normal(&a, shape, 3, MLX_FLOAT32, 0.0f, 1.0f,
                                  mlx_array_empty, s) != 0)
            {
                // fallback zeros
                mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
            }
            mlx_stream_free(s);
            // append value (moves reference)
            if (mlx_vector_array_append_value(sample, a) != 0)
            {
                fprintf(stderr, "failed to append sample array\n");
                mlx_array_free(a);
                mlx_vector_array_free(sample);
                mlx_vector_vector_array_free(facies_pyramids);
                return 1;
            }
            // note: mlx_vector_array_append_value copies the value; free local
            // reference
            mlx_array_free(a);
        }
        if (mlx_vector_vector_array_append_value(facies_pyramids, sample) != 0)
        {
            fprintf(stderr, "failed to append sample vector\n");
            mlx_vector_array_free(sample);
            mlx_vector_vector_array_free(facies_pyramids);
            return 1;
        }
        mlx_vector_array_free(sample);
    }

    // create empty wells/seismic
    mlx_vector_vector_array wells = mlx_vector_vector_array_new();
    mlx_vector_vector_array seismic = mlx_vector_vector_array_new();

    /* In synthetic mode we leave wells empty (no mapping available) */

    facies_dataset *ds = NULL;
    if (facies_dataset_new(&ds, facies_pyramids, wells, seismic) != 0)
    {
        fprintf(stderr, "failed to create facies_dataset\n");
        mlx_vector_vector_array_free(facies_pyramids);
        mlx_vector_vector_array_free(wells);
        mlx_vector_vector_array_free(seismic);
        return 1;
    }

    facies_dataloader *dl = NULL;
    if (facies_dataloader_new(&dl, ds, (size_t)batch_size, false, false,
                              (unsigned int)time(NULL)) != 0)
    {
        fprintf(stderr, "failed to create facies_dataloader\n");
        facies_dataset_free(ds);
        mlx_vector_vector_array_free(facies_pyramids);
        mlx_vector_vector_array_free(wells);
        mlx_vector_vector_array_free(seismic);
        return 1;
    }

    printf("Starting C-native trainer: %d samples, %d scales, batch %d\n",
           num_samples, num_scales, batch_size);

    /* Initialize Python bridge visualizer and background worker (best-effort).
     * These calls are no-ops if Python headers are unavailable or initialization
     * fails. */
    pybridge_create_visualizer(num_scales, ".", NULL, 1);
    pybridge_create_background_worker(2, 32);

    mlx_stream s = mlx_default_cpu_stream_new();

    int batch_idx = 0;
    while (1)
    {
        mlx_vector_array out_facies = mlx_vector_array_new();
        mlx_vector_array out_wells = mlx_vector_array_new();
        mlx_vector_array out_seismic = mlx_vector_array_new();
        int rc =
            facies_dataloader_next(dl, &out_facies, &out_wells, &out_seismic, s);
        if (rc == 2)
        {
            // finished
            mlx_vector_array_free(out_facies);
            mlx_vector_array_free(out_wells);
            mlx_vector_array_free(out_seismic);
            break;
        }
        else if (rc != 0)
        {
            fprintf(stderr, "dataloader_next error: %d\n", rc);
            mlx_vector_array_free(out_facies);
            mlx_vector_array_free(out_wells);
            mlx_vector_array_free(out_seismic);
            break;
        }
        // For demo, inspect number of stacked scales
        size_t nsc = mlx_vector_array_size(out_facies);
        printf(" batch %d: stacked scales = %zu\n", batch_idx, nsc);
        /* Emit a lightweight JSON metrics object consumed by the Python visualizer.
         */
        char metrics[1024];
        /* Build a simple per-scale metrics JSON with placeholder values. */
        int off = 0;
        off += snprintf(metrics + off, sizeof(metrics) - off, "{");
        for (int sc = 0; sc < num_scales; ++sc)
        {
            off += snprintf(metrics + off, sizeof(metrics) - off,
                            "\"%d\":{\"d_total\":%g,\"d_real\":%g,\"d_fake\":%g,\"g_"
                            "total\":%g,\"g_adv\":%g,\"g_rec\":%g}",
                            sc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            if (sc + 1 < num_scales)
                off += snprintf(metrics + off, sizeof(metrics) - off, ",");
        }
        off += snprintf(metrics + off, sizeof(metrics) - off, "}");
        pybridge_update_visualizer_from_json(batch_idx, metrics,
                                             batch_idx * batch_size);
        // free outputs
        mlx_vector_array_free(out_facies);
        mlx_vector_array_free(out_wells);
        mlx_vector_array_free(out_seismic);
        batch_idx++;
    }

    mlx_stream_free(s);
    facies_dataloader_free(dl);
    facies_dataset_free(ds);
    mlx_vector_vector_array_free(facies_pyramids);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);

    /* no wells mapping allocated in synthetic mode */

    printf("C-native trainer finished; processed %d batches\n", batch_idx);
    return 0;
}

int c_trainer_run_full(const char *input_path, const char *output_path,
                       int manual_seed, int use_wells, int use_seismic,
                       int num_train_pyramids, int num_scales,
                       int num_img_channels, int crop_size, int batch_size,
                       int save_interval)
{
    if (!input_path || !output_path)
    {
        fprintf(stderr, "input_path and output_path are required for real-data training\n");
        return -1;
    }

    int desired_num = num_train_pyramids > 0 ? num_train_pyramids : 1;
    int num_samples = desired_num;
    int stop_scale = num_scales - 1;

    char cache_dir[PATH_MAX];
    /* Use a C-specific cache directory under the repo .cache to avoid
       clashing with Python joblib caches while keeping files nearby. */
    snprintf(cache_dir, PATH_MAX, "./.cache/c_pyramids_cache");
    /* create cache dir */
    if (mkdir(cache_dir, 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "Could not create cache dir %s: %s\n", cache_dir,
                strerror(errno));
        /* continue, attempt to use existing dir */
    }

    /* Ensure a function-level .npz cache exists and get its path and sample count. */
    char cache_npz[PATH_MAX] = {0};
    int actual_samples = 0;
    if (ensure_function_cache(input_path, cache_dir, desired_num, stop_scale,
                              crop_size, num_img_channels, use_wells, use_seismic,
                              manual_seed, cache_npz, sizeof(cache_npz), &actual_samples) != 0)
    {
        fprintf(stderr, "Failed to prepare function cache for %s\n", input_path);
        return -1;
    }
    num_samples = actual_samples > 0 ? actual_samples : desired_num;
    mlx_vector_vector_array facies_pyramids = mlx_vector_vector_array_new();

    mlx_stream s = mlx_default_cpu_stream_new();

    /* Load pyramids from .npz function cache into memory via reader API */
    for (int si = 0; si < num_samples; ++si)
    {
        mlx_vector_array sample = mlx_vector_array_new();
        for (int sc = 0; sc < num_scales; ++sc)
        {
            char member[64];
            snprintf(member, sizeof(member), "sample_%d/facies_%d.npy", si, sc);
            mlx_io_reader reader;
            int prc = npz_extract_member_to_mlx_reader(cache_npz, member, &reader);
            mlx_array a = mlx_array_new();
            if (prc == 0)
            {
                if (mlx_load_reader(&a, reader, s) != 0)
                {
                    int shape[3] = {crop_size, crop_size, num_img_channels};
                    mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
                }
                mlx_io_reader_free(reader);
            }
            else
            {
                int shape[3] = {crop_size, crop_size, num_img_channels};
                mlx_zeros(&a, shape, 3, MLX_FLOAT32, s);
            }
            if (mlx_vector_array_append_value(sample, a) != 0)
            {
                fprintf(stderr, "failed to append facies array for sample %d scale %d\n", si, sc);
                mlx_array_free(a);
                mlx_vector_array_free(sample);
                mlx_vector_vector_array_free(facies_pyramids);
                mlx_stream_free(s);
                return 1;
            }
            mlx_array_free(a);
        }
        if (mlx_vector_vector_array_append_value(facies_pyramids, sample) != 0)
        {
            fprintf(stderr, "failed to append sample vector\n");
            mlx_vector_array_free(sample);
            mlx_vector_vector_array_free(facies_pyramids);
            mlx_stream_free(s);
            return 1;
        }
        mlx_vector_array_free(sample);
    }

    /* create empty wells/seismic (collate expects same API) */
    /* build wells mapping and create wells pyramids */
    mlx_vector_vector_array wells = mlx_vector_vector_array_new();
    mlx_vector_vector_array seismic = mlx_vector_vector_array_new();

    int32_t *wcols = NULL;
    int32_t *wcounts = NULL;
    int wmap_n = 0;
    char **wimage_files = NULL;
    int wimage_count = 0;
    if (use_wells)
    {
        if (datasets_load_wells_mapping(input_path, "wells", &wcols, &wcounts,
                                        &wmap_n, &wimage_files,
                                        &wimage_count) != 0)
        {
            /* mapping unavailable: leave wells empty */
            wcols = NULL;
            wcounts = NULL;
            wmap_n = 0;
        }
    }

    facies_dataset *ds = NULL;
    if (facies_dataset_new(&ds, facies_pyramids, wells, seismic) != 0)
    {
        fprintf(stderr, "failed to create facies_dataset\n");
        mlx_vector_vector_array_free(facies_pyramids);
        mlx_vector_vector_array_free(wells);
        mlx_vector_vector_array_free(seismic);
        mlx_stream_free(s);
        return 1;
    }

    facies_dataloader *dl = NULL;
    if (facies_dataloader_new(&dl, ds, (size_t)batch_size, false, false,
                              (unsigned int)time(NULL)) != 0)
    {
        fprintf(stderr, "failed to create facies_dataloader\n");
        facies_dataset_free(ds);
        mlx_vector_vector_array_free(facies_pyramids);
        mlx_vector_vector_array_free(wells);
        mlx_vector_vector_array_free(seismic);
        mlx_stream_free(s);
        return 1;
    }

    printf("Starting C-native trainer (real data): %d samples, %d scales, batch "
           "%d\n",
           num_samples, num_scales, batch_size);

    /* Initialize optional Python bridge visualizer and background worker */
    pybridge_create_visualizer(num_scales, output_path ? output_path : ".", NULL,
                               1);
    pybridge_create_background_worker(2, 32);

    /* Create an MLX Base Manager to compute accurate per-scale metrics when
     * available */
    TrainningOptions *t = mlx_options_new_trainning_defaults();
    MLXTrainOptions train_opts = {0};
    mlx_options_to_mlx_train_opts(t, &train_opts);
    /* override number of scales to match loaded pyramids */
    train_opts.num_parallel_scales = num_scales;
    mlx_options_free_trainning(t);

    MLXBaseManager *mgr = mlx_base_manager_create_with_faciesgan(&train_opts);
    MLXFaciesGAN *fg = NULL;
    if (mgr)
    {
        mlx_base_manager_init_scales(mgr, 0, num_scales);
        fg = (MLXFaciesGAN *)mlx_base_manager_get_user_ctx(mgr);
    }

    int batch_idx = 0;
    while (1)
    {
        mlx_vector_array out_facies = mlx_vector_array_new();
        mlx_vector_array out_wells = mlx_vector_array_new();
        mlx_vector_array out_seismic = mlx_vector_array_new();
        int rc =
            facies_dataloader_next(dl, &out_facies, &out_wells, &out_seismic, s);
        if (rc == 2)
        {
            mlx_vector_array_free(out_facies);
            mlx_vector_array_free(out_wells);
            mlx_vector_array_free(out_seismic);
            break;
        }
        else if (rc != 0)
        {
            fprintf(stderr, "dataloader_next error: %d\n", rc);
            mlx_vector_array_free(out_facies);
            mlx_vector_array_free(out_wells);
            mlx_vector_array_free(out_seismic);
            break;
        }
        size_t nsc = mlx_vector_array_size(out_facies);
        printf(" batch %d: stacked scales = %zu\n", batch_idx, nsc);

        /* If MLX model context is available, collect true per-scale metrics */
        if (fg && nsc > 0)
        {
            /* build facies_pyramid array (mlx_array* per scale) */
            mlx_array *facies_store = (mlx_array *)malloc(sizeof(mlx_array) * nsc);
            mlx_array **facies_pyramid =
                (mlx_array **)malloc(sizeof(mlx_array *) * nsc);
            for (size_t si = 0; si < nsc; ++si)
            {
                mlx_array tmp;
                if (mlx_vector_array_get(&tmp, out_facies, si) != 0)
                {
                    /* create dummy zeros if unavailable */
                    int shape[3] = {crop_size, crop_size, num_img_channels};
                    mlx_zeros(&facies_store[si], shape, 3, MLX_FLOAT32, s);
                    facies_pyramid[si] = &facies_store[si];
                }
                else
                {
                    /* copy tmp into store */
                    facies_store[si] = tmp;
                    facies_pyramid[si] = &facies_store[si];
                }
            }

            /* wells and seismic pyramids (optional) */
            mlx_array *wells_store = NULL;
            mlx_array **wells_pyramid = NULL;
            if (mlx_vector_array_size(out_wells) > 0)
            {
                size_t wn = mlx_vector_array_size(out_wells);
                wells_store = (mlx_array *)malloc(sizeof(mlx_array) * wn);
                wells_pyramid = (mlx_array **)malloc(sizeof(mlx_array *) * wn);
                for (size_t wi = 0; wi < wn; ++wi)
                {
                    mlx_array tmpw;
                    if (mlx_vector_array_get(&tmpw, out_wells, wi) == 0)
                    {
                        wells_store[wi] = tmpw;
                        wells_pyramid[wi] = &wells_store[wi];
                    }
                    else
                    {
                        wells_pyramid[wi] = NULL;
                    }
                }
            }

            mlx_array *seis_store = NULL;
            mlx_array **seis_pyramid = NULL;
            if (mlx_vector_array_size(out_seismic) > 0)
            {
                size_t sn = mlx_vector_array_size(out_seismic);
                seis_store = (mlx_array *)malloc(sizeof(mlx_array) * sn);
                seis_pyramid = (mlx_array **)malloc(sizeof(mlx_array *) * sn);
                for (size_t si = 0; si < sn; ++si)
                {
                    mlx_array tmps;
                    if (mlx_vector_array_get(&tmps, out_seismic, si) == 0)
                    {
                        seis_store[si] = tmps;
                        seis_pyramid[si] = &seis_store[si];
                    }
                    else
                    {
                        seis_pyramid[si] = NULL;
                    }
                }
            }

            int *active_scales = (int *)malloc(sizeof(int) * nsc);
            for (size_t si = 0; si < nsc; ++si)
                active_scales[si] = (int)si;
            int indexes_arr[1] = {0};
            MLXResults *res = NULL;
            /* Use default training hyperparams for metric computation */
            float lambda_diversity = 0.0f;
            float well_loss_penalty = 10.0f;
            float alpha = 10.0f;
            float lambda_grad = 0.1f;

            if (mlx_faciesgan_collect_metrics_and_grads(
                    fg, indexes_arr, 1, active_scales, (int)nsc, facies_pyramid, NULL,
                    wells_pyramid, NULL, seis_pyramid, lambda_diversity,
                    well_loss_penalty, alpha, lambda_grad, &res) == 0 &&
                res)
            {
                /* Build JSON metrics and send to Python visualizer */
                char metrics[4096];
                int off = 0;
                off += snprintf(metrics + off, sizeof(metrics) - off, "{");
                for (int sc = 0; sc < res->n_scales; ++sc)
                {
                    MLXScaleResults *sr = &res->scales[sc];
                    float g_total = 0.0f, g_adv = 0.0f, g_rec = 0.0f;
                    if (sr->metrics.total)
                    {
                        mlx_array_eval(*sr->metrics.total);
                        const float *pdata = mlx_array_data_float32(*sr->metrics.total);
                        if (pdata)
                            g_total = pdata[0];
                    }
                    if (sr->metrics.fake)
                    {
                        mlx_array_eval(*sr->metrics.fake);
                        const float *pdata = mlx_array_data_float32(*sr->metrics.fake);
                        if (pdata)
                            g_adv = pdata[0];
                    }
                    if (sr->metrics.rec)
                    {
                        mlx_array_eval(*sr->metrics.rec);
                        const float *pdata = mlx_array_data_float32(*sr->metrics.rec);
                        if (pdata)
                            g_rec = pdata[0];
                    }
                    off += snprintf(metrics + off, sizeof(metrics) - off,
                                    "\"%d\":{\"d_total\":%g,\"d_real\":%g,\"d_fake\":%g,"
                                    "\"g_total\":%g,\"g_adv\":%g,\"g_rec\":%g}",
                                    sr->scale, 0.0, 0.0, 0.0, g_total, g_adv, g_rec);
                    if (sc + 1 < res->n_scales)
                        off += snprintf(metrics + off, sizeof(metrics) - off, ",");
                }
                off += snprintf(metrics + off, sizeof(metrics) - off, "}");
                pybridge_update_visualizer_from_json(batch_idx, metrics,
                                                     batch_idx * batch_size);
                mlx_results_free(res);
            }

            /* cleanup temporary stores */
            free(facies_store);
            free(facies_pyramid);
            if (wells_store)
            {
                free(wells_store);
                free(wells_pyramid);
            }
            if (seis_store)
            {
                free(seis_store);
                free(seis_pyramid);
            }
            free(active_scales);
        }
        mlx_vector_array_free(out_facies);
        mlx_vector_array_free(out_wells);
        mlx_vector_array_free(out_seismic);
        batch_idx++;
    }

    mlx_stream_free(s);
    facies_dataloader_free(dl);
    facies_dataset_free(ds);
    mlx_vector_vector_array_free(facies_pyramids);
    mlx_vector_vector_array_free(wells);
    mlx_vector_vector_array_free(seismic);

    printf("C-native trainer finished; processed %d batches\n", batch_idx);
    return 0;
}

int c_trainer_run_with_opts(const TrainningOptions *opts)
{
    if (!opts)
        return -1;
    const char *input_path = opts->input_path ? opts->input_path : NULL;
    const char *output_path = opts->output_path ? opts->output_path : NULL;
    int manual_seed = opts->manual_seed;
    int use_wells = opts->use_wells ? 1 : 0;
    int use_seismic = opts->use_seismic ? 1 : 0;
    int num_train_pyramids = opts->num_train_pyramids;
    int num_scales = opts->stop_scale + 1;
    int num_img_channels = opts->num_img_channels;
    int crop_size = opts->crop_size;
    int batch_size = opts->batch_size;
    int save_interval = opts->save_interval;
    return c_trainer_run_full(input_path, output_path, manual_seed, use_wells,
                              use_seismic, num_train_pyramids, num_scales,
                              num_img_channels, crop_size, batch_size,
                              save_interval);
}
