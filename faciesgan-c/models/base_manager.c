#include "base_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
/* Ensure memcpy prototype available under build flags */
#include <stddef.h>
/* When used with the faciesgan adapter, the user_ctx holds a
 * MLXFaciesGAN instance; free it here to avoid leaking model-owned
 * mlx_array allocations when the manager is freed. This keeps the
 * simple adapter path working without requiring callers to free the
 * user_ctx separately. */
#include "facies_gan.h"
#include "utils.h"

struct MLXBaseManager
{
    MLXTrainOptions opts;
    MLXBaseManagerCallbacks cbs;
    void *user_ctx;

    /* shapes: array of pointers to int arrays, with dims stored */
    int **shapes;
    size_t *shapes_ndim;
    size_t shapes_count;

    /* noise amplitudes */
    double *noise_amps;
    size_t noise_amps_count;

    /* active scales list */
    int *active_scales;
    size_t active_scales_count;
};

MLXBaseManager *mlx_base_manager_create(const MLXTrainOptions *opts, const MLXBaseManagerCallbacks *cbs)
{
    MLXBaseManager *m = (MLXBaseManager *)calloc(1, sizeof(MLXBaseManager));
    if (!m)
        return NULL;
    if (opts)
        m->opts = *opts;
    if (cbs)
        m->cbs = *cbs;
    return m;
}

void mlx_base_manager_set_user_ctx(MLXBaseManager *mgr, void *ctx)
{
    if (!mgr)
        return;
    mgr->user_ctx = ctx;
}

void *mlx_base_manager_get_user_ctx(MLXBaseManager *mgr)
{
    return mgr ? mgr->user_ctx : NULL;
}

void mlx_base_manager_free(MLXBaseManager *mgr)
{
    if (!mgr)
        return;
    /* If a facies-gang context was attached, free it so any nested
     * model resources (mlx_array weights) are released. */
    if (mgr->user_ctx)
    {
        MLXFaciesGAN *fg = (MLXFaciesGAN *)mgr->user_ctx;
        mlx_faciesgan_free(fg);
        mgr->user_ctx = NULL;
    }
    for (size_t i = 0; i < mgr->shapes_count; ++i)
    {
        free(mgr->shapes[i]);
    }
    free(mgr->shapes);
    free(mgr->shapes_ndim);
    free(mgr->noise_amps);
    free(mgr->active_scales);
    free(mgr);
}

int mlx_base_manager_add_shape(MLXBaseManager *mgr, const int *shape, size_t ndim)
{
    if (!mgr || !shape || ndim == 0)
        return -1;
    int **nshapes = realloc(mgr->shapes, sizeof(int *) * (mgr->shapes_count + 1));
    size_t *nndim = realloc(mgr->shapes_ndim, sizeof(size_t) * (mgr->shapes_count + 1));
    if (!nshapes || !nndim)
        return -1;
    mgr->shapes = nshapes;
    mgr->shapes_ndim = nndim;
    mgr->shapes[mgr->shapes_count] = (int *)malloc(sizeof(int) * ndim);
    if (!mgr->shapes[mgr->shapes_count])
        return -1;
    memcpy(mgr->shapes[mgr->shapes_count], shape, sizeof(int) * ndim);
    mgr->shapes_ndim[mgr->shapes_count] = ndim;
    mgr->shapes_count += 1;
    return 0;
}

size_t mlx_base_manager_shapes_count(MLXBaseManager *mgr)
{
    return mgr ? mgr->shapes_count : 0;
}

void mlx_base_manager_init_generator_for_scale(MLXBaseManager *mgr, int scale)
{
    int num_feature = mgr->opts.num_feature;
    int min_num_feature = mgr->opts.min_num_feature;
    if (mgr->cbs.create_generator_scale)
    {
        mgr->cbs.create_generator_scale(mgr, scale, num_feature, min_num_feature);
    }
    if (mgr->cbs.finalize_generator_scale)
    {
        int prev_is_spade = 0;
        int curr_is_spade = 0;
        int reinit = prev_is_spade || curr_is_spade;
        mgr->cbs.finalize_generator_scale(mgr, scale, reinit);
    }
    /* mark active */
    int *tmp = realloc(mgr->active_scales, sizeof(int) * (mgr->active_scales_count + 1));
    if (tmp)
    {
        mgr->active_scales = tmp;
        mgr->active_scales[mgr->active_scales_count++] = scale;
    }
}

void mlx_base_manager_init_discriminator_for_scale(MLXBaseManager *mgr, int scale)
{
    int num_feature = mgr->opts.num_feature;
    int min_num_feature = mgr->opts.min_num_feature;
    if (mgr->cbs.create_discriminator_scale)
    {
        mgr->cbs.create_discriminator_scale(mgr, num_feature, min_num_feature);
    }
    if (mgr->cbs.finalize_discriminator_scale)
    {
        mgr->cbs.finalize_discriminator_scale(mgr, scale);
    }
}

void mlx_base_manager_init_scales(MLXBaseManager *mgr, int start_scale, int num_scales)
{
    for (int s = start_scale; s < start_scale + num_scales; ++s)
    {
        mlx_base_manager_init_generator_for_scale(mgr, s);
        mlx_base_manager_init_discriminator_for_scale(mgr, s);
    }
}

static int dir_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

int mlx_base_manager_save_scale(MLXBaseManager *mgr, int scale, const char *path)
{
    if (!mgr || !path)
        return -1;
    if (!dir_exists(path))
    {
        if (mlx_create_dirs(path) != 0)
            return -1;
    }
    if (mgr->cbs.save_generator_state)
        mgr->cbs.save_generator_state(mgr, path, scale);
    if (mgr->cbs.save_discriminator_state)
        mgr->cbs.save_discriminator_state(mgr, path, scale);
    /* write amp if present */
    if ((size_t)scale < mgr->noise_amps_count)
    {
        char amp_path[1024];
        snprintf(amp_path, sizeof(amp_path), "%s/amp.txt", path);
        FILE *f = fopen(amp_path, "w");
        if (f)
        {
            fprintf(f, "%f\n", mgr->noise_amps[scale]);
            fclose(f);
        }
    }
    if (mgr->cbs.save_shape)
        mgr->cbs.save_shape(mgr, path, scale);
    return 0;
}

int mlx_base_manager_load(MLXBaseManager *mgr, const char *path, int load_shapes, int until_scale, int load_discriminator, int load_wells)
{
    if (!mgr || !path)
        return 0;
    int scale = 0;
    char scale_path[1024];
    while (1)
    {
        snprintf(scale_path, sizeof(scale_path), "%s/%d", path, scale);
        if (!dir_exists(scale_path))
            break;
        if (until_scale >= 0 && scale > until_scale)
            break;

        if (mgr->cbs.load_generator_state)
        {
            if (mgr->cbs.create_generator_scale)
            {
                mlx_base_manager_init_generator_for_scale(mgr, scale);
            }
            mgr->cbs.load_generator_state(mgr, scale_path, scale);
        }

        if (load_discriminator && mgr->cbs.load_discriminator_state)
        {
            if (mgr->cbs.create_discriminator_scale)
            {
                mlx_base_manager_init_discriminator_for_scale(mgr, scale);
            }
            mgr->cbs.load_discriminator_state(mgr, scale_path, scale);
        }

        /* amplitude */
        char ampfile[1024];
        snprintf(ampfile, sizeof(ampfile), "%s/amp.txt", scale_path);
        if (access(ampfile, F_OK) == 0)
        {
            if (mgr->cbs.load_amp)
                mgr->cbs.load_amp(mgr, scale_path);
            else
            {
                /* read default amp */
                FILE *f = fopen(ampfile, "r");
                if (f)
                {
                    double v = 1.0;
                    if (fscanf(f, "%lf", &v) == 1)
                    {
                        double *na = realloc(mgr->noise_amps, sizeof(double) * (mgr->noise_amps_count + 1));
                        if (na)
                        {
                            mgr->noise_amps = na;
                            mgr->noise_amps[mgr->noise_amps_count++] = v;
                        }
                    }
                    fclose(f);
                }
            }
        }

        if (load_shapes && mgr->cbs.load_shape)
            mgr->cbs.load_shape(mgr, scale_path, scale);
        if (load_wells && mgr->cbs.load_wells)
            mgr->cbs.load_wells(mgr, scale_path);

        scale += 1;
    }
    return scale;
}
