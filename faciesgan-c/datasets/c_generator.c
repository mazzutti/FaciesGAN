#include "c_generator.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
/* Avoid using stb_image_resize2.h (missing in third_party); implement a
    small bilinear resizer in-place instead of depending on stb_image_resize. */

#include "../io/npz_unzip.h"
#include <mlx/c/array.h>
#include <mlx/c/io.h>
#include <mlx/c/stream.h>

int generate_pyramids_c(const char *input_path, const char *cache_dir, int num_samples, int stop_scale, int crop_size, int num_img_channels, int use_wells, int use_seismic, int seed)
{
    // List facies image files
    char **files = NULL;
    int count = 0;
    if (datasets_list_image_files(input_path, "facies", &files, &count) != 0 || count == 0)
    {
        if (files)
        {
            for (int i = 0; i < count; ++i)
                free(files[i]);
            free(files);
        }
        return -1;
    }

    // Ensure cache dir exists
    struct stat st = {0};
    if (stat(cache_dir, &st) != 0)
    {
        if (mkdir(cache_dir, 0755) != 0 && errno != EEXIST)
            return -1;
    }

    // For each sample up to num_samples, load image, create scales, save .npy per scale
    for (int si = 0; si < num_samples && si < count; ++si)
    {
        const char *img_path = files[si];
        // Build sample dir
        char sample_dir[PATH_MAX];
        snprintf(sample_dir, PATH_MAX, "%s/sample_%d", cache_dir, si);
        if (stat(sample_dir, &st) != 0)
        {
            if (mkdir(sample_dir, 0755) != 0 && errno != EEXIST)
                return -1;
        }

        int w, h, comp;
        unsigned char *data = stbi_load(img_path, &w, &h, &comp, num_img_channels);
        if (!data)
        {
            fprintf(stderr, "stbi_load failed for %s\n", img_path);
            return -1;
        }

        // Create scales: simple powers of two down to crop_size
        int sc = 0;
        for (int scale = stop_scale; scale >= 0; --scale)
        {
            int target = crop_size >> (stop_scale - scale);
            if (target <= 0)
                target = crop_size;
            unsigned char *resized = malloc((size_t)target * target * num_img_channels);
            if (!resized)
            {
                stbi_image_free(data);
                return -1;
            }
            // simple bilinear resize (avoids dependency on stb_image_resize2.h)
            {
                int iw = w, ih = h, ow = target, oh = target, c = num_img_channels;
                for (int oy = 0; oy < oh; ++oy)
                {
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
                    for (int ox = 0; ox < ow; ++ox)
                    {
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
                        for (int ch = 0; ch < c; ++ch)
                        {
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
            if (!fdata)
            {
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
            if (mlx_array_set_data(&a, fdata, shape, 3, MLX_FLOAT32) != 0)
            {
                mlx_stream_free(s);
                free(fdata);
                free(resized);
                stbi_image_free(data);
                return -1;
            }

            char fname[PATH_MAX];
            snprintf(fname, PATH_MAX, "%s/facies_%d.npy", sample_dir, sc);
            if (mlx_save(fname, a) != 0)
            {
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
            sc++;
        }

        stbi_image_free(data);
    }

    for (int i = 0; i < count; ++i)
        free(files[i]);
    free(files);
    return 0;
}
