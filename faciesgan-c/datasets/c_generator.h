#ifndef FACIESGAN_DATASETS_C_GENERATOR_H
#define FACIESGAN_DATASETS_C_GENERATOR_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

    /* Generate pyramids in C using stb_image + stb_image_resize when available.
     * Writes per-sample per-scale .npy files into cache_dir/sample_<i>/facies_<scale>.npy
     * Returns 0 on success, non-zero otherwise.
     */
    int generate_pyramids_c(const char *input_path, const char *cache_dir, int num_samples, int stop_scale, int crop_size, int num_img_channels, int use_wells, int use_seismic, int seed);

#ifdef __cplusplus
}
#endif

#endif
