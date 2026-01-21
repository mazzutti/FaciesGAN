#include "func_cache.h"
#include "c_generator.h"
#include "io/npz_create.h"
#include "io/npz_unzip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
#include <stdint.h>

static uint64_t fnv1a_hash(const char *s)
{
    uint64_t h = 14695981039346656037ULL;
    const unsigned char *p = (const unsigned char *)s;
    while (*p)
    {
        h ^= (uint64_t)(*p++);
        h *= 1099511628211ULL;
    }
    return h;
}

int ensure_function_cache(const char *input_path, const char *cache_dir, int desired_num,
                          int stop_scale, int crop_size, int num_img_channels,
                          int use_wells, int use_seismic, int manual_seed,
                          char *out_cache_npz, size_t out_len, int *out_num_samples)
{
    if (!input_path || !cache_dir || !out_cache_npz || !out_num_samples)
        return -1;

    char keybuf[512];
    snprintf(keybuf, sizeof(keybuf), "%s|scales=%d|crop=%d|ch=%d|w=%d|s=%d|seed=%d",
             input_path, stop_scale, crop_size, num_img_channels, use_wells, use_seismic, manual_seed);
    uint64_t key = fnv1a_hash(keybuf);
    char cache_npz[PATH_MAX];
    snprintf(cache_npz, PATH_MAX, "%s/func_cache_%016llx.npz", cache_dir, (unsigned long long)key);

    /* If cache exists, probe members to count available samples */
    if (access(cache_npz, F_OK) == 0)
    {
        int available = 0;
        for (int i = 0; i < desired_num; ++i)
        {
            char member[64];
            snprintf(member, sizeof(member), "sample_%d/facies_0.npy", i);
            mlx_io_reader reader;
            if (npz_extract_member_to_mlx_reader(cache_npz, member, &reader) == 0)
            {
                available++;
                mlx_io_reader_free(reader);
            }
            else
                break;
        }
        *out_num_samples = available > 0 ? (available < desired_num ? available : desired_num) : 0;
        strncpy(out_cache_npz, cache_npz, out_len);
        return 0;
    }

    /* Otherwise generate using native C generator into temporary sample_* dirs */
    int num_samples = desired_num > 0 ? desired_num : 1;
    int grc = generate_pyramids_c(input_path, cache_dir, num_samples, stop_scale,
                                  crop_size, num_img_channels, use_wells ? 1 : 0,
                                  use_seismic ? 1 : 0, manual_seed);
    if (grc != 0)
    {
        fprintf(stderr, "generate_pyramids_c failed rc=%d\n", grc);
        return grc;
    }

    /* Package generated files into .npz */
    int idx = 0;
    int n_members = num_samples * (stop_scale + 1) + 1;
    char **names = (char **)malloc(sizeof(char *) * n_members);
    void **bufs = (void **)malloc(sizeof(void *) * n_members);
    size_t *sizes = (size_t *)malloc(sizeof(size_t) * n_members);
    if (!names || !bufs || !sizes)
    {
        fprintf(stderr, "Out of memory creating packaging arrays\n");
        return -1;
    }

    for (int si = 0; si < num_samples; ++si)
    {
        char sample_dir[PATH_MAX];
        snprintf(sample_dir, PATH_MAX, "%s/sample_%d", cache_dir, si);
        for (int sc = 0; sc < stop_scale + 1; ++sc)
        {
            char fname[PATH_MAX];
            snprintf(fname, PATH_MAX, "%s/facies_%d.npy", sample_dir, sc);
            FILE *f = fopen(fname, "rb");
            if (!f)
            {
                fprintf(stderr, "Failed to open %s\n", fname);
                for (int j = 0; j < idx; ++j)
                {
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
            if (!buf)
            {
                fclose(f);
                fprintf(stderr, "Out of memory reading %s\n", fname);
                for (int j = 0; j < idx; ++j)
                {
                    free(names[j]);
                    free(bufs[j]);
                }
                free(names);
                free(bufs);
                free(sizes);
                return -1;
            }
            if (fread(buf, 1, (size_t)fsize, f) != (size_t)fsize)
            {
                fclose(f);
                free(buf);
                fprintf(stderr, "Failed to read %s\n", fname);
                for (int j = 0; j < idx; ++j)
                {
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
        }
    }

    names[idx] = strdup("meta.json");
    bufs[idx] = strdup("{}");
    sizes[idx] = strlen((const char *)bufs[idx]);

    if (npz_create_from_memory(cache_npz, (const char **)names, (const void **)bufs, sizes, idx + 1) != 0)
    {
        fprintf(stderr, "npz_create_from_memory failed\n");
        for (int j = 0; j <= idx; ++j)
        {
            free(names[j]);
            free(bufs[j]);
        }
        free(names);
        free(bufs);
        free(sizes);
        return -1;
    }

    for (int j = 0; j <= idx; ++j)
    {
        free(names[j]);
        free(bufs[j]);
    }
    free(names);
    free(bufs);
    free(sizes);

    /* remove temporary sample_* dirs */
    for (int si = 0; si < num_samples; ++si)
    {
        for (int sc = 0; sc < stop_scale + 1; ++sc)
        {
            char fname[PATH_MAX];
            snprintf(fname, PATH_MAX, "%s/sample_%d/facies_%d.npy", cache_dir, si, sc);
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
