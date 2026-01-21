#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/stream.h>
#include <mlx/c/io.h>

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s <out_dir> <num_samples> <num_scales> <crop_size> <channels>\n", argv[0]);
        return 2;
    }
    const char *out = argv[1];
    int num_samples = atoi(argv[2]);
    int num_scales = atoi(argv[3]);
    int crop = atoi(argv[4]);
    int channels = atoi(argv[5]);

    struct stat st = {0};
    if (stat(out, &st) != 0)
    {
        if (mkdir(out, 0755) != 0 && errno != EEXIST)
        {
            perror("mkdir out");
            return 1;
        }
    }

    mlx_stream s = mlx_default_cpu_stream_new();

    for (int si = 0; si < num_samples; ++si)
    {
        char sample_dir[PATH_MAX];
        snprintf(sample_dir, PATH_MAX, "%s/sample_%d", out, si);
        if (stat(sample_dir, &st) != 0)
        {
            if (mkdir(sample_dir, 0755) != 0 && errno != EEXIST)
            {
                perror("mkdir sample");
                mlx_stream_free(s);
                return 1;
            }
        }

        for (int sc = 0; sc < num_scales; ++sc)
        {
            int shape[3] = {crop, crop, channels};
            mlx_array a = mlx_array_new();
            size_t elems = (size_t)crop * (size_t)crop * (size_t)channels;
            float *buf = (float *)malloc(elems * sizeof(float));
            if (!buf)
            {
                mlx_array_free(a);
                mlx_stream_free(s);
                return 1;
            }
            for (size_t i = 0; i < elems; ++i)
                buf[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

            if (mlx_array_set_data(&a, buf, shape, 3, MLX_FLOAT32) != 0)
            {
                free(buf);
                mlx_array_free(a);
                mlx_stream_free(s);
                return 1;
            }

            char fname[PATH_MAX];
            snprintf(fname, PATH_MAX, "%s/facies_%d.npy", sample_dir, sc);
            if (mlx_save(fname, a) != 0)
            {
                mlx_array_free(a);
                mlx_stream_free(s);
                return 1;
            }

            mlx_array_free(a);
            free(buf);
        }
    }

    mlx_stream_free(s);
    return 0;
}
