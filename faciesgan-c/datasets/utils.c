#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <ctype.h>
#include <stdio.h>
#include <limits.h>

static int has_image_ext(const char *name)
{
    const char *ext = strrchr(name, '.');
    if (!ext)
        return 0;
    ++ext;
    char low[16];
    size_t i = 0;
    while (ext[i] && i + 1 < sizeof(low))
    {
        low[i] = (char)tolower((unsigned char)ext[i]);
        i++;
    }
    low[i] = '\0';
    return (strcmp(low, "png") == 0 || strcmp(low, "jpg") == 0 || strcmp(low, "jpeg") == 0 || strcmp(low, "bmp") == 0);
}

static int has_model_ext(const char *name)
{
    const char *ext = strrchr(name, '.');
    if (!ext)
        return 0;
    ++ext;
    return (strcmp(ext, "pt") == 0 || strcmp(ext, "pth") == 0);
}

int datasets_list_image_files(const char *data_root, const char *subdir, char ***files, int *count)
{
    if (!data_root || !subdir || !files || !count)
        return -1;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", data_root, subdir);
    DIR *d = opendir(path);
    if (!d)
    {
        *files = NULL;
        *count = 0;
        return 0;
    }
    struct dirent *ent;
    char **list = NULL;
    int n = 0;
    while ((ent = readdir(d)) != NULL)
    {
        if (ent->d_type == DT_DIR)
            continue;
        if (!has_image_ext(ent->d_name))
            continue;
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        char *s = strdup(full);
        if (!s)
            continue;
        char **tmp = realloc(list, sizeof(char *) * (n + 1));
        if (!tmp)
        {
            free(s);
            break;
        }
        list = tmp;
        list[n++] = s;
    }
    closedir(d);
    /* sort */
    if (n > 1)
    {
        qsort(list, n, sizeof(char *), (int (*)(const void *, const void *))strcmp);
    }
    *files = list;
    *count = n;
    return 0;
}

int datasets_list_model_files(const char *data_root, const char *subdir, char ***files, int *count)
{
    if (!data_root || !subdir || !files || !count)
        return -1;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", data_root, subdir);
    DIR *d = opendir(path);
    if (!d)
    {
        *files = NULL;
        *count = 0;
        return 0;
    }
    struct dirent *ent;
    char **list = NULL;
    int n = 0;
    while ((ent = readdir(d)) != NULL)
    {
        if (ent->d_type == DT_DIR)
            continue;
        if (!has_model_ext(ent->d_name))
            continue;
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        char *s = strdup(full);
        if (!s)
            continue;
        char **tmp = realloc(list, sizeof(char *) * (n + 1));
        if (!tmp)
        {
            free(s);
            break;
        }
        list = tmp;
        list[n++] = s;
    }
    closedir(d);
    if (n > 1)
        qsort(list, n, sizeof(char *), (int (*)(const void *, const void *))strcmp);
    *files = list;
    *count = n;
    return 0;
}

int datasets_generate_scales(const TrainningOptions *opts, int channels_last, DatasetScale **out, int *out_count)
{
    if (!opts || !out || !out_count)
        return -1;
    int stop_scale = opts->stop_scale;
    int crop_size = opts->crop_size;
    int max_size = opts->max_size > 0 ? opts->max_size : crop_size;
    int min_size = opts->min_size > 0 ? opts->min_size : 1;
    int batch = opts->batch_size > 0 ? opts->batch_size : 1;
    int channels = opts->num_img_channels > 0 ? opts->num_img_channels : 1;

    int nscales = stop_scale + 1;
    DatasetScale *arr = malloc(sizeof(DatasetScale) * nscales);
    if (!arr)
        return -1;

    double scale_factor = 1.0;
    if (stop_scale > 0)
    {
        scale_factor = pow((double)min_size / (double)((max_size < crop_size ? max_size : crop_size)), 1.0 / (double)stop_scale);
    }

    for (int i = 0; i < nscales; ++i)
    {
        double s = pow(scale_factor, (double)(stop_scale - i));
        double base = (double)((max_size < crop_size ? max_size : crop_size)) * s;
        int out_wh = (int)(round(base));
        if (out_wh % 2 != 0)
            out_wh += 1;
        arr[i].batch = batch;
        arr[i].height = out_wh;
        arr[i].width = out_wh;
        arr[i].channels = channels;
    }

    *out = arr;
    *out_count = nscales;
    return 0;
}
