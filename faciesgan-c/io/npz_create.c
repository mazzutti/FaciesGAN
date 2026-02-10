#include "npz_create.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>

#include <time.h>
#include "miniz.h"
#include "miniz_zip.h"

int npz_create_from_memory(const char *npz_path, const char **member_names, const void **member_bufs, const size_t *member_sizes, int n_members)
{
    if (!npz_path || !member_names || !member_bufs || !member_sizes || n_members <= 0)
        return -1;

    mz_zip_archive za;
    memset(&za, 0, sizeof(za));
    if (!mz_zip_writer_init_file(&za, npz_path, 0))
        return -1;
    for (int i = 0; i < n_members; ++i)
    {
        if (!mz_zip_writer_add_mem(&za, member_names[i], member_bufs[i], member_sizes[i], MZ_BEST_COMPRESSION))
        {
            mz_zip_writer_end(&za);
            return -1;
        }
    }
    if (!mz_zip_writer_finalize_archive(&za))
    {
        mz_zip_writer_end(&za);
        return -1;
    }
    mz_zip_writer_end(&za);
    return 0;
}

int npz_pack_npy_dir(const char *dir, const char *archive_name) {
    if (!dir || !archive_name)
        return -1;

    /* Build output path: <dir>/<archive_name> */
    char npz_path[4096];
    snprintf(npz_path, sizeof(npz_path), "%s/%s", dir, archive_name);

    /* Collect .npy filenames from the directory */
    DIR *d = opendir(dir);
    if (!d)
        return -1;

    char **npy_paths = NULL;
    char **npy_names = NULL;
    int n = 0, cap = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        size_t len = strlen(ent->d_name);
        if (len < 5 || strcmp(ent->d_name + len - 4, ".npy") != 0)
            continue;
        if (n >= cap) {
            cap = cap == 0 ? 16 : cap * 2;
            npy_paths = (char **)realloc(npy_paths, cap * sizeof(char *));
            npy_names = (char **)realloc(npy_names, cap * sizeof(char *));
        }
        char fullpath[4096];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dir, ent->d_name);
        npy_paths[n] = strdup(fullpath);
        npy_names[n] = strdup(ent->d_name);
        n++;
    }
    closedir(d);

    if (n == 0) {
        free(npy_paths);
        free(npy_names);
        return 0; /* nothing to pack */
    }

    /* Create zip archive from the .npy files on disk */
    mz_zip_archive za;
    memset(&za, 0, sizeof(za));
    if (!mz_zip_writer_init_file(&za, npz_path, 0)) {
        for (int i = 0; i < n; ++i) {
            free(npy_paths[i]);
            free(npy_names[i]);
        }
        free(npy_paths);
        free(npy_names);
        return -1;
    }

    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (!mz_zip_writer_add_file(&za, npy_names[i], npy_paths[i],
                                    NULL, 0, MZ_BEST_COMPRESSION)) {
            ok = 0;
            break;
        }
    }

    if (ok)
        ok = mz_zip_writer_finalize_archive(&za);
    mz_zip_writer_end(&za);

    /* Remove individual .npy files on success */
    if (ok) {
        for (int i = 0; i < n; ++i)
            unlink(npy_paths[i]);
    }

    for (int i = 0; i < n; ++i) {
        free(npy_paths[i]);
        free(npy_names[i]);
    }
    free(npy_paths);
    free(npy_names);
    return ok ? 0 : -1;
}
