#include "npz_create.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
