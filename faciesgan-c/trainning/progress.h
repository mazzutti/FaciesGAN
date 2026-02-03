// Simple terminal progress bar (tqdm-like) for C
#ifndef FACIESGAN_PROGRESS_H
#define FACIESGAN_PROGRESS_H

#include <stddef.h>

/* Initialize progress bar with total iterations and optional width (0 = auto).
 * desc may be NULL. Returns 0 on success. */
int fg_progress_init(const char *desc, size_t total, int width);

/* Update progress to `done` (0..total). Should be called monotonically. */
void fg_progress_update(size_t done);

/* Finish and clear progress bar. */
void fg_progress_finish(void);

#endif
