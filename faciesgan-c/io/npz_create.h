#ifndef MLX_C_NPZ_CREATE_H
#define MLX_C_NPZ_CREATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Create an .npz file at `npz_path` with provided members.
 * - `member_names`: array of member names (e.g. "state.json")
 * - `member_bufs`: array of pointers to member data
 * - `member_sizes`: array of sizes for each member
 * - `n_members`: number of members
 * Returns 0 on success.
int npz_create_from_memory(const char *npz_path, const char **member_names, const void **member_bufs, const size_t *member_sizes, int n_members);

#ifdef __cplusplus
}
#endif

#endif
