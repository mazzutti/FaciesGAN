#include "../datasets/collate.h"
#include "../datasets/dataloader.h"
#include "../utils_extra.h"
#include <mlx/c/array.h>
#include <mlx/c/vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct batch_sampler_ctx_s {
  size_t n;
  size_t *indices;
  size_t pos;
} batch_sampler_ctx;

static int example_batch_sampler_next(void *ctx, size_t *out_indices,
                                      int max_count, int *out_count) {
  batch_sampler_ctx *b = (batch_sampler_ctx *)ctx;
  if (!b || !out_indices || !out_count)
    return 1;
  if (b->pos >= b->n)
    return 2; // finished
  int filled = 0;
  while (filled < max_count && b->pos < b->n) {
    out_indices[filled++] = b->indices[b->pos++];
  }
  *out_count = filled;
  return 0;
}

static int example_batch_sampler_reset(void *ctx) {
  batch_sampler_ctx *b = (batch_sampler_ctx *)ctx;
  if (!b)
    return 1;
  b->pos = 0;
  return 0;
}

static mlx_array make_array_with_value(float v) {
  int shape[4] = {1, 1, 2, 2};
  int ndim = 4;
  size_t elems = 1;
  for (int i = 0; i < ndim; ++i)
    elems *= (size_t)shape[i];
  float *buf = (float *)malloc(sizeof(float) * elems);
  for (size_t i = 0; i < elems; ++i)
    buf[i] = v;
  mlx_array a = mlx_array_new_data(buf, shape, ndim, MLX_FLOAT32);
  free(buf);
  return a;
}

int main(void) {
  printf("Dataloader batch_sampler test\n");
  const size_t n_samples = 6;
  /* build facies pyramids: one scale per sample */
  mlx_vector_vector_array facies = mlx_vector_vector_array_new();
  for (size_t i = 0; i < n_samples; ++i) {
    mlx_vector_array sample = mlx_vector_array_new();
    mlx_array a = make_array_with_value((float)(i + 1));
    if (mlx_vector_array_append_value(sample, a)) {
      fprintf(stderr, "append failed\n");
      return 1;
    }
    mlx_array_free(a);
    if (mlx_vector_vector_array_append_value(facies, sample)) {
      fprintf(stderr, "append vv failed\n");
      return 1;
    }
    mlx_vector_array_free(sample);
  }

  /* empty wells/seismic */
  mlx_vector_vector_array wells = mlx_vector_vector_array_new();
  mlx_vector_vector_array seismic = mlx_vector_vector_array_new();

  facies_dataset *ds = NULL;
  if (facies_dataset_new(&ds, facies, wells, seismic) != 0) {
    fprintf(stderr, "failed to create dataset\n");
    return 1;
  }

  /* prepare batch sampler ctx: fixed batches of size 2 */
  size_t *idxs = (size_t *)malloc(sizeof(size_t) * n_samples);
  for (size_t i = 0; i < n_samples; ++i)
    idxs[i] = i;
  batch_sampler_ctx bctx = {n_samples, idxs, 0};

  facies_dataloader *dl = NULL;
  if (facies_dataloader_new_ex(
          &dl, ds, 2, false, false, 0, 2, 2, false, 2000, NULL, NULL, false,
          NULL, NULL, NULL, example_batch_sampler_next, &bctx,
          example_batch_sampler_reset, NULL, NULL, 0, NULL, NULL) != 0) {
    fprintf(stderr, "failed to create dataloader\n");
    return 1;
  }

  int rc = 0;
  int iter = 0;
  mlx_stream s = mlx_default_cpu_stream_new();
  while (1) {
    mlx_vector_array facs = mlx_vector_array_new();
    mlx_vector_array wells_out = mlx_vector_array_new();
    mlx_vector_array seis_out = mlx_vector_array_new();
    rc = facies_dataloader_next(dl, &facs, &wells_out, &seis_out, s);
    if (rc == 2)
      break; /* finished */
    if (rc != 0) {
      fprintf(stderr, "next failed rc=%d\n", rc);
      break;
    }
    size_t nscales = mlx_vector_array_size(facs);
    printf("batch %d: nscales=%zu\n", iter++, nscales);
    if (nscales > 0) {
      mlx_array arr = mlx_array_new();
      if (mlx_vector_array_get(&arr, facs, 0) == 0) {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(arr, &buf, &elems, &ndim, &shape) == 0) {
          printf("  first scale array ndim=%d shape[0]=%d elems=%zu "
                 "first_val=%f\n",
                 ndim, shape[0], elems, buf[0]);
          free(buf);
          free(shape);
        }
        mlx_array_free(arr);
      }
    }
    mlx_vector_array_free(facs);
    mlx_vector_array_free(wells_out);
    mlx_vector_array_free(seis_out);
  }

  facies_dataloader_free(dl);
  mlx_stream_free(s);
  facies_dataset_free(ds);
  free(idxs);
  printf("done\n");
  return 0;
}
