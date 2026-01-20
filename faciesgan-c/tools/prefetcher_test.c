#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../datasets/prefetcher.h"
#include "../utils_extra.h"
#include <mlx/c/array.h>

int main(void) {
  int scales[2] = {1, 2};
  PrefetcherHandle ph = prefetcher_create(8, -1, scales, 2);
  if (!ph) {
    fprintf(stderr, "prefetcher_create failed\n");
    return 1;
  }

  /* Prepare simple per-scale arrays (two scales) */
  mlx_array facies_arrs[2];
  mlx_array wells_arrs[2];
  mlx_array seismic_arrs[2];
  facies_arrs[0] = mlx_array_new();
  facies_arrs[1] = mlx_array_new();
  wells_arrs[0] = mlx_array_new();
  wells_arrs[1] = mlx_array_new();
  seismic_arrs[0] = mlx_array_new();
  seismic_arrs[1] = mlx_array_new();

  float f0[] = {1.0f};
  int fshape0[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&facies_arrs[0], f0, fshape0, 4);
  float f1[] = {2.0f};
  int fshape1[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&facies_arrs[1], f1, fshape1, 4);

  float w0[] = {0.0f, 1.0f, -1.0f, 0.0f};
  int wshape0[] = {1, 1, 1, 4};
  mlx_array_from_float_buffer(&wells_arrs[0], w0, wshape0, 4);
  float w1[] = {0.0f};
  int wshape1[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&wells_arrs[1], w1, wshape1, 4);

  float s0[] = {0.5f};
  int sshape0[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&seismic_arrs[0], s0, sshape0, 4);
  float s1[] = {0.6f};
  int sshape1[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&seismic_arrs[1], s1, sshape1, 4);
  int rc = 0;
  /* signal no more pushes so background preload can finish */
  /* Push first batch */
  rc = prefetcher_push_mlx(ph, facies_arrs, 2, wells_arrs, 2, NULL, 0,
                           seismic_arrs, 2);
  if (rc != 0) {
    fprintf(stderr, "prefetcher_push_mlx failed %d\n", rc);
  }
  /* Prepare second batch with different values to validate ordering */
  mlx_array facies_b[2];
  mlx_array wells_b[2];
  mlx_array seismic_b[2];
  facies_b[0] = mlx_array_new();
  facies_b[1] = mlx_array_new();
  wells_b[0] = mlx_array_new();
  wells_b[1] = mlx_array_new();
  seismic_b[0] = mlx_array_new();
  seismic_b[1] = mlx_array_new();
  float fb0[] = {3.0f};
  int fbshape0[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&facies_b[0], fb0, fbshape0, 4);
  float fb1[] = {4.0f};
  int fbshape1[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&facies_b[1], fb1, fbshape1, 4);
  float wb0[] = {0.0f, 0.0f, 0.0f, 0.0f};
  int wbshape0[] = {1, 1, 1, 4};
  mlx_array_from_float_buffer(&wells_b[0], wb0, wbshape0, 4);
  float wb1[] = {0.0f};
  int wbshape1[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&wells_b[1], wb1, wbshape1, 4);
  float sb0[] = {0.7f};
  int sbshape0[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&seismic_b[0], sb0, sbshape0, 4);
  float sb1[] = {0.8f};
  int sbshape1[] = {1, 1, 1, 1};
  mlx_array_from_float_buffer(&seismic_b[1], sb1, sbshape1, 4);

  rc = prefetcher_push_mlx(ph, facies_b, 2, wells_b, 2, NULL, 0, seismic_b, 2);
  if (rc != 0) {
    fprintf(stderr, "prefetcher_push_mlx (batch2) failed %d\n", rc);
  }
  /* free local arrays; prefetcher owns its copies */
  for (int i = 0; i < 2; ++i) {
    mlx_array_free(facies_b[i]);
    mlx_array_free(wells_b[i]);
    mlx_array_free(seismic_b[i]);
  }

  /* signal no more pushes so background preload can finish */
  prefetcher_mark_finished(ph);

  /* iterator will be created after freeing local arrays */
  /* free our local arrays; prefetcher holds its own copies */
  for (int i = 0; i < 2; ++i) {
    mlx_array_free(facies_arrs[i]);
    mlx_array_free(wells_arrs[i]);
    mlx_array_free(seismic_arrs[i]);
  }

  /* create iterator after buffers freed */
  PrefetcherIteratorHandle it = prefetcher_iterator_create(ph);
  if (!it) {
    fprintf(stderr, "iterator create failed\n");
    prefetcher_destroy(ph);
    return 1;
  }

  /* Iterate using the iterator API (preload + next) */
  PrefetchedPyramidsBatch *b;
  int batch_idx = 0;
  float expected[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  int failures = 0;

  prefetcher_iterator_preload(it);
  while ((b = prefetcher_iterator_next(it)) != NULL) {
    printf("batch n_scales=%d (idx=%d)\n", b->n_scales, batch_idx);
    for (int i = 0; i < b->n_scales; ++i) {
      /* Validate facies first element */
      if (b->facies && mlx_array_ndim(b->facies[i]) > 0) {
        float *buf = NULL;
        size_t elems = 0;
        int ndim = 0;
        int *shape = NULL;
        if (mlx_array_to_float_buffer(b->facies[i], &buf, &elems, &ndim,
                                      &shape) == 0) {
          float got = buf[0];
          float exp = expected[batch_idx][i];
          if (got != exp) {
            printf("  facies mismatch scale %d: got %f expected %f\n", i, got,
                   exp);
            failures++;
          }
          free(buf);
          if (shape)
            free(shape);
        }
      } else {
        printf("  facies missing for scale %d\n", i);
        failures++;
      }

      /* Validate masks: if wells present we expect a mask array */
      if (b->wells) {
        if (!b->masks) {
          printf("  expected masks but masks=NULL\n");
          failures++;
        } else {
          float *mbuf = NULL;
          size_t mele = 0;
          int mnd = 0;
          int *mshape = NULL;
          if (mlx_array_to_float_buffer(b->masks[i], &mbuf, &mele, &mnd,
                                        &mshape) == 0) {
            if (mele > 0) {
              float mv = mbuf[0];
              if (!(mv == 0.0f || mv == 1.0f)) {
                printf("  unexpected mask value %f\n", mv);
                failures++;
              }
            }
            free(mbuf);
            if (mshape)
              free(mshape);
          }
        }
      } else {
        if (b->masks) {
          printf("  unexpected masks present when wells=NULL\n");
          failures++;
        }
      }
    }

    prefetcher_free_pyramids(b);
    batch_idx++;
  }

  printf("batches seen=%d failures=%d\n", batch_idx, failures);
  prefetcher_iterator_destroy(it);
  prefetcher_destroy(ph);
  return 0;
}
