#include "checkpoint.h"
#include "array_helpers.h"
#include "optimizer.h"
#include "train_step.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple binary checkpoint format:
 * [magic 8 bytes] "MLXCKPT\0"
 * [version int32]
 * [gen_count int32]
 * for each gen param: dtype(int32), ndim(int32), shape(int32*ndim),
 * nbytes(int64), data(bytes) [disc_count int32] for each disc param: same
 * format

static const char *MAGIC = "MLXCKPT";

static int write_array(FILE *f, const mlx_array *a) {
  if (!f || !a)
    return -1;
  mlx_array arr = *a;
  int dtype = (int)mlx_array_dtype(arr);
  int ndim = (int)mlx_array_ndim(arr);
  const int *shape = mlx_array_shape(arr);
  size_t nbytes = mlx_array_nbytes(arr);
  if (fwrite(&dtype, sizeof(int), 1, f) != 1)
    return -1;
  if (fwrite(&ndim, sizeof(int), 1, f) != 1)
    return -1;
  if (ndim > 0) {
    if (fwrite((void *)shape, sizeof(int), ndim, f) != (size_t)ndim)
      return -1;
  }
  long long nb = (long long)nbytes;
  if (fwrite(&nb, sizeof(long long), 1, f) != 1)
    return -1;
  /* ensure data is available */
  mlx_array_eval(arr);
  const char *data = NULL;
  switch (mlx_array_dtype(arr)) {
  case MLX_FLOAT32:
    data = (const char *)mlx_array_data_float32(arr);
    break;
  case MLX_FLOAT64:
    data = (const char *)mlx_array_data_float64(arr);
    break;
  default:
    data = (const char *)mlx_array_data_float32(arr);
    break;
  }
  if (!data)
    return -1;
  if (fwrite(data, 1, (size_t)nb, f) != (size_t)nb)
    return -1;
  return 0;
}

static int read_array(FILE *f, mlx_array *out) {
  if (!f || !out)
    return -1;
  int dtype = 0;
  if (fread(&dtype, sizeof(int), 1, f) != 1)
    return -1;
  int ndim = 0;
  if (fread(&ndim, sizeof(int), 1, f) != 1)
    return -1;
  int *shape = NULL;
  if (ndim > 0) {
    if (mlx_alloc_int_array(&shape, ndim) != 0)
      return -1;
    if (fread(shape, sizeof(int), ndim, f) != (size_t)ndim) {
      mlx_free_int_array(&shape, &ndim);
      return -1;
    }
  }
  long long nb = 0;
  if (fread(&nb, sizeof(long long), 1, f) != 1) {
    if (shape)
      mlx_free_int_array(&shape, &ndim);
    return -1;
  }
  void *buf = malloc((size_t)nb);
  if (!buf) {
    if (shape)
      mlx_free_int_array(&shape, &ndim);
    return -1;
  }
  if (fread(buf, 1, (size_t)nb, f) != (size_t)nb) {
    free(buf);
    if (shape)
      mlx_free_int_array(&shape, &ndim);
    return -1;
  }
  int rc = mlx_array_set_data(out, buf, shape, ndim, (mlx_dtype)dtype);
  free(buf);
  if (shape)
    mlx_free_int_array(&shape, &ndim);
  return rc == 0 ? 0 : -1;
}

int mlx_checkpoint_save(const char *path, MLXFaciesGAN *m) {
  if (!path || !m)
    return -1;
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  /* magic */
  if (fwrite(MAGIC, 1, 6, f) != 6) {
    fclose(f);
    return -1;
  }
  int version = 1;
  fwrite(&version, sizeof(int), 1, f);

  /* write shapes (if present) */
  int *shapes = NULL;
  int n_shapes = 0;
  if (&mlx_faciesgan_get_shapes_flat) {
    if (mlx_faciesgan_get_shapes_flat(m, &shapes, &n_shapes) == 0 && shapes &&
        n_shapes > 0) {
      fwrite(&n_shapes, sizeof(int), 1, f);
      /* each shape is 4 ints */
      fwrite(shapes, sizeof(int), n_shapes * 4, f);
      {
        int _sh_n = n_shapes * 4;
        mlx_free_int_array(&shapes, &_sh_n);
      }
    } else {
      int z = 0;
      fwrite(&z, sizeof(int), 1, f);
    }
  } else {
    int z = 0;
    fwrite(&z, sizeof(int), 1, f);
  }

  /* write noise amps */
  float *amps = NULL;
  int n_amps = 0;
  if (&mlx_faciesgan_get_noise_amps) {
    if (mlx_faciesgan_get_noise_amps(m, &amps, &n_amps) == 0 && amps &&
        n_amps > 0) {
      fwrite(&n_amps, sizeof(int), 1, f);
      fwrite(amps, sizeof(float), n_amps, f);
      mlx_free_float_buf(&amps, &n_amps);
    } else {
      int z = 0;
      fwrite(&z, sizeof(int), 1, f);
    }
  } else {
    int z = 0;
    fwrite(&z, sizeof(int), 1, f);
  }

  /* generator params */
  MLXGenerator *g = mlx_faciesgan_build_generator(m);
  int gen_n = 0;
  mlx_array **gen_params = NULL;
  if (g)
    gen_params = mlx_generator_get_parameters(g, &gen_n);
  fwrite(&gen_n, sizeof(int), 1, f);
  for (int i = 0; i < gen_n; ++i) {
    if (write_array(f, gen_params[i]) != 0) {
      if (gen_params)
        mlx_generator_free_parameters_list(gen_params);
      fclose(f);
      return -1;
    }
  }
  if (gen_params)
    mlx_generator_free_parameters_list(gen_params);

  /* discriminator params */
  MLXDiscriminator *d = mlx_faciesgan_build_discriminator(m);
  int disc_n = 0;
  mlx_array **disc_params = NULL;
  if (d)
    disc_params = mlx_discriminator_get_parameters(d, &disc_n);
  fwrite(&disc_n, sizeof(int), 1, f);
  for (int i = 0; i < disc_n; ++i) {
    if (write_array(f, disc_params[i]) != 0) {
      if (disc_params)
        mlx_discriminator_free_parameters_list(disc_params);
      fclose(f);
      return -1;
    }
  }
  if (disc_params)
    mlx_discriminator_free_parameters_list(disc_params);

  fclose(f);
  return 0;
}

int mlx_checkpoint_load(const char *path, MLXFaciesGAN *m) {
  if (!path || !m)
    return -1;
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;
  char magic[7] = {0};
  if (fread(magic, 1, 6, f) != 6) {
    fclose(f);
    return -1;
  }
  if (strncmp(magic, MAGIC, 6) != 0) {
    fclose(f);
    return -1;
  }
  int version = 0;
  if (fread(&version, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }

  /* generator */
  int gen_n = 0;
  if (fread(&gen_n, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  MLXGenerator *g = mlx_faciesgan_build_generator(m);
  int cur_gen_n = 0;
  mlx_array **gen_params = NULL;
  if (g)
    gen_params = mlx_generator_get_parameters(g, &cur_gen_n);
  if (cur_gen_n != gen_n) {
    /* mismatch: still read values but cannot apply */
    for (int i = 0; i < gen_n; ++i) {
      mlx_array tmp = mlx_array_new();
      read_array(f, &tmp);
      mlx_array_free(tmp);
    }
  } else {
    for (int i = 0; i < gen_n; ++i) {
      mlx_array tmp = mlx_array_new();
      if (read_array(f, &tmp) != 0) {
        if (gen_params)
          mlx_generator_free_parameters_list(gen_params);
        fclose(f);
        return -1;
      }
      /* set param data */
      mlx_array_set(gen_params[i], tmp);
      mlx_array_free(tmp);
    }
  }
  if (gen_params)
    mlx_generator_free_parameters_list(gen_params);

  /* discriminator */
  int disc_n = 0;
  if (fread(&disc_n, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  MLXDiscriminator *d = mlx_faciesgan_build_discriminator(m);
  int cur_disc_n = 0;
  mlx_array **disc_params = NULL;
  if (d)
    disc_params = mlx_discriminator_get_parameters(d, &cur_disc_n);
  if (cur_disc_n != disc_n) {
    for (int i = 0; i < disc_n; ++i) {
      mlx_array tmp = mlx_array_new();
      read_array(f, &tmp);
      mlx_array_free(tmp);
    }
  } else {
    for (int i = 0; i < disc_n; ++i) {
      mlx_array tmp = mlx_array_new();
      if (read_array(f, &tmp) != 0) {
        if (disc_params)
          mlx_discriminator_free_parameters_list(disc_params);
        fclose(f);
        return -1;
      }
      mlx_array_set(disc_params[i], tmp);
      mlx_array_free(tmp);
    }
  }
  if (disc_params)
    mlx_discriminator_free_parameters_list(disc_params);

  /* read shapes */
  int n_shapes = 0;
  if (fread(&n_shapes, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (n_shapes > 0) {
    int *shapes = NULL;
    int shapes_n = n_shapes * 4;
    if (mlx_alloc_int_array(&shapes, shapes_n) != 0) {
      fclose(f);
      return -1;
    }
    if (fread(shapes, sizeof(int), shapes_n, f) != (size_t)shapes_n) {
      mlx_free_int_array(&shapes, &shapes_n);
      fclose(f);
      return -1;
    }
    mlx_faciesgan_set_shapes(m, shapes, n_shapes);
    mlx_free_int_array(&shapes, &shapes_n);
  }

  /* read noise amps */
  int n_amps = 0;
  if (fread(&n_amps, sizeof(int), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (n_amps > 0) {
    float *amps = NULL;
    if (mlx_alloc_float_buf(&amps, n_amps) != 0) {
      fclose(f);
      return -1;
    }
    if (fread(amps, sizeof(float), n_amps, f) != (size_t)n_amps) {
      mlx_free_float_buf(&amps, &n_amps);
      fclose(f);
      return -1;
    }
    mlx_faciesgan_set_noise_amps(m, amps, n_amps);
    mlx_free_float_buf(&amps, &n_amps);
  }

  fclose(f);
  return 0;
}
