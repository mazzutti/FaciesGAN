#include "cli/cli_args.h"
#include "cli/cli_opts.h"
#include "cli/options_builder.h"
#include "datasets/data_files.h"
#include "options.h"
#include "trainer.h"
#include "trainning/mlx_trainer_api.h"
#include "utils.h"

#include "datasets/dataloader.h"
#include "datasets/func_cache.h"
#include "io/npz_unzip.h"
#include "main.h"
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <mlx/c/array.h>
#include <mlx/c/stream.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  char *output_path = NULL;
  char *output_fullpath = NULL;
  int gpu_device = 0;

  CLIArgs parsed_args;
  CLIArgs_init(&parsed_args);
  int parse_rc = CLIArgs_parse_from_argv(&parsed_args, argc, argv);
  if (parse_rc == 2) {
    CLIArgs_free(&parsed_args);
    return 0;
  }
  if (parse_rc != 0) {
    CLIArgs_free(&parsed_args);
    return 1;
  }

  TrainningOptions *topt = trainning_options_from_cli(&parsed_args);

  CLIArgs_free(&parsed_args);
  if (!topt)
    return 1;
  char timestamp[TIMESTAMP_BUFSZ];
  format_timestamp(timestamp, sizeof(timestamp));

  char final_out[PATH_BUFSZ];
  if (output_fullpath) {
    strncpy(final_out, output_fullpath, sizeof(final_out) - 1);
    final_out[sizeof(final_out) - 1] = '\0';
  } else {
    const char *base = output_path ? output_path : topt->output_path;
    join_path(final_out, sizeof(final_out), base, timestamp);
  }

  free(topt->output_path);
  topt->output_path = strdup(final_out);

  ensure_dir(topt->output_path);

  write_options_json(topt, topt->wells_mask_columns, topt->wells_mask_count);

  FaciesGANTrainer *ftr = facies_trainer_new(topt, gpu_device, ".checkpoints");
  if (!ftr) {
    mlx_options_free_trainning(topt);
    return 1;
  }
  int rc = facies_trainer_run(ftr);
  facies_trainer_destroy(ftr);
  mlx_options_free_trainning(topt);
  return rc;
}
