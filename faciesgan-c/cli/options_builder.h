#ifndef FACIESGAN_OPTIONS_BUILDER_H
#define FACIESGAN_OPTIONS_BUILDER_H

#include "cli/cli_args.h"
#include "options.h"

TrainningOptions *trainning_options_from_cli(CLIArgs *args);

#endif
