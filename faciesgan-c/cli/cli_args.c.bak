#include "cli_args.h"
#include "cli_defs.h"
#include "cli_help.h"
#include "cli_opts.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void CLIArgs_init(CLIArgs *args) {
    if (!args)
        return;
    memset(args, 0, sizeof(*args));
    args->manual_seed = -1;
    args->stop_scale = -1;
    args->num_img_channels = -1;
    args->crop_size = -1;
    args->batch_size = -1;
    args->num_parallel_scales = -1;
    args->num_train_pyramids = -1;
    args->num_workers = -1;
    args->save_interval = -1;
    args->noise_channels = -1;
    args->img_color_lo = -1;
    args->img_color_hi = -1;
    args->num_features = -1;
    args->min_num_features = -1;
    args->kernel_size = -1;
    args->num_layers = -1;
    args->stride = -1;
    args->padding_size = -1;
    args->noise_amp = -1.0;
    args->min_noise_amp = -1.0;
    args->scale0_noise_amp = -1.0;
    args->generator_steps = -1;
    args->discriminator_steps = -1;
    args->lambda_grad = -1.0;
    args->alpha = -1;
    args->lambda_diversity = -1.0;
    args->beta1 = -1.0;
    args->lr_g = -1.0;
    args->lr_d = -1.0;
    args->lr_decay = -1;
    args->num_real_facies = -1;
    args->num_generated_per_real = -1;
    args->num_iter = -1;
}

void CLIArgs_free(CLIArgs *args) {
    if (!args)
        return;
    free(args->input_path);
    free(args->output_path);
    /* If wells_mask_columns still owned by args, free it */
    if (args->wells_mask_columns)
        free(args->wells_mask_columns);
    memset(args, 0, sizeof(*args));
}

void CLIArgs_set_input_path(CLIArgs *args, const char *path) {
    if (!args)
        return;
    free(args->input_path);
    args->input_path = path ? strdup(path) : NULL;
}

void CLIArgs_set_output_path(CLIArgs *args, const char *path) {
    if (!args)
        return;
    free(args->output_path);
    args->output_path = path ? strdup(path) : NULL;
}

int CLIArgs_add_well_mask_column(CLIArgs *args, int col) {
    if (!args)
        return -1;
    int *p = (int *)realloc(args->wells_mask_columns,
                            sizeof(int) * (args->wells_mask_count + 1));
    if (!p)
        return -1;
    args->wells_mask_columns = p;
    args->wells_mask_columns[args->wells_mask_count++] = col;
    return 0;
}

int CLIArgs_parse_from_argv(CLIArgs *args, int argc, char **argv) {
    if (!args)
        return 1;

    /* use shared long-options from cli_defs.c */
    struct option *long_options = facies_long_options;

    int opt;
    while ((opt = getopt_long(argc, argv, "hi:o:n:", long_options, NULL)) != -1) {
        switch (opt) {
        case 'h':
            /* Show help and signal caller that help was printed. */
            print_help();
            return 2; /* help requested */
        case 'i':
            CLIArgs_set_input_path(args, optarg);
            break;
        case 'o':
            CLIArgs_set_output_path(args, optarg);
            break;
        case OPT_OUTPUT_PATH:
            CLIArgs_set_output_path(args, optarg);
            break;
        case 'n':
            args->num_iter = atoi(optarg);
            break;
        case OPT_USE_CPU:
            args->use_cpu = true;
            break;
        case OPT_GPU_DEVICE:
            args->gpu_device = atoi(optarg);
            break;
        case OPT_INPUT_PATH:
            CLIArgs_set_input_path(args, optarg);
            break;
        case OPT_MANUAL_SEED:
            args->manual_seed = atoi(optarg);
            break;
        case OPT_STOP_SCALE:
            args->stop_scale = atoi(optarg);
            break;
        case OPT_NUM_IMG_CHANNELS:
            args->num_img_channels = atoi(optarg);
            break;
        case OPT_CROP_SIZE:
            args->crop_size = atoi(optarg);
            break;
        case OPT_BATCH_SIZE:
            args->batch_size = atoi(optarg);
            break;
        case OPT_NUM_PARALLEL_SCALES:
            args->num_parallel_scales = atoi(optarg);
            break;
        case OPT_NOISE_CHANNELS:
            args->noise_channels = atoi(optarg);
            break;
        case OPT_IMG_COLOR_RANGE: {
            if (strchr(optarg, ',') != NULL) {
                if (sscanf(optarg, "%d,%d", &args->img_color_lo, &args->img_color_hi) !=
                        2) {
                    args->img_color_lo = 0;
                    args->img_color_hi = 255;
                }
            } else {
                args->img_color_lo = atoi(optarg);
                if (optind < argc && argv[optind][0] != '-') {
                    args->img_color_hi = atoi(argv[optind]);
                    optind++;
                } else {
                    args->img_color_hi = args->img_color_lo;
                }
            }
        }
        break;
        case OPT_NUM_FEATURES:
            args->num_features = atoi(optarg);
            break;
        case OPT_MIN_NUM_FEATURES:
            args->min_num_features = atoi(optarg);
            break;
        case OPT_KERNEL_SIZE:
            args->kernel_size = atoi(optarg);
            break;
        case OPT_NUM_LAYERS:
            args->num_layers = atoi(optarg);
            break;
        case OPT_STRIDE:
            args->stride = atoi(optarg);
            break;
        case OPT_PADDING_SIZE:
            args->padding_size = atoi(optarg);
            break;
        case OPT_NOISE_AMP:
            args->noise_amp = atof(optarg);
            break;
        case OPT_MIN_NOISE_AMP:
            args->min_noise_amp = atof(optarg);
            break;
        case OPT_SCALE0_NOISE_AMP:
            args->scale0_noise_amp = atof(optarg);
            break;
        case OPT_GENERATOR_STEPS:
            args->generator_steps = atoi(optarg);
            break;
        case OPT_DISCRIMINATOR_STEPS:
            args->discriminator_steps = atoi(optarg);
            break;
        case OPT_LAMBDA_GRAD:
            args->lambda_grad = atof(optarg);
            break;
        case OPT_ALPHA:
            args->alpha = atoi(optarg);
            break;
        case OPT_LAMBDA_DIVERSITY:
            args->lambda_diversity = atof(optarg);
            break;
        case OPT_BETA1:
            args->beta1 = atof(optarg);
            break;
        case OPT_LR_G:
            args->lr_g = atof(optarg);
            break;
        case OPT_LR_D:
            args->lr_d = atof(optarg);
            break;
        case OPT_LR_DECAY:
            args->lr_decay = atoi(optarg);
            break;
        case OPT_SAVE_INTERVAL:
            args->save_interval = atoi(optarg);
            break;
        case OPT_NUM_REAL_FACIES:
            args->num_real_facies = atoi(optarg);
            break;
        case OPT_NUM_GENERATED_PER_REAL:
            args->num_generated_per_real = atoi(optarg);
            break;
        case OPT_NUM_TRAIN_PYRAMIDS:
            args->num_train_pyramids = atoi(optarg);
            break;
        case OPT_NUM_WORKERS:
            args->num_workers = atoi(optarg);
            break;
        case OPT_USE_PROFILER:
            args->use_profiler = true;
            break;
        case OPT_NUM_ITER:
            args->num_iter = atoi(optarg);
            break;
        case OPT_USE_WELLS:
            args->use_wells = true;
            break;
        case OPT_WELLS_MASK_COLUMNS: {
            char *tokbuf = strdup(optarg);
            if (tokbuf) {
                char *tok = strtok(tokbuf, ",");
                while (tok) {
                    CLIArgs_add_well_mask_column(args, atoi(tok));
                    tok = strtok(NULL, ",");
                }
                free(tokbuf);
            }
            while (optind < argc && argv[optind][0] != '-') {
                CLIArgs_add_well_mask_column(args, atoi(argv[optind]));
                optind++;
            }
        }
        break;
        case OPT_WELL_LOSS_PENALTY:
            break;
        case OPT_USE_SEISMIC:
            args->use_seismic = true;
            break;
        case OPT_NO_TENSORBOARD:
            args->no_tensorboard = true;
            break;
        case OPT_NO_PLOT_FACIES:
            args->no_plot_facies = true;
            break;
        case OPT_COMPILE_BACKEND:
            args->compile_backend = true;
            break;
        case OPT_USE_MLX:
            args->use_mlx = true;
            break;
        case OPT_HAND_OFF_TO_C:
            args->hand_off_to_c = true;
            break;
        default:
            break;
        }
    }

    return 0;
}
