// Minimal C port of main.py: CLI and optional hand-off to compiled example
#include "options.h"
#include "utils.h"
#include "trainning/c_trainer.h"
#include "data_files.h"

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

enum
{
    OPT_USE_CPU = 1000,
    OPT_GPU_DEVICE,
    OPT_INPUT_PATH,
    OPT_MANUAL_SEED,
    OPT_OUTPUT_FULLPATH,
    OPT_OUTPUT_PATH,
    OPT_STOP_SCALE,
    OPT_NUM_IMG_CHANNELS,
    OPT_IMG_COLOR_RANGE,
    OPT_CROP_SIZE,
    OPT_BATCH_SIZE,
    OPT_NUM_FEATURES,
    OPT_MIN_NUM_FEATURES,
    OPT_KERNEL_SIZE,
    OPT_NUM_LAYERS,
    OPT_STRIDE,
    OPT_PADDING_SIZE,
    OPT_NOISE_AMP,
    OPT_MIN_NOISE_AMP,
    OPT_SCALE0_NOISE_AMP,
    OPT_NUM_PARALLEL_SCALES,
    OPT_NOISE_CHANNELS,
    OPT_USE_PROFILER,
    OPT_NUM_ITER,
    OPT_GAMMA,
    OPT_LR_G,
    OPT_LR_D,
    OPT_LR_DECAY,
    OPT_BETA1,
    OPT_BETA2,
    OPT_EPS,
    OPT_ADAM_BIAS_CORRECTION,
    OPT_WEIGHT_DECAY,
    OPT_GENERATOR_STEPS,
    OPT_DISCRIMINATOR_STEPS,
    OPT_LAMBDA_GRAD,
    OPT_ALPHA,
    OPT_LAMBDA_DIVERSITY,
    OPT_SAVE_INTERVAL,
    OPT_NUM_REAL_FACIES,
    OPT_NUM_GENERATED_PER_REAL,
    OPT_NUM_TRAIN_PYRAMIDS,
    OPT_NUM_WORKERS,
    OPT_USE_WELLS,
    OPT_WELLS_MASK_COLUMNS,
    OPT_WELL_LOSS_PENALTY,
    OPT_USE_SEISMIC,
    OPT_NO_TENSORBOARD,
    OPT_NO_PLOT_FACIES,
    OPT_COMPILE_BACKEND,
    OPT_USE_MLX,
    OPT_HAND_OFF_TO_C,
};

static void print_help(void)
{
    printf("faciesgan usage:\n");
    printf("  -h, --help                          Show this help text\n");
    printf("  -i <path>, --input-path <path>     Input data root (must contain %s/)\n", DF_DIR_FACIES);
    printf("  -o <path>, --output-path <path>    Output directory (default: results)\n");
    printf("      --output-fullpath <path>       Use exact output path (no timestamp)\n");
    printf("  -n <n>, --num-iter <n>             Iterations per scale\n");
    printf("      --use-cpu                      Run on CPU\n");
    printf("      --gpu-device <id>              GPU device id (default 0)\n");
    printf("      --manual-seed <seed>           RNG manual seed\n");
    printf("      --stop-scale <n>               Stop training at this scale\n");
    printf("      --num-img-channels <n>         Number of image channels\n");
    printf("      --img-color-range lo,hi        Image color range (lo,hi)\n");
    printf("      --crop-size <px>               Crop size for training\n");
    printf("      --batch-size <n>               Batch size\n");
    printf("      --num-parallel-scales <n>      Number of parallel scales\n");
    printf("      --num-train-pyramids <n>       Number of training pyramids\n");
    printf("      --num-workers <n>              Data loader worker count\n");
    printf("      --use-profiler                 Enable profiler hooks\n");
    printf("      --noise-channels <n>           Noise channels for generator\n");
    printf("      --num-features <n>             Base feature count for generator\n");
    printf("      --min-num-features <n>         Minimum feature maps\n");
    printf("      --kernel-size <n>              Convolution kernel size\n");
    printf("      --num-layers <n>               Number of layers in blocks\n");
    printf("      --stride <n>                   Convolution stride\n");
    printf("      --padding-size <n>             Padding size for convolutions\n");
    printf("      --noise-amp <f>                Noise amplitude\n");
    printf("      --min-noise-amp <f>            Minimum noise amplitude\n");
    printf("      --scale0-noise-amp <f>         Scale-0 noise amplitude\n");
    printf("      --generator-steps <n>          Generator steps per iteration\n");
    printf("      --discriminator-steps <n>      Discriminator steps per iteration\n");
    printf("      --lambda-grad <f>              Gradient penalty weight\n");
    printf("      --alpha <f>                    Alpha (used for some losses)\n");
    printf("      --lambda-diversity <f>         Diversity loss weight\n");
    printf("      --beta1 <f>                    Adam beta1\n");
    printf("      --lr-g <f>                     Generator learning rate\n");
    printf("      --lr-d <f>                     Discriminator learning rate\n");
    printf("      --lr-decay <n>                 Learning-rate decay (epochs)\n");
    printf("      --save-interval <n>            Save model every n iterations\n");
    printf("      --num-real-facies <n>          Number of real facies classes\n");
    printf("      --num-generated-per-real <n>   Generated samples per real facies\n");
    printf("      --wells-mask-columns c1,c2..   Columns to mask from well logs\n");
    printf("      --use-wells                    Use well data during training\n");
    printf("      --well-loss-penalty <f>        Penalty weight for well-loss (ignored)\n");
    printf("      --use-seismic                  Use seismic data during training\n");
    printf("      --no-tensorboard               Disable tensorboard logging\n");
    printf("      --no-plot-facies               Disable facies plotting\n");
    printf("      --compile-backend              Compile C backend before running\n");
    printf("      --use-mlx                      Use MLX backend instead of CUDA\n");
    printf("      --hand-off-to-c                Hand off execution to compiled C trainer\n");
}

int main(int argc, char **argv)
{
    /* Simple CLI variables (only commonly-used options are parsed here). */
    char *input_path = NULL;
    char *output_path = NULL;
    char *output_fullpath = NULL;
    int manual_seed = -1;
    int gpu_device = 0;
    bool use_cpu = false;
    bool use_mlx = false;
    bool use_wells = false;
    bool use_seismic = false;
    bool no_tensorboard = false;
    bool no_plot_facies = false;
    bool compile_backend = false;
    bool hand_off_to_c = false;
    bool use_profiler = false;
    int num_iter = -1;
    int stop_scale = -1;
    int num_img_channels = -1;
    int crop_size = -1;
    int batch_size = -1;
    int num_parallel_scales = -1;
    int num_train_pyramids = -1;
    int num_workers = -1;
    int save_interval = -1;
    int noise_channels = -1;
    int img_color_lo = -1;
    int img_color_hi = -1;
    int num_features = -1;
    int min_num_features = -1;
    int kernel_size = -1;
    int num_layers = -1;
    int stride = -1;
    int padding_size = -1;
    double noise_amp = -1.0;
    double min_noise_amp = -1.0;
    double scale0_noise_amp = -1.0;
    int generator_steps = -1;
    int discriminator_steps = -1;
    double lambda_grad = -1.0;
    double alpha = -1.0;
    double lambda_diversity = -1.0;
    double beta1 = -1.0;
    double lr_g = -1.0;
    double lr_d = -1.0;
    int lr_decay = -1;
    int num_real_facies = -1;
    int num_generated_per_real = -1;

    int *wells_mask_columns = NULL;
    size_t wells_mask_count = 0;

    static struct option long_options[] = {
        {"use-cpu", no_argument, 0, OPT_USE_CPU},
        {"gpu-device", required_argument, 0, OPT_GPU_DEVICE},
        {"input-path", required_argument, 0, OPT_INPUT_PATH},
        {"manual-seed", required_argument, 0, OPT_MANUAL_SEED},
        {"output-fullpath", required_argument, 0, OPT_OUTPUT_FULLPATH},
        {"output-path", required_argument, 0, OPT_OUTPUT_PATH},
        {"stop-scale", required_argument, 0, OPT_STOP_SCALE},
        {"num-img-channels", required_argument, 0, OPT_NUM_IMG_CHANNELS},
        {"crop-size", required_argument, 0, OPT_CROP_SIZE},
        {"batch-size", required_argument, 0, OPT_BATCH_SIZE},
        {"num-parallel-scales", required_argument, 0, OPT_NUM_PARALLEL_SCALES},
        {"num-train-pyramids", required_argument, 0, OPT_NUM_TRAIN_PYRAMIDS},
        {"num-workers", required_argument, 0, OPT_NUM_WORKERS},
        {"use-profiler", no_argument, 0, OPT_USE_PROFILER},
        {"noise-channels", required_argument, 0, OPT_NOISE_CHANNELS},
        {"num-features", required_argument, 0, OPT_NUM_FEATURES},
        {"min-num-features", required_argument, 0, OPT_MIN_NUM_FEATURES},
        {"kernel-size", required_argument, 0, OPT_KERNEL_SIZE},
        {"num-layers", required_argument, 0, OPT_NUM_LAYERS},
        {"stride", required_argument, 0, OPT_STRIDE},
        {"padding-size", required_argument, 0, OPT_PADDING_SIZE},
        {"noise-amp", required_argument, 0, OPT_NOISE_AMP},
        {"min-noise-amp", required_argument, 0, OPT_MIN_NOISE_AMP},
        {"scale0-noise-amp", required_argument, 0, OPT_SCALE0_NOISE_AMP},
        {"generator-steps", required_argument, 0, OPT_GENERATOR_STEPS},
        {"discriminator-steps", required_argument, 0, OPT_DISCRIMINATOR_STEPS},
        {"lambda-grad", required_argument, 0, OPT_LAMBDA_GRAD},
        {"alpha", required_argument, 0, OPT_ALPHA},
        {"lambda-diversity", required_argument, 0, OPT_LAMBDA_DIVERSITY},
        {"beta1", required_argument, 0, OPT_BETA1},
        {"lr-g", required_argument, 0, OPT_LR_G},
        {"lr-d", required_argument, 0, OPT_LR_D},
        {"lr-decay", required_argument, 0, OPT_LR_DECAY},
        {"save-interval", required_argument, 0, OPT_SAVE_INTERVAL},
        {"num-real-facies", required_argument, 0, OPT_NUM_REAL_FACIES},
        {"num-generated-per-real", required_argument, 0, OPT_NUM_GENERATED_PER_REAL},
        {"num-iter", required_argument, 0, OPT_NUM_ITER},
        {"wells-mask-columns", required_argument, 0, OPT_WELLS_MASK_COLUMNS},
        {"use-wells", no_argument, 0, OPT_USE_WELLS},
        {"well-loss-penalty", required_argument, 0, OPT_WELL_LOSS_PENALTY},
        {"use-seismic", no_argument, 0, OPT_USE_SEISMIC},
        {"no-tensorboard", no_argument, 0, OPT_NO_TENSORBOARD},
        {"no-plot-facies", no_argument, 0, OPT_NO_PLOT_FACIES},
        {"compile-backend", no_argument, 0, OPT_COMPILE_BACKEND},
        {"use-mlx", no_argument, 0, OPT_USE_MLX},
        {"hand-off-to-c", no_argument, 0, OPT_HAND_OFF_TO_C},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hi:o:n:", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_help();
            return 0;
        case 'i':
            input_path = optarg;
            break;
        case 'o':
            output_path = optarg;
            break;
        case 'n':
            num_iter = atoi(optarg);
            break;
        case OPT_USE_CPU:
            use_cpu = true;
            break;
        case OPT_GPU_DEVICE:
            gpu_device = atoi(optarg);
            break;
        case OPT_INPUT_PATH:
            input_path = optarg;
            break;
        case OPT_MANUAL_SEED:
            manual_seed = atoi(optarg);
            break;
        case OPT_OUTPUT_FULLPATH:
            output_fullpath = optarg;
            break;
        case OPT_OUTPUT_PATH:
            output_path = optarg;
            break;
        case OPT_STOP_SCALE:
            stop_scale = atoi(optarg);
            break;
        case OPT_NUM_IMG_CHANNELS:
            num_img_channels = atoi(optarg);
            break;
        case OPT_CROP_SIZE:
            crop_size = atoi(optarg);
            break;
        case OPT_BATCH_SIZE:
            batch_size = atoi(optarg);
            break;
        case OPT_NUM_PARALLEL_SCALES:
            num_parallel_scales = atoi(optarg);
            break;
        case OPT_NOISE_CHANNELS:
            noise_channels = atoi(optarg);
            break;
        case OPT_IMG_COLOR_RANGE:
        {
            /* accept either "lo,hi" or "lo hi" */
            if (strchr(optarg, ',') != NULL)
            {
                if (sscanf(optarg, "%d,%d", &img_color_lo, &img_color_hi) != 2)
                {
                    img_color_lo = 0;
                    img_color_hi = 255;
                }
            }
            else
            {
                img_color_lo = atoi(optarg);
                if (optind < argc && argv[optind][0] != '-')
                {
                    img_color_hi = atoi(argv[optind]);
                    optind++;
                }
                else
                {
                    img_color_hi = img_color_lo;
                }
            }
        }
        break;
        case OPT_NUM_FEATURES:
            num_features = atoi(optarg);
            break;
        case OPT_MIN_NUM_FEATURES:
            min_num_features = atoi(optarg);
            break;
        case OPT_KERNEL_SIZE:
            kernel_size = atoi(optarg);
            break;
        case OPT_NUM_LAYERS:
            num_layers = atoi(optarg);
            break;
        case OPT_STRIDE:
            stride = atoi(optarg);
            break;
        case OPT_PADDING_SIZE:
            padding_size = atoi(optarg);
            break;
        case OPT_NOISE_AMP:
            noise_amp = atof(optarg);
            break;
        case OPT_MIN_NOISE_AMP:
            min_noise_amp = atof(optarg);
            break;
        case OPT_SCALE0_NOISE_AMP:
            scale0_noise_amp = atof(optarg);
            break;
        case OPT_GENERATOR_STEPS:
            generator_steps = atoi(optarg);
            break;
        case OPT_DISCRIMINATOR_STEPS:
            discriminator_steps = atoi(optarg);
            break;
        case OPT_LAMBDA_GRAD:
            lambda_grad = atof(optarg);
            break;
        case OPT_ALPHA:
            alpha = atof(optarg);
            break;
        case OPT_LAMBDA_DIVERSITY:
            lambda_diversity = atof(optarg);
            break;
        case OPT_BETA1:
            beta1 = atof(optarg);
            break;
        case OPT_LR_G:
            lr_g = atof(optarg);
            break;
        case OPT_LR_D:
            lr_d = atof(optarg);
            break;
        case OPT_LR_DECAY:
            lr_decay = atoi(optarg);
            break;
        case OPT_SAVE_INTERVAL:
            save_interval = atoi(optarg);
            break;
        case OPT_NUM_REAL_FACIES:
            num_real_facies = atoi(optarg);
            break;
        case OPT_NUM_GENERATED_PER_REAL:
            num_generated_per_real = atoi(optarg);
            break;
        case OPT_NUM_TRAIN_PYRAMIDS:
            num_train_pyramids = atoi(optarg);
            break;
        case OPT_NUM_WORKERS:
            num_workers = atoi(optarg);
            break;
        case OPT_USE_PROFILER:
            use_profiler = true;
            break;
        case OPT_NUM_ITER:
            num_iter = atoi(optarg);
            break;
        case OPT_USE_WELLS:
            use_wells = true;
            break;
        case OPT_WELLS_MASK_COLUMNS:
        {
            /* Accept comma-separated or space separated values. */
            char *tokbuf = strdup(optarg);
            if (tokbuf)
            {
                char *tok = strtok(tokbuf, ",");
                while (tok)
                {
                    wells_mask_columns = (int *)realloc(wells_mask_columns, sizeof(int) * (wells_mask_count + 1));
                    wells_mask_columns[wells_mask_count++] = atoi(tok);
                    tok = strtok(NULL, ",");
                }
                free(tokbuf);
            }
            /* consume following numeric args until next option */
            while (optind < argc && argv[optind][0] != '-')
            {
                wells_mask_columns = (int *)realloc(wells_mask_columns, sizeof(int) * (wells_mask_count + 1));
                wells_mask_columns[wells_mask_count++] = atoi(argv[optind]);
                optind++;
            }
        }
        break;
        case OPT_WELL_LOSS_PENALTY:
            /* ignored for now, kept for parity */
            break;
        case OPT_USE_SEISMIC:
            use_seismic = true;
            break;
        case OPT_NO_TENSORBOARD:
            no_tensorboard = true;
            break;
        case OPT_NO_PLOT_FACIES:
            no_plot_facies = true;
            break;
        case OPT_COMPILE_BACKEND:
            compile_backend = true;
            break;
        case OPT_USE_MLX:
            use_mlx = true;
            break;
        case OPT_HAND_OFF_TO_C:
            hand_off_to_c = true;
            break;
        default:
            break;
        }
    }

    /* Build a TrainningOptions structure using defaults then override from CLI */
    TrainningOptions *topt = mlx_options_new_trainning_defaults();
    if (!topt)
        return 1;

    if (input_path)
    {
        free(topt->input_path);
        topt->input_path = strdup(input_path);
    }
    if (manual_seed >= 0)
        topt->manual_seed = manual_seed;
    if (stop_scale >= 0)
        topt->stop_scale = stop_scale;
    if (num_img_channels > 0)
        topt->num_img_channels = num_img_channels;
    if (crop_size > 0)
        topt->crop_size = crop_size;
    if (batch_size > 0)
        topt->batch_size = batch_size;
    if (num_parallel_scales > 0)
        topt->num_parallel_scales = num_parallel_scales;
    if (num_train_pyramids > 0)
        topt->num_train_pyramids = num_train_pyramids;
    if (num_workers >= 0)
        topt->num_workers = num_workers;
    if (noise_channels > 0)
        topt->noise_channels = noise_channels;
    if (img_color_lo >= 0)
    {
        topt->img_color_min = img_color_lo;
        topt->img_color_max = img_color_hi >= 0 ? img_color_hi : img_color_lo;
    }
    if (num_features > 0)
        topt->num_feature = num_features;
    if (min_num_features > 0)
        topt->min_num_feature = min_num_features;
    if (kernel_size > 0)
        topt->kernel_size = kernel_size;
    if (num_layers > 0)
        topt->num_layer = num_layers;
    if (stride >= 0)
        topt->stride = stride;
    if (padding_size >= 0)
        topt->padding_size = padding_size;
    if (noise_amp >= 0.0)
        topt->noise_amp = noise_amp;
    if (min_noise_amp >= 0.0)
        topt->min_noise_amp = min_noise_amp;
    if (scale0_noise_amp >= 0.0)
        topt->scale0_noise_amp = scale0_noise_amp;
    if (generator_steps > 0)
        topt->generator_steps = generator_steps;
    if (discriminator_steps > 0)
        topt->discriminator_steps = discriminator_steps;
    if (lambda_grad >= 0.0)
        topt->lambda_grad = lambda_grad;
    if (alpha >= 0.0)
        topt->alpha = (int)alpha;
    if (lambda_diversity >= 0.0)
        topt->lambda_diversity = lambda_diversity;
    if (beta1 >= 0.0)
        topt->beta1 = beta1;
    if (lr_g >= 0.0)
        topt->lr_g = lr_g;
    if (lr_d >= 0.0)
        topt->lr_d = lr_d;
    if (lr_decay >= 0)
        topt->lr_decay = lr_decay;
    if (num_real_facies > 0)
        topt->num_real_facies = num_real_facies;
    if (num_generated_per_real > 0)
        topt->num_generated_per_real = num_generated_per_real;
    if (num_iter > 0)
        topt->num_iter = num_iter;
    if (save_interval > 0)
        topt->save_interval = save_interval;

    topt->use_wells = use_wells;
    topt->use_seismic = use_seismic;
    topt->use_cpu = use_cpu;
    topt->use_mlx = use_mlx;
    topt->gpu_device = gpu_device;
    topt->enable_tensorboard = !no_tensorboard;
    topt->enable_plot_facies = !no_plot_facies;
    topt->compile_backend = compile_backend;
    topt->use_profiler = use_profiler;
    topt->hand_off_to_c = hand_off_to_c;

    /* Determine final output path: if output_fullpath provided use it
       else join provided output_path (or default) with timestamp. */
    char timestamp[TIMESTAMP_BUFSZ];
    format_timestamp(timestamp, sizeof(timestamp));

    char final_out[PATH_BUFSZ];
    if (output_fullpath)
    {
        strncpy(final_out, output_fullpath, sizeof(final_out) - 1);
        final_out[sizeof(final_out) - 1] = '\0';
    }
    else
    {
        const char *base = output_path ? output_path : topt->output_path;
        join_path(final_out, sizeof(final_out), base, timestamp);
    }

    free(topt->output_path);
    topt->output_path = strdup(final_out);

    ensure_dir(topt->output_path);

    write_options_json(topt, wells_mask_columns, wells_mask_count);

    char logpath[PATH_BUFSZ];
    join_path(logpath, sizeof(logpath), topt->output_path, "log.txt");
    FILE *lf = fopen(logpath, "a");
    if (lf)
    {
        fprintf(lf, "FaciesGAN run initialized: %s\n", topt->output_path);
        fclose(lf);
    }

    /* Print header similar to Python's main.py */
    char device_str[128];
    if (use_mlx)
        snprintf(device_str, sizeof(device_str), "MLX (gpu %d)", gpu_device);
    else if (use_cpu)
        snprintf(device_str, sizeof(device_str), "cpu");
    else
        snprintf(device_str, sizeof(device_str), "gpu:%d", gpu_device);

    printf("\n============================================================\n");
    printf("PARALLEL LAPGAN TRAINING\n");
    printf("============================================================\n");
    printf("Device: %s\n", device_str);
    printf("Training scales: %d to %d\n", 0, topt->stop_scale);
    printf("Parallel scales: %d\n", topt->num_parallel_scales);
    printf("Iterations per scale: %d\n", topt->num_iter);
    printf("Output path: %s\n", topt->output_path);
    printf("============================================================\n\n");

    printf("Launching C-native trainer...\n");
    int rc = c_trainer_run_with_opts(topt);

    mlx_options_free_trainning(topt);
    if (wells_mask_columns)
        free(wells_mask_columns);
    return rc;
}
