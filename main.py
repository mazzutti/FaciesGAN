import json
import os
import random
from datetime import datetime

import torch
from dateutil import tz

from argparse import ArgumentParser
from config import CHECKPOINT_PATH, OPT_FILE
from log import init_output_logging
from options import TrainningOptions
from train import Trainer


def get_arguments() -> ArgumentParser:
    parser = ArgumentParser()

    # workspace:
    parser.add_argument(
        "--use-cpu", action="store_true", help="use cpu")
    parser.add_argument(
        "--gpu-device", type=int, help="which GPU to use", default=0)
    parser.add_argument(
        "--input-path", help="input facie path", required=True)

    # load, input, save configurations:
    parser.add_argument("--manual-seed", type=int, help="manual seed")
    parser.add_argument(
        "--output-path", help="output folder path", default="facies_gan")
    parser.add_argument("--stop-scale", type=int, help="stop scale", default=6)
    parser.add_argument(
        "--facie-num-channels", type=int, help="facie number of channels", default=3
    )
    parser.add_argument(
        "--img-color-range",
        type=int,
        nargs=2,
        help="range of values in the input facie",
        default=[0, 255],
    )
    parser.add_argument(
        "--crop-size", type=int, help="crop size to train the facie", default=256)
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Total batch size - e.g: num_gpus = 2, batch_size = 128 then, effectively, 64",
    )

    # networks hyperparameters:
    parser.add_argument(
        "--num-features",
        type=int,
        help="initial number of features in each layer",
        default=32,
    )
    parser.add_argument(
        "--min-num-features",
        type=int,
        help="minimal number of features in each layer",
        default=32,
    )
    parser.add_argument(
        "--kernel-size", type=int, help="kernel size", default=3)
    parser.add_argument(
        "--num-layers", type=int, help="number of layers in each scale", default=5)
    parser.add_argument(
        "--stride", help="stride", default=1)
    parser.add_argument(
        "--padding-size", type=int, help="net pad size", default=0)

    # pyramid parameters:
    parser.add_argument(
        "--noise-amp", type=float, help="adaptive noise cont weight", default=0.1)

    # optimization hyperparameters:
    parser.add_argument(
        "--num-iter", type=int, default=2000, help="number of epochs to train per scale"
    )
    parser.add_argument(
        "--gamma", type=float, help="scheduler gamma", default=0.9)
    parser.add_argument(
        "--lr-g", type=float, default=5e-5, help="learning rate, default=5e-8")
    parser.add_argument("--lr-d", type=float, default=5e-5, help="learning rate, default=5e-8")
    parser.add_argument(
        "--lr-decay", type=int, default=1000, help="number of epochs before lr decay"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
    parser.add_argument(
        "--generator-steps", type=int, help="Generator inner steps", default=3)
    parser.add_argument(
        "--discriminator-steps", type=int, help="Discriminator inner steps", default=3
    )
    parser.add_argument(
        "--lambda-grad", type=float, help="gradient penalty weight", default=0.1)
    parser.add_argument(
        "--alpha", type=float, help="reconstruction loss weight", default=10)
    parser.add_argument(
        "--save-interval", type=int, help="save log interval", default=100)
    parser.add_argument(
        "--num-real-facies",
        type=int,
        help="Number of real facies to use in the grid plot",
        default=5,
    )
    parser.add_argument(
        "--num-generated-per-real",
        type=int,
        help="Number of generated facies per real facies to use in the grid plot",
        default=5,
    )
    parser.add_argument(
        "--num-train-pyramids",
        type=int,
        help="Number of train pyramids to use in the FaciesGAN training",
        default=200,
    )

    parser.add_argument(
        "--wells",
        type=int,
        help="list of well indices to train the model from",
        nargs="+",
        default=tuple(),
    )

    parser.add_argument(
        "--regen-npy-gz",
        action="store_false",
        type=bool,
        help="regenerate the npy.gz files from the input facies"
    )

    return parser


if __name__ == "__main__":
    argument_parser = get_arguments()
    options = argument_parser.parse_args(namespace=TrainningOptions())

    if options.manual_seed is not None:
        random.seed(options.manual_seed)
        torch.manual_seed(options.manual_seed)  # type: ignore

    timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
    options.output_path = os.path.join(
        CHECKPOINT_PATH, f"{timestamp}_{options.output_path}")
    options.start_scale = 0

    os.makedirs(options.output_path, exist_ok=True)

    # Save the input parameters options
    with open(os.path.join(options.output_path, OPT_FILE), "w") as file:
        json.dump(vars(options), file, indent=4)  # type: ignore

    init_output_logging(os.path.join(options.output_path, "log.txt"))

    device = torch.device(
        f"cuda:{options.gpu_device}"
        if torch.cuda.is_available()
        else f"mps:{options.gpu_device}" 
            if torch.backends.mps.is_available() else "cpu"
    )

    trainer = Trainer(device, options)
    trainer.train()
