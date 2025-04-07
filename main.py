import os

import torch
import random
import json
import argparse

from datetime import datetime
from dateutil import tz

from train import Trainer
from log import init_output_logging
from config import CHECKPOINT_PATH, OPT_FILE

def get_arguments():
    parser = argparse.ArgumentParser()

    # workspace:
    parser.add_argument("--use_cpu", action="store_true", help="use cpu")
    parser.add_argument("--gpu_device", type=int, help="which GPU to use", default=0)
    parser.add_argument("--input_path", help="input facie path", required=True)

    # load, input, save configurations:
    parser.add_argument("--manual_seed", type=int, help="manual seed")
    parser.add_argument("--out_path", help="output folder path", default="facies_gan")
    parser.add_argument("--stop_scale", type=int, help="stop scale", default=4)
    parser.add_argument("--facie_num_channels", type=int, help="facie number of channels", default=1)
    parser.add_argument(
        "--img_color_range", type=int, nargs=2, help="range of values in the input facie", default=[0, 255]
    )
    parser.add_argument("--crop_size", type=int, help="crop size to train the facie", default=256)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Total batch size - e.g: num_gpus = 2, batch_size = 128 then, effectively, 64')

    # networks hyperparameters:
    parser.add_argument("--num_feature", type=int, help="initial number of features in each layer", default=32)
    parser.add_argument("--min_num_feature", type=int, help="minimal number of features in each layer", default=32)
    parser.add_argument("--kernel_size", type=int, help="kernel size", default=3)
    parser.add_argument("--num_layer", type=int, help="number of layers in each scale", default=5)
    parser.add_argument("--stride", help="stride", default=1)
    parser.add_argument("--padding_size", type=int, help="net pad size", default=0)

    # pyramid parameters:
    parser.add_argument("--noise_amp", type=float, help="adaptive noise cont weight", default=0.1)
    parser.add_argument("--min_size", type=int, help="facie minimal size at the coarser scale", default=12)
    parser.add_argument("--max_size", type=int, help="facie minimal size at the coarser scale", default=256)

    # optimization hyperparameters:
    parser.add_argument("--num_iter", type=int, default=2000, help="number of epochs to train per scale")
    parser.add_argument("--gamma", type=float, help="scheduler gamma", default=0.9)
    parser.add_argument("--lr_g", type=float, default=5e-5, help="learning rate, default=5e-8")
    parser.add_argument("--lr_d", type=float, default=5e-5, help="learning rate, default=5e-8")
    parser.add_argument("--lr_decay", type=int, default=1000, help="number of epochs before lr decay")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
    parser.add_argument("--generator_steps", type=int, help="Generator inner steps", default=3)
    parser.add_argument("--discriminator_steps", type=int, help="Discriminator inner steps", default=3)
    parser.add_argument("--lambda_grad", type=float, help="gradient penalty weight", default=0.1)
    parser.add_argument("--alpha", type=float, help="reconstruction loss weight", default=10)
    parser.add_argument("--save_interval", type=int, help="save log interval", default=100)
    parser.add_argument("--num_real_facies", type=int,
                        help="Number of real facies to use in the grid plot", default=5)
    parser.add_argument("--num_generated_per_real", type=int,
                        help="Number of generated facies per real facie to use in the grid plot", default=5)
    parser.add_argument("--num_train_facies", type=int,
                        help="Number of train facies to use in the FaciesGAN training", default=200)
    parser.add_argument("--wells", type=int,
                        help="list of well indices to train the model from",
                        nargs='+',
                        default=tuple())

    return parser


if __name__ == "__main__":
    argument_parser = get_arguments()
    options = argument_parser.parse_args()

    if options.manual_seed is not None:
        random.seed(options.manual_seed)
        torch.manual_seed(options.manual_seed)

    timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
    options.out_path = os.path.join(CHECKPOINT_PATH, f"{timestamp}_{options.out_path}")
    options.start_scale = 0

    os.makedirs(options.out_path, exist_ok=True)

    # Save the input parameters options
    with open(os.path.join(options.out_path, OPT_FILE), "w") as file:
        json.dump(vars(options), file, indent=4) # type: ignore

    init_output_logging(os.path.join(options.out_path, "log.txt"))
    print(vars(options))

    device = torch.device(
        f"cuda:{options.gpu_device}" if torch.cuda.is_available()
        else f"mps:{options.gpu_device}" if torch.backends.mps.is_available()
        else "cpu"
    )

    trainer = Trainer(device, options)
    trainer.train()
