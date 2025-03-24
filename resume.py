import os
import torch
import random
import json
import argparse
import glob

from types import SimpleNamespace
from train import Trainer
from log import init_output_logging
from config import G_FILE, RESULT_FACIES_PATH, OPT_FILE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tuning", action="store_true", help="fine-tune the models")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="checkpoint path to continue the training",
        required=True
    )
    parser.add_argument("--num_iter", type=int, help="number of epochs for fine-tuning")
    parser.add_argument("--start_scale", type=int, default=0, help="start scale for fine-tuning")

    arguments = parser.parse_args()

    if arguments.fine_tuning and arguments.num_iter is None:
        raise ValueError("Number of iterations required for fine-tuning.")

    # Load the saved input parameter options for the trained models
    with open(os.path.join(arguments.checkpoint_path, OPT_FILE), "r") as f:
        options = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

    options.out_path = arguments.checkpoint_path

    init_output_logging(os.path.join(options.out_path, "log.txt"))

    if options.manual_seed is not None:
        random.seed(options.manual_seed)
        torch.manual_seed(options.manual_seed)

    if arguments.finetuning:
        print("Fine-Tuning: %d iter\n" % arguments.num_iter)
        options.num_iter = arguments.num_iter

    device = torch.device(
        f"cuda:{options.gpu_device}" if torch.cuda.is_available()
        else f"mps:{options.gpu_device}" if torch.backends.mps.is_available()
        else "cpu"
    )
    trainer = Trainer(device, options, arguments.finetuning, arguments.checkpoint_path)

    if arguments.fine_tuning:
        trainer.load(arguments.checkpoint_path, arguments.start_scale - 1)
    else:
        # Get last saved scale path
        last_scale = max(map(int, next(os.walk(arguments.checkpoint_path))[1]))
        last_scale_path = os.path.join(arguments.checkpoint_path, str(last_scale))

        # If the last scale folder was created, but no models were saved, remove the folder
        if not os.path.isfile(os.path.join(last_scale_path, G_FILE)):
            for file in glob.glob(os.path.join(last_scale_path, RESULT_FACIES_PATH, "*")):
                os.remove(file)
            os.removedirs(os.path.join(last_scale_path, RESULT_FACIES_PATH))
            for file in glob.glob(os.path.join(last_scale_path, "*")):
                os.remove(file)
            os.removedirs(last_scale_path)

        trainer.load(arguments.checkpoint_path)

    trainer.train()
