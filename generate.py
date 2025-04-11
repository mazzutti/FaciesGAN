import copy
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import json
import time
import tifffile as tif

from argparse import ArgumentParser
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances

from facies_dataset import FaciesDataset
from log import format_time
from models.facies_gan import FaciesGAN
from config import OPT_FILE
from types import SimpleNamespace

from utils import torch2np
from sklearn.manifold import MDS

def generate_facies(model: FaciesGAN, how_many: int, model_path: str, options: SimpleNamespace) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate facies using the FaciesGAN model.

        Args:
            model (FaciesGAN): The FaciesGAN model instance.
            how_many (int): Number of facies to generate.
            model_path (str): Path to the model.
            options (SimpleNamespace): Options for the model.

        Returns:
            Tuple[List[np.ndarray], List[int]]: A tuple containing a list of generated facies as numpy arrays and a list of mask indexes.
        """
        model.load(model_path, load_discriminator=False, load_masked_facies=False)
        model.generator.eval()

        mask_indexes = [random.choice(options.wells) for _ in range(how_many)]

        noises = model.get_noise(mask_indexes, rec=options.rec)

        with torch.no_grad():
            generated_facies = [
                torch2np(gen_facie.unsqueeze(0), denormalize=True)
                for gen_facie in model.generator(noises, model.noise_amp, in_facie=None)
            ]
        return generated_facies, mask_indexes


def plot_mds(fake_facies, mask_indexes, options):

    fake_facies = np.stack(fake_facies, 0).squeeze(-1)
    real_facies = dataset.facies_pyramid[-1]
    real_facies = np.reshape(torch2np(real_facies, denormalize=True), [dataset.facies_pyramid[-1].shape[0], -1])
    fake_facies = np.reshape(fake_facies, [len(mask_indexes), -1])

    real_facies_similarities = euclidean_distances(real_facies)
    fake_facies_similarities = euclidean_distances(fake_facies)
    mds = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=np.random.RandomState(seed=3),
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )
    real_facies_reduced = mds.fit((real_facies_similarities + real_facies_similarities.T) / 2).embedding_
    real_facies_reduced = real_facies_reduced[options.wells]
    fake_facies_reduced = mds.fit((fake_facies_similarities + fake_facies_similarities.T) / 2).embedding_
    plt.scatter(real_facies_reduced[:, 0], real_facies_reduced[:, 1])
    plt.scatter(fake_facies_reduced[:, 0], fake_facies_reduced[:, 1])
    plt.title("MDS Visualization of FaciesGAN generated facies")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(('Real Facies', 'Fake Facies'), loc='upper right')
    plt.show()

    # fake_facies = np.stack(fake_facies, 0).squeeze(-1)
    # real_facies = dataset.facies_pyramid[-1]
    # real_facies = np.reshape(torch2np(real_facies, denormalize=True), [200, -1])
    # fake_facies = np.reshape(fake_facies, [len(mask_indexes), -1])
    #
    # real_facies_similarities = euclidean_distances(real_facies)
    # fake_facies_similarities = euclidean_distances(fake_facies)
    # mds = MDS(
    #     n_components=2,
    #     max_iter=3000,
    #     eps=1e-9,
    #     random_state=np.random.RandomState(seed=3),
    #     dissimilarity="precomputed",
    #     n_jobs=1,
    #     normalized_stress="auto",
    # )
    # real_facies_reduced = mds.fit((real_facies_similarities + real_facies_similarities.T) / 2).embedding_
    # fake_facies_reduced = mds.fit((fake_facies_similarities + fake_facies_similarities.T) / 2).embedding_
    # sc = plt.scatter(real_facies_reduced[:, 0], real_facies_reduced[:, 1])
    # # plt.scatter(fake_facies_reduced[:, 0], fake_facies_reduced[:, 1])
    # plt.title("MDS Visualization of FaciesGAN generated facies")
    # plt.xlabel("MDS Dimension 1")
    # # plt.ylabel("MDS Dimension 2")
    # # plt.legend(('Real Facies', 'Fake Facies'), loc='upper right')
    # import mplcursors
    # cursor = mplcursors.cursor([sc], hover=True)
    #
    # def label_func(sel):
    #     print(sel.index)
    #     sel.annotation.set_text(str(sel.index))
    #
    # cursor.connect("add", label_func)
    # plt.show()
    # pass




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--how_many", help="how many facies", type=int, required=True)
    parser.add_argument("--model_path", help="models path", type=str, required=True)
    parser.add_argument("--out_path", help="path to save the generated facie", type=str)
    parser.add_argument("--use_gpu", help="use available GPU", action="store_true")
    parser.add_argument("--plot_mds", help="plot the multi dimensional scaling", action="store_true")
    parser.add_argument(
        "--wells",
        help="list of well indices to generate facies from",
        type=int,
        nargs='+',
        default=tuple(range(200)),
    )
    parser.add_argument(
        "--rec",
        help="generate a sample with the reconstruction noise. The reconstruction sample will have the same size as the TI",
        action="store_true",
    )
    parser.add_argument("--gpu_device", help="GPU device", type=int, default=0)
    parser.add_argument(
        "--plot_well_mask",
        help="Add/plot also the well masks on each generated facies",
        action="store_true"
    )

    arguments = parser.parse_args()

    if arguments.out_path is None:
        arguments.out_path = arguments.model_path

    if not os.path.exists(arguments.out_path):
        os.makedirs(arguments.out_path)

    device = torch.device(f"cuda:{arguments.gpu_device}" if torch.cuda.is_available()
                          else f"mps:{arguments.gpu_device}" if torch.backends.mps.is_available()
                          else f"cpu:{arguments.gpu_device}")

    with open(os.path.join(arguments.model_path, OPT_FILE), "r") as f:
        args = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        args.use_gpu = arguments.use_gpu
        args.rec = arguments.rec
        args.wells = arguments.wells
        args.device = device
        args.data_dir = 'data'

    start_time = time.time()

    print("Generating facies...")

    options = copy.copy(args)
    # if arguments.plot_mds:
    #     options.num_train_facies = len(options.wells)

    dataset: FaciesDataset = FaciesDataset(options, ceiling=False)
    masked_facies = []
    for i in range(len(dataset.facies_pyramid)):
        masked_facies.append(torch.stack([mask * facie
              for mask, facie in zip(dataset.masks_pyramid[i], dataset.facies_pyramid[i])], dim=0))
    faciesGAN = FaciesGAN(args.device, options=args, masked_facies=masked_facies)
    facies, mi = generate_facies(faciesGAN, arguments.how_many, arguments.model_path, args)


    if arguments.plot_mds: plot_mds(facies, mi, args)
    if arguments.plot_well_mask:
        for i, (facie, masked_facie) in enumerate(zip(facies, [masked_facies[-1][i] for i in mi]), 1):
            masked_facie = np.squeeze(masked_facie.numpy())
            mask_index = np.argmax(np.sum(np.squeeze(masked_facie) != 0, axis=0))
            fig, axes = plt.subplots(1, 1)
            axes.imshow(facie.squeeze() , cmap='gray')
            axes.scatter(
                np.full((masked_facie.shape[0],), mask_index),
                np.arange(0, masked_facie.shape[0]),
                c=np.astype(masked_facie[:, mask_index] >= 0.5, np.int8),
                s=1, marker='s', cmap='plasma', label="Facies Mask"
            )
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axis('off')

            fig.savefig(os.path.join(arguments.out_path, f"generated_facie_{i}.tif"))
            plt.close(fig)
    else:
        for i, facie in enumerate(facies, 1):
            tif.imwrite(os.path.join(arguments.out_path, f"generated_facie_{i}.tif"), facie)

    print(f"Facies generated at '{os.path.join(arguments.out_path, 'generated_facie_[1, 2, ...].tif')}'.")
    print(f"Total time: {format_time(int(time.time() - start_time))}")
