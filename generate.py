import copy
import json
import os
import random
import time
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import tifffile as tif
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

from config import OPT_FILE
from facies_dataset import FaciesDataset
from log import format_time
from models.facies_gan import FaciesGAN
from utils import torch2np


def generate_facies(
    model: FaciesGAN, how_many: int, model_path: str, options: SimpleNamespace
) -> tuple[list[np.ndarray], list[int]]:
    """
    Generate facies using the FaciesGAN model.

    Args:
        model (FaciesGAN): The FaciesGAN model instance.
        how_many (int): Number of facies to generate.
        model_path (str): Path to the model.
        options (SimpleNamespace): Options for the model.

    Returns:
        tuple[list[np.ndarray], list[int]]: A tuple with generated facies (numpy arrays) and mask indexes.
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


def plot_mds(
    fake_facies: list[NDArray[np.float32]],
    mask_indexes: list[int],
    options: SimpleNamespace,
) -> None:

    # fake_facies parameter is a list of numpy arrays. Do not rebind the parameter
    # to an ndarray (which would violate the annotated type). Use a new local
    # variable for the stacked ndarray representation.
    fake_facies_arr = np.stack(fake_facies, 0).squeeze(-1)
    real_facies = dataset.facies_pyramid[-1]
    real_facies = np.reshape(
        torch2np(real_facies, denormalize=True),
        [dataset.facies_pyramid[-1].shape[0], -1],
    )
    fake_facies_arr = np.reshape(fake_facies_arr, [len(mask_indexes), -1])

    real_facies_similarities = euclidean_distances(real_facies)
    fake_facies_similarities = euclidean_distances(fake_facies_arr)
    mds = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=np.random.RandomState(seed=3),
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )
    real_facies_reduced = mds.fit(
        (real_facies_similarities + real_facies_similarities.T) / 2
    ).embedding_
    real_facies_reduced = real_facies_reduced[options.wells]
    fake_facies_reduced = mds.fit(
        (fake_facies_similarities + fake_facies_similarities.T) / 2
    ).embedding_
    plt.scatter(real_facies_reduced[:, 0], real_facies_reduced[:, 1])
    plt.scatter(fake_facies_reduced[:, 0], fake_facies_reduced[:, 1])
    plt.title("MDS Visualization of FaciesGAN generated facies")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(("Real Facies", "Fake Facies"), loc="upper right")  # type: ignore
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
    parser.add_argument(
        "--plot_mds", help="plot the multi dimensional scaling", action="store_true"
    )
    parser.add_argument(
        "--wells",
        help="list of well indices to generate facies from",
        type=int,
        nargs="+",
        default=tuple(range(200)),
    )
    parser.add_argument(
        "--rec",
        help=(
            "generate a sample with the reconstruction noise. "
            "The reconstruction sample will have the same size as the TI"
        ),
        action="store_true",
    )
    parser.add_argument("--gpu_device", help="GPU device", type=int, default=0)
    parser.add_argument(
        "--plot_well_mask",
        help="Add/plot also the well masks on each generated facies",
        action="store_true",
    )

    arguments = parser.parse_args()

    if arguments.out_path is None:
        arguments.out_path = arguments.model_path

    if not os.path.exists(arguments.out_path):
        os.makedirs(arguments.out_path)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{arguments.gpu_device}")
    elif torch.backends.mps.is_available():
        device = torch.device(f"mps:{arguments.gpu_device}")
    else:
        device = torch.device(f"cpu:{arguments.gpu_device}")

    with open(os.path.join(arguments.model_path, OPT_FILE), "r") as f:
        args = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
        args.use_gpu = arguments.use_gpu
        args.rec = arguments.rec
        args.wells = arguments.wells
        args.device = device
        args.data_dir = "data"

    start_time = time.time()

    print("Generating facies...")

    options = copy.copy(args)
    # if arguments.plot_mds:
    #     options.num_train_facies = len(options.wells)

    dataset: FaciesDataset = FaciesDataset(options, ceiling=False)
    masked_facies: list[torch.Tensor] = []
    for i in range(len(dataset.facies_pyramid)):
        masked_facies.append(
            torch.stack(
                [
                    mask * facie
                    for mask, facie in zip(dataset.masks_pyramid[i], dataset.facies_pyramid[i])
                ],
                dim=0,
            )
        )
    faciesGAN = FaciesGAN(args.device, options=args, masked_facies=masked_facies)
    facies, mi = generate_facies(faciesGAN, arguments.how_many, arguments.model_path, args)

    if arguments.plot_mds:
        plot_mds(facies, mi, args)
    if arguments.plot_well_mask:
        for i, (facie, masked_facie) in enumerate(
            zip(facies, [masked_facies[-1][i] for i in mi]), 1
        ):
            masked_facie_arr = np.squeeze(masked_facie.numpy())
            mask_index = np.argmax(np.sum(np.squeeze(masked_facie) != 0, axis=0))
            # create figure and axes for plotting
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(1, 1)
            axes.imshow(facie.squeeze(), cmap="gray")  # type: ignore
            # Ensure numeric dtypes for matplotlib scatter to avoid analyzer/runtime issues
            x_coords = np.full(masked_facie_arr.shape[0], mask_index, dtype=float)
            y_coords = np.arange(masked_facie_arr.shape[0], dtype=float)
            colors = (masked_facie_arr[:, mask_index] >= 0.5).astype(np.int8)
            axes.scatter(  # type: ignore
                x_coords,
                y_coords,
                c=colors,
                s=1,
                marker=MarkerStyle("s"),
                cmap="plasma",
                label="Facies Mask",
            )
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axis("off")  # type: ignore

            out_file = os.path.join(arguments.out_path, f"generated_facie_{i}.tif")
            fig.savefig(out_file)  # type: ignore
            plt.close(fig)
    else:
        for i, facie in enumerate(facies, 1):
            tif.imwrite(os.path.join(arguments.out_path, f"generated_facie_{i}.tif"), facie)

    generated_pattern = os.path.join(arguments.out_path, "generated_facie_[1, 2, ...].tif")
    print(f"Facies generated at '{generated_pattern}'.")
    print(f"Total time: {format_time(int(time.time() - start_time))}")
