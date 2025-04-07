from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.stats as st


from config import RESULTS_DIR
from ops import torch2np


def plot_mask(mask: np.ndarray, real_facie: np.ndarray, axis: plt.Axes) -> None:
    """
    Plot the facies masked_facie on the given axis.

    Args:
        mask (np.ndarray): The facies masked_facie array.
        real_facie (np.ndarray): The real array.
        axis (matplotlib.axes.Axes): The axis to plot on.
    """
    mask_sum = np.sum(np.squeeze(mask), axis=0)
    mask_index = np.where(mask_sum == np.max(mask_sum))[0]
    axis.scatter(
        np.stack([np.full((mask.shape[0],), i) for i in mask_index]),
        np.stack([np.arange(0, mask.shape[0]) for _ in mask_index]),
        c=np.stack([
            np.astype(real_facie[:, i] >= 0.5, np.int8) for i in mask_index
        ]), s=1, marker='s', cmap='plasma', label="Facies Mask")

def plot_generated_facies(
        fake_facies: list[torch.Tensor],
        real_facies: torch.Tensor,
        masks: torch.Tensor,
        stage: int,
        index: int,
        out_dir: str = RESULTS_DIR,
        save: bool = False
    ) -> None:
    """
    Plot and optionally save generated facies alongside real facies and masks.

    Args:
        fake_facies (list of torch.Tensor): List of generated facies for each real facie.
        real_facies (torch.Tensor): Tensor of real facies.
        masks (torch.Tensor): Tensor of masks corresponding to the real facies.
        stage (int): Current stage of the process.
        index (int): Index for saving the plot.
        out_dir (str, optional): Directory to save the plot. Defaults to './outputs'.
        save (bool, optional): Whether to save the plot. Defaults to False.
    """
    num_real_facies = real_facies.size(0)
    num_generated_per_real = fake_facies[0].shape[0]
    fig, axes = plt.subplots(num_real_facies, num_generated_per_real + 1, figsize=(12, num_real_facies * 2.5))
    fake_facies = [torch2np(fake_facie, denormalize=True) for fake_facie in fake_facies]
    real_facies = torch2np(real_facies, denormalize=True, ceiling=True)
    masks = torch2np(masks)
    for i in range(num_real_facies):
        axes[i, 0].imshow(real_facies[i], cmap='YlGn')
        axes[i, 0].set_title(f'Well {i + 1}')
        plot_mask(masks[i], np.squeeze(real_facies[i]), axes[i, 0])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].axis('off')

        # Plot generated facies (remaining columns)
        for j in range(num_generated_per_real):
            axes[i, j + 1].imshow(fake_facies[i][j], cmap='gray')
            axes[i, j + 1].set_title(f'Gen {j + 1}')
            plot_mask(masks[i], np.squeeze(real_facies[i]), axes[i, j + 1])
            axes[i, j + 1].axis('off')

    # Add a main title for the plot
    plt.suptitle(f"Stage {stage} - Well Log, Real vs Generated Facies", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    if save: fig.savefig(f'{out_dir}/gen_{stage}_{index}.tif')
    plt.close()


def get_best_distribution(data: np.ndarray) -> Tuple[str, float, Tuple]:
    """
    Identify the best fitting distribution for the given data.

    Args:
        data (np.ndarray): The data to fit the distributions to.

    Returns:
        Tuple[str, float, Tuple]: The name of the best fitting distribution, the p-value, and the parameters
        of the best fit.
    """
    dist_names = ["norm", "exponweib", "pareto", "genextreme"]
    dist_results = []
    params = {}


    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data.flatten())
        params[dist_name] = param

        # Applying the Kolmogorov-Smirnov test
        result  = st.kstest(data, dist_name, args=param)
        dist_results.append((dist_name, np.sum(result.pvalue)))

    # Select the best fitted distribution
    best_dist, best_p = max(dist_results, key=lambda item: item[1])

    print(f"Best fitting distribution: {best_dist}")
    print(f"Best p value: {best_p}")
    print(f"Parameters for the best fit: {params[best_dist]}")

    return best_dist, best_p, params[best_dist]