from typing import Any

import matplotlib
# Only set Agg backend if matplotlib hasn't been configured yet
# This allows training_visualizer.py to use interactive backends
if matplotlib.get_backend() == 'agg':
    matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch

from config import RESULTS_DIR
from ops import torch2np


def plot_mask(mask: np.ndarray, real_facie: np.ndarray, axis: plt.Axes) -> None:
    """Plot well locations on a facies visualization with visual markers.

    Displays well positions as vertical black lines, yellow side markers,
    and arrows to mark the location without changing the actual facies colors.

    Parameters
    ----------
    mask : np.ndarray
        Well mask array indicating well column positions.
    real_facie : np.ndarray
        Real facies array for determining colors at well locations.
    axis : plt.Axes
        Matplotlib axis to plot on.
    """
    mask_sum = np.sum(np.squeeze(mask), axis=0)
    mask_index = np.where(mask_sum == np.max(mask_sum))[0]

    # If there are no mask indices, nothing to plot
    if mask_index.size == 0:
        return
    
    for well_x in mask_index:
        # Add thin yellow vertical lines on left and right sides (keeping well data in middle)
        axis.axvline(x=int(well_x) - 0.5, color='yellow', linewidth=0.8, alpha=0.9, linestyle='-') # type: ignore
        axis.axvline(x=int(well_x) + 0.5, color='yellow', linewidth=0.8, alpha=0.9, linestyle='-') # type: ignore
        
        # Add arrow at the top pointing down to the well location
        arrow_y = -5  # Position above the image
        axis.annotate( # type: ignore
            '', 
            xy=(int(well_x), 0),  # Arrow points to top of well
            xytext=(int(well_x), arrow_y),  # Arrow starts above
            arrowprops=dict(
                arrowstyle='->',
                color='red',
                lw=2,
                mutation_scale=20
            )
        )


def plot_generated_facies(
    fake_facies: list[torch.Tensor],
    real_facies: torch.Tensor,
    masks: torch.Tensor,
    stage: int,
    index: int,
    out_dir: str = RESULTS_DIR,
    save: bool = False,
) -> None:
    """Plot and optionally save generated facies alongside real facies with well locations.

    Creates a grid visualization comparing generated facies realizations with
    real facies and well constraint overlays.

    Parameters
    ----------
    fake_facies : list[torch.Tensor]
        List of generated facies tensors, one tensor per real facie containing
        multiple generated samples.
    real_facies : torch.Tensor
        Tensor of real facies images.
    masks : torch.Tensor
        Tensor of well location masks corresponding to the real facies.
    stage : int
        Current training stage/scale for labeling.
    index : int
        Iteration index for filename when saving.
    out_dir : str, optional
        Directory to save the plot. Defaults to RESULTS_DIR.
    save : bool, optional
        Whether to save the plot to disk. Defaults to False.
    """
    num_real_facies = real_facies.size(0)
    num_generated_per_real = fake_facies[0].shape[0]
    fig, axes = plt.subplots(
        num_real_facies, num_generated_per_real + 1, figsize=(12, num_real_facies * 2.5)
    )
    # Ensure axes is always 2D for consistent indexing
    if num_real_facies == 1:
        axes = axes.reshape(1, -1)
    fake_facies_arr = [torch2np(fake_facie, denormalize=True) for fake_facie in fake_facies]
    np_real_facies = torch2np(real_facies, denormalize=True, ceiling=True)
    np_masks = torch2np(masks)
    for i in range(num_real_facies):
        axes[i, 0].imshow(np_real_facies[i], cmap="YlGn")
        axes[i, 0].set_title(f"Well {i + 1}")
        plot_mask(np_masks[i], np.squeeze(np_real_facies[i]), axes[i, 0])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].axis("off")

        # Plot generated facies (remaining columns)
        for j in range(num_generated_per_real):
            axes[i, j + 1].imshow(fake_facies_arr[i][j], cmap="gray")
            axes[i, j + 1].set_title(f"Gen {j + 1}")
            plot_mask(np_masks[i], np.squeeze(np_real_facies[i]), axes[i, j + 1])
            axes[i, j + 1].axis("off")

    # Add a main title for the plot
    plt.suptitle(f"Stage {stage} - Well Log, Real vs Generated Facies", fontsize=16, y=0.99)  # type: ignore
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    # plt.show()
    if save:
        fig.savefig(f"{out_dir}/gen_{stage}_{index}.png")  # type: ignore
    plt.close()


def get_best_distribution(data: np.ndarray) -> tuple[str, float, tuple[Any, ...]]:
    """
    Identify the best fitting distribution for the given data.

    Args:
        data (np.ndarray): The data to fit the distributions to.

    Returns:
        tuple[str, float, tuple[Any, ...]]: The name of the best fitting distribution, the p-value, and the parameters
        of the best fit.
    """
    dist_names = ["norm", "exponweib", "pareto", "genextreme"]
    # Explicitly type these containers so static analysis can infer types for methods like .append
    dist_results: list[tuple[str, float]] = []
    params: dict[str, tuple[Any, ...]] = {}

    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data.flatten())
        params[dist_name] = param

        # Applying the Kolmogorov-Smirnov test
        # Use flattened data for the KS test to match the fit input
        result = st.kstest(data.flatten(), dist_name, args=param)
        # pvalue may be a scalar or array-like; coerce to float
        pval = float(np.sum(result.pvalue))
        dist_results.append((dist_name, pval))

    # Select the best fitted distribution
    best_dist, best_p = max(dist_results, key=lambda item: item[1])

    print(f"Best fitting distribution: {best_dist}")
    print(f"Best p value: {best_p}")
    print(f"Parameters for the best fit: {params[best_dist]}")

    return best_dist, best_p, params[best_dist]
