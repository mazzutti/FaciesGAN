import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets.geological_facies_dataset import GeologicalFaciesDataset


def denormalize(tensor, mean=GeologicalFaciesDataset.MEAN, std=GeologicalFaciesDataset.STD):
    return tensor * std + mean

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def formatted_print(notice, value):
    print('{0:<40}{1:<40}'.format(notice, value))


def save_checkpoint(state, check_list, log_dir, epoch=0):
    check_file = os.path.join(log_dir, 'model_{}.ckpt'.format(epoch))
    torch.save(state, check_file)
    check_list.write('model_{}.ckpt\n'.format(epoch))


def scatter_ij(ax, ij, vals):
    ij0, ij1 = ij[np.squeeze(np.where(vals == 0))[0]], ij[np.squeeze(np.where(vals != 0))[0]]
    ax.scatter(ij0[:, 1], ij0[:, 0], marker='.', s=1, c='C1')
    ax.scatter(ij1[:, 1], ij1[:, 0], marker='.', s=1, c='C0')


def plot_mask(mask_index, real_image,  axis):
    axis.scatter(
        np.full((real_image.shape[0],), mask_index),
        np.arange(0, real_image.shape[0]),
        c=np.astype(real_image[:, mask_index] > 0, np.int8), s=1, marker='s', cmap='plasma', label="Facies Mask")

def plot_generated_images(fake_images, real_images, mask_indexes,  stage, index, out_dir='./results', save=False):
    num_real_images = real_images.size(0)
    num_generated_per_real = fake_images[0].size(0)
    fig, axes = plt.subplots(num_real_images, num_generated_per_real + 1, figsize=(12, num_real_images * 2.5))
    fake_images = [np.permute_dims(fake_image.cpu().detach().numpy(), (0, 2, 3, 1)) for fake_image in fake_images]
    fake_images = [(fake_image > 0).astype(np.int32) for fake_image in fake_images]
    real_images = np.permute_dims(real_images.cpu().detach().numpy(), (0, 2, 3, 1))
    mask_indexes = mask_indexes.cpu().detach().numpy()
    for i in range(num_real_images):
        axes[i, 0].imshow(real_images[i], cmap='YlGn')  # Assuming one channel (e.g., well log)
        axes[i, 0].set_title(f'Well {i + 1}')
        plot_mask(mask_indexes[i], real_images[i], axes[i, 0])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].axis('off')

        # Plot generated facies (remaining columns)
        for j in range(num_generated_per_real):
            axes[i, j + 1].imshow(fake_images[i][j], cmap='gray')
            axes[i, j + 1].set_title(f'Gen {j + 1}')
            plot_mask(mask_indexes[i], real_images[i], axes[i, j + 1])
            axes[i, j + 1].axis('off')

    # Add a main title for the plot
    plt.suptitle(f"Stage {stage} - Well Log, Real vs Generated Facies", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    if save: plt.savefig(f'{out_dir}/gen_{stage}_{index}.png')
    plt.close()

def plot_image_on_axis(axis, image):
    axis.imshow(image, cmap='gray')
    axis.axis('off')