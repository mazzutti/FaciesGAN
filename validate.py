import os

from torch.nn.functional import  pad

import utils
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from ops import compute_mse_g_loss


def validate(data_loader, generator, discriminator, stage, args):

    generator.eval()
    discriminator.eval()

    indexes = torch.randint(0, len(data_loader.dataset), (args.num_real_samples,))


    images = [
        torch.stack([img.to(args.device) for img in sublist[indexes]])
        for sublist in data_loader.dataset.images_list[:stage + 1]
    ]

    mask_indexes = [
        torch.stack([mask.to(args.device) for mask in sublist[indexes]])
        for sublist in data_loader.dataset.mask_indexes_list[:stage + 1]
    ]

    z_rec = [
        pad((torch.randn if i == 0 else torch.zeros)
            (images[0].size(0), args.in_channels, s, s),
            [args.in_padding] * 4, value=0).to(args.device)
        for i, s in enumerate(args.scales_list)
    ]

    image_fake_list = []

    for i in range(args.num_real_samples):
        with torch.no_grad():

            image_rec_list = generator(z_rec,
                 mask_indexes=[mask_index[i].repeat(args.num_gen_per_sample, 1, 1, 1) for mask_index in mask_indexes],
                 images=[image[i].repeat(args.num_gen_per_sample, 1, 1, 1) for image in images])

            # Calculate RMSE for each scale using list comprehension
            rmse_list =  torch.tensor([1.0] +  [
                torch.sqrt(compute_mse_g_loss(
                    image_rec_list[i], images[i], mask_indexes[i])).item() / (100.0 if args.validation else 1.0)
                for i in range(1, stage + 1)
            ]).to(args.device)
            if len(rmse_list) > 1: rmse_list[-1] = 0.0

            z_list = [
                pad(rmse * torch.randn(args.num_gen_per_sample, args.in_channels,
                   args.scales_list[r], args.scales_list[r]).to(args.device), [args.in_padding] * 4, value=0)
                for r, rmse in enumerate(rmse_list)
            ]

            images_fake = generator(z_list)
            image_fake_list.append(torch.clamp(images_fake[stage], -0.5, 0.5))

    utils.plot_generated_images(image_fake_list, images[stage], mask_indexes[stage], stage, stage)

