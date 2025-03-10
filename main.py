import argparse

import numpy as np
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.geological_facies_dataset import GeologicalFaciesDataset
from models.generator import Generator
from models.discriminator import Discriminator
from train import *
from utils import makedirs, formatted_print
from validate import *

def main():

    parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
    parser.add_argument('--data_dir', default='data/facies_data', help='path to dataset')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--logs_dir', default='./logs')
    parser.add_argument('--cond', default=None)
    parser.add_argument('--gantype', default='zerogp',
                        help='type of GAN loss', choices=['wgangp', 'zerogp', 'lsgan'])
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
    parser.add_argument('--val_batch', default=1, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--kernel_size', default=5, type=int, choices=[3, 5, 7, 9])
    parser.add_argument('--total_iter', default=250, type=int, help='total num of iteration')
    parser.add_argument('--decay_lr', default=500, type=int, help='learning rate change iteration times')
    parser.add_argument('--validation', default=False, type=bool)
    parser.add_argument('--num_real_samples', default=5, type=int)
    parser.add_argument('--num_gen_per_sample', default=5, type=int)

    args = parser.parse_args()
    args.scales_list = [8, 16, 32, 64, 128, 256]
    args.num_scales = len(args.scales_list) - 1
    args.in_padding = 5
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    makedirs(args.logs_dir)
    makedirs(args.results_dir)

    formatted_print('Batch SIZE:', args.batch_size)
    formatted_print('Logs DIR:', args.logs_dir)
    formatted_print('Results DIR:', args.results_dir)
    formatted_print('GAN TYPE:', args.gantype)

    main_worker(args)

def main_worker(args):
    ################
    # Define model #
    ################
    discriminator = Discriminator(args.in_channels, args.kernel_size).to(args.device)
    generator = Generator(args.scales_list, args.in_channels, args.kernel_size).to(args.device)

    ######################
    # Loss and Optimizer #
    ######################
    discriminator_optimizer = Adam(discriminator.subs[0].parameters(), 1e-4, (0.5, 0.95))
    generator_optimizer = Adam(generator.subs[0].parameters(), 1e-4, (0.5, 0.95))

    ###########
    # Dataset #
    ###########

    dataset = GeologicalFaciesDataset(args.data_dir, args.scales_list)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ######################
    # Validate and Train #
    ######################

    args.stage = 0
    for stage in range(args.stage, args.num_scales + 1):
        train(data_loader, generator, discriminator, generator_optimizer, discriminator_optimizer, stage, args)
        validate(data_loader, generator, discriminator, stage, args)
        if stage < args.num_scales:
            discriminator.progress()
            generator.progress()
            networks = [discriminator, generator]
            networks = [x.to(args.device) for x in networks]
            discriminator, generator, = networks
            for net_idx in range(generator.current_scale):
                for param in generator.subs[net_idx].parameters():
                    param.requires_grad = False
                for param in discriminator.subs[net_idx].parameters():
                    param.requires_grad = False
            discriminator_optimizer = Adam(
                discriminator.subs[discriminator.current_scale].parameters(), 1e-4, (0.5, 0.95))
            generator_optimizer = Adam(
                generator.subs[generator.current_scale].parameters(),1e-4, (0.5, 0.95))

    ##############
    # Save model #
    ##############
    data_iterator = iter(data_loader)
    images, masks = next(data_iterator)
    images = [img.to(args.device) for img in images[:args.num_scales + 1]]

    z_rec = [
        pad((torch.randn if i == 0 else torch.zeros)
            (images[0].size(0), args.in_channels, s, s),
            [args.in_padding] * 4, value=0).to(args.device)
        for i, s in enumerate(args.scales_list)
    ]

    with torch.no_grad():
        image_rec_list = generator(z_rec)
        rmse_list = [1.0] + [
            torch.sqrt(compute_mse_g_loss(image_rec_list[i], images[i])).item() / (100.0 if args.validation else 1.0)
            for i in range(1, args.num_scales + 1)
        ]
        if len(rmse_list) > 1: rmse_list[-1] = 0.0
    np.savetxt(os.path.join(args.logs_dir, 'rmse_list.txt'), np.array(rmse_list))
    torch.save(generator, os.path.join(args.logs_dir, 'gen.pkl'))


def run(cond_args, gslib_pre: str = '', cond_file: str = None, use_fiter: bool = False):
    ijs, vals = None, None
    if cond_file is not None:
        ij, vals = utils.load_condfile(cond_file)
        ijs = [(ij * (cond_args.scales_list[zeros_idx] / cond_args.scales_list[-1])).astype(int) for zeros_idx in
               range(cond_args.num_scales + 1)]
        if use_fiter:
            vals = vals * 2
            # ijs, vals = helper.gen_filter(ijs, vals)
        vals = torch.from_numpy(vals.astype(np.float32)).to(cond_args.device)
    # for gen_nums in range(1):
    #     x_fake_list = cond_args.generator(cond_args.z_list, ijs=ijs, vals=vals)
    #     for scale in range(cond_args.num_scales + 1):
    #         helper.save_tensor_to_gslib(x_fake_list[scale], cond_args.save_folder,
    #                                     file_names=["{0}_{1}.gslib".format(gslib_pre, scale + 1)])
    pass

if __name__ == '__main__':
    main()
