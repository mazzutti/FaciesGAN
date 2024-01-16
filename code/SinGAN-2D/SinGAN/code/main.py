import argparse
import datasets.DataSets as DataSets
import torch
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os
from models.generator import Generator
from models.discriminator import Discriminator
from train import *
from validation import *

parser = argparse.ArgumentParser(description='PyTorch Simultaneous Training')
parser.add_argument('--data_dir', default='../data/Gslibdata/250TI.gslib', help='path to dataset')
parser.add_argument('--res_dir', default='./results')
parser.add_argument('--log_dir', default='./logs')
parser.add_argument('--condi', default=None)
parser.add_argument('--gantype', default='zerogp', help='type of GAN loss', choices=['wgangp', 'zerogp', 'lsgan'])
parser.add_argument('--batch_size', default=1, type=int,help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--val_batch', default=1, type=int)
parser.add_argument('--img_ch', default=1, type=int)
parser.add_argument('--kernel', default=5, type=int,choices=[3,5,7,9])
parser.add_argument('--total_iter', default=250, type=int, help='total num of iter')
parser.add_argument('--decay_lr', default=500, type=int, help='learning rate change iter times')
parser.add_argument('--validation',default=False,type=bool)

def main():
    args = parser.parse_args()
    args.size_list = [25,33,44,60,80,108,144,192,250]
    args.num_scale = len(args.size_list)-1
    args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    makedirs(args.log_dir)
    makedirs(args.res_dir)

    formatted_print('Batch Size:', args.batch_size)
    formatted_print('Log DIR:', args.log_dir)
    formatted_print('Result DIR:', args.res_dir)
    formatted_print('GAN TYPE:', args.gantype)

    main_worker(args)

def main_worker(args):
    ################
    # Define model #
    ################
    discriminator = Discriminator(args.img_ch,args.kernel)
    generator = Generator(args.size_list,args.img_ch,args.kernel)
    networks = [discriminator, generator]
    networks = [x.to(args.device) for x in networks]
    discriminator, generator, = networks

    ######################
    # Loss and Optimizer #
    ######################
    d_opt = torch.optim.Adam(discriminator.sub_discriminators[0].parameters(), 5e-4, (0.5, 0.999))
    g_opt = torch.optim.Adam(generator.sub_generators[0].parameters(), 5e-4, (0.5, 0.999))
    
    ###########
    # Dataset #
    ###########
    train_loader=DataSets.get_dataloader(args.data_dir)

    ######################
    # Validate and Train #
    ######################
    z_fix_list = [F.pad(torch.randn(args.batch_size, args.img_ch, args.size_list[0], args.size_list[0]), [5, 5, 5, 5], value=0)]
    zero_list = [F.pad(torch.zeros(args.batch_size, args.img_ch, args.size_list[zeros_idx], args.size_list[zeros_idx]),
                       [5, 5, 5, 5], value=0) for zeros_idx in range(1, args.num_scale + 1)]
    z_fix_list = z_fix_list + zero_list

    args.stage = 0
    for stage in range(args.stage, args.num_scale+1):
        trainSinGAN(train_loader, networks, {"d_opt": d_opt, "g_opt": g_opt}, stage, args, {"z_rec": z_fix_list})
        validateSinGAN(train_loader, networks, stage, args, {"z_rec": z_fix_list})
        if stage<args.num_scale:
            discriminator.progress()
            generator.progress()
            networks = [discriminator, generator]
            networks = [x.to(args.device) for x in networks]
            discriminator, generator, = networks
            for net_idx in range(generator.current_scale):
                for param in generator.sub_generators[net_idx].parameters():
                    param.requires_grad = False
                for param in discriminator.sub_discriminators[net_idx].parameters():
                    param.requires_grad = False

            d_opt = torch.optim.Adam(discriminator.sub_discriminators[discriminator.current_scale].parameters(),
                                    5e-4, (0.5, 0.999))
            g_opt = torch.optim.Adam(generator.sub_generators[generator.current_scale].parameters(),
                                        5e-4, (0.5, 0.999))
    
    
    ##############
    # Save model #
    ##############
    val_it = iter(train_loader)
    x_in = next(val_it)
    x_in = x_in.to(args.device)
    x_org = x_in
    x_in = F.interpolate(x_in, (args.size_list[0], args.size_list[0]), mode='bilinear', align_corners=True)
    x_in_list = [x_in]
    for stage in range(1, args.num_scale + 1):
        x_tmp = F.interpolate(x_org, (args.size_list[stage], args.size_list[stage]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)
    with torch.no_grad():
        x_rec_list = generator(z_fix_list)
        rmse_list = [1.0]
        for rmseidx in range(1, args.num_scale + 1):
            rmse = torch.sqrt(F.mse_loss(x_rec_list[rmseidx], x_in_list[rmseidx]))
            # if args.validation:
            #     rmse /= 100.0
            rmse_list.append(rmse.item())
        if len(rmse_list) > 1:
            rmse_list[-1] = 0.0
    np.savetxt(os.path.join(args.log_dir,'rmse_list.txt'),np.array(rmse_list))
    #torch.save(discriminator,os.path.join(args.logdir,'disc.pkl'))
    torch.save(generator,os.path.join(args.log_dir,'gen.pkl'))

def run(condi_args,gslib_pre:str='',condi_file:str=None,use_fiter:bool=False):
    if condi_args.save_folder is None:
        return
    ijs, vals=None,None
    if condi_file is not None:
        ij, vals = utils.load_condfile(condi_file)
        ijs = [(ij*(condi_args.size_list[zeros_idx]/condi_args.size_list[-1])).astype(int) for zeros_idx in range(condi_args.num_scale + 1)]
        if use_fiter:
            vals = vals * 2
            #ijs, vals = helper.gen_filter(ijs, vals)
        vals = torch.from_numpy(vals.astype(np.float32)).to(condi_args.device)
    for gen_nums in range(1):
        x_fake_list = condi_args.generator(condi_args.z_list,ijs=ijs,vals=vals)
        for scale in range(condi_args.num_scale + 1):
            helper.save_tensor_to_gslib(x_fake_list[scale],condi_args.save_folder,file_names=["{0}_{1}.gslib".format(gslib_pre,scale+1)])

if __name__ == '__main__':
    main()
