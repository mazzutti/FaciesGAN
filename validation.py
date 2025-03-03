import utils
from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils

from utils import *

def validateSinGAN(data_loader, networks, stage, args, additional=None):
    D = networks[0]
    G = networks[1]
    D.eval()
    G.eval()
    val_it = iter(data_loader)

    z_rec = additional['z_rec']

    x_in = next(val_it)
    x_in = x_in.to(args.device)
    x_org = x_in

    x_in = F.interpolate(x_in, (args.size_list[stage], args.size_list[stage]), mode='bilinear', align_corners=True)
    vutils.save_image(x_in.detach().cpu(), os.path.join(args.res_dir, 'ORG_{}.png'.format(stage)),
                      nrow=1, normalize=True)
    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = F.interpolate(x_org, (args.size_list[xidx], args.size_list[xidx]), mode='bilinear', align_corners=True)
        x_in_list.append(x_tmp)

    for z_idx in range(len(z_rec)):
        z_rec[z_idx] = z_rec[z_idx].to(args.device)

    with torch.no_grad():
        x_rec_list = G(z_rec)

        # calculate rmse for each scale
        rmse_list = [1.0]
        for rmseidx in range(1, stage + 1):
            rmse = torch.sqrt(F.mse_loss(x_rec_list[rmseidx], x_in_list[rmseidx]))
            if args.validation:
                rmse /= 100.0
            rmse_list.append(rmse)
        if len(rmse_list) > 1:
            rmse_list[-1] = 0.0
        
        vutils.save_image(torch.clamp(x_rec_list[-1].detach().cpu(), -0.5, 0.5), os.path.join(args.res_dir, 'REC_{}.png'.format(stage)),
                          nrow=1, normalize=True)
        
        z_list = [F.pad(rmse_list[z_idx] * torch.randn(args.batch_size, args.img_ch, args.size_list[z_idx],
                        args.size_list[z_idx]).to(args.device),
                        [5, 5, 5, 5], value=0) for z_idx in range(stage + 1)]
        x_fake_list = G(z_list)
        for xi in range(len(x_fake_list)):
            scale=x_fake_list[xi].shape[-1]/args.size_list[-1]
            utils.save_pic(args.condi, torch.clamp(x_fake_list[xi].detach().cpu(), -0.5, 0.5), args.res_dir, scale, stage, xi)



