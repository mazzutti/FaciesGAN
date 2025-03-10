from torch import autograd
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits

def compute_mse_g_loss(fake_images, real_images, mask_columns=None):
    if mask_columns is not None:
        fake_images = fake_images.clone()
        real_images = real_images.clone()
        g_mask_loss = mse_loss(fake_images[:, :, :, mask_columns], real_images[:, :, :, mask_columns])
        fake_images[:, :, :,  mask_columns] = real_images[:, :, :,  mask_columns] = 0
        g_loss = mse_loss(fake_images, real_images)
        g_loss = g_loss +  g_mask_loss
    else:
        g_loss = mse_loss(fake_images, real_images)
    return g_loss

def compute_g_loss(args, g_fake_logit, g_rec_loss, g_rmse_loss, ones):
    g_loss = None
    if args.gantype == 'wgangp':
        g_loss = compute_wgangp_g_loss(g_fake_logit, g_rec_loss, g_rmse_loss)
    elif args.gantype == 'zerogp':
        g_loss = compute_zerogp_g_loss(g_fake_logit, g_rec_loss, g_rmse_loss, ones)
    elif args.gantype == 'lsgan':
        g_loss = compute_lsgan_g_loss(g_fake_logit, g_rec_loss, g_rmse_loss, ones)
    return g_loss

def get_discriminator_loss(args, discriminator, d_real_logit, d_fake_logit, image, image_fake_list):
    if args.gantype == 'wgangp':
        return compute_wgangp_loss(discriminator, d_real_logit, d_fake_logit, image, image_fake_list, args)
    elif args.gantype == 'zerogp':
        return compute_zerogp_loss(d_real_logit, d_fake_logit, image, args)
    elif args.gantype == 'lsgan':
        return compute_lsgan_loss(d_real_logit, d_fake_logit, args)
    else:
        raise NotImplementedError

def compute_wgangp_g_loss(g_fake_logit, g_rec_loss, g_rmse_loss):
    g_fake = -torch.sum(torch.mean(g_fake_logit, (2, 3)))
    return g_fake + 10.0 * g_rec_loss +  20 * g_rmse_loss

def compute_zerogp_g_loss(g_fake_logit, g_rec_loss, g_rmse_loss, ones):
    g_fake = binary_cross_entropy_with_logits(g_fake_logit, ones, reduction='none').mean()
    return 0.001 * g_fake + 0.01 * g_rec_loss + g_rmse_loss

def compute_lsgan_g_loss(g_fake_logit, g_rec_loss, g_rmse_loss, ones):
    g_fake = mse_loss(g_fake_logit, 0.9 * ones)
    return g_fake + 50.0 * g_rec_loss + 100 * g_rmse_loss

def compute_wgangp_loss(discriminator, d_real_logit, d_fake_logit, image, image_fake_list,  args):
    d_fake = torch.sum(torch.mean(d_fake_logit, (2, 3)))
    d_real = -torch.sum(torch.mean(d_real_logit, (2, 3)))
    d_gp = compute_grad_gp_wgan(discriminator, image, image_fake_list[-1], args.device)
    return d_real + d_fake + 10.0 * d_gp + 10.0


def compute_zerogp_loss(d_real_logit, d_fake_logit, image, args):
    zeros = torch.zeros_like(d_fake_logit).to(args.device)
    ones = torch.ones_like(d_real_logit).to(args.device)
    d_fake = binary_cross_entropy_with_logits(d_fake_logit, zeros, reduction='none').mean()
    d_real = binary_cross_entropy_with_logits(d_real_logit, ones, reduction='none').mean()
    d_gp = compute_grad_gp(torch.mean(d_real_logit, (2, 3)), image)
    return d_real + d_fake + 10.0 * d_gp


def compute_lsgan_loss(d_real_logit, d_fake_logit, args):
    zeros = torch.zeros_like(d_fake_logit).to(args.device)
    ones = torch.ones_like(d_real_logit).to(args.device)
    d_fake = mse_loss(d_fake_logit, zeros)
    d_real = mse_loss(d_real_logit, 0.9 * ones)
    return d_real + d_fake

def compute_grad_gp(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return torch.sum(reg)


def compute_grad_gp_wgan(D, x_real, x_fake, gpu):
    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(gpu)

    x_interpolate = ((1 - alpha) * x_real + alpha * x_fake).detach()
    x_interpolate.requires_grad = True
    d_inter_logit = D(x_interpolate)
    grad = torch.autograd.grad(d_inter_logit, x_interpolate,
                               grad_outputs=torch.ones_like(d_inter_logit), create_graph=True)[0]

    norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)

    d_gp = ((norm - 1) ** 2).mean()
    return d_gp
