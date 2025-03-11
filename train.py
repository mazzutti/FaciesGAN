from torch.nn.functional import pad
from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from ops import compute_g_loss, get_discriminator_loss, compute_mse_g_loss


def train(data_loader, generator, discriminator, generator_optimizer, discriminator_optimizer, stage, args):
    discriminator.train()
    generator.train()
    data_loader.dataset.shuffle()

    total_iter = args.total_iter
    decay_lr = args.decay_lr

    d_iter = 1
    g_iter = 1
    num_of_batchs = len(data_loader.dataset) // args.batch_size + 1

    train_iterator = trange(0, total_iter, initial=0, total=total_iter)

    z_list = []
    d_loss, g_loss, g_rmse_loss = None, None, None
    for i in train_iterator:
        data_iterator = iter(data_loader)
        for batch_id, (images, mask_indexes) in enumerate(data_iterator):
            mask_indexes = [mask.to(args.device) for mask in mask_indexes[:stage + 1]]
            images = [img.to(args.device) for img in images[:stage + 1]]
            if i == decay_lr:
                for param_group in discriminator_optimizer.param_groups: param_group['lr'] *= 0.9
                for param_group in generator_optimizer.param_groups: param_group['lr'] *= 0.9
            for _ in range(g_iter):
                generator_optimizer.zero_grad()
                z_rec = [
                    pad((torch.randn if i == 0 else torch.zeros)
                        (images[0].size(0), args.in_channels, s, s),
                        [args.in_padding] * 4, value=0).to(args.device)
                    for i, s in enumerate(args.scales_list)
                ]
                image_rec_list = generator(z_rec, mask_indexes=mask_indexes, images=images)
                g_rec_loss = compute_mse_g_loss(image_rec_list[-1], images[stage], mask_indexes[-1])
                g_rmse_loss = torch.tensor([1.0] + [torch.sqrt(compute_mse_g_loss(
                    image_rec_list[j], images[j], mask_indexes[j])) for j in range(1, stage + 1)]).to(args.device)

                z_list = [
                    pad(g_rmse_loss[z_idx] * torch.randn(
                        images[0].size(0),
                        args.in_channels,
                        args.scales_list[z_idx],
                        args.scales_list[z_idx]).to(args.device),
                          [args.kernel_size] * 4, value=0) for z_idx in range(stage + 1)
                ]

                image_fake_list = generator(z_list)
                g_fake_logit = discriminator(image_fake_list[-1])
                ones = torch.ones_like(g_fake_logit).to(args.device)
                g_loss = compute_g_loss(args, g_fake_logit, g_rec_loss, torch.sum(g_rmse_loss),  ones)
                g_loss.backward(retain_graph=True)
                generator_optimizer.step()

            # Update discriminator
            for _ in range(d_iter):
                images[stage].requires_grad = True
                discriminator_optimizer.zero_grad()
                image_fake_list = generator(z_list)
                d_fake_logit = discriminator(image_fake_list[-1].detach())
                d_real_logit = discriminator(images[stage])
                d_loss = get_discriminator_loss(args,
                    discriminator, d_real_logit, d_fake_logit, images[stage], image_fake_list)
                d_loss.backward()
                discriminator_optimizer.step()

            train_iterator.set_description(
                f'Stage: [{stage}/{args.num_scales}] Batch: [{str(batch_id + 1).rjust(2)}/{num_of_batchs}] Avg Loss: discriminator '
                f'[{d_loss.item():.3f}] generator [{g_loss.item():.3f}] RMSE [{g_rmse_loss[-1]:.3f}] |')