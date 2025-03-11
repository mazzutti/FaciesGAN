import torch
from torch import nn
from torch.nn.functional import interpolate, pad


class Generator(nn.Module):
    def __init__(self, scales_list, in_channels=1, kernel_size=3, padding=5, out_channels=32):
        super(Generator, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales_list = scales_list
        self.current_scale = 0
        self.out_padding = self.kernel_size // 2 - 1
        self.in_padding = padding
        self.stride = 1
        self.resizers = [
            lambda x, s=scale: interpolate(
                x, (s, s), mode='bilinear', align_corners=True
            )
            for scale in self.scales_list[1:]
        ]
        self.subs = nn.ModuleList()
        self.__add_generator()

    def __add_generator(self):
        self.subs.append(
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.out_padding),
                    nn.BatchNorm2d(self.out_channels),
                    nn.LeakyReLU(2e-1)),
                *[nn.Sequential(
                    nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.out_padding),
                    nn.BatchNorm2d(self.out_channels),
                    nn.LeakyReLU(2e-1)
                ) for _ in range(3)],
                nn.Sequential(
                    nn.Conv2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.out_padding),
                    nn.Tanh())
            )
        )

    def forward(self, z, mask_indexes=None, images=None):
        if mask_indexes is not None and images is not None:
            for i in range(self.current_scale + 1):
                z[i][:, :, self.in_padding:-self.in_padding,
                    mask_indexes[i]+self.in_padding] = images[i][:, :, :, mask_indexes[i]]
        fake_inter_image = self.subs[0](z[0])
        fake_image_list = [fake_inter_image]

        for i in range(1, self.current_scale + 1):
            fake_inter_image = self.resizers[i-1](fake_inter_image)
            prev_fake_image = fake_inter_image
            fake_inter_image = pad(fake_inter_image, [self.in_padding] * 4, value=0)
            fake_inter_image = fake_inter_image + z[i]
            gen = self.subs[i](fake_inter_image)
            fake_inter_image = gen + prev_fake_image
            fake_image_list.append(fake_inter_image)
        return fake_image_list

    def progress(self):
        self.current_scale += 1
        if self.current_scale % 4 == 0: self.out_channels *= 2
        self.__add_generator()
        if self.current_scale % 4 != 0 and self.current_scale >= 1:
            self.subs[-1].load_state_dict(self.subs[-2].state_dict())
