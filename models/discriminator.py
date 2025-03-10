from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels = 1, kernel_size = 3):
        super(Discriminator, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = 32
        self.current_scale = 0
        self.in_channels = in_channels
        self.stride = 1
        self.padding = kernel_size // 2
        self.subs = nn.ModuleList()
        self.__add_discriminator()

    def __add_discriminator(self):
        self.subs.append(
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
                    nn.LeakyReLU(2e-1)),
                *[nn.Sequential(
                    nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
                    nn.BatchNorm2d(self.out_channels),
                    nn.LeakyReLU(2e-1)
                ) for _ in range(3)],
                nn.Sequential(
                    nn.Conv2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.padding),
                    nn.Tanh())
            )
        )

    def forward(self, x):
        return self.subs[self.current_scale](x)

    def progress(self):
        self.current_scale += 1
        if self.current_scale % 4 == 0: self.out_channels *= 2
        self.__add_discriminator()
        if self.current_scale % 4 != 0 and self.current_scale >= 1:
            self.subs[-1].load_state_dict(self.subs[-2].state_dict())
