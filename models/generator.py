from torch import nn
from torch.nn import functional as F
import torch
import os
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, size_list ,img_ch=1,kernel:int=3):
        super(Generator, self).__init__()
        self.nf = 32
        self.kernel=kernel
        self.current_scale = 0
        self.img_ch=img_ch
        self.size_list = size_list
        print(self.size_list)

        self.sub_generators = nn.ModuleList()

        first_generator = nn.ModuleList()

        first_generator.append(nn.Sequential(nn.Conv2d(self.img_ch, self.nf, self.kernel, 1,self.kernel//2-1),
                                             nn.BatchNorm2d(self.nf),
                                             nn.LeakyReLU(2e-1)))
        for _ in range(3):
            first_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, self.kernel, 1,self.kernel//2-1),
                                                 nn.BatchNorm2d(self.nf),
                                                 nn.LeakyReLU(2e-1)))

        first_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.img_ch, self.kernel, 1,self.kernel//2-1),
                                             nn.Tanh()))

        first_generator = nn.Sequential(*first_generator)

        self.sub_generators.append(first_generator)

    def forward(self, z, img=None,ijs=None,vals=None):
        x_list = []
        x_first = self.sub_generators[0](z[0])

        if (ijs is not None) and (vals is not None):
            ij=ijs[0]
            x_first[...,ij[:,0],ij[:,1]]=vals

        x_list.append(x_first)
        if img is not None:
            x_inter = img
        else:
            x_inter = x_first

        for i in range(1, self.current_scale + 1): #for i in range(1, 1) is NULL
            x_inter = F.interpolate(x_inter, (self.size_list[i], self.size_list[i]), mode='bilinear', align_corners=True)

            x_prev = x_inter
            x_inter = F.pad(x_inter, [5, 5, 5, 5], value=0)
            x_inter = x_inter + z[i]
            gen = self.sub_generators[i](x_inter)
            x_inter =gen  + x_prev

            if (ijs is not None) and (vals is not None):
                ij=ijs[i]
                x_inter[...,ij[:,0],ij[:,1]]=vals

            x_list.append(x_inter)

        return x_list

    def progress(self):
        self.current_scale += 1

        if self.current_scale % 4 == 0:
            self.nf *= 2

        tmp_generator = nn.ModuleList()
        tmp_generator.append(nn.Sequential(nn.Conv2d(self.img_ch, self.nf, self.kernel, 1,self.kernel//2-1),
                                           nn.BatchNorm2d(self.nf),
                                           nn.LeakyReLU(2e-1)))

        for _ in range(3):
            tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.nf, self.kernel, 1,self.kernel//2-1),
                                               nn.BatchNorm2d(self.nf),
                                               nn.LeakyReLU(2e-1)))

        tmp_generator.append(nn.Sequential(nn.Conv2d(self.nf, self.img_ch, self.kernel, 1,self.kernel//2-1),
                                           nn.Tanh()))

        tmp_generator = nn.Sequential(*tmp_generator)

        if self.current_scale % 4 != 0:
            prev_generator = self.sub_generators[-1]

            # Initialize layers via copy
            if self.current_scale >= 1:
                tmp_generator.load_state_dict(prev_generator.state_dict())

        self.sub_generators.append(tmp_generator)
        print("GENERATOR PROGRESSION DONE")
