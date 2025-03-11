import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class GeologicalFaciesDataset(Dataset):
    THRESHOLD = 155
    IMG_WIDTH = 256.0
    IMG_HEIGHT = 256.0
    MEAN = 0.5
    STD = 1.0

    def __init__(self, imgs_dir, scales_list, shuffle=True):
        self.imgs_dir = imgs_dir
        self.scales_list = scales_list
        self.image_files = [f for f in os.listdir(self.imgs_dir) if f.endswith('.png')]
        self.images_list = [torch.empty((0, 1, scale, scale)) for scale in scales_list]
        self.mask_indexes_list = [torch.empty((0, 1, scale, scale), dtype=torch.int32) for scale in scales_list]
        self.normalizer = transforms.Normalize(self.MEAN, self.STD)
        self.resizers = [transforms.Resize(scale) for scale in scales_list[:-1]]
        self.image_torchizer = transforms.Compose(
            [
                transforms.Resize((self.scales_list[-1], self.scales_list[-1])),
                transforms.ToTensor(),
                lambda x: (x > 0).float(),
                transforms.Normalize(self.MEAN, self.STD)
            ]
        )

        self.mask_torchizer = transforms.Compose(
            [
                transforms.Resize((self.scales_list[-1], self.scales_list[-1])),
                transforms.ToTensor(),
            ]
        )

        mask_indexes = [[] for _ in scales_list]
        images = [[] for _ in scales_list]
        for image_file in self.image_files:
            image_path = os.path.join(self.imgs_dir, image_file)
            self.__load_image(image_path, images, mask_indexes)

        for i in range(len(self.scales_list)):
            self.images_list[i] = torch.stack(images[i], dim=0).squeeze(1)
            self.mask_indexes_list[i] = torch.stack(mask_indexes[i], dim=0)

        if shuffle:
            self.shuffle()

    def __len__(self):
        return self.images_list[0].shape[0]

    def __getitem__(self, idx):
        return tuple(images[idx] for images in self.images_list), tuple(masks[idx] for masks in self.mask_indexes_list)

    def shuffle(self):
        idxs = torch.randperm(self.__len__())
        for i in range(len(self.scales_list)):
            self.images_list[i] = self.images_list[i][idxs]
            self.mask_indexes_list[i] = self.mask_indexes_list[i][idxs]

    def __load_image(self, image_file,  images, mask_indexes, image_size=(IMG_WIDTH, IMG_WIDTH)):
        """Loads an image, generates masks_list, and converts them into tensors."""
        image = self.image_torchizer(Image.open(image_file)
            .convert("L").crop((0.0, 0.0, image_size[0], image_size[1]))).unsqueeze(0)
        mask = self.mask_torchizer(Image.open(image_file)
            .convert("L").crop((image_size[0], 0, 2 * image_size[0], image_size[1]))).unsqueeze(0)

        for i, scale in enumerate(self.scales_list[:-1]):
            scaled_image = self.resizers[i](image)
            channels_idxs = torch.where(scaled_image > 0)
            scaled_image = torch.full_like(scaled_image, -self.MEAN)
            scaled_image[channels_idxs] = self.MEAN
            images[i].append(scaled_image)
        images[-1].append(image)

        self.__get_mask_indexes(images, mask_indexes, mask)


    def __get_mask_indexes(self, images, mask_indexes, mask):
        for i, image in enumerate(images[:-1]):
            mi = self.__get_scaled_mask_index(mask, i)
            mask_indexes[i].append(mi)
        mask_index = self.__get_main_mask_index(mask)
        mask_indexes[-1].append(mask_index)


    @staticmethod
    def __get_main_mask_index(mask):
        sum_mask = torch.sum(mask > 0.14, dim=2)
        max_sum, _ = torch.max(sum_mask, dim=2, keepdim=True)
        return torch.mean(torch.where(sum_mask == max_sum)[2].float()).int()


    def __get_scaled_mask_index(self, mask, scale):
        scaled_mask = self.resizers[scale](mask).unsqueeze(0).unsqueeze(0)
        max_mask = scaled_mask == torch.max(scaled_mask)
        columns_sum = torch.sum(max_mask, dim=4)
        return np.argmax(columns_sum) if torch.any(columns_sum > 0) else -1
