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
        self.masks_list = [torch.empty((0, 1, scale, scale), dtype=torch.int32) for scale in scales_list]
        self.normalizer = transforms.Normalize(self.MEAN, self.STD)
        self.resizers = [transforms.Resize(scale) for scale in scales_list[:-1]]
        self.torchizer = transforms.Compose(
            [
                transforms.Resize((self.scales_list[-1], self.scales_list[-1])),
                transforms.ToTensor(),
                lambda x: (x > 0).float(),
                transforms.Normalize(self.MEAN, self.STD)
            ]
        )

        for image_file in self.image_files:
            image_path = os.path.join(self.imgs_dir, image_file)
            self.__load_image(image_path)

        if shuffle:
            self.shuffle()

    def __len__(self):
        return self.images_list[0].shape[0]

    def __getitem__(self, idx):
        return tuple(images[idx] for images in self.images_list), tuple(masks[idx] for masks in self.masks_list)

    def shuffle(self):
        idxs = torch.randperm(self.__len__())
        for i in range(len(self.scales_list)):
            self.images_list[i] = self.images_list[i][idxs]
            self.masks_list[i] = self.masks_list[i][idxs]

    def __load_image(self, image_file, image_size=(IMG_WIDTH, IMG_WIDTH)):
        """Loads an image, generates masks_list, and converts them into tensors."""
        image = self.torchizer(Image.open(image_file)
           .convert("L").crop((0.0, 0.0, image_size[0], image_size[1]))).unsqueeze(0)
        masks = [[] for _ in self.scales_list]
        images = [[] for _ in self.scales_list]
        masks_indexes = [[] for _ in self.scales_list]
        num_masks_indexes = self.__generate_masks_indexes(image, masks_indexes)
        if num_masks_indexes > 0:
            for i, scale in enumerate(self.scales_list[:-1]):
                scaled_image = self.resizers[i](image)
                channels_idxs = torch.where(scaled_image > 0)
                scaled_image = torch.full_like(scaled_image, -self.MEAN)
                scaled_image[channels_idxs] = self.MEAN
                images[i].extend([scaled_image for _ in range(num_masks_indexes)])
            images[-1].extend([image for _ in range(num_masks_indexes)])
            self.__generate_masks(masks, images, masks_indexes)
            for i in range(len(self.scales_list)):
                self.images_list[i] = torch.cat([self.images_list[i], torch.stack(images[i], dim=0).squeeze(1)], dim=0)
                self.masks_list[i] = torch.cat([self.masks_list[i], torch.stack(masks[i], dim=0).squeeze(1)], dim=0)


    @staticmethod
    def __generate_masks(masks_lists, images_lists, masks_indexes):
        for i, image_list in enumerate(images_lists):
            for j, image in enumerate(image_list):
                mask = torch.zeros_like(image, dtype=torch.int8)
                mask[..., :, masks_indexes[i][j]] = (image[..., :, masks_indexes[i][j]] > 0).type(torch.int8)
                masks_lists[i].append(mask)


    def __generate_masks_indexes(self, image, masks_indexes):
        num_masks_indexes = 0
        for i in range(image.shape[2]):
            y = image[0, :, :, i]
            mask_idxs = torch.where(y > torch.min(y))
            if len(mask_idxs[0]) > self.THRESHOLD:
                mask = torch.zeros_like(image)
                mask[0, 0, mask_idxs[1], mask_idxs[0] + i] = torch.ones_like(y[mask_idxs])
                scaled_mask_index = self.__generate_single_mask_index(mask, 0)
                if scaled_mask_index > 0:
                    masks_indexes[0].append(scaled_mask_index)
                    masks_indexes[-1].append(i)
                    for s in range(1, len(self.scales_list[1:])):
                        masks_indexes[s].append(self.__generate_single_mask_index(mask, s))
                    num_masks_indexes += 1
        return num_masks_indexes

    def __generate_single_mask_index(self, mask, scale):
        scaled_mask = self.resizers[scale](mask)
        columns_sum = torch.sum(scaled_mask > 0, dim=2)
        return np.argmax(columns_sum) if torch.any(columns_sum > 0) else -1
