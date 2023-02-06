import os
import numpy as np
from PIL import Image
from typing import List, Callable, Tuple


import torch
import torch.nn as nn
import torchvision.transforms as tf
import torchvision.transforms.functional as F


# Transforms
class Compose:
    def __init__(self,
                 transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self,
                 lr: Image.Image,
                 hr: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            lr, hr = transform(lr, hr)

        return lr, hr


class ToTensor:
    def __init__(self,
                 rgb_range: int = 1):
        self.rgb_range = rgb_range

    def __call__(self,
                 lr: Image.Image,
                 hr: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rgb_range != 1:
            lr = F.pil_to_tensor(lr).float()
            hr = F.pil_to_tensor(hr).float()
        else:
            lr = F.to_tensor(np.array(lr))
            hr = F.to_tensor(np.array(hr))

        return lr, hr


class Normalize:
    def __init__(self,
                 mean: List[float] = (0.4488, 0.4371, 0.4040),
                 std: List[float] = (1.0, 1.0, 1.0),
                 rgb_range: int = 1):
        self.mean = mean
        self.std = std
        self.rgb_range = rgb_range
        self.mean_shift = self.rgb_range * torch.Tensor(self.mean) / torch.Tensor(self.std)
        self.norm = tf.Normalize(mean=mean,
                                 std=std)

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rgb_range != 1:
            return img, gt
        else:
            img = self.norm(img)
            return img, gt


class CenterCrop:
    def __init__(self,
                 crop_size: int,
                 scale: int):
        self.crop_size = crop_size
        self.scale = scale

    def __call__(self,
                 lr: Image.Image,
                 hr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        hr = F.center_crop(hr, (self.crop_size * self.scale, self.crop_size * self.scale))
        lr = F.center_crop(lr, (self.crop_size, self.crop_size))

        return lr, hr


# Dataset
class SRDataset(nn.Dataset):
    def __init__(self, lr_images_dir, hr_images_dir, transform=None):
        self.lr_images_dir = lr_images_dir
        self.hr_images_dir = hr_images_dir
        self.transform = transform
        self.lr_images = os.listdir(lr_images_dir)
        self.hr_images = os.listdir(hr_images_dir)

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        lr_image_path = os.path.join(self.lr_images_dir, self.lr_images[index])
        hr_image_path = os.path.join(self.hr_images_dir, self.hr_images[index])

        lr_image = Image.open(lr_image_path)
        hr_image = Image.open(hr_image_path)

        if self.transform:
            lr_image, hr_image = self.transform(lr_image, hr_image)

        return lr_image, hr_image, self.lr_images[index]