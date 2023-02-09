import os
import numpy as np
from PIL import Image
from typing import List, Callable, Tuple


import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from utils import util_image as util


# Transforms
class Compose:
    def __init__(self,
                 transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self,
                 lr: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            lr = transform(lr)

        return lr


class ToTensor:
    def __init__(self,
                 rgb_range: int = 1):
        self.rgb_range = rgb_range

    def __call__(self,
                 lr: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rgb_range != 1:
            lr = F.pil_to_tensor(lr).float()
        else:
            lr = F.to_tensor(np.array(lr))

        return lr
    

class CenterCrop:
    def __init__(self,
                 crop_size: int,
                 scale: int):
        self.h, self.w = crop_size
        self.scale = scale

    def __call__(self,
                 lr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        lr = F.center_crop(lr, (int(self.h/self.scale), int(self.w/self.scale)))

        return lr


# Dataset
class SRDataset(data.Dataset):
    def __init__(self, lr_images_dir, transform=None, n_channels=3):
        self.lr_images_dir = lr_images_dir
        self.lr_images = os.listdir(lr_images_dir)
        self.transform = transform
        self.n_channels = n_channels

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        img_path = os.path.join(self.lr_images_dir, self.lr_images[index])
       
        img_L = util.imread_uint(img_path, n_channels=self.n_channels)
        img_L = util.uint2tensor3(img_L)


        if self.transform:
            img_L = self.transform(img_L)

        return img_L, self.lr_images[index]