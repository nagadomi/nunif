from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)
import os
from os import path
import torch.nn.functional as F
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from ...base_depth_model import BaseDepthModel
import random


def random_resized_crop(size, *images):
    i, j, h, w = T.RandomResizedCrop.get_params(images[0], scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
    results = []
    for im in images:
        results.append(TF.resized_crop(im, i, j, h, w, size=(size, size),
                                       interpolation=InterpolationMode.BILINEAR, antialias=False))

    return tuple(results)


def random_hflip(*images):
    do_flip = random.choice([True, False])
    if not do_flip:
        return images

    results = []
    for im in images:
        results.append(TF.hflip(im))

    return tuple(results)


def center_crop(size, *images):
    results = []
    for im in images:
        results.append(TF.center_crop(im))

    return tuple(results)


def resize(size, *images):
    results = []
    for im in images:
        resized = F.interpolate(
            im.unsqueeze(0),
            (size, size),
            mode="bilinear",
            antialias=False,
            align_corners=True
        ).squeeze(0)
        results.append(resized)

    return tuple(results)


def load_image(depth_path, training):
    depth, _ = BaseDepthModel.load_depth(depth_path)
    if training:
        mask_dir = "DUTS-TR-Mask"
    else:
        mask_dir = "DUTS-TE-Mask"
    mask_path = path.join(path.dirname(depth_path), "..", "..", mask_dir, path.basename(depth_path))
    mask, _ = load_image_simple(mask_path, color="gray")
    mask = TF.to_tensor(mask)

    return depth, mask


class DSODDataset(Dataset):
    def __init__(self, input_dir, mask_dir, size, training):
        super().__init__()
        self.size = size
        self.training = training
        self.mask_dir = mask_dir
        self.files = []
        for d in os.listdir(input_dir):
            self.files += ImageLoader.listdir(path.join(input_dir, d))
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        depth, mask = load_image(self.files[index], self.training)

        if self.training:
            depth, mask = resize(self.size + 64, depth, mask)
            depth, mask = random_hflip(depth, mask)
            depth, mask = random_resized_crop(self.size, depth, mask)
        else:
            depth, mask = resize(self.size, depth, mask)

        mask = (mask > 0.5).float()

        return depth, mask, index
