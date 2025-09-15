import random
import os
from os import path
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from torchvision.transforms import (
    functional as TF,
)
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from iw3.dilation import dilate_outer, dilate_inner


SIZE = 256  # % 8 == 0
SEQ = 12


class RightDilate():
    def __init__(self, max_step=20, p=0.5):
        self.max_step = max_step
        self.p = p

    def __call__(self, mask):
        if random.uniform(0, 1) < self.p:
            n_step = random.randint(1, self.max_step)
            mask = dilate_outer(mask, n_iter=n_step)
        return mask


class LeftDilate():
    def __init__(self, max_step=4, p=0.25):
        self.max_step = max_step
        self.p = p

    def __call__(self, mask):
        if random.uniform(0, 1) < self.p:
            n_step = random.randint(1, self.max_step)
            mask = dilate_inner(mask, n_iter=n_step)
        return mask


class RandomColorJitter():
    def __call__(self, x):
        if random.uniform(0, 1) < 0.5:
            return x

        grayscale = random.uniform(0, 1) < 0.05
        if grayscale:
            x = x.mean(dim=1, keepdim=True).expand_as(x).contiguous()
            return x

        scale_r = random.uniform(0.7, 0.9)
        scale_g = random.uniform(0.7, 0.9)
        scale_b = random.uniform(0.7, 0.9)
        bias_r = random.uniform(0, 1 - scale_r)
        bias_g = random.uniform(0, 1 - scale_g)
        bias_b = random.uniform(0, 1 - scale_b)
        scale = torch.tensor([scale_r, scale_g, scale_b], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        bias = torch.tensor([bias_r, bias_g, bias_b], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)

        return (x * scale + bias).clamp(0, 1)


def crop(images, i, j, h, w):
    results = []
    for im in images:
        results.append(im[:, :, i:i + h, j:j + w])

    return tuple(results)


def random_crop(size, *images):
    i, j, h, w = T.RandomCrop.get_params(images[0][0], (size, size))
    return crop(images, i, j, h, w)


def random_hard_example_crop(size, n, *images):
    assert n > 0
    results = []
    for _ in range(n):
        crop_images = random_crop(size, *images)
        mask = crop_images[-1]
        mask_sum = mask.float().sum().item()
        results.append((mask_sum, crop_images))

    results = sorted(results, key=lambda v: v[0], reverse=True)
    return results[0][1]


def center_crop(size, *images):
    h, w = images[0].shape[-2:]
    h_i = (h - size) // 2
    w_i = (w - size) // 2
    return tuple([im[:, :, h_i:h_i + size, w_i: w_i + size] for im in images])


def fixed_hard_example_crop(size, *images):
    results = []
    H, W = images[0].shape[-2:]
    h = 0
    for h in range(0, H - size + 1, size // 4):
        for w in range(0, W - size + 1, size // 4):
            crop_images = crop(images, h, w, size, size)
            mask = crop_images[-1]
            mask_sum = mask.float().sum().item()
            results.append((mask_sum, crop_images))

    results = sorted(results, key=lambda v: v[0], reverse=True)
    return results[0][1]


class VideoInpaintDataset(Dataset):
    def __init__(self, input_dir, model_offset, model_sequence_offset, training):
        super().__init__()
        self.training = training
        self.model_offset = model_offset
        self.model_sequence_offset = model_sequence_offset
        self.folders = self.load_folders(input_dir)
        if not self.folders:
            raise RuntimeError(f"{input_dir} is empty")
        self.right_dilate = RightDilate()
        self.left_dilate = LeftDilate()
        self.color_jitter = RandomColorJitter()

    @staticmethod
    def load_folders(input_dir):
        folders = []
        for fn in os.listdir(input_dir):
            fn = path.join(input_dir, fn)
            if path.isdir(fn):
                folders.append(fn)
        return folders

    @staticmethod
    def load_files(input_dir, seq, training):
        files = ImageLoader.listdir(input_dir)

        reverse = training and random.uniform(0, 1) < 0.5
        masks = sorted([fn for fn in files if fn.endswith("_M.png")], reverse=reverse)

        files = []
        for fn in masks:
            rgb_file = fn.replace("_M.png", "_C.png")
            if path.exists(rgb_file):
                files.append(rgb_file)
            else:
                raise RuntimeError(f"{rgb_file} not found")

        assert len(files) >= seq
        if training:
            start_i = random.randint(0, len(files) - seq)
        else:
            start_i = (len(files) - seq) // 2

        return files[start_i:start_i + seq], masks[start_i:start_i + seq]

    def worker_init(self, worker_id):
        pass

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        x, mask = self.load_files(folder, SEQ, self.training)

        x = torch.stack([TF.to_tensor(load_image_simple(fn, color="rgb")[0]) for fn in x])
        mask = torch.stack([TF.to_tensor(load_image_simple(fn, color="gray")[0]) for fn in mask])

        if self.training:
            x = self.color_jitter(x)
            mask = self.right_dilate(mask)
            mask = self.left_dilate(mask)
            x, mask = random_hard_example_crop(SIZE, 4, x, mask)
        else:
            x, mask = fixed_hard_example_crop(SIZE, x, mask)

        y = x.clone()
        y = y[:, :,
              self.model_offset: self.model_offset + (y.shape[-2] - self.model_offset * 2),
              self.model_offset: self.model_offset + (y.shape[-1] - self.model_offset * 2)]
        if self.model_sequence_offset > 0:
            y = y[self.model_sequence_offset:-self.model_sequence_offset]

        mask = mask > 0.5

        return x, mask, y, index


if __name__ == "__main__":
    pass
