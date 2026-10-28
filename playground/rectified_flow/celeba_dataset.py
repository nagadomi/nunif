import random

import torch
import torchvision.transforms as T
from torchvision.datasets import CelebA
from torchvision.transforms import functional as TF

IMAGE_SIZE = 112


class RandomCrop:
    def __init__(self, min_size=168):
        self.min_size = min_size

    def __call__(self, x):
        w, h = x.size
        min_w = min(self.min_size, w)
        min_h = min(self.min_size, h)
        size_w = random.randint(min_w, w)
        size_h = random.randint(min_h, h)
        sw = random.randint(0, w - size_w)
        sh = random.randint(0, h - size_h)

        assert sw + size_w <= w
        assert sh + size_h <= h

        return TF.crop(x, sh, sw, size_h, size_w)


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        super().__init__()
        if split == "train":
            transform = T.Compose(
                [
                    T.CenterCrop((178, 178)),
                    RandomCrop(min_size=168),
                    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1)], p=0.25),
                    T.RandomHorizontalFlip(),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.CenterCrop((168, 168)),
                    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                ]
            )
        self.celeba = CelebA(root, split, target_type="bbox", transform=transform, download=True)

    def __len__(self):
        return len(self.celeba)

    def sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(self, num_samples=num_samples, replacement=True)

    def __getitem__(self, i):
        x, *_ = self.celeba[i]
        x = TF.to_tensor(x)
        x = x * 2 - 1.0
        return x, 0.0
