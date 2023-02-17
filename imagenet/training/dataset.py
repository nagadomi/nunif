import torch
from torchvision.datasets import ImageNet
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode
)


class Normalize():
    def __init__(self, mode="imagenet"):
        if mode == "none":
            self.f = lambda x: x
        elif mode == "imagenet":
            self.f = lambda x: TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif mode == "gcn":
            self.f = lambda x: (x - x.mean(dim=[1, 2], keepdim=True)) / (x.std(dim=[1, 2], keepdim=True) + 1e-6)
        elif mode == "center":
            self.f = lambda x: (x - 0.5) * 2.
        else:
            raise NotImplementedError()

    def __call__(self, x):
        return self.f(x)


class ImageNetDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root, split, resize=256, size=224, norm="imagenet"):
        assert resize >= size
        if split == "train":
            transform = T.Compose([
                T.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.RandomCrop(size),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                Normalize(mode=norm)
            ])
        else:
            transform = T.Compose([
                T.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(size),
                T.ToTensor(),
                Normalize(mode=norm)
            ])
        self.imagenet = ImageNet(root=root, split=split, transform=transform)

    def __len__(self):
        return len(self.imagenet)

    def sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def __getitem__(self, i):
        x, y = self.imagenet[i]
        return x, y
