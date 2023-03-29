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


class Resize():
    def __init__(self, size, mode):
        assert mode in {"resize", "reflect"}
        assert isinstance(size, int)
        self.size = size
        self.mode = mode

    def __call__(self, x):
        if self.mode == "resize":
            # just resize
            return TF.resize(x, self.size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        elif self.mode == "reflect":
            # resize with preserve aspect ratio
            w, h = x.size
            if w >= h:
                scale_factor = self.size / w
                new_h, new_w = int(h * scale_factor), self.size
            else:
                scale_factor = self.size / h
                new_h, new_w = self.size, int(w * scale_factor)
            x = TF.resize(x, (new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)

            # reflection pad for small side
            w, h = x.size
            pad_l = pad_t = pad_r = pad_b = 0
            if self.size > w:
                border = (self.size - w)
                pad_l = border // 2
                pad_r = border // 2 + (border % 2)
            if self.size > h:
                border = (self.size - h)
                pad_t = border // 2
                pad_b = border // 2 + (border % 2)
            if pad_l + pad_t + pad_r + pad_b != 0:
                x = TF.pad(x, (pad_l, pad_t, pad_r, pad_b), padding_mode="reflect")

            assert x.size == (self.size, self.size)
            return x


class ImageNetDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root, split, resize=256, size=224, norm="imagenet", resize_mode="reflect"):
        assert resize >= size
        if split == "train":
            transform = T.Compose([
                Resize(resize, resize_mode),
                T.RandomCrop(size),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                Normalize(mode=norm)
            ])
        else:
            transform = T.Compose([
                Resize(resize, resize_mode),
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
