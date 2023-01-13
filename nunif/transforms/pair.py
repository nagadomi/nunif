from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode
)
import random


def same_size(a, b):
    if isinstance(a, Image.Image):
        return a.size == b.size
    else:
        return a.size() == b.size()


class Identity():
    def __init__(self):
        pass

    def __call__(self, x, y):
        return x, y


class RandomCrop():
    def __init__(self, size, y_offset=0, y_scale=1):
        assert (y_scale in {1, 2, 4})
        self.size = (size, size)
        self.y_offset = y_offset
        self.y_scale = y_scale

    def __call__(self, x, y):
        assert ((not self.y_scale == 1) or same_size(x, y))
        i, j, h, w = T.RandomCrop.get_params(x, self.size)
        x = TF.crop(x, i, j, h, w)
        y = TF.crop(
            y,
            int(i * self.y_scale) + self.y_offset,
            int(j * self.y_scale) + self.y_offset,
            int(h * self.y_scale) - self.y_offset * 2,
            int(w * self.y_scale) - self.y_offset * 2)

        return x, y


class CenterCrop():
    def __init__(self, size, y_offset=0, y_scale=1):
        assert (y_scale in {1, 2, 4})
        self.size = (size, size)
        self.y_offset = y_offset
        self.y_scale = y_scale

    def __call__(self, x, y):
        assert ((not self.y_scale == 1) or same_size(x, y))
        x = TF.center_crop(x, self.size)
        y = TF.center_crop(
            y,
            (int(self.size[0] * self.y_scale) - self.y_offset * 2,
             int(self.size[1] * self.y_scale) - self.y_offset * 2))

        return x, y


class RandomHardExampleCrop():
    def __init__(self, size, y_offset=0, y_scale=1, samples=4):
        assert (y_scale in {1, 2, 4})
        self.size = (size, size)
        self.y_offset = y_offset
        self.y_scale = y_scale
        self.samples = samples

    def __call__(self, x, y):
        assert ((not self.y_scale == 1) or same_size(x, y))
        rects = []
        yt = TF.to_tensor(y)
        for _ in range(self.samples):
            i, j, h, w = T.RandomCrop.get_params(x, self.size)
            rect = TF.crop(
                yt,
                int(i * self.y_scale) + self.y_offset,
                int(j * self.y_scale) + self.y_offset,
                int(h * self.y_scale) - self.y_offset * 2,
                int(w * self.y_scale) - self.y_offset * 2)
            color_stdv = rect.permute(1, 2, 0).reshape(-1, 3).std(dim=0).sum().item()
            rects.append(((i, j, h, w), color_stdv))

        i, j, h, w = max(rects, key=lambda v: v[1])[0]
        x = TF.crop(x, i, j, h, w)
        y = TF.crop(
            y,
            int(i * self.y_scale) + self.y_offset,
            int(j * self.y_scale) + self.y_offset,
            int(h * self.y_scale) - self.y_offset * 2,
            int(w * self.y_scale) - self.y_offset * 2)

        return x, y


class RandomFlip():
    def __call__(self, x, y):
        if random.uniform(0, 1) > 0.5:
            x = TF.rotate(x, 90, interpolation=InterpolationMode.NEAREST)
            y = TF.rotate(y, 90, interpolation=InterpolationMode.NEAREST)
        steps = random.choice([[], [TF.hflip], [TF.vflip], [TF.vflip, TF.hflip]])
        for f in steps:
            x = f(x)
            y = f(y)
        return x, y


class RandomApply():
    def __init__(self, p, transform):
        self.p = p
        self.transform = transform

    def __call__(self, x, y):
        if random.uniform(0, 1) < self.p:
            return self.transform(x, y)
        else:
            return x, y


class RandomChoice():
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p
        if self.p is None:
            self.p = [1] * len(self.transforms)

    def __call__(self, x, y):
        transform = random.choices(self.transforms, weights=self.p, k=1)[0]
        return transform(x, y)


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for transform in self.transforms:
            x, y = transform(x, y)
        return x, y
