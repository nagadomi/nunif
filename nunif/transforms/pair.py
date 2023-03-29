import math
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode
)
import random
from .std import pad as safe_pad


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
        assert (y_scale in {1, 2, 4, 8})
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
        assert (y_scale in {1, 2, 4, 8})
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
        assert (y_scale in {1, 2, 4, 8})
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
            color_stdv = rect.std(dim=[1, 2]).sum().item()
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


class RandomSafeRotate():
    def __init__(self, angle_min=-45, angle_max=45, y_scale=1):
        self.y_scale = y_scale
        self.angle_min = angle_min
        self.angle_max = angle_max

    def __call__(self, x, y):
        pad_x = (math.ceil(max(x.size) * math.sqrt(2)) - max(x.size)) // 2
        pad_y = pad_x * self.y_scale
        angle = random.uniform(self.angle_min, self.angle_max)
        rot_x = TF.rotate(safe_pad(x, (x.size[1] + pad_x * 2, x.size[0] + pad_x * 2)), angle=angle,
                          interpolation=InterpolationMode.BICUBIC)
        rot_y = TF.rotate(safe_pad(y, (y.size[1] + pad_y * 2, y.size[0] + pad_y * 2)), angle=angle,
                          interpolation=InterpolationMode.BICUBIC)
        x = TF.center_crop(rot_x, (x.size[1], x.size[0]))
        y = TF.center_crop(rot_y, (y.size[1], y.size[0]))

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
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p

    def __call__(self, x, y):
        if random.uniform(0, 1) < self.p:
            for f in self.transforms:
                x, y = f(x, y)
            return x, y
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
