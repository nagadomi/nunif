# overlay
import random

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps


def _random_crop(x, width, height):
    i, j, h, w = T.RandomCrop.get_params(x, (height, width))
    rect = TF.crop(x, i, j, h, w)
    return rect


def mixup(a, b=None, alpha=0.5):
    # a, b: PIL image
    if b is None:
        # a and b are the same image
        # make grid
        hflip = ImageOps.mirror(a)
        b = Image.new(a.mode, (a.width * 2, a.height * 2), "black")
        b.paste(a, (0, 0))
        b.paste(hflip, (a.width, 0))
        b.paste(hflip, (0, a.height))
        b.paste(a, (a.width, a.height))

    # crop to the same size
    width = min(a.width, b.height)
    height = min(a.height, b.height)
    if a.width != width or a.height != height:
        a = _random_crop(a, width, height)
    if b.width != width or b.height != height:
        b = _random_crop(b, width, height)

    # composite
    out = Image.blend(b, a, alpha)

    return out


class RandomOverlay:
    def __call__(self, a, b=None):
        return mixup(a, b, alpha=random.uniform(0.0, 1.0))


def _test():
    import time

    a = Image.open("tests/images/bottle_448.png")
    b = Image.open("tests/images/lighthouse_448.png")

    if True:
        for i in range(2):
            out = mixup(a, b)
            out.show()
            time.sleep(2)

    if True:
        transform = RandomOverlay()
        for i in range(2):
            out = transform(b)
            out.show()
            time.sleep(2)


if __name__ == "__main__":
    _test()
