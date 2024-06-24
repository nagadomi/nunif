from PIL import Image, ImageDraw, ImageOps
import random
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)


def _random_crop(x, width, height):
    i, j, h, w = T.RandomCrop.get_params(x, (height, width))
    rect = TF.crop(x, i, j, h, w)
    return rect


def generate_random_mask(width, height, mask_min=0.2, mask_max=0.5, rotate_p=0.2):
    width, height = width * 2, height * 2

    mask = Image.new("L", (width, height), "black")
    gc = ImageDraw.Draw(mask)

    w, h = int(random.uniform(mask_min, mask_max) * width), int(random.uniform(mask_min, mask_max) * height)
    x, y = random.randint(-w // 2, width - 1 - w // 2), random.randint(-h // 2, height - 1 - h // 2)
    xy = (x, y, x + w, y + h)
    mask_type = random.choice(["ellipse", "rectangle", "rounded_rectangle"])
    if mask_type == "rectangle":
        gc.rectangle(xy, fill="white")
    elif mask_type == "rounded_rectangle":
        radius = random.randint(0, min(w, h) // 2)
        gc.rounded_rectangle(xy, radius=radius, fill="white")
    else:
        gc.ellipse(xy, fill="white")

    if random.uniform(0, 1) < rotate_p:
        angle = random.uniform(0, 360)
        mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.BILINEAR)

    mask = TF.resize(mask, size=(height // 2, width // 2), interpolation=InterpolationMode.BILINEAR)

    return mask


def cutmix(a, b=None, mask_min=0.2, mask_max=0.5, rotate_p=0.2):
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
        a, b = b, a

    # crop to the same size
    width = min(a.width, b.height)
    height = min(a.height, b.height)
    if a.width != width or a.height != height:
        a = _random_crop(a, width, height)
    if b.width != width or b.height != height:
        b = _random_crop(b, width, height)
    # composite
    mask = generate_random_mask(width, height, mask_min=mask_min, mask_max=mask_max, rotate_p=rotate_p)
    out = Image.composite(a, b, mask)
    return out


class CutMix():
    def __init__(self, mask_min=0.2, mask_max=0.5, rotate_p=0.2):
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.rotate_p = rotate_p

    def __call__(self, a, b=None):
        return cutmix(a, b, mask_min=self.mask_min, mask_max=self.mask_max, rotate_p=self.rotate_p)


if __name__ == "__main__":
    import time
    a = Image.open("cc0/bottle.jpg")
    b = Image.open("cc0/lighthouse.jpg")

    if False:
        for i in range(10):
            out = cutmix(a, b)
            out.show()
            time.sleep(1)

    if True:
        transform = CutMix()
        for i in range(10):
            out = transform(b)
            out.show()
            time.sleep(1)
