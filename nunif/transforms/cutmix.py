from PIL import Image, ImageDraw
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


def cutmix(a, b, mask_min=0.2, mask_max=0.5, rotate_p=0.2):
    # a, b: PIL image

    # crop to the same size
    width = min(a.width, b.height)
    height = min(a.height, b.height)
    a = _random_crop(a, width, height)
    b = _random_crop(b, width, height)
    # composite
    mask = generate_random_mask(width, height, mask_min=mask_min, mask_max=mask_max, rotate_p=rotate_p)
    out = Image.composite(a, b, mask)
    return out


if __name__ == "__main__":
    import time
    a = Image.open("cc0/bottle.jpg")
    b = Image.open("cc0/lighthouse.jpg")

    for i in range(10):
        out = cutmix(a, b)
        out.show()
        time.sleep(1)
