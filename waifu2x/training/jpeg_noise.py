# JPEG Noise level processing porting from original waifu2x
import random
from io import BytesIO
from PIL import Image
from torchvision.transforms import functional as TF
import torch
from os import path


# p of random apply
NR_RATE = {
    "art": {
        0: 0.65,
        1: 0.65,
        2: 0.65,
        3: 1.0,
    },
    "photo": {
        0: 0.3,
        1: 0.3,
        2: 0.6,
        3: 1.0,
    }
}
JPEG_CHROMA_SUBSAMPLING_RATE = 0.5

# fixed quality for validation
EVAL_QUALITY = {
    "art": {
        0: [85 + (95 - 85) // 2],
        1: [65 + (85 - 65) // 2],
        2: [37 + (70 - 37) // 2, 37 + (70 - 37) // 2 - (5 + (10 - 5) // 2)],
        3: [37 + (70 - 37) // 2, 37 + (70 - 37) // 2 - (5 + (10 - 5) // 2)],
    },
    "photo": {
        0: [85 + (95 - 85) // 2],
        1: [37 + (70 - 37) // 2],
        2: [37 + (70 - 37) // 2, 37 + (70 - 37) // 2 - (5 + (10 - 5) // 2)],
        3: [37 + (70 - 37) // 2, 37 + (70 - 37) // 2 - (5 + (10 - 5) // 2)],
    }
}


# Use custom qtables
if path.exists(path.join(path.dirname(__file__), "qtables.pth")):
    OLD_QTABLES = torch.load(path.join(path.dirname(__file__), "qtables.pth"))
else:
    OLD_QTABLES = None


def choose_validation_jpeg_quality(index, style, noise_level):
    mod100 = index % 100
    if mod100 > int(NR_RATE[style][noise_level] * 100):
        return [], None
    if index % 2 == 0:
        subsampling = "4:2:0"
    else:
        subsampling = "4:4:4"
    return EVAL_QUALITY[style][noise_level], subsampling


def add_jpeg_noise(x, quality, subsampling):
    assert x.mode == "RGB"
    with BytesIO() as buff:
        x.save(buff, format="jpeg", quality=quality, subsampling=subsampling)
        buff.seek(0)
        x = Image.open(buff)
        x.load()
        return x


def add_jpeg_noise_qtable(x):
    assert x.mode == "RGB"
    with BytesIO() as buff:
        x.save(buff, format="jpeg", qtables=random.choice(OLD_QTABLES))
        buff.seek(0)
        x = Image.open(buff)
        x.load()
        return x


def choose_jpeg_quality(style, noise_level):
    qualities = []
    if noise_level == 0:
        qualities.append(random.randint(85, 95))
    elif noise_level == 1:
        if style == "art":
            qualities.append(random.randint(65, 85))
        else:
            qualities.append(random.randint(37, 70))
    elif noise_level in {2, 3}:
        # 2 and 3 are the same, NR_RATE is different
        r = random.uniform(0, 1)
        if r > 0.4:
            qualities.append(random.randint(27, 70))
        elif r > 0.1:
            # nunif: Add high quality patterns
            if random.uniform(0, 1) < 0.05:
                quality1 = random.randint(37, 95)
            else:
                quality1 = random.randint(37, 70)
            quality2 = quality1 - random.randint(5, 10)
            qualities.append(quality1)
            qualities.append(quality2)
        else:
            # nunif: Add high quality patterns
            if random.uniform(0, 1) < 0.05:
                quality1 = random.randint(52, 95)
            else:
                quality1 = random.randint(52, 70)
            quality2 = quality1 - random.randint(5, 15)
            quality3 = quality1 - random.randint(15, 25)
            qualities.append(quality1)
            qualities.append(quality2)
            qualities.append(quality3)

    return qualities


def shift_jpeg_block(x, y, x_shift=None):
    # nunif: Add random crop before the second jpeg
    y_scale = y.size[0] / x.size[0]
    assert y_scale in {1, 2, 4}
    y_scale = int(y_scale)
    x_w, x_h = x.size
    y_w, y_h = y.size
    if x_shift is None:
        if random.uniform(0, 0.5) < 0.5:
            x_h_shift = random.randint(0, 7)
            x_w_shift = random.randint(0, 7)
        else:
            x_h_shift = x_w_shift = 0
    else:
        x_h_shift = x_w_shift = x_shift

    if x_h_shift > 0 or x_w_shift > 0:
        y_h_shift = x_h_shift * y_scale
        y_w_shift = x_w_shift * y_scale
        x = TF.crop(x, x_h_shift, x_w_shift, x_h - x_h_shift, x_w - x_w_shift)
        y = TF.crop(y, y_h_shift, y_w_shift, y_h - y_h_shift, y_w - y_w_shift)
        assert y.size[0] == x.size[0] * y_scale and y.size[1] == x.size[1] * y_scale

    return x, y


class RandomJPEGNoiseX():
    def __init__(self, style, noise_level, random_crop=False):
        assert noise_level in {0, 1, 2, 3} and style in {"art", "photo"}
        self.noise_level = noise_level
        self.style = style
        self.random_crop = random_crop

    def __call__(self, x, y):
        if random.uniform(0, 1) > NR_RATE[self.style][self.noise_level]:
            # do nothing
            return x, y
        if OLD_QTABLES and self.noise_level == 3 and random.uniform(0, 1) < 0.02:
            x = add_jpeg_noise_qtable(x)
            if random.uniform(0, 1) < 0.2:
                x = add_jpeg_noise_qtable(x)
            return x, y
        if (self.noise_level == 3 and random.uniform(0, 1) < 0.95) or random.uniform(0, 1) < 0.75:
            # use noise_level noise
            qualities = choose_jpeg_quality(self.style, self.noise_level)
        else:
            # use lower noise_level noise
            # this is the fix for a problem in the original waifu2x
            # that lower level noise cannot be denoised with higher level denoise model.
            noise_level = random.randint(0, self.noise_level)
            qualities = choose_jpeg_quality(self.style, noise_level)

        assert len(qualities) > 0

        if random.uniform(0, 1) < JPEG_CHROMA_SUBSAMPLING_RATE:
            subsampling = "4:2:0"
        else:
            subsampling = "4:4:4"

        # scale factor
        y_scale = y.size[0] / x.size[0]
        assert y_scale in {1, 2, 4}
        y_scale = int(y_scale)
        if self.random_crop and len(qualities) > 1:
            random_crop = True
        else:
            random_crop = False

        for i, quality in enumerate(qualities):
            x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)
            if random_crop and i != len(qualities) - 1:
                x, y = shift_jpeg_block(x, y)
        return x, y


if __name__ == "__main__":
    print("** train")
    for style in ["art", "photo"]:
        for noise_level in [0, 1, 2, 3]:
            for _ in range(10):
                n = random.randint(0, noise_level)
                print(style, noise_level, choose_jpeg_quality(style, n))
    print("** validation")
    for style in ["art", "photo"]:
        for noise_level in [0, 1, 2, 3]:
            for index in range(100):
                print(style, noise_level, choose_validation_jpeg_quality(index, style, noise_level))
