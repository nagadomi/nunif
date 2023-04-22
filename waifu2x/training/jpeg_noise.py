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
        1: 0.6,
        2: 1.0,
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
        0: [90],
        1: [80],
        2: [70],
        3: [60, 90],
    }
}


# Use custom qtables
QTABLE_FILE = path.join(path.dirname(__file__), "_qtables_1.pth")
if path.exists(QTABLE_FILE):
    QTABLES = torch.load(QTABLE_FILE, weights_only=True)
else:
    QTABLES = None


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
        x.save(buff, format="jpeg", qtables=random.choice(QTABLES), subsampling="4:2:0")
        buff.seek(0)
        x = Image.open(buff)
        x.load()
        return x


def choose_jpeg_quality(style, noise_level):
    qualities = []
    if style == "art":
        if noise_level == 0:
            qualities.append(random.randint(85, 95))
        elif noise_level == 1:
            qualities.append(random.randint(65, 85))
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
    elif style == "photo":
        if noise_level == 0:
            qualities.append(random.randint(85, 95))
        elif noise_level == 1:
            if random.uniform(0, 1) < 0.5:
                qualities.append(random.randint(37, 70))
            else:
                qualities.append(random.randint(90, 98))
        else:
            if noise_level == 3 or random.uniform(0, 1) < 0.6:
                if random.uniform(0, 1) < 0.05:
                    quality1 = random.randint(52, 95)
                else:
                    quality1 = random.randint(37, 70)
                qualities.append(quality1)
                if quality1 >= 70 and random.uniform(0, 1) < 0.2:
                    qualities.append(random.randint(70, 90))
            else:
                qualities.append(random.randint(90, 98))
    else:
        raise NotImplementedError()

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


LAPLACIAN_KERNEL = torch.tensor([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0],
], dtype=torch.float32).reshape(1, 1, 3, 3)


def sharpen(x, strength=0.1):
    grad = torch.nn.functional.conv2d(x.mean(dim=0, keepdim=True).unsqueeze(0),
                                      weight=LAPLACIAN_KERNEL, stride=1, padding=1).squeeze(0)
    x = x + grad * strength
    x = torch.clamp(x, 0., 1.)
    return x


def sharpen_noise(original_x, noise_x, strength=0.1):
    """ shapen (noise added image - original image) diff
    """
    original_x = TF.to_tensor(original_x)
    noise_x = TF.to_tensor(noise_x)
    noise = noise_x - original_x
    noise = sharpen(noise, strength=strength)
    x = torch.clamp(original_x + noise, 0., 1.)
    x = TF.to_pil_image(x)
    return x


def sharpen_noise_all(x, strength=0.1):
    """ just sharpen image
    """
    x = TF.to_tensor(x)
    x = sharpen(x, strength=strength)
    x = TF.to_pil_image(x)
    return x


class RandomJPEGNoiseX():
    def __init__(self, style, noise_level, random_crop=False):
        assert noise_level in {0, 1, 2, 3} and style in {"art", "photo"}
        self.noise_level = noise_level
        self.style = style
        self.random_crop = random_crop

    def __call__(self, x, y):
        original_x = x
        if random.uniform(0, 1) > NR_RATE[self.style][self.noise_level]:
            # do nothing
            return x, y
        if self.style == "photo" and QTABLES and self.noise_level in {2, 3} and random.uniform(0, 1) < 0.25:
            x = add_jpeg_noise_qtable(x)
            strength_factor = 1. if self.noise_level == 3 else 0.75
            if random.uniform(0, 1) < 0.5:
                if random.uniform(0, 1) < 0.25:
                    x = sharpen_noise(original_x, x,
                                      strength=random.uniform(0.05, 0.2) * strength_factor)
                else:
                    # I do not want to use this
                    # because it means applying blur (inverse of sharpening) to the output.
                    # However, without this,
                    # it is difficult to remove noise applying sharpness filter after JPEG compression.
                    x = sharpen_noise_all(x, strength=random.uniform(0.1, 0.4) * strength_factor)
                    if random.uniform(0, 1) < 0.25:
                        x = add_jpeg_noise(x, quality=random.randint(80, 95), subsampling="4:2:0")
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
            if (i == 0 and self.style == "photo" and self.noise_level in {2, 3} and random.uniform(0, 1) < 0.2):
                if random.uniform(0, 1) < 0.75:
                    x = sharpen_noise(original_x, x, strength=random.uniform(0.05, 0.2))
                else:
                    x = sharpen_noise_all(x, strength=random.uniform(0.1, 0.3))
            if random_crop and i != len(qualities) - 1:
                x, y = shift_jpeg_block(x, y)
        return x, y


def _test_noise_level():
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


def _test_noise_sharpen():
    from nunif.utils import pil_io
    import argparse
    import cv2

    def show(name, im):
        cv2.imshow(name, pil_io.to_cv2(im))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input file")
    args = parser.parse_args()
    im, _ = pil_io.load_image_simple(args.input)

    show("original", im)
    while True:
        noise = add_jpeg_noise_qtable(im)
        if False:
            x = sharpen_noise_all(im, noise, 0.2)
        else:
            x = sharpen_noise_all(noise, 0.3)
        show("noise", noise)
        show("sharpen", x)
        c = cv2.waitKey(0)
        if c in {ord("q"), ord("x")}:
            break


if __name__ == "__main__":
    # _test_noise_level()
    _test_noise_sharpen()
