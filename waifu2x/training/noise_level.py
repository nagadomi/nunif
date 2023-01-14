# JPEG Noise level processing porting from original waifu2x
import random
from io import BytesIO
from PIL import Image


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
        2: [37 + (70 - 37) // 2],
        3: [37 + (70 - 37) // 2],
    }
}


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
                quality1 = random.randint(37, 70)
                quality2 = quality1 - random.randint(5, 10)
                qualities.append(quality1)
                qualities.append(quality2)
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
        else:
            # 1,2,3 are the same, NR_RATE is different
            qualities.append(random.randint(37, 70))

    return qualities


class RandomJPEGNoiseX():
    def __init__(self, style, noise_level):
        assert noise_level in {0, 1, 2, 3} and style in {"art", "photo"}
        self.noise_level = noise_level
        self.style = style

    def __call__(self, x, y):
        if random.uniform(0, 1) > NR_RATE[self.style][self.noise_level]:
            # do nothing
            return x, y

        if random.uniform(0, 1) < 0.75:
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
        for quality in qualities:
            x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)

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
