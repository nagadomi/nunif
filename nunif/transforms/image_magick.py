import numpy as np
from .. utils import wand_io


def to_wand_image(float_tensor):
    return wand_io.to_image(float_tensor)


def to_tensor(im):
    return wand_io.to_tensor(im)


# ref: https://github.com/tsurumeso/waifu2x-chainer/blob/master/lib/iproc.py


YUV420 = "2x2,1x1,1x1"
YUV444 = "1x1,1x1,1x1"


def jpeg_noise(x, sampling_factor, quality):
    color = "rgb" if x.shape[0] == 3 else "gray"
    im = to_wand_image(x)
    im.depth = 8
    im.options["jpeg:sampling-factor"] = sampling_factor
    im.compression_quality = quality
    im, _ = wand_io.decode_image(im.make_blob("jpeg"), color=color)
    with im:
        return to_tensor(im)


def resize(x, size, filter_type, blur=1):
    if isinstance(size, (list, tuple)):
        h, w = size
    else:
        h = w = size
    with to_wand_image(x) as im:
        im.resize(w, h, filter_type, blur)
        return to_tensor(im)


def scale(x, scale_factor, filter_type, blur=1):
    h, w = int(x.shape[1] * scale_factor), int(x.shape[2] * scale_factor)
    with to_wand_image(x) as im:
        im.resize(w, h, filter_type, blur)
        return to_tensor(im)


def random_filter_resize(x, size, filters, blur_min=1, blur_max=1, rng=None):
    if isinstance(size, (list, tuple)):
        h, w = size
    else:
        h = w = size
    rng = rng or np.random
    filter_type = filters[rng.randint(0, len(filters))]
    if blur_min == blur_max:
        blur = blur_min
    else:
        blur = rng.uniform(blur_min, blur_max)
    with to_wand_image(x) as im:
        im.resize(w, h, filter_type, blur)
        return to_tensor(im)


def random_filter_scale(x, scale_factor, filters, blur_min, blur_max, rng=None):
    h, w = int(x.shape[1] * scale_factor), int(x.shape[2] * scale_factor)
    return random_filter_resize(x, (h, w), filters, blur_min, blur_max, rng)


def random_jpeg_noise(x, sampling_factors, quality_min, quality_max, rng=None):
    rng = rng or np.random
    quality = rng.randint(quality_min, quality_max)
    sampling_factor = sampling_factors[rng.randint(0, len(sampling_factors))]
    return jpeg_noise(x, sampling_factor=sampling_factor, quality=quality)
