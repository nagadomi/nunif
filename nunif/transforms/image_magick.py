from wand.image import Image as WandImage
from torchvision.transforms import functional as TF
from PIL import Image as PILImage
import io
import numpy as np


# ref: https://github.com/tsurumeso/waifu2x-chainer/blob/master/lib/iproc.py


def to_wand_image(float_tensor):
    with io.BytesIO() as buf:
        TF.to_pil_image(float_tensor).save(buf, "BMP")
        return WandImage(blob=buf.getvalue())


def to_tensor(im):
    with io.BytesIO(im.make_blob("BMP")) as buf:
        return TF.to_tensor(PILImage.open(buf))


YUV420 = "2x2,1x1,1x1"
YUV444 = "1x1,1x1,1x1"


def jpeg_noise(x, sampling_factor, quality):
    im = to_wand_image(x)
    im.depth = 8
    im.options["jpeg:sampling-factor"] = sampling_factor
    im.compression_quality = quality
    with io.BytesIO(im.make_blob("jpeg")) as buf:
        return TF.to_tensor(PILImage.open(buf))


def resize(x, size, filter_type, blur=1):
    if isinstance(size, (list, tuple)):
        h, w = size
    else:
        h = w = size
    im = to_wand_image(x)
    im.resize(w, h, filter_type, blur)
    return to_tensor(im)


def scale(x, scale_factor, filter_type, blur=1):
    h, w = int(x.shape[1] * scale_factor), int(x.shape[2] * scale_factor)
    im = to_wand_image(x)
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
    im = to_wand_image(x)
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
