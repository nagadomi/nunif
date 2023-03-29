import torch
from wand.image import Image, IMAGE_TYPES
from wand.api import library
from wand.color import Color
import io
from PIL import ImageCms
from ..transforms.functional import quantize256


sRGB_profile = ImageCms.core.profile_tobytes(ImageCms.createProfile("sRGB"))


GRAYSCALE_TYPES = {
    "grayscale",
    "grayscalematte",
    "grayscalealpha",
}
GRAYSCALE_ALPHA_TYPE = "grayscalealpha" if "grayscalealpha" in IMAGE_TYPES else "grayscalematte"
GRAYSCALE_TYPE = "grayscale"
RGBA_TYPE = "truecoloralpha" if "truecoloralpha" in IMAGE_TYPES else "truecolormatte"
RGB_TYPE = "truecolor"
GAMMA_LCD = 45454


def decode_image(blob, filename=None, color=None, keep_alpha=False):
    im = Image(blob=blob)

    # normalize, build meta data
    meta = {}
    meta["engine"] = "wand"
    meta["filename"] = filename
    meta["depth"] = im.depth
    meta["grayscale"] = im.colorspace == "gray" or im.type in GRAYSCALE_TYPES
    meta["gamma"] = None
    gamma = int(library.MagickGetImageGamma(im.wand) * 100000)
    if gamma != 0 and gamma != GAMMA_LCD:
        meta["gamma"] = gamma

    meta["icc_profile"] = im.profiles["ICC"]
    if meta["icc_profile"]:
        im.profiles["ICC"] = sRGB_profile
        del im.profiles["ICC"]

    if im.colorspace == "cmyk":
        im.transform_colorspace("srgb")
        im.colorspace = "srgb"

    if im.colorspace == "gray" or im.type in GRAYSCALE_TYPES:
        if im.alpha_channel:
            im.type = GRAYSCALE_ALPHA_TYPE
        else:
            im.type = GRAYSCALE_TYPE
    else:
        if im.alpha_channel:
            im.type = RGBA_TYPE
        else:
            im.type = RGB_TYPE
    if color is not None:
        if color == "gray" and im.colorspace != "gray":
            im.transform_colorspace("gray")
        elif color == "rgb" and im.colorspace not in {"srgb", "rgb"}:
            im.transform_colorspace("srgb")
    if not keep_alpha:
        im.alpha_channel = False

    return im, meta


def load_image(filename, color=None, keep_alpha=False):
    assert (color is None or color in {"rgb", "gray"})
    with open(filename, "rb") as f:
        return decode_image(f.read(), filename, color=color, keep_alpha=keep_alpha)


def predict_storage(dtype, int_type="short"):
    if dtype in {torch.float, torch.float32, torch.float16}:
        storage = "float"
    elif dtype in {torch.double, torch.float64}:
        storage = "double"
    elif dtype == torch.uint8:
        storage = "char"
    else:
        storage = int_type
    return storage


def to_tensor(im, return_alpha=False, dtype=torch.float32):
    if im.type in {RGB_TYPE, RGBA_TYPE}:
        channel_map = "RGB"
    elif im.type in {GRAYSCALE_TYPE, GRAYSCALE_ALPHA_TYPE}:
        channel_map = "R"
    else:
        assert (im.type in {RGB_TYPE, RGBA_TYPE, GRAYSCALE_TYPE, GRAYSCALE_ALPHA_TYPE})

    storage = predict_storage(dtype)
    w, h = im.size
    ch = len(channel_map)
    data = im.export_pixels(0, 0, w, h, channel_map=channel_map, storage=storage)
    x = torch.tensor(data, dtype=dtype).view(h, w, ch).permute(2, 0, 1).contiguous()
    del data
    if return_alpha:
        if im.alpha_channel:
            data = im.export_pixels(0, 0, w, h, channel_map="A", storage=storage)
            alpha = torch.tensor(data, dtype=dtype).view(h, w, 1).permute(2, 0, 1).contiguous()
            del data
        else:
            alpha = None
        return x, alpha
    else:
        return x


def to_image(x, alpha=None, depth=8):
    assert (alpha is None or
            (x.dtype == alpha.dtype and alpha.shape[0] == 1 and
             alpha.shape[1] == x.shape[1] and alpha.shape[2] == x.shape[2]))
    ch, h, w = x.shape
    assert (ch in {1, 3})
    if ch == 1:
        if alpha is not None:
            x = torch.cat((x, alpha), dim=0)
            channel_map = "IA"
        else:
            channel_map = "I"
    else:
        if alpha is not None:
            x = torch.cat((x, alpha), dim=0)
            channel_map = "RGBA"
        else:
            channel_map = "RGB"

    if depth == 8:
        x = quantize256(x)
        storage = "char"
    else:
        storage = predict_storage(x.dtype, int_type="long")

    return Image.from_array(x.permute(1, 2, 0).numpy(), channel_map=channel_map, storage=storage)


def restore(im, meta):
    if meta is None:
        return im

    assert (meta["engine"] == "wand")

    if meta["grayscale"] and im.colorspace != "gray":
        im.transform_colorspace("gray")
    if meta["gamma"] is not None:
        library.MagickSetImageGamma(im.wand, meta["gamma"] / 100000)
    if meta["icc_profile"]:
        im.profiles["ICC"] = sRGB_profile
        im.profiles["ICC"] = meta["icc_profile"]
    if meta["depth"] != im.depth:
        im.depth = meta["depth"]

    return im


def encode_image(im, format="png", meta=None):
    with im.convert(format) as out, io.BytesIO() as fp:
        if meta is not None:
            out = restore(out, meta)
        else:
            out.depth = 8
        if format in {"jpg", "jpeg"}:
            out.options["jpeg:sampling-factor"] = "1x1,1x1,1x1"
            out.compression_quality = 95
            out.background_color = Color('white')
            out.alpha_channel = "remove"

        elif format == "webp":
            library.MagickSetOption(out.wand, b"webp:lossless", b"true")

        out.save(file=fp)
        return fp.getvalue()


def save_image(im, filename, format="png", meta=None):
    with open(filename, "wb") as fp:
        fp.write(encode_image(im, format=format, meta=meta))
