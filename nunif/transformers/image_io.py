from PIL import Image


def load_image_rgb(filename):
    with open(filename, "rb") as f:
        im = Image.open(f)
        if im.mode != "RGB":
            im = im.convert("RGB")
    return Image.open(f).convert("RGB")


def _load_image(im, filename):
    meta_data = {}
    im.load()
    if im.mode in ("RGBA", "LA"):
        meta_data["alpha"] = im.getchannel("A")
    meta_data["filename"] = filename
    meta_data["mode"] = im.mode
    meta_data["gamma"] = im.info.get("gamma")
    meta_data["icc_profile"] = im.info.get("icc_profile")
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im, meta_data


def load_image(filename):
    with open(filename, "rb") as f:
        im = Image.open(f)
        return _load_image(im, filename)


def decode_image(buff):
    pass
