import os


def set_image_ext(filename, format):
    format = format.lower()
    filename = os.path.splitext(filename)[0]
    if format == "png":
        return filename + ".png"
    elif format == "webp":
        return filename + ".webp"
    elif format in {"jpg", "jpeg"}:
        return filename + ".jpg"
    else:
        raise NotImplementedError(f"{format}")
