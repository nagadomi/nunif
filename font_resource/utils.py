from nunif.logger import logger
from .metadata import FontInfo, VALIDATE_FONT_SIZE, FONT_MAP
from os import path


FONT_DIR = path.join(path.dirname(__file__), "fonts")


def normalize_font_name(font_name):
    return font_name.replace("_", " ")


def native_path(posix_path):
    if path.sep == "/":
        return posix_path
    else:
        return posix_path.replace("/", path.sep)


def load_font(
        font_name,
        validate_cmap=False, validate_font_size=VALIDATE_FONT_SIZE,
        font_dir=FONT_DIR):
    font_name = normalize_font_name(font_name)
    if font_name not in FONT_MAP:
        logger.error(f"load_fonts: unable to load `{font_name}`")
        return None

    font = FontInfo.load(path.join(font_dir, native_path(FONT_MAP[font_name])))
    if validate_cmap:
        font.validate_cmap(font_size=validate_font_size)
    return font


def load_fonts(
        font_names, 
        validate_cmap=False, validate_font_size=VALIDATE_FONT_SIZE,
        font_dir=FONT_DIR):
    fonts = []
    for font_name in font_names:
        font = load_font(
            font_name,
            validate_cmap=validate_cmap, validate_font_size=validate_font_size,
            font_dir=font_dir)
        if font is not None:
            fonts.append(font)
    return fonts
