from nunif.logger import logger
from .metadata import FontInfo, VALIDATE_FONT_SIZE, FONT_MAP
from os import path


FONT_DIR = path.join(path.dirname(__file__), "fonts")


def normalize_font_name(font_name):
    return font_name.replace("_", " ")


def load_fonts(
        font_names, 
        validate_cmap=False, validate_font_size=VALIDATE_FONT_SIZE,
        font_dir=FONT_DIR):
    fonts = []
    for name in font_names:
        name = normalize_font_name(name)
        if name in FONT_MAP:
            font = FontInfo.load(path.join(font_dir, FONT_MAP[name]))
            if validate_cmap:
                font.validate_cmap(font_size=validate_font_size)
            fonts.append(font)
        else:
            logger.error(f"load_fonts: unable to load `{name}`")
    return fonts
