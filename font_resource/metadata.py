from os import path


# TODO: make this list
# Use only Hiragana, Katakana, Alphabet and Kanji (No symbol characters due to rendering problem)
LV1_FONT_NAMES = ()
# LV1 + `ー`(Katakana-Hiragana Prolonged Sound Mark)
LV2_FONT_NAMES = ()
# No `―`(Horizontal bar, Dash in Japanese)
LV3_FONT_NAMES = ()
# Full support for vertical and horizontal layouts in all glyphs
LVF_FONT_NAMES = ()


DEFAULT_FONT_DIR = path.abspath(path.join(path.dirname(__file__), "fonts"))


"""
Default Font names
TODO: This list is temporary to be updated.

1. Support vertical layout
2. Included in LVF_FONT_NAMES
"""
DEFAULT_FONT_NAMES = (
    'Noto Sans JP',
    'Noto Sans JP Light',
    'Noto Sans JP Bold',
    'Noto Serif JP',
    'Noto Serif JP Light',
    'Noto Serif JP Bold',
    'Shippori Antique B1 Regular',
    'Shippori Mincho Regular',
    'Shippori Mincho B1 Bold',
    'Kiwi Maru Regular',
    'Kiwi Maru Light',
    'Kosugi Maru Regular',
    'Kosugi Regular',
    'M PLUS 1p Regular',
    'M PLUS 1p Bold',
    "Klee One Regular",
    "Klee One SemiBold",
)


def is_bold_font(font_name):
    font_name = font_name.lower()
    return any(postfix in font_name for postfix in ("bold", "heavy", "black"))
