from os import path
import threading
import random
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont
from .font_map import FONT_MAP

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


INVISIBLE_CODES = {
    # ASCII Control character
    *[c for c in range(0, 0x20 + 1)],
    0x7F,  # DEL

    # Unicode
    0x0009,  # CHARACTER TABULATION
    0x0020,  # SPACE
    0x00A0,  # NO-BREAK SPACE
    0x00AD,  # SOFT HYPHEN
    0x034F,  # COMBINING GRAPHEME JOINER
    0x061C,  # ARABIC LETTER MARK
    0x115F,  # HANGUL CHOSEONG FILLER
    0x1160,  # HANGUL JUNGSEONG FILLER
    0x17B4,  # KHMER VOWEL INHERENT AQ
    0x17B5,  # KHMER VOWEL INHERENT AA
    0x180E,  # MONGOLIAN VOWEL SEPARATOR
    0x2000,  # EN QUAD
    0x2001,  # EM QUAD
    0x2002,  # EN SPACE
    0x2003,  # EM SPACE
    0x2004,  # THREE-PER-EM SPACE
    0x2005,  # FOUR-PER-EM SPACE
    0x2006,  # SIX-PER-EM SPACE
    0x2007,  # FIGURE SPACE
    0x2008,  # PUNCTUATION SPACE
    0x2009,  # THIN SPACE
    0x200A,  # HAIR SPACE
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x202F,  # NARROW NO-BREAK SPACE
    0x205F,  # MEDIUM MATHEMATICAL SPACE
    0x2060,  # WORD JOINER
    0x2061,  # FUNCTION APPLICATION
    0x2062,  # INVISIBLE TIMES
    0x2063,  # INVISIBLE SEPARATOR
    0x2064,  # INVISIBLE PLUS
    0x206A,  # INHIBIT SYMMETRIC SWAPPING
    0x206B,  # ACTIVATE SYMMETRIC SWAPPING
    0x206C,  # INHIBIT ARABIC FORM SHAPING
    0x206D,  # ACTIVATE ARABIC FORM SHAPING
    0x206E,  # NATIONAL DIGIT SHAPES
    0x206F,  # NOMINAL DIGIT SHAPES
    0x3000,  # IDEOGRAPHIC SPACE
    0x2800,  # BRAILLE PATTERN BLANK
    0x3164,  # HANGUL FILLER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE
    0xFFA0,  # HALFWIDTH HANGUL FILLER
    0x1D159,  # MUSICAL SYMBOL NULL NOTEHEAD
    0x1D173,  # MUSICAL SYMBOL BEGIN BEAM
    0x1D174,  # MUSICAL SYMBOL END BEAM
    0x1D175,  # MUSICAL SYMBOL BEGIN TIE
    0x1D176,  # MUSICAL SYMBOL END TIE
    0x1D177,  # MUSICAL SYMBOL BEGIN SLUR
    0x1D178,  # MUSICAL SYMBOL END SLUR
    0x1D179,  # MUSICAL SYMBOL BEGIN PHRASE
    0x1D17A,  # MUSICAL SYMBOL END PHRASE

    # Bidirectional Class
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,

    0x2028,  # LINE SEPARATOR
    0x2029,  # PARAGRAPH SEPARATOR
    # BOM
    0xfff9, 0xfffa, 0xfffb,
}

# NOTE: https://learn.microsoft.com/en-us/typography/opentype/spec/name#name-ids
FONT_NAME_ID = {
    "Name": 4,
    "Family Name": 1,
    "Version": 5,
    "Copyright Notice": 0,
    "Trademark": 7,
    "Manufacturer Name": 8,
    "Designer": 9,
    "Description": 10,
    "URL Vendor": 11,
    "URL Designer": 12,
    "License Description": 13,
    "License URL": 14,
}
VALIDATE_FONT_SIZE = 16


class FontInfo():
    def __init__(self, ttfont, file_path, name, cmap):
        self.file_path = file_path
        self.ttfont = ttfont
        self.cmap = cmap
        self.name = name
        self.lock = threading.RLock()

    @classmethod
    def load(cls, file_path, validate_cmap=False, validate_font_size=VALIDATE_FONT_SIZE):
        ttfont = TTFont(file_path)
        name = ttfont["name"].getDebugName(FONT_NAME_ID["Name"])
        cmap = set()
        for code in ttfont.getBestCmap():
            cmap.add(code)
        font = FontInfo(ttfont=ttfont, name=name,
                        cmap=cmap, file_path=file_path)
        if validate_cmap:
            font.validate_cmap(font_size=validate_font_size)
        return font

    def get_metadata(self, name_id=None, name=None):
        if name_id is not None:
            return self.ttfont["name"].getDebugName(name_id)
        elif name is not None:
            return self.ttfont["name"].getDebugName(FONT_NAME_ID[name])

    def validate_cmap(self, font_size=VALIDATE_FONT_SIZE):
        """
        Most Free Fonts contain many character codes that cannot be rendered.
        It causes mislabeling the generated training data.
        This method tests each code for rendering and removes the character codes that cannot be rendered.
        """
        font = ImageFont.truetype(self.file_path, size=font_size, index=0,
                                  layout_engine=ImageFont.Layout.RAQM)

        def render_test(code):
            font_image = Image.Image()._new(font.getmask(chr(code), mode="L"))
            if font_image.size[0] == 0 or font_image.size[1] == 0:
                return False
            min_value, max_value = font_image.getextrema()
            if min_value == 0 and max_value == 0:
                return False
            font_image.close()
            return True
        invalid_codes = set(code for code in self.cmap
                            if code not in INVISIBLE_CODES and not render_test(code))
        # print([(hex(c), chr(c)) for c in invalid_codes])
        self.cmap -= invalid_codes
        return invalid_codes

    def drawable(self, text):
        return all([ord(c) in self.cmap for c in text])

    def __repr__(self):
        return f"FontInfo(name={self.name}, file_path={self.file_path})"


class ImageFonts():
    def __init__(self):
        self.font_map = {True: {True: {}, False: {}}, False: {True: {}, False: {}}}

    def add(self, code, image_path, vertical, length=1, bold=False, prob=1.0):
        image = Image.open(image_path, mode="L")
        image.load()
        self.font_map[vertical][bold][code] = {
            "image": image,
            "vertical": vertical,
            "length": length,
            "bold": bold,
            "prob": prob
        }

    def has_code(self, code, vertical, bold=None):
        if bold is None:
            return (code in self.font_map[vertical][False] or
                    code in self.font_map[vertical][True])
        else:
            return code in self.font_map[vertical][bold]

    def has_code_random(self, code, vertical, bold=None):
        rec = self.get_record(code, vertical, bold)
        if rec is None:
            return False
        return random.uniform(0, 1) < rec["prob"]

    def get_record(self, code, vertical, bold=False):
        if not bold:
            if self.has_code(code, vertical, False):
                return self.font_map[vertical][False]
            elif self.has_code(code, vertical, True):
                return self.font_map[vertical][True]
            else:
                return None
        else:
            if self.has_code(code, vertical, True):
                return self.font_map[vertical][True]
            elif self.has_code(code, vertical, False):
                return self.font_map[vertical][False]
            else:
                return None

    def get(self, code, font_size, vertical, bold=False):
        rec = self.get_record(code, vertical, bold)
        if rec is None:
            return None
        return rec["image"].resize((font_size, font_size), resample=Image.BILINEAR), rec["length"]
