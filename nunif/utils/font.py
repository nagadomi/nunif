from dataclasses import dataclass
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import threading
import random


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


@dataclass
class CharBox():
    label: str
    x: int
    y: int
    width: int
    height: int
    has_letter_spacing: bool


class CharDraw():
    def __init__(self, font_info, font_size, vertical, bold=None, image_fonts=None,
                 lang="ja", test_text="常用漢字"):
        self.font_info = font_info
        self.font_size = font_size
        self.vertical = vertical
        self.lang = lang
        self.bold = bold
        self.font = ImageFont.truetype(self.font_info.file_path, size=font_size,
                                       layout_engine=ImageFont.Layout.RAQM)
        self.image_fonts = image_fonts if image_fonts is not None else ImageFonts()
        if self.vertical:
            self.direction = "ttb"
            # TODO: should we use left top?
            w, h = self.font.getbbox(test_text, direction=self.direction, language=self.lang)[2:]
            self.char_size = max(w, font_size)
        else:
            self.direction = "ltr"
            w, h = self.font.getbbox(test_text, direction=self.direction, language=self.lang)[2:]
            self.char_size = max(h, font_size)

    def can_render(self, code):
        return (code in self.font_info.cmap or self.image_fonts.has_code(code, self.vertical))

    def draw_image(self, gc, x, y, image, stroke_width=0, color="white"):
        # TODO: test
        if stroke_width == 0:
            gc.bitmap((x, y), image, fill=color)
        else:
            bold = image.filter(ImageFilter.MaxFilter(1 + stroke_width * 2))
            gc.bitmap((x, y), bold, fill=color)

    def draw(self, gc, x, y, code, label=None, stroke_width=0, color="white"):
        text = chr(code)
        if label is None:
            label = text
        image_font = code_len = None
        if gc is not None:
            if code not in self.font_info.cmap:
                # use image font
                image_font, code_len = self.image_fonts.get(code, self.font_size, self.vertical, self.bold)
                self.draw_image(gc, x, y, image_font, stroke_width, color)
            else:
                if self.image_fonts.has_code_random(code, self.vertical, self.bold):
                    image_font, code_len = self.image_fonts.get(code, self.font_size, self.vertical, self.bold)
                    self.draw_image(gc, x, y, image_font, stroke_width, color)
                else:
                    gc.text((x, y), text + "　", font=self.font, fill=color, stroke_width=stroke_width,
                            direction=self.direction, anchor=None, language=self.lang)
        boxes = []
        if image_font is not None:
            w, h = image_font.size
            if self.vertical:
                h = h // code_len
            else:
                w = w // code_len
            for i in range(code_len):
                boxes.append(CharBox(label=label,
                                     x=x, y=y, width=w, height=h,
                                     has_letter_spacing=(i == code_len - 1)))
        else:
            if self.vertical:
                w, h = self.font.getbbox(text, direction=self.direction, language=self.lang)[2:]
                w = max(w, self.char_size)
            else:
                w, h = self.font.getbbox(text, direction=self.direction, language=self.lang)[2:]
                h = max(h, self.char_size)
            # TODO: allow randomize white space size
            boxes.append(CharBox(label=label,
                                 x=x, y=y, width=int(w), height=int(h),
                                 has_letter_spacing=True))

        return boxes


@dataclass
class LineBox():
    label: str
    x: int
    y: int
    width: int
    height: int


class SimpleLineDraw(CharDraw):
    def __init__(self, font_info, font_size, vertical, lang="ja"):
        self.font_info = font_info
        self.font_size = font_size
        self.vertical = vertical
        self.lang = lang
        self.font = ImageFont.truetype(self.font_info.file_path, size=font_size,
                                       layout_engine=ImageFont.Layout.RAQM)
        if self.vertical:
            self.direction = "ttb"
        else:
            self.direction = "ltr"

    def can_render(self, text):
        return all([ord(c) in self.font_info.cmap for c in text])

    def draw(self, gc, x, y, text, label=None, stroke_width=0, color="white"):
        if label is None:
            label = text
        if gc is not None:
            gc.text((x, y), text + "　", font=self.font, fill=color, stroke_width=stroke_width,
                    direction=self.direction, anchor=None, language=self.lang)

        w, h = self.font.getbbox(text, stroke_width=stroke_width,
                                 direction=self.direction, language=self.lang)[2:]
        return LineBox(label=label, x=x, y=y, width=w, height=h)


def _test_font():
    info = FontInfo.load("font_resource/fonts/Kosugi_Maru/KosugiMaru-Regular.ttf")
    print(info)
    print([chr(c) for c in info.validate_cmap(font_size=32)])


def _test_draw():
    text = """
吾輩（わがはい）は猫である。
名前はまだ無い。
どこで生れたかとんと見当（けんとう）がつかぬ。
I'm a cat!!
"""
    font_info = FontInfo.load("./font_resource/fonts/Noto_Sans_JP/NotoSansJP-Regular.otf")
    font_size = 20

    # CharDraw

    # horizontal
    im = Image.new("L", (512, 128), (0,))
    gc = ImageDraw.Draw(im)
    draw = CharDraw(font_info, font_size, vertical=False)
    sx = sy = x = y = 8
    letter_spacing = int(font_size * 0.1)
    line_spacing = int(font_size * 0.1)
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        stroke_width = 0  # if i % 2 == 0 else 1
        color = (200,) if i % 2 == 0 else "white"
        x = sx
        h = 0
        for c in line:
            boxes = draw.draw(gc, x, y, ord(c), label=c, stroke_width=stroke_width, color=color)
            for box in boxes:
                gc.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), outline=(128,))
                x += box.width
                if not str.isascii(c) and box.has_letter_spacing:
                    x += letter_spacing
                h = max(h, box.height)
        y += h + line_spacing
    im.show()

    # vertical
    im = Image.new("L", (128, 512), (0,))
    gc = ImageDraw.Draw(im)
    draw = CharDraw(font_info, font_size, vertical=True)
    sx = sy = x = y = 8
    letter_spacing = int(font_size * 0.1)
    line_spacing = int(font_size * 0.1)

    for i, line in enumerate(reversed(text.splitlines())):
        line = line.strip()
        stroke_width = 0  # if i % 2 == 0 else 1
        color = (200,) if i % 2 == 0 else "white"
        y = sy
        w = 0
        for c in line:
            boxes = draw.draw(gc, x, y, ord(c), label=c, stroke_width=stroke_width, color=color)
            for box in boxes:
                gc.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), outline=(128,))
                y += box.height
                if not str.isascii(c) and box.has_letter_spacing:
                    y += letter_spacing
                w = max(w, box.width)
        x += w + line_spacing
    im.show()

    # SimpleLineDraw

    # horizontal
    im = Image.new("L", (512, 128), (0,))
    gc = ImageDraw.Draw(im)
    draw = SimpleLineDraw(font_info, font_size, vertical=False)
    x = y = 8
    line_spacing = int(font_size * 0.1)
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        stroke_width = 0  # if i % 2 == 0 else 1
        color = (200,) if i % 2 == 0 else "white"
        h = 0
        box = draw.draw(gc, x, y, line, label=c, stroke_width=stroke_width, color=color)
        gc.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), outline=(128,))
        h = max(h, box.height)
        y += h + line_spacing
    im.show()

    # vertical
    im = Image.new("L", (128, 512), (0,))
    gc = ImageDraw.Draw(im)
    draw = SimpleLineDraw(font_info, font_size, vertical=True)
    x = y = 8
    line_spacing = int(font_size * 0.1)
    for i, line in enumerate(reversed(text.splitlines())):
        line = line.strip()
        stroke_width = 0  # if i % 2 == 0 else 1
        color = (200,) if i % 2 == 0 else "white"
        h = 0
        box = draw.draw(gc, x, y, line, label=c, stroke_width=stroke_width, color=color)
        gc.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), outline=(128,))
        w = max(w, box.width)
        x += w + line_spacing
    im.show()


if __name__ == "__main__":
    # _test_font()
    _test_draw()
