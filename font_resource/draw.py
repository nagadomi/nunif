from dataclasses import dataclass
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from .metadata import ImageFonts, FontInfo


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

    def drawable(self, code):
        return (code in self.font_info.cmap or self.image_fonts.has_code(code, self.vertical))

    def draw_image(self, gc, x, y, image, stroke_width=0, color="white"):
        # TODO: test
        if stroke_width == 0:
            gc.bitmap((x, y), image, fill=color)
        else:
            bold = image.filter(ImageFilter.MaxFilter(1 + stroke_width * 2))
            gc.bitmap((x, y), bold, fill=color)

    def draw_text(self, gc, x, y, text, stroke_width, color):
        gc.text((x, y), text + "　", font=self.font, fill=color, stroke_width=stroke_width,
                direction=self.direction, anchor=None, language=self.lang)

    def draw(self, gc, x, y, code, label=None, stroke_width=0, color="white",
             shadow_color=None, shadow_width=None):
        text = chr(code)
        if label is None:
            label = text
        image_font = code_len = None
        if gc is not None:
            # TODO: draw_image with shadow
            if code not in self.font_info.cmap:
                # use image font
                image_font, code_len = self.image_fonts.get(code, self.font_size, self.vertical, self.bold)
                self.draw_image(gc, x, y, image_font, stroke_width, color)
            else:
                if self.image_fonts.has_code_random(code, self.vertical, self.bold):
                    image_font, code_len = self.image_fonts.get(code, self.font_size, self.vertical, self.bold)
                    self.draw_image(gc, x, y, image_font, stroke_width, color)
                else:
                    if shadow_color is not None:
                        if shadow_width is None:
                            shadow_stroke_width = stroke_width + (2 + self.font_size // 8)
                        else:
                            shadow_stroke_width = stroke_width + shadow_width
                        self.draw_text(gc, x, y, text, shadow_stroke_width, shadow_color)
                        if shadow_stroke_width > 4:
                            self.draw_text(gc, x, y, text, shadow_stroke_width // 2, shadow_color)
                    self.draw_text(gc, x, y, text, stroke_width, color)
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

    def drawable(self, text):
        return all([ord(c) in self.font_info.cmap for c in text])

    def draw_text(self, gc, x, y, text, stroke_width, color):
        gc.text((x, y), text + "　", font=self.font, fill=color, stroke_width=stroke_width,
                direction=self.direction, anchor=None, language=self.lang)

    def draw(self, gc, x, y, text, label=None, stroke_width=0, color="white",
             shadow_color=None, shadow_width=None):
        if label is None:
            label = text
        if gc is not None:
            if shadow_color is not None:
                if shadow_width is None:
                    shadow_stroke_width = stroke_width + (2 + self.font_size // 8)
                else:
                    shadow_stroke_width = stroke_width + shadow_width
                self.draw_text(gc, x, y, text, stroke_width=shadow_stroke_width, color=shadow_color)
                if shadow_stroke_width > 4:
                    self.draw_text(gc, x, y, text, stroke_width=shadow_stroke_width // 2, color=shadow_color)
            self.draw_text(gc, x, y, text, stroke_width=stroke_width, color=color)
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
    im = Image.new("L", (512, 128), (128,))
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
            boxes = draw.draw(gc, x, y, ord(c), label=c, stroke_width=stroke_width, color=color, shadow_color=0)
            for box in boxes:
                gc.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), outline=(64,))
                x += box.width
                if box.has_letter_spacing:
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
    im = Image.new("L", (128, 512), (128,))
    gc = ImageDraw.Draw(im)
    draw = SimpleLineDraw(font_info, font_size, vertical=True)
    x = y = 8
    line_spacing = int(font_size * 0.1)
    for i, line in enumerate(reversed(text.splitlines())):
        line = line.strip()
        stroke_width = 0  # if i % 2 == 0 else 1
        color = (200,) if i % 2 == 0 else "white"
        h = 0
        box = draw.draw(gc, x, y, line, label=c, stroke_width=stroke_width, color=color, shadow_color=64)
        gc.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), outline=(0,))
        w = max(w, box.width)
        x += w + line_spacing
    im.show()


if __name__ == "__main__":
    # _test_font()
    _test_draw()
