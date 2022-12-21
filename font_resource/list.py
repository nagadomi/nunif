from nunif.utils.font import FontInfo
from glob import glob
from os import path
from pprint import pprint


def main():
    font_dir = path.join(path.dirname(__file__), "fonts")
    fonts = []
    for file_path in glob(path.join(font_dir, "**", "*.*")):
        ext = path.splitext(file_path)[-1].lower()
        if ext in {".ttf", ".otf"}:
            font = FontInfo.load(file_path)
            fonts.append(font)

    pprint({font.name: path.relpath(font.file_path, start=font_dir)
            for font in sorted(fonts, key=lambda font: font.name)},
           sort_dicts=False)


if __name__ == "__main__":
    main()
