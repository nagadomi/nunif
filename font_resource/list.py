from nunif.utils.font import FontInfo, FONT_NAME_ID
from glob import glob
from os import path
from pprint import pprint
import argparse
import html
import re


def escape(s):
    if not isinstance(s, str):
        return s
    s = html.escape(s)
    s = re.sub(r"([|`_*\[\]])", r"\\\1", s)
    return s


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata", action="store_true", help="show metadata")
    parser.add_argument("--markdown", action="store_true", help="markdown format")
    args = parser.parse_args()

    font_dir = path.join(path.dirname(__file__), "fonts")
    fonts = []
    for file_path in glob(path.join(font_dir, "**", "*.*")):
        ext = path.splitext(file_path)[-1].lower()
        if ext in {".ttf", ".otf"}:
            font = FontInfo.load(file_path)
            fonts.append(font)
    if args.metadata:
        data = {}
        for font in sorted(fonts, key=lambda font: font.name):
            data[font.name] = {}
            for meta_name in FONT_NAME_ID:
                value = font.get_metadata(name=meta_name)
                data[font.name][meta_name] = value
            data[font.name]["File"] = path.basename(font.file_path)
            # data[font.name]["Number of Glyphs"] = len(font.cmap)
        if args.markdown:
            print("# Font List\n")
            for name, meta in data.items():
                print(f"## {escape(name)}\n")
                print("| Name | Description |")
                print("| -----| ----------- |")
                for key, value in meta.items():
                    print(f"| {escape(key)} | {escape(value)} |")
                print("")
        else:
            pprint(data, sort_dicts=False)
    else:
        data = {font.name: path.relpath(font.file_path, start=font_dir)
                for font in sorted(fonts, key=lambda font: font.name)}
        if args.markdown:
            print("# Font List")
            print("| Font Name | File |")
            print("| --------- | ---- |")
            for name, file_path in data.items():
                print(f"| {escape(name)} | {escape(file_path)} |")
        else:
            pprint(data, sort_dicts=False)


if __name__ == "__main__":
    main()
