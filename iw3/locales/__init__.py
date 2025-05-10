import os
from os import path
import yaml


def load_locales(locale_dir):
    files = [path.join(locale_dir, f) for f in os.listdir(locale_dir) if f.endswith(".yml")]
    locales = {}
    for locale_file in files:
        with open(locale_file, mode="r", encoding="utf-8") as f:
            locale = yaml.load(f.read(), Loader=yaml.SafeLoader)
            if "_LOCALE" in locale:
                name = locale["_LOCALE"]
                names = name if isinstance(name, (list, tuple)) else [name]
                for name in names:
                    locales[name] = locale
    return locales


LOCALES = load_locales(path.dirname(__file__))
