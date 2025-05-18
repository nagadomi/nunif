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
                locale["_LOCALE"] = names
                for name in names:
                    locales[name] = locale
    return locales


def save_language_setting(config_path, lang):
    with open(config_path, mode="w", encoding="utf-8") as f:
        f.write(lang)


def load_language_setting(config_path):
    if path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return f.read().strip()
    return None


LOCALES = load_locales(path.dirname(__file__))
