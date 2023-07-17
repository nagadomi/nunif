import os
from os import path
import yaml
from .. web.webgen.gen import load_locales as load_webgen_locales


WEBGEN_TERMS = [
    "artwork", "scan", "photo",
    "noise_reduction", "nr_none", "nr_low", "nr_medium", "nr_high", "nr_highest",
    "upscaling", "up_none",
]


def merge_en(webgen_locales, lang):
    t = webgen_locales["en"].copy()
    t.update(webgen_locales[lang])
    return t


def merge_locales(webgen_locales, locale, lang):
    webgen_locale = merge_en(webgen_locales, lang)
    assert len([term for term in WEBGEN_TERMS if term not in webgen_locale]) == 0
    webgen_locale = {term: webgen_locale[term] for term in WEBGEN_TERMS}
    locale.update(webgen_locale)


def load_locales(locale_dir, webgen_locale_dir):
    webgen_locales = load_webgen_locales(webgen_locale_dir)
    files = [path.join(locale_dir, f) for f in os.listdir(locale_dir) if f.endswith(".yml")]
    locales = {}
    for locale_file in files:
        with open(locale_file, mode="r", encoding="utf-8") as f:
            locale = yaml.load(f.read(), Loader=yaml.SafeLoader)
            lang = path.splitext(path.basename(locale_file))[0]
            merge_locales(webgen_locales, locale, lang)
            if "_LOCALE_LINUX" in locale:
                name = locale["_LOCALE_LINUX"]
                names = name if isinstance(name, (list, tuple)) else [name]
                for name in names:
                    locales[name] = locale
            if "_LOCALE_WINDOWS" in locale:
                name = locale["_LOCALE_WINDOWS"]
                names = name if isinstance(name, (list, tuple)) else [name]
                for name in names:
                    locales[name] = locale
    return locales


LOCALES = load_locales(path.dirname(__file__),
                       path.join(path.dirname(__file__), "..", "web", "webgen", "locales"))
