# Font downloader
# python3 -m font_resource.download_google_fonts


import requests
import os
from os import path
from urllib.parse import quote_plus as url_encode
import shutil
from nunif.utils.downloader import ArchiveDownloader


# Google fonts
GOOGLE_FONTS = [
    "Noto Sans JP",
    "Noto Serif JP",
    "Zen Old Mincho",
    "M PLUS Rounded 1c",
    "M PLUS 1p",
    "M PLUS 1",
    "M PLUS 2",
    "M PLUS 1 Code",
    "Sawarabi Mincho",
    "Sawarabi Gothic",
    "Kosugi Maru",
    "Kosugi",
    "Zen Maru Gothic",
    "Zen Kaku Gothic New",
    "Klee One",
    "Shippori Mincho",
    "Shippori Mincho B1",
    "Shippori Antique",
    "Shippori Antique B1",
    "BIZ UDPGothic",
    "BIZ UDPMincho",
    "BIZ UDGothic",
    "BIZ UDMincho",
    "IBM Plex Sans JP",
    "Kiwi Maru",
    "Zen Antique",
    "Zen Old Mincho",
    "Zen Kaku Gothic Antique",
    "Zen Kurenaido",
    "Zen Antique Soft",
    "Kaisei Opti",
    "Kaisei Decol",
    "Kaisei Tokumin",
    "Yomogi",
    "Murecho",
    "Hina Mincho",
    "Yuji Syuku",
    "Yuji Boku",
    "Yuji Mai",
    "New Tegomin",
    "Kaisei HarunoUmi",
    "Yusei Magic",
    "DotGothic16",
    "Dela Gothic One",
    "RocknRoll One",
    "Reggae One",
    "Potta One",
    "Train One",
    "Mochiy Pop One",
    "Mochiy Pop P One",
    "Rampart One",
    "Stick",
    "Hachi Maru Pop",
]
GOOGLE_FONT_DOWNLOAD_URL = "https://fonts.google.com/download?family=%s"


def name_to_url(name):
    return GOOGLE_FONT_DOWNLOAD_URL % url_encode(name)


def name_to_filename(name):
    return name.replace(' ', '_')


def check_google_font_urls():
    """ validate font name for develop
    """
    for name in GOOGLE_FONTS:
        url = name_to_url(name)
        res = requests.head(url, allow_redirects=True)
        if res.status_code != 200:
            print("Error", name, res)
    print("check done")


class GoogleFontDownloader(ArchiveDownloader):
    def run_all(self):
        output_dir = self.kwargs.get("output_dir")
        for name in GOOGLE_FONTS:
            dst = path.join(output_dir, name_to_filename(name))
            if path.exists(dst):
                print(f"{name}: skip")
                continue
            url = name_to_url(name)
            self.reset_param(url=url, name=name, format=self.format)
            self.run()

    def handle(self, src):
        output_dir = self.kwargs.get("output_dir")
        dst = path.join(output_dir, name_to_filename(self.name))
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    output_dir = path.join(path.dirname(__file__), "fonts")
    os.makedirs(output_dir, exist_ok=True)
    downloader = GoogleFontDownloader(format="zip", output_dir=output_dir)
    downloader.run_all()


if __name__ == "__main__":
    if True:
        main()
    else:
        check_google_font_urls()
