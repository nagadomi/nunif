# Aozora bunko downloader
# python3 -m text_resource.aozora.download


import requests
import os
from os import path
from urllib.parse import quote_plus as url_encode
import shutil
from nunif.utils.downloader import ArchiveDownloader


AOZORA_TEXT_URL = "https://github.com/aozorahack/aozorabunko_text/archive/master.zip"
AOZORA_CSV_URL = "http://www.aozora.gr.jp/index_pages/list_person_all.zip"


class AozoraTextDownloader(ArchiveDownloader):
    def handle(self, src):
        dst = path.join(self.kwargs.get("output_dir"), "cards")
        shutil.copytree(path.join(src, "aozorabunko_text-master", "cards"), dst, dirs_exist_ok=True)


class AozoraCSVDownloader(ArchiveDownloader):
    def handle(self, src):
        dst = self.kwargs.get("output_dir")
        shutil.copyfile(path.join(src, self.name), path.join(dst, self.name))


def main():
    output_dir = path.join(path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    downloader = AozoraCSVDownloader(AOZORA_CSV_URL, name="list_person_all.csv",
                                     format="zip", output_dir=output_dir)
    downloader.run()

    downloader = AozoraTextDownloader(AOZORA_TEXT_URL, name="aozorabunko_text", 
                                      format="zip", output_dir=output_dir)
    downloader.run()


if __name__ == "__main__":
    main()
