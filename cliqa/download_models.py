import shutil
from os import path
from nunif.utils.downloader import ArchiveDownloader
from nunif.logger import logger


class ModelDownloader(ArchiveDownloader):
    def handle(self, src):
        src = path.join(src, "pretrained_models")
        dst = path.join(path.dirname(__file__), "pretrained_models")
        logger.debug(f"Downloder: {self.name}: copytree: {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    MODEL_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/cliqa_pretrained_models_20230419.zip"
    downloder = ModelDownloader(MODEL_URL, name="CLIQA Models", format="zip")
    downloder.run()


if __name__ == "__main__":
    main()
