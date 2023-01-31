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


if __name__ == "__main__":
    MODEL_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_pretrained_models_20230131.zip"
    downloder = ModelDownloader(MODEL_URL, name="Waifu2x Models", format="zip")
    downloder.run()
