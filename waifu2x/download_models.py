import shutil
from os import path
from nunif.utils.downloader import ArchiveDownloader
from nunif.logger import logger
from .model_dir import MODEL_DIR

# Base
VERSION = "20250502"
VERSION_FILE = path.join(MODEL_DIR, VERSION)
MODEL_URL = f"https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_pretrained_models_{VERSION}.zip"

# Diff
PATCH1_VERSION = "20260816"
PATCH1_VERSION_FILE = path.join(MODEL_DIR, PATCH1_VERSION)
PATCH1_MODEL_URL = f"https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_pretrained_models_swin_unet_v3_art_{PATCH1_VERSION}.zip"


class ModelDownloader(ArchiveDownloader):
    def handle(self, src):
        src = path.join(src, "pretrained_models")
        dst = MODEL_DIR
        logger.debug(f"Downloder: {self.name}: copytree: {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    if not path.exists(VERSION_FILE):
        downloder = ModelDownloader(MODEL_URL, name="Waifu2x Models", format="zip")
        downloder.run()
        with open(VERSION_FILE, mode="w") as f:
            f.write(VERSION)

    if not path.exists(PATCH1_VERSION_FILE):
        downloder = ModelDownloader(PATCH1_MODEL_URL, name="Waifu2x Models Patch 1", format="zip")
        downloder.run()
        with open(PATCH1_VERSION_FILE, mode="w") as f:
            f.write(VERSION)

if __name__ == "__main__":
    main()
