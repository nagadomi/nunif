import os
from os import path
import shutil
from nunif.utils.downloader import ArchiveDownloader
from nunif.logger import logger
from .dir_config import VERSION_DIR


IW3_PLAYER_ROOT = path.dirname(__file__)  # TODO: package
URL_TEMPALTE = "https://github.com/nagadomi/nunif/releases/download/iw3_player_assets/iw3_player_{name}_{version}.zip"
ASSET_VERSIONS = {
    "lib": "20260212",
    "fonts": "20260215",
    "lut": "20260217",
    "environments": "20260213",
    "example": "20260217",
}
ASSERT_CLEAN_DIR = {
    "lib": [path.join(IW3_PLAYER_ROOT, "public", "lib")],
}


class AssetDownloader(ArchiveDownloader):
    def handle(self, src):
        dst = IW3_PLAYER_ROOT
        logger.debug(f"Downloder: {self.name}: copytree: {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    os.makedirs(VERSION_DIR, exist_ok=True)
    ignore_file = path.join(VERSION_DIR, "ignore")
    if path.exists(ignore_file):
        return

    for name, version in ASSET_VERSIONS.items():
        version_file = path.join(VERSION_DIR, f"{name}_{version}")
        if not path.exists(version_file):
            clean_dirs = ASSERT_CLEAN_DIR.get(name, [])
            for clean_dir in clean_dirs:
                if path.commonpath([IW3_PLAYER_ROOT, clean_dir]) == IW3_PLAYER_ROOT and path.exists(clean_dir):
                    shutil.rmtree(clean_dir)

            url = URL_TEMPALTE.format(name=name, version=version)
            downloder = AssetDownloader(url, name=f"iw3_player_{name}_{version}", format="zip")
            downloder.run()
            with open(version_file, mode="w") as f:
                f.write(version)


if __name__ == "__main__":
    main()
