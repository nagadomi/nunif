import requests
import shutil
from tqdm import tqdm
from tempfile import NamedTemporaryFile, mkdtemp
import posixpath
from abc import ABC, abstractmethod
from ..logger import logger


class Downloader(ABC):
    def __init__(self, url, name=None, format=None, archive=False):
        self.url = url
        self.name = name or posixpath.basename(url)
        self.archive = archive
        self.format = format
        self.delete_tmp = True

    def name(self):
        return self.name

    def run(self, block_size=2 * 1024 * 1024, show_progress=True):
        response = requests.get(self.url, allow_redirects=True, stream=True)
        # TODO: total_size == 0
        total_size = int(response.headers.get("content-length", 0))
        logger.debug(f"Downloader: {self.name}: url={self.url}, size={total_size}")
        with NamedTemporaryFile(prefix="nunif-", delete=self.delete_tmp) as tmp:
            progress_bar = tqdm(desc=self.name, total=total_size, unit='iB', unit_scale=True,
                                ncols=80, disable=not show_progress)
            for data in response.iter_content(block_size):
                tmp.write(data)
                progress_bar.update(len(data))
            progress_bar.close()
            tmp.seek(0)
            if self.archive:
                tmp_dir = mkdtemp(prefix="nunif-")
                try:
                    options = {} if self.format is None else {"format": self.format}
                    shutil.unpack_archive(filename=tmp.name, extract_dir=tmp_dir, **options)
                    self.handle_directory(tmp_dir)
                finally:
                    if self.delete_tmp:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                self.handle_file(src=tmp.name, file=tmp)

    @abstractmethod
    def handle_file(self, src, file=None):
        pass

    @abstractmethod
    def handle_directory(self, src):
        pass


class FileDownloader(Downloader):
    def __init__(self, url, name=None):
        super().__init__(url, name=name, archive=False)

    def handle_directory(self, src):
        raise NotImplementedError()


class ArchiveDownloader(Downloader):
    def __init__(self, url, name=None, format=None):
        super().__init__(url, name=name, format=format, archive=True)

    def handle_file(self, src, file=None):
        raise NotImplementedError()
