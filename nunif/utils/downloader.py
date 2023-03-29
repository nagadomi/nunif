import requests
import shutil
import os
from tqdm import tqdm
from tempfile import NamedTemporaryFile, mkdtemp
from abc import ABC, abstractmethod
from ..logger import logger


class Downloader(ABC):
    def __init__(self, url=None, name=None, format=None, archive=False, **kwargs):
        self.reset_param(url=url, name=name, format=format, archive=archive)
        self.delete_tmp = True
        self.kwargs = kwargs

    def reset_param(self, url, name=None, format=None, archive=False):
        self.url = url
        self.name = name
        self.archive = archive
        self.format = format

    def run(self, block_size=2 * 1024 * 1024, show_progress=True):
        response = requests.get(self.url, allow_redirects=True, stream=True)
        # TODO: total_size == 0
        total_size = int(response.headers.get("Content-Length", 0))
        # TOTO: rfc6266
        if response.url != self.url:
            self.url = response.url

        logger.debug(f"Downloader: {self.name}: url={self.url}, size={total_size}")
        tmp = None
        try:
            with NamedTemporaryFile(prefix="nunif-", delete=False) as tmp:
                progress_bar = tqdm(desc=self.name, total=total_size, unit='iB', unit_scale=True,
                                    ncols=80, disable=not show_progress)
                for data in response.iter_content(block_size):
                    tmp.write(data)
                    progress_bar.update(len(data))
                progress_bar.close()
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
        finally:
            if self.delete_tmp:
                if tmp is not None:
                    os.unlink(tmp.name)

    @abstractmethod
    def handle_file(self, src, file):
        pass

    @abstractmethod
    def handle_directory(self, src):
        pass

    @abstractmethod
    def handle(self, src):
        pass


class FileDownloader(Downloader):
    def __init__(self, url=None, name=None, **kwargs):
        super().__init__(url=url, name=name, archive=False, **kwargs)

    def reset_param(self, url, name=None, archive=False):
        super().reset_param(url=url, name=name, archive=archive)

    def handle_file(self, src, file):
        self.handle(src)

    def handle_directory(self, src):
        raise NotImplementedError()


class ArchiveDownloader(Downloader):
    def __init__(self, url=None, name=None, format=None, **kwargs):
        super().__init__(url=url, name=name, format=format, archive=True, **kwargs)

    def reset_param(self, url, name=None, format=None, archive=True):
        super().reset_param(url=url, name=name, format=format, archive=archive)

    def handle_file(self, src, file):
        raise NotImplementedError()

    def handle_directory(self, src):
        self.handle(src)
