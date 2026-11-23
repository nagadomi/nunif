import os
import shutil
import threading
import urllib.request
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any

from tqdm import tqdm

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

    def run(
        self,
        block_size: int = 512 * 1024,
        show_progress: bool = True,
        stop_event: threading.Event | None = None,
        tqdm_fn: Any = None,
        tmp_dir: str | None = None,
    ) -> bool:
        tmp = None
        tqdm_cls = tqdm if tqdm_fn is None else tqdm_fn
        try:
            with urllib.request.urlopen(self.url) as response:
                if response.geturl() != self.url:
                    self.url = response.geturl()
                total_size = int(response.getheader("Content-Length", 0))
                logger.debug(f"Downloader: {self.name}: url={self.url}, size={total_size}")

                with NamedTemporaryFile(prefix="nunif-", dir=tmp_dir, delete=False) as tmp:
                    progress_bar = tqdm_cls(
                        desc=self.name,
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        ncols=80,
                        disable=not show_progress,
                    )
                    downloaded_size = 0
                    while True:
                        if stop_event is not None and stop_event.is_set():
                            progress_bar.close()
                            return False
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        tmp.write(chunk)
                        downloaded_size += len(chunk)
                        progress_bar.update(len(chunk))
                    progress_bar.close()
                    if total_size > 0 and downloaded_size != total_size:
                        raise RuntimeError(
                            f"Incomplete download for {self.name}: "
                            f"expected {total_size} bytes, got {downloaded_size} bytes"
                        )

                if self.archive:
                    extract_dir = mkdtemp(prefix="nunif-")
                    try:
                        options = {} if self.format is None else {"format": self.format}
                        shutil.unpack_archive(filename=tmp.name, extract_dir=extract_dir, **options)
                        self.handle_directory(extract_dir)
                    finally:
                        if self.delete_tmp:
                            shutil.rmtree(extract_dir, ignore_errors=True)
                else:
                    self.handle_file(src=tmp.name, file=tmp)
            return True
        finally:
            if self.delete_tmp:
                if tmp is not None and os.path.exists(tmp.name):
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

    def reset_param(self, url, name=None, format=None, archive=False):
        super().reset_param(url=url, name=name, format=format, archive=archive)

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


class CheckpointDownloader(FileDownloader):
    """Download a file into a temporary file in the target directory and replace atomically."""

    def __init__(self, url: str, dst_path: str, name: str | None = None, **kwargs):
        super().__init__(url=url, name=name or os.path.basename(dst_path), **kwargs)
        self.dst_path = dst_path

    def run(
        self,
        block_size: int = 512 * 1024,
        show_progress: bool = True,
        stop_event: threading.Event | None = None,
        tqdm_fn: Any = None,
        tmp_dir: str | None = None,
    ) -> bool:
        dst_dir = os.path.dirname(self.dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
        if tmp_dir is None and dst_dir:
            tmp_dir = dst_dir
        return super().run(
            block_size=block_size,
            show_progress=show_progress,
            stop_event=stop_event,
            tqdm_fn=tqdm_fn,
            tmp_dir=tmp_dir,
        )

    def handle(self, src: str) -> None:
        os.replace(src, self.dst_path)
