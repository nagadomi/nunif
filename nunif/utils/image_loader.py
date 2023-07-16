import os
import sys
from time import sleep
from threading import Thread, Event
from queue import Queue
from .. logger import logger
from . import pil_io
import mimetypes


# Add missing mimetypes
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/vnd.ms-dds", ".dds")


# TODO: there are other extensions that can be used
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.dds')


def image_load_task(q, stop_flag, files, max_queue_size, load_func):
    for f in files:
        if stop_flag.is_set():
            break
        while q.qsize() >= max_queue_size:
            if stop_flag.is_set():
                q.put(None)
                return
            sleep(0.001)
        try:
            im, meta = load_func(f)
        except KeyboardInterrupt:
            raise
        except:  # noqa: E722
            logger.error(f"ImageLoader: load error: {f}, {sys.exc_info()[:2]}")
            im, meta = None, None
        q.put((im, meta))
    q.put(None)


def list_images(directory, extensions=IMG_EXTENSIONS):
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[-1].lower() in extensions
    )


class ImageLoader():
    @classmethod
    def listdir(cls, directory, extensions=IMG_EXTENSIONS):
        return list_images(directory, extensions=IMG_EXTENSIONS)

    def __init__(self, directory=None, files=None, max_queue_size=256,
                 load_func=pil_io.load_image,
                 load_func_kwargs=None,
                 extensions=IMG_EXTENSIONS):
        assert (directory is not None or files is not None)
        if files is not None:
            self.files = files
        else:
            self.files = ImageLoader.listdir(directory, extensions=extensions)
        self.max_queue_size = max_queue_size
        self.load_func = lambda x: load_func(x, **(load_func_kwargs or {}))
        self.proc = None
        self.queue = Queue()
        self.stop_flag = Event()

    def terminate(self):
        if self.proc:
            self.stop_flag.set()
            self.proc.join()
            self.proc = None
            self.stop_flag.clear()
            self.queue = Queue()

    def start(self):
        if self.proc is None:
            self.stop_flag.clear()
            self.proc = Thread(target=image_load_task,
                               args=(self.queue, self.stop_flag, self.files,
                                     self.max_queue_size, self.load_func))
            self.proc.start()

    def __del__(self):
        self.terminate()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

    def __next__(self):
        if self.proc is None:
            self.start()
        while True:
            ret = self.queue.get()
            if ret is None:
                self.proc.join()
                self.proc = None
                self.stop_flag.clear()
                raise StopIteration()
            elif ret[0] is None:
                continue
            else:
                return ret
