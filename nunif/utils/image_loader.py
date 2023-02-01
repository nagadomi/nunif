import glob
import os
import sys
from time import sleep
from threading import Thread, Event
from queue import Queue
from .. logger import logger
from . import pil_io


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MAX_IMAGE_QUEUE = 256


def image_load_task(q, stop_flag, files, max_queue_size, load_func):
    for f in files:
        while q.qsize() >= max_queue_size:
            if stop_flag.is_set():
                q.put(None)
                return
            sleep(0.001)
        try:
            im, meta = load_func(f)
        except:  # noqa: E722
            logger.error(f"ImageLoader: load error: {f}, {sys.exc_info()[:2]}")
            im, meta = None, None
        q.put((im, meta))
    q.put(None)


def list_images(directory):
    return sorted([f for f in glob.glob(os.path.join(directory, "*"))
                   if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS])


class ImageLoader():
    @classmethod
    def listdir(cls, directory):
        return list_images(directory)

    def __init__(self, directory=None, files=None, max_queue_size=256,
                 load_func=pil_io.load_image,
                 load_func_kwargs=None):
        assert (directory is not None or files is not None)
        if files is not None:
            self.files = files
        else:
            self.files = ImageLoader.listdir(directory)
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


class DummyImageLoader():
    """ I don't remember what this is :(
    """
    def __init__(self, n):
        self.n = n
        self.i = 0

    def terminate(self):
        self.i = 0

    def start(self):
        pass

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
        return len(self.n)

    def __next__(self):
        if self.i < self.n:
            return (None, None)
        else:
            self.i = 0
            raise StopIteration()


SEP = "."


def basename_without_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def filename2key(filename, subdir_level=0):
    filename = os.path.abspath(filename)
    if subdir_level > 0:
        subdirs = []
        basename = basename_without_ext(filename)
        for _ in range(subdir_level):
            filename = os.path.dirname(filename)
            subdirs.insert(0, os.path.basename(filename))
        return SEP.join(subdirs + [basename])
    else:
        return basename_without_ext(filename)
