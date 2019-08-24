import glob
import os
from time import sleep
from threading import Thread
from queue import Queue
import numpy as np
from .. transformers import load_image


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MAX_IMAGE_QUEUE = 256


def image_load_task(q, files, max_queue_size):
    for f in files:
        while q.qsize() >= max_queue_size:
            sleep(0.01)
        try:
            im, meta = load_image(f)
        except:
            im, meta = None, None
        q.put((im, meta))
    q.put(None)


class ImageLoader():
    def __init__(self, directory=None, files=None, max_queue_size=256):
        assert(directory is not None or files is not None)
        if files is not None:
            self.files = files
        else:
            self.files = np.array([f for f in glob.glob(os.path.join(directory, "*")) if os.path.splitext(f)[-1] in IMG_EXTENSIONS])
        self.max_queue_size = max_queue_size
        self.proc = None
        self.queue = Queue()

    def __iter__(self):
        return self

    def __next__(self):
        if self.proc is None:
            self.proc = Thread(target=image_load_task, args=(self.queue, self.files, self.max_queue_size))
            self.proc.start()
        ret = self.queue.get()
        if ret is None:
            self.proc.join()
            self.proc = None
            raise StopIteration()
        else:
            return ret
