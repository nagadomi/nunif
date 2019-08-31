import glob
import os
from time import sleep
from threading import Thread
from queue import Queue
from PIL import Image, ImageCms, PngImagePlugin
import io
import struct


sRGB_profile = ImageCms.createProfile("sRGB")


def _load_image(im, filename):
    meta = {}
    im.load()
    if im.mode in ("L", "RGB", "P"):
        if isinstance(im.info.get('transparency'), bytes):
            im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        meta["alpha"] = im.getchannel("A")
    meta["filename"] = filename
    meta["mode"] = im.mode
    meta["gamma"] = im.info.get("gamma")
    meta["icc_profile"] = im.info.get("icc_profile")
    if meta['icc_profile'] is not None:
        io_handle = io.BytesIO(meta['icc_profile'])
        src_profile = ImageCms.ImageCmsProfile(io_handle)
        dst_profile = sRGB_profile
        im = ImageCms.profileToProfile(im, src_profile, dst_profile)

    if im.mode != "RGB":
        im = im.convert("RGB")
    return im, meta


def load_image(filename):
    with open(filename, "rb") as f:
        im = Image.open(f)
        return _load_image(im, filename)


def load_image_rgb(filename):
    im, meta = load_image(filename)
    return im


def save_image(im, meta, filename):
    if meta["icc_profile"] is not None:
        io_handle = io.BytesIO(meta['icc_profile'])
        src_profile = sRGB_profile
        dst_profile = ImageCms.ImageCmsProfile(io_handle)
        im = ImageCms.profileToProfile(im, src_profile, dst_profile)

    pnginfo = PngImagePlugin.PngInfo()
    if meta["gamma"] is not None:
        pnginfo.add(b"gAMA", struct.pack(">I", int(meta["gamma"] * 100000)))
    im.save(filename, icc_profile=meta["icc_profile"], pnginfo=pnginfo)


def decode_image(buff):
    pass


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
    @classmethod
    def listdir(cls, directory):
        return [f for f in glob.glob(os.path.join(directory, "*")) if os.path.splitext(f)[-1] in IMG_EXTENSIONS]

    def __init__(self, directory=None, files=None, max_queue_size=256):
        assert(directory is not None or files is not None)
        if files is not None:
            self.files = files
        else:
            self.files = ImageLoader.listdir(directory)
        self.max_queue_size = max_queue_size
        self.proc = None
        self.queue = Queue()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

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
