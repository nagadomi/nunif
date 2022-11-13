import glob
import os
from time import sleep
from threading import Thread, Event
from queue import Queue
from PIL import Image, ImageCms, PngImagePlugin
import torch
from torchvision.transforms import functional as TF
import io
import struct
import snappy
import numpy as np
from .. logger import logger
sRGB_profile = ImageCms.createProfile("sRGB")


def _load_image(im, filename):
    meta = {}
    im.load()
    if im.mode in ("L", "I", "RGB", "P"):
        transparency = im.info.get('transparency')
        if isinstance(transparency, bytes) or isinstance(transparency, int):
            im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA", "IA"):
        meta["alpha"] = im.getchannel("A")
    meta["filename"] = filename
    meta["mode"] = im.mode
    meta["gamma"] = im.info.get("gamma")
    meta["icc_profile"] = im.info.get("icc_profile")
    if meta['icc_profile'] is not None:
        with io.BytesIO(meta['icc_profile']) as io_handle:
            src_profile = ImageCms.ImageCmsProfile(io_handle)
            dst_profile = sRGB_profile
            im = ImageCms.profileToProfile(im, src_profile, dst_profile)

    if im.mode != "RGB":
        # FIXME: `I to RGB` is broken
        im = im.convert("RGB")

    return im, meta


def load_image(filename):
    if os.path.splitext(filename)[-1] == ".sz":
        im = load_image_snappy(filename, Image.Image)
        return _load_image(im, filename)
    else:
        with open(filename, "rb") as f:
            im = Image.open(f)
            return _load_image(im, filename)


def decode_image(buff, filename=None):
    with io.BytesIO(buff) as data:
        im = Image.open(data)
        return _load_image(im, filename)


def load_image_rgb(filename):
    im, meta = load_image(filename)
    return im


def save_image_snappy(x, filename):
    with open(filename, "wb") as f:
        if isinstance(x, Image.Image):
            x = TF.to_tensor(im).mul_(255).byte()
        elif isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor)):
            x = x.mul(255).byte()  # expect: float tensor
        elif isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)):
            pass
        else:
            raise ValueError("Unknown input format")
        header = struct.pack("!LLL", x.shape[0], x.shape[1], x.shape[2])
        image_data = snappy.compress(x.numpy())
        f.write(header)
        f.write(image_data)


def load_image_snappy(filename, dtype=torch.FloatTensor):
    with open(filename, "rb") as f:
        header = f.read(3 * 4)
        image_data = f.read()
        s1, s2, s3 = struct.unpack("!LLL", header)
        x = torch.ByteTensor(np.frombuffer(snappy.decompress(image_data), dtype=np.uint8)).reshape(s1, s2, s3)
        if dtype is Image.Image:
            return TF.to_pil_image(x)
        elif dtype is torch.FloatTensor:
            return x.float().div_(255)
        elif dtype is torch.ByteTensor:
            return x
        else:
            raise ValueError("Unknown dtype")


def encode_image_snappy(x):
    with io.BytesIO() as f:
        if isinstance(x, Image.Image):
            x = TF.to_tensor(im).mul_(255).byte()
        elif isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor)):
            x = x.mul(255).byte()  # expect: float tensor
        elif isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)):
            pass
        else:
            raise ValueError("Unknown input format")
        header = struct.pack("!LLL", x.shape[0], x.shape[1], x.shape[2])
        image_data = snappy.compress(x.numpy())
        f.write(header)
        f.write(image_data)
        return f.getvalue()


def decode_image_snappy(buf, dtype=torch.FloatTensor):
    with io.BytesIO(buf) as f:
        header = f.read(3 * 4)
        image_data = f.read()
        s1, s2, s3 = struct.unpack("!LLL", header)
        x = torch.ByteTensor(np.frombuffer(snappy.decompress(image_data), dtype=np.uint8)).reshape(s1, s2, s3)
        if dtype is Image.Image:
            return TF.to_pil_image(x)
        elif dtype is torch.FloatTensor:
            return x.float().div_(255)
        elif dtype is torch.ByteTensor:
            return x
        else:
            raise ValueError("Unknown dtype")


def save_image(im, meta, filename, format="png", compress_level=6):
    # TODO: support non PNG format
    meta = meta if meta is not None else {"gamma": None, "icc_profile": None}
    if meta["icc_profile"] is not None:
        with io.BytesIO(meta['icc_profile']) as io_handle:
            src_profile = sRGB_profile
            dst_profile = ImageCms.ImageCmsProfile(io_handle)
            im = ImageCms.profileToProfile(im, src_profile, dst_profile)

    pnginfo = PngImagePlugin.PngInfo()
    if meta["gamma"] is not None:
        pnginfo.add(b"gAMA", struct.pack(">I", int(meta["gamma"] * 100000)))

    im.save(filename, format=format,
            icc_profile=meta["icc_profile"], pnginfo=pnginfo,
            compress_level=compress_level)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.sz')
MAX_IMAGE_QUEUE = 256


def image_load_task(q, stop_flag, files, max_queue_size):
    for f in files:
        while q.qsize() >= max_queue_size:
            if stop_flag.is_set():
                q.put(None)
                return
            sleep(0.001)
        try:
            im, meta = load_image(f)
        except:
            logger.error(f"Failed to load image: {f}")
            im, meta = None, None
        q.put((im, meta))
    q.put(None)


class ImageLoader():
    @classmethod
    def listdir(cls, directory):
        return [f for f in glob.glob(os.path.join(directory, "*")) if os.path.splitext(f)[-1] in IMG_EXTENSIONS]

    def __init__(self, directory=None, files=None, max_queue_size=256):
        assert (directory is not None or files is not None)
        if files is not None:
            self.files = files
        else:
            self.files = ImageLoader.listdir(directory)
        self.max_queue_size = max_queue_size
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
                               args=(self.queue, self.stop_flag, self.files, self.max_queue_size))
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


def basename_without_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


SEP = "."


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


if __name__ == "__main__":
    import time

    def save_image_fast(im, filename):
        im.save(filename, compress_level=1)

    def save_image_pth(im, filename):
        x = TF.to_tensor(im).mul_(255).byte()
        torch.save(x, filename)

    src = "tmp/miku_CC_BY-NC.jpg"
    im = load_image_rgb(src)
    buf = encode_image_snappy(im)
    im = decode_image_snappy(buf, Image.Image)

    save_image_fast(im, "tmp/miku_fast.png")
    save_image_pth(im, "tmp/miku_fast.pth")
    save_image_snappy(im, "tmp/miku_fast.sz")

    # im = load_image_snappy("tmp/miku_fast.sz", dtype=Image.Image)
    # im.show()
    N = 3000

    t = time.time()
    for _ in range(N):
        a = load_image_snappy("tmp/miku_fast.sz", dtype=torch.FloatTensor)
    print(f"save_image_snappy: {time.time() - t}")

    t = time.time()
    for _ in range(N):
        a = load_image_rgb(src)
    print(f"save_image: {time.time() - t}")

    t = time.time()
    for _ in range(N):
        a = load_image_rgb("tmp/miku_fast.png")
    print(f"save_image_fast: {time.time() - t}")

    t = time.time()
    for _ in range(N):
        a = torch.load("tmp/miku_fast.pth").float().mul_(255)
    print(f"save_image_pth: {time.time() - t}")


"""
save_image_snappy: 3.7878315448760986
save_image: 12.264774322509766
save_image_fast: 15.478008508682251
save_image_pth: 3.5136613845825195

-rw-r--r-- 1 nagadomi nagadomi   78998  8月 22 00:38 miku_CC_BY-NC.jpg
-rw-r--r-- 1 nagadomi nagadomi  246365  9月  3 09:42 miku_fast.png
-rw-r--r-- 1 nagadomi nagadomi 1080341  9月  3 09:42 miku_fast.pth
-rw-r--r-- 1 nagadomi nagadomi  380019  9月  3 09:42 miku_fast.sz
"""
