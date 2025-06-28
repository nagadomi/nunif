import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
)
from nunif.utils.image_loader import ImageLoader
from nunif.utils import pil_io
from nunif.training.sampler import HardExampleSampler, MiningMethod
from os import path
import random
from PIL import Image
from tqdm import tqdm
from ...backward_warp import make_input_tensor
from concurrent.futures import ThreadPoolExecutor


def load_images(org_file, side=None):
    dirname = path.dirname(org_file)
    basename = path.basename(org_file)

    im_org, _ = pil_io.load_image_simple(org_file, color="rgb")
    # depth is 16bit int image
    im_depth = Image.open(path.join(dirname, basename.replace("_C.png", "_D.png")))
    im_depth.load()
    if side is None:
        im_left, _ = pil_io.load_image_simple(
            path.join(dirname, basename.replace("_C.png", "_L.png")), color="rgb")
        im_right, _ = pil_io.load_image_simple(
            path.join(dirname, basename.replace("_C.png", "_R.png")), color="rgb")
        im_mask_left, _ = pil_io.load_image_simple(
            path.join(dirname, basename.replace("_C.png", "_ML.png")), color="gray")
        im_mask_right, _ = pil_io.load_image_simple(
            path.join(dirname, basename.replace("_C.png", "_MR.png")), color="gray")

        if not all([im_org, im_depth, im_left, im_right, im_mask_left, im_mask_right]):
            raise RuntimeError(f"load error {org_file}")
        assert im_org.size == im_depth.size == im_left.size == im_right.size == im_mask_left.size == im_mask_right.size
        return im_org, im_depth, im_left, im_right, im_mask_left, im_mask_right
    else:
        if side == "left":
            im_side, _ = pil_io.load_image_simple(
                path.join(dirname, basename.replace("_C.png", "_L.png")), color="rgb")
            im_mask, _ = pil_io.load_image_simple(
                path.join(dirname, basename.replace("_C.png", "_ML.png")), color="gray")
        else:
            im_side, _ = pil_io.load_image_simple(
                path.join(dirname, basename.replace("_C.png", "_R.png")), color="rgb")
            im_mask, _ = pil_io.load_image_simple(
                path.join(dirname, basename.replace("_C.png", "_MR.png")), color="gray")
        if not all([im_org, im_depth, im_side, im_mask]):
            raise RuntimeError(f"load error {org_file}")
        assert im_org.size == im_depth.size and im_org.size == im_side.size == im_mask.size
        return im_org, im_depth, im_side, im_mask


def depth_pil_to_tensor(im_depth, depth_min, depth_max):
    # depth is already normalized. expect depth_min=0 and depth_max=0xffff
    # pil_to_tensor() -> torch.uint16
    depth = TF.pil_to_tensor(im_depth).to(torch.float32)
    depth = torch.clamp((depth - depth_min) / (depth_max - depth_min), 0, 1)
    depth = torch.nan_to_num(depth)
    return depth


def random_crop(size, *images):
    i, j, h, w = T.RandomCrop.get_params(images[0], (size, size))
    results = []
    for im in images:
        results.append(TF.crop(im, i, j, h, w))

    return tuple(results)


def center_crop(size, *images):
    results = []
    for im in images:
        results.append(TF.center_crop(im, (size, size)))

    return tuple(results)


def filter_weak_convergence(files, datset_dir):
    name = path.basename(datset_dir)
    cache_file = path.join(datset_dir, "..", f"weak_convergence_{name}.txt")
    print("cache_file", cache_file)
    if not path.exists(cache_file):

        def get_convergence(fn):
            with Image.open(fn) as im:
                convergence = float(im.text["sbs_convergence"])
                if 0.3 <= convergence <= 0.7:
                    return [path.basename(fn)]
            return []

        weak_convergence_files = set()
        futures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for fn in tqdm(files, ncols=80, desc="filter weak_convergence"):
                futures.append(pool.submit(get_convergence, fn))
                if len(futures) > 100:
                    for f in futures:
                        weak_convergence_files.update(f.result())
                    futures.clear()
            for f in futures:
                weak_convergence_files.update(f.result())
            futures.clear()

        with open(cache_file, mode="w", encoding="utf-8") as f:
            for fn in weak_convergence_files:
                f.write(fn + "\n")

    with open(cache_file, mode="r", encoding="utf-8") as f:
        hit_names = set(f.read().split("\n"))

    new_files = []
    for fn in files:
        if path.basename(fn) in hit_names:
            new_files.append(fn)

    print("filter_weak_convergence", len(files), len(new_files))
    return new_files


class SBSDataset(Dataset):
    def __init__(self, input_dir, size, model_offset, symmetric, training, weak_convergence=False):
        super().__init__()
        self.size = size
        self.training = training
        self.symmetric = symmetric
        self.model_offset = model_offset
        self.files = [fn for fn in ImageLoader.listdir(input_dir) if fn.endswith("_C.png")]
        if weak_convergence:
            self.files = filter_weak_convergence(self.files, input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")

    def worker_init(self, worker_id):
        pass

    def create_sampler(self, num_samples):
        return HardExampleSampler(
            torch.ones((len(self),), dtype=torch.double),
            num_samples=num_samples,
            method=MiningMethod.LINEAR,
            history_size=4,
            scale_factor=4.,
        )

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_metadata(im_depth):
        depth_max = int(im_depth.text["sbs_depth_max"])
        depth_min = int(im_depth.text["sbs_depth_min"])
        original_image_width = int(im_depth.text["sbs_width"])
        divergence = float(im_depth.text["sbs_divergence"])
        convergence = float(im_depth.text["sbs_convergence"])
        if "sbs_mapper" in im_depth.text:
            mapper = im_depth.text["sbs_mapper"]
        else:
            mapper = "pow2"
        return depth_max, depth_min, original_image_width, divergence, convergence, mapper

    def _getitem_symmetric(self, index):
        im_org, im_depth, im_left, im_right, im_mask_left, im_mask_right = load_images(self.files[index])
        (depth_max, depth_min, original_image_width,
         divergence, convergence, mapper) = self.get_metadata(im_depth)

        if self.size != im_org.height:
            raise ValueError("--symmetric does not support --size option")

        depth = depth_pil_to_tensor(im_depth, depth_min=depth_min, depth_max=depth_max)
        x = make_input_tensor(
            TF.to_tensor(im_org),
            depth,
            divergence, convergence,
            original_image_width,
            mapper=mapper,
        )
        left = TF.to_tensor(TF.crop(im_left, self.model_offset, self.model_offset,
                                    im_left.height - self.model_offset * 2,
                                    im_left.width - self.model_offset * 2))
        right = TF.to_tensor(TF.crop(im_right, self.model_offset, self.model_offset,
                                     im_right.height - self.model_offset * 2,
                                     im_right.width - self.model_offset * 2))
        mask_left = TF.to_tensor(TF.crop(im_mask_left, self.model_offset, self.model_offset,
                                         im_mask_left.height - self.model_offset * 2,
                                         im_mask_left.width - self.model_offset * 2))
        mask_right = TF.to_tensor(TF.crop(im_mask_right, self.model_offset, self.model_offset,
                                          im_mask_right.height - self.model_offset * 2,
                                          im_mask_right.width - self.model_offset * 2))
        y = torch.cat([left, right], dim=0)
        mask = torch.cat([mask_left, mask_right], dim=0)

        return x, (y, mask), index

    def _getitem(self, index):
        if self.training:
            side = random.choice(["left", "right"])
        else:
            side = "left"
        im_org, im_depth, im_side, im_mask = load_images(self.files[index], side)
        (depth_max, depth_min, original_image_width,
         divergence, convergence, mapper) = self.get_metadata(im_depth)

        if side == "right":
            im_org = TF.hflip(im_org)
            im_depth = TF.hflip(im_depth)
            im_side = TF.hflip(im_side)
            im_mask = TF.hflip(im_mask)

        if self.size != im_org.height:
            if self.training:
                im_org, im_depth, im_side, im_mask = random_crop(self.size, im_org, im_depth, im_side, im_mask)
            else:
                im_org, im_depth, im_side, im_mask = center_crop(self.size, im_org, im_depth, im_side, im_mask)

        depth = depth_pil_to_tensor(im_depth, depth_min=depth_min, depth_max=depth_max)
        x = make_input_tensor(
            TF.to_tensor(im_org),
            depth,
            divergence, convergence,
            original_image_width,
            mapper=mapper,
        )
        y = TF.to_tensor(TF.crop(im_side, self.model_offset, self.model_offset,
                                 im_side.height - self.model_offset * 2,
                                 im_side.width - self.model_offset * 2))
        mask = TF.to_tensor(TF.crop(im_mask, self.model_offset, self.model_offset,
                                    im_mask.height - self.model_offset * 2,
                                    im_mask.width - self.model_offset * 2))
        return x, (y, mask), index

    def __getitem__(self, index):
        if self.symmetric:
            return self._getitem_symmetric(index)
        else:
            return self._getitem(index)
