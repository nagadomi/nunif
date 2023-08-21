import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    functional as TF,
)
from nunif.utils.image_loader import ImageLoader
from nunif.utils import pil_io
from nunif.training.sampler import HardExampleSampler, MiningMethod
from os import path
import random
from PIL import Image
from ... import utils as US


def load_images(org_file, side):
    dirname = path.dirname(org_file)
    basename = path.basename(org_file)

    im_org, _ = pil_io.load_image_simple(org_file, color="rgb")
    # depth is 16bit int image
    im_depth = Image.open(path.join(dirname, basename.replace("_C.png", "_D.png")))
    im_depth.load()
    if side == "left":
        im_side, _ = pil_io.load_image_simple(
            path.join(dirname, basename.replace("_C.png", "_L.png")), color="rgb")
    else:
        im_side, _ = pil_io.load_image_simple(
            path.join(dirname, basename.replace("_C.png", "_R.png")), color="rgb")

    if not all([im_org, im_depth, im_side]):
        raise RuntimeError(f"load error {org_file}")
    assert im_org.size == im_depth.size and im_org.size == im_side.size

    return im_org, im_depth, im_side


class SBSDataset(Dataset):
    def __init__(self, input_dir, model_offset, training):
        super().__init__()
        self.training = training
        self.model_offset = model_offset
        self.files = [fn for fn in ImageLoader.listdir(input_dir) if fn.endswith("_C.png")]
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

    def __getitem__(self, index):
        if self.training:
            side = random.choice(["left", "right"])
        else:
            side = "left"
        im_org, im_depth, im_side = load_images(self.files[index], side)
        depth_max = int(im_depth.text["sbs_depth_max"])
        depth_min = int(im_depth.text["sbs_depth_min"])
        original_image_width = int(im_depth.text["sbs_width"])
        divergence = float(im_depth.text["sbs_divergence"])
        convergence = float(im_depth.text["sbs_convergence"])
        if side == "right":
            im_org = TF.hflip(im_org)
            im_depth = TF.hflip(im_depth)
            im_side = TF.hflip(im_side)

        x = US.make_input_tensor(
            TF.to_tensor(im_org),
            TF.to_tensor(im_depth),
            divergence, convergence,
            original_image_width,
            depth_min=depth_min, depth_max=depth_max
        )
        y = TF.to_tensor(TF.crop(im_side, self.model_offset, self.model_offset,
                                 im_side.height - self.model_offset * 2,
                                 im_side.width - self.model_offset * 2))

        return x, y, index
