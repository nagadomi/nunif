import random
from os import path
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from torchvision.transforms import (
    functional as TF,
)
import nunif.transforms.pair as TP
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
from nunif.training.sampler import HardExampleSampler, MiningMethod
from iw3.dilation import dilate_outer, dilate_inner


SIZE = 256  # % 8 == 0


class RightDilate():
    def __init__(self, max_step=20, p=0.5):
        self.max_step = max_step
        self.p = p

    def __call__(self, mask):
        if random.uniform(0, 1) < self.p:
            n_step = random.randint(1, self.max_step)
            mask = dilate_outer(mask.unsqueeze(0), n_iter=n_step).squeeze(0)
        return mask


class LeftDilate():
    def __init__(self, max_step=4, p=0.25):
        self.max_step = max_step
        self.p = p

    def __call__(self, mask):
        if random.uniform(0, 1) < self.p:
            n_step = random.randint(1, self.max_step)
            mask = dilate_inner(mask.unsqueeze(0), n_iter=n_step).squeeze(0)
        return mask


class InpaintDataset(Dataset):
    def __init__(self, input_dir, model_offset, training):
        super().__init__()
        self.training = training
        self.model_offset = model_offset
        self.load_files(input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        self.gt_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.25),
            T.RandomGrayscale(p=0.05),
        ])
        self.random_crop = TP.RandomHardExampleCrop(SIZE)
        self.center_crop = TP.CenterCrop(SIZE)
        self.right_dilate = RightDilate()
        self.left_dilate = LeftDilate()

    def load_files(self, input_dir):
        files = ImageLoader.listdir(input_dir)
        self.masks = [fn for fn in files if fn.endswith("_M.png")]
        self.files = []
        for fn in self.masks:
            rgb_file = fn.replace("_M.png", "_C.png")
            if path.exists(rgb_file):
                self.files.append(rgb_file)
            else:
                raise RuntimeError(f"{rgb_file} not found")

    def create_sampler(self, num_samples):
        return HardExampleSampler(
            torch.ones((len(self),), dtype=torch.double),
            num_samples=num_samples,
            method=MiningMethod.LINEAR,
            history_size=4,
            scale_factor=4.,
        )

    def worker_init(self, worker_id):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        im, _ = load_image_simple(self.files[index], color="rgb")
        mask, _ = load_image_simple(self.masks[index], color="gray")

        x = TF.to_tensor(im)
        mask = TF.to_tensor(mask)
        if self.training:
            im = self.gt_transform(im)
            mask = self.right_dilate(mask)
            mask = self.left_dilate(mask)

        if self.training:
            mask, x = self.random_crop(mask, x)
        else:
            mask, x = self.center_crop(mask, x)

        y = x.clone()
        y = TF.crop(y, self.model_offset, self.model_offset,
                    y.shape[-2] - self.model_offset * 2,
                    y.shape[-1] - self.model_offset * 2)

        mask = mask > 0

        return x, mask, y, index


if __name__ == "__main__":
    pass
