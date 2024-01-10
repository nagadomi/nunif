from os import path
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from nunif.utils.image_loader import list_images
from nunif.utils import pil_io
from nunif.modules import ClampLoss
from nunif.transforms.std import RandomFlip, RandomSRHardExampleCrop, add_jpeg_noise
from nunif.training.env import RegressionEnv
from nunif.training.trainer import Trainer
from nunif.transforms import image_magick as IM


INTERPOLATION_MODES = (
    "sinc",
    "lanczos",
    "triangle",
    "catrom",
)
VALIDATION_INTERPOLATION_MODE = "catrom"
MIN_SCALE_FACTOR = 0.5
MAX_SCALE_FACTOR = 2.0
MAX_SCALE_FACTOR_LARGE = 4.0


def resize(im, scale_factor, filter_type):
    if isinstance(scale_factor, (list, tuple)):
        scale_factor_h, scale_factor_w = scale_factor
    else:
        scale_factor_h = scale_factor_w = scale_factor
    scale_factor_h = round(scale_factor_h, 2)
    scale_factor_w = round(scale_factor_w, 2)

    if scale_factor_h == 1.0 and scale_factor_w == 1.0:
        return TF.to_tensor(im), 1.0

    h = int(im.height * scale_factor_h)
    w = int(im.width * scale_factor_w)
    im = IM.resize(TF.to_tensor(im), (h, w), filter_type)
    scale_factor = max(scale_factor_h, scale_factor_w)
    return im, min(max(scale_factor_w, 1.), 2.)


def random_resize(im):
    method = random.choice(["none", "downscale", "upscale", "upscale"])
    if method == "none":
        return TF.to_tensor(im), 1.
    elif method == "upscale":
        max_scale_factor = MAX_SCALE_FACTOR_LARGE if random.uniform(0, 1) < 0.1 else MAX_SCALE_FACTOR
        keep_aspect = random.uniform(0, 1) < 0.8
        if keep_aspect:
            scale_factor_w = scale_factor_h = random.uniform(1., max_scale_factor)
        else:
            scale_factor_w = random.uniform(1., max_scale_factor)
            scale_factor_h = random.uniform(1., max_scale_factor)

        filter_type = random.choice(INTERPOLATION_MODES)
        return resize(im, [scale_factor_h, scale_factor_w], filter_type)
    elif method == "downscale":
        scale_factor = random.uniform(MIN_SCALE_FACTOR, 1.)
        filter_type = random.choice(INTERPOLATION_MODES)
        return resize(im, scale_factor, filter_type)


class ResizeDataset(Dataset):
    def __init__(self, input_dir, training):
        super().__init__()
        self.training = training
        self.files = list_images(input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        self.gt_transform = T.Compose([
            RandomSRHardExampleCrop(256 + 16),
        ])
        self.random_crop = T.Compose([RandomSRHardExampleCrop(128), RandomFlip()])
        self.random_grayscale = T.RandomGrayscale(p=0.01)
        self.center_crop = T.CenterCrop(128)
        self.validation_settings = self._make_validation_settings()

    @staticmethod
    def _make_validation_settings(n=200):
        # Use the same settings as training as possible
        settings = []
        method_types = ["none", "downscale", "upscale", "upscale"]
        method_split = n // len(method_types)
        methods = ["none"] * method_split
        methods += ["downscale"] * method_split
        methods += ["upscale"] * method_split
        methods += ["upscale"] * (n - method_split * 3)
        assert len(methods) == n

        downscale_factors = torch.linspace(0.5, 0.99, 20)
        upscale_factors = torch.linspace(1.01, 2.0, 20)
        for i in range(n):
            method = methods[i]
            if method == "none":
                settings.append({"scale_factor": 1.})
            elif method == "downscale":
                settings.append({"scale_factor": downscale_factors[i % len(downscale_factors)].item()})
            elif method == "upscale":
                settings.append({"scale_factor": upscale_factors[i % len(upscale_factors)].item()})

        return settings

    def create_sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def __len__(self):
        return len(self.files)

    def get_item_train(self, x, index):
        x = self.gt_transform(x)
        x = self.random_grayscale(x)  # before resize
        x, scale_factor = random_resize(x)  # x is tensor
        x = self.random_grayscale(x)  # after resize
        if random.uniform(0, 1) < 0.5:
            jpeg_quality = random.randint(80, 99)
            jpeg_subsampling = random.choice(["4:4:4", "4:2:0"])
            x = TF.to_pil_image(x)
            x = add_jpeg_noise(x, quality=jpeg_quality, subsampling=jpeg_subsampling)
            x = TF.to_tensor(x)
        x = self.random_crop(x)
        y = torch.tensor((scale_factor, ), dtype=torch.float32)

        return x, y

    def get_item_eval(self, x, index):
        setting = self.validation_settings[index % len(self.validation_settings)]
        x = self.gt_transform(x)
        x, scale_factor = resize(x, setting["scale_factor"], VALIDATION_INTERPOLATION_MODE)
        x = self.center_crop(x)
        y = torch.tensor((scale_factor, ), dtype=torch.float32)
        return x, y

    def __getitem__(self, index):
        x, _ = pil_io.load_image_simple(self.files[index], color="rgb")
        if self.training:
            return self.get_item_train(x, index)
        else:
            return self.get_item_eval(x, index)


class ResizeTrainer(Trainer):
    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = ResizeDataset(path.join(self.args.data_dir, "train"), training=True)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.create_sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            return loader
        else:
            dataset = ResizeDataset(path.join(self.args.data_dir, "eval"), training=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader

    def create_env(self):
        criterion = ClampLoss(nn.L1Loss(), 1., 2.).to(self.device)
        return RegressionEnv(self.model, criterion)


def train(args):
    trainer = ResizeTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "cliqa.resize",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="cliqa.scale_factor", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")

    parser.set_defaults(
        batch_size=64,
        optimizer="adam",
        learning_rate=0.0001,
        scheduler="cosine",
        learning_rate_cycles=12,
        learning_rate_decay=0.98,
        learning_rate_decay_step=[1],
        max_epoch=400,
        momentum=0.9,
        weight_decay=0,
    )
    parser.set_defaults(handler=train)

    return parser
