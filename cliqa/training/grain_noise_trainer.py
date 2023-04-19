import sys
from os import path
import argparse
import random
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from torchvision.transforms import (
    functional as TF,
)
from nunif.utils.image_loader import ImageLoader
from nunif.utils import pil_io
from nunif.models import create_model as nunif_create_model
from nunif.modules import ClampLoss
from nunif.transforms.std import add_jpeg_noise, RandomFlip
from nunif.training.env import BaseEnv
from nunif.training.trainer import Trainer
from waifu2x.training.photo_noise import gaussian_noise_variants, grain_noise1, grain_noise2, structured_noise


def calc_noise_level(x, x_org):
    x = (x.mean(dim=0) * 255. + 0.49).long().float()
    x_org = (x_org.mean(dim=0) * 255. + 0.49).long().float()
    mse = (((x - x_org) ** 2).mean()).item()
    # 10 * math.log10((255 * 255) / 0.65025) == 50.0
    mse = max(mse, 0.65025)
    psnr = 10 * math.log10((255 * 255) / mse)
    psnr = min(max(psnr, 0), 50)
    noise_level = 50.0 - psnr
    return noise_level


NOISE_METHODS = [
    gaussian_noise_variants,
    grain_noise1,
    grain_noise2,
    structured_noise]


class GrainNoiseDataset(Dataset):
    def __init__(self, input_dir, training):
        super().__init__()
        self.training = training
        self.files = ImageLoader.listdir(input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        self.gt_transform = T.Compose([
            T.RandomCrop((136, 136)),
            T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1)], p=0.5),
        ])
        self.random_crop = T.Compose([T.RandomCrop((128, 128)), RandomFlip()])
        self.random_grayscale = T.RandomGrayscale(p=0.02)
        self.center_crop = T.CenterCrop((128, 128))
        self.validation_settings = self._make_validation_settings()

    @staticmethod
    def _make_validation_settings(n=200, seed=71):
        # Use the same settings as training as possible
        rng = np.random.RandomState(seed)
        settings = []
        for _ in range(n):
            p = rng.uniform()
            if p < 0.1:
                strength = 0
            elif p < 0.2:
                strength = rng.uniform(0.0, 0.05)
            elif p < 0.3:
                strength = rng.uniform(0.2, 0.3)
            else:
                strength = rng.uniform(0.05, 0.2)
            # NOTE: np.randint returns a different range than random.randint.
            noise_method = rng.randint(0, len(NOISE_METHODS))
            if rng.uniform() < 0.5:
                jpeg_quality = rng.randint(80, 99 + 1)
                jpeg_subsampling = "4:4:4" if rng.uniform() < 0.5 else "4:2:0"
            else:
                jpeg_quality = 100
                jpeg_subsampling = None
            settings.append({"strength": strength,
                             "noise_method": noise_method,
                             "jpeg_quality": jpeg_quality,
                             "jpeg_subsampling": jpeg_subsampling})
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
        x = TF.to_tensor(x)
        x_org = x
        p = random.uniform(0, 1)
        if p < 0.1:
            strength = 0
        elif p < 0.2:
            strength = random.uniform(0.0, 0.05)
        elif p < 0.3:
            strength = random.uniform(0.2, 0.3)
        else:
            strength = random.uniform(0.05, 0.2)

        if strength > 0:
            f = random.choice(NOISE_METHODS)
            x = f(x, strength=strength)

        noise_level = torch.tensor((calc_noise_level(x, x_org),), dtype=torch.float32)
        if random.uniform(0, 1) < 0.5:
            jpeg_quality = random.randint(80, 99)
            jpeg_subsampling = random.choice(["4:4:4", "4:2:0"])
            x = TF.to_pil_image(x)
            x = self.random_grayscale(x)
            x = add_jpeg_noise(x, quality=jpeg_quality, subsampling=jpeg_subsampling)
        else:
            x = TF.to_pil_image(x)
            x = self.random_grayscale(x)
        x = self.random_crop(x)
        x = TF.to_tensor(x)

        return x, noise_level

    def get_item_eval(self, x, index):
        x = self.center_crop(x)
        x = TF.to_tensor(x)
        x_org = x
        setting = self.validation_settings[index % len(self.validation_settings)]
        strength = setting["strength"]
        noise_method = NOISE_METHODS[setting["noise_method"]]
        jpeg_quality = setting["jpeg_quality"]
        jpeg_subsampling = setting["jpeg_subsampling"]

        if strength > 0:
            x = noise_method(x, strength=strength)
        noise_level = torch.tensor((calc_noise_level(x, x_org),), dtype=torch.float32)

        if jpeg_quality < 100:
            x = add_jpeg_noise(TF.to_pil_image(x), quality=jpeg_quality, subsampling=jpeg_subsampling)
            x = TF.to_tensor(x)

        return x, noise_level

    def __getitem__(self, index):
        x, _ = pil_io.load_image_simple(self.files[index], color="rgb")
        if self.training:
            return self.get_item_train(x, index)
        else:
            return self.get_item_eval(x, index)


class GrainNoiseEnv(BaseEnv):
    def __init__(self, model, criterion, device):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.criterion = criterion

    def clear_loss(self):
        self.sum_loss = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.clear_loss()

    def train_step(self, data):
        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        with self.autocast():
            z = self.model(x)
            loss = self.criterion(z, y)
        self.sum_loss += loss.item()
        self.sum_step += 1

        return loss

    def train_end(self):
        loss = self.sum_loss / self.sum_step
        print(f"loss: {loss}")
        return loss

    def eval_begin(self):
        self.model.eval()
        self.clear_loss()

    def eval_step(self, data):
        return self.train_step(data)

    def eval_end(self, file=sys.stdout):
        return self.train_end()


class GrainNoiseTrainer(Trainer):
    def create_model(self):
        model = nunif_create_model(self.args.arch, device_ids=self.args.gpu)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = GrainNoiseDataset(path.join(self.args.data_dir, "train"), training=True)
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
            dataset = GrainNoiseDataset(path.join(self.args.data_dir, "eval"), training=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader

    def create_env(self):
        criterion = ClampLoss(nn.L1Loss(), 0, 50).to(self.device)
        return GrainNoiseEnv(self.model, criterion, self.device)


def train(args):
    trainer = GrainNoiseTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "cliqa.grain",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="cliqa.grain_noise_level", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")

    parser.set_defaults(
        batch_size=64,
        optimizer="adam",
        learning_rate=0.00005,
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
