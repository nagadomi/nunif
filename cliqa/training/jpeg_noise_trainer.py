import sys
from os import path
import argparse
import random
import torch
import torch.nn as nn
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


class JPEGDataset(Dataset):
    def __init__(self, input_dir, training):
        super().__init__()
        self.training = training
        self.files = ImageLoader.listdir(input_dir)
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        self.gt_transform = T.Compose([
            T.RandomCrop((256, 256)),
            T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1)], p=0.5),
            T.RandomGrayscale(p=0.02)
        ])
        self.random_crop = T.Compose([T.RandomCrop((128, 128)), RandomFlip()])
        self.center_crop = T.CenterCrop((128, 128))

    def create_sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def __len__(self):
        return len(self.files)

    def get_item_train(self, x, index):
        x = self.gt_transform(x)
        p = random.uniform(0, 1)
        if p < 0.1:
            quality = 100
            subsampling = "4:4:4"
        elif p < 0.2:
            quality = random.randint(0, 40)
            subsampling = random.choice(["4:4:4", "4:2:0"])
            x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)
        else:
            quality = random.randint(40, 99)
            subsampling = random.choice(["4:4:4", "4:2:0"])
            x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)

        subsampling = torch.tensor((1 if subsampling == "4:2:0" else 0,), dtype=torch.float32)
        quality = torch.tensor((quality,), dtype=torch.float32)

        x = self.random_crop(x)
        x = TF.to_tensor(x)

        return x, quality, subsampling

    def get_item_eval(self, x, index):
        x = self.center_crop(x)
        if index % 10 == 0:
            quality = 100
            subsampling = "4:4:4"
        elif index % 9 == 0:
            quality = (index % 20) * 2
            subsampling = "4:4:4" if index % 2 == 0 else "4:2:0"
            x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)
        else:
            quality = 40 + (index % 30) * 2
            subsampling = "4:4:4" if index % 2 == 0 else "4:2:0"
            x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)

        subsampling = torch.tensor((1 if subsampling == "4:2:0" else 0,), dtype=torch.float32)
        quality = torch.tensor((quality,), dtype=torch.float32)

        x = TF.to_tensor(x)

        return x, quality, subsampling

    def __getitem__(self, index):
        x, _ = pil_io.load_image_simple(self.files[index], color="rgb")
        if self.training:
            return self.get_item_train(x, index)
        else:
            return self.get_item_eval(x, index)


class JPEGEnv(BaseEnv):
    def __init__(self, model, quality_criterion, subsampling_criterion, device):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.quality_criterion = quality_criterion
        self.subsampling_criterion = subsampling_criterion

    def clear_loss(self):
        self.sum_quality_loss = 0
        self.sum_subsampling_loss = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.clear_loss()

    def train_step(self, data):
        x, y_quality, y_subsampling, *_ = data
        x = self.to_device(x)
        y_quality = self.to_device(y_quality)
        y_subsampling = self.to_device(y_subsampling)

        with self.autocast():
            quality, subsampling = self.model(x)
            quality_loss = self.quality_criterion(quality, y_quality)
            subsampling_loss = self.subsampling_criterion(subsampling, y_subsampling)

        self.sum_quality_loss += quality_loss.item()
        self.sum_subsampling_loss += subsampling_loss.item()
        self.sum_step += 1

        loss = quality_loss + subsampling_loss

        return loss

    def train_end(self):
        quality_loss = self.sum_quality_loss / self.sum_step
        subsampling_loss = self.sum_subsampling_loss / self.sum_step
        print(f"quality_loss: {quality_loss}, subsampling_loss: {subsampling_loss}")
        return quality_loss

    def eval_begin(self):
        self.model.eval()
        self.clear_loss()

    def eval_step(self, data):
        return self.train_step(data)

    def eval_end(self, file=sys.stdout):
        return self.train_end()


class JPEGTrainer(Trainer):
    def create_model(self):
        model = nunif_create_model(self.args.arch, device_ids=self.args.gpu)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = JPEGDataset(path.join(self.args.data_dir, "train"), training=True)
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
            dataset = JPEGDataset(path.join(self.args.data_dir, "eval"), training=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader

    def create_env(self):
        quality_criterion = ClampLoss(nn.L1Loss(), 0, 100).to(self.device)
        subsampling_criterion = nn.BCEWithLogitsLoss().to(self.device)
        return JPEGEnv(self.model, quality_criterion, subsampling_criterion, self.device)


def train(args):
    trainer = JPEGTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "cliqa.jpeg",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="cliqa.jpeg_quality", help="network arch")
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
