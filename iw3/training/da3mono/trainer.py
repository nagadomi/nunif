from os import path
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import create_model
from nunif.training.env import I2IEnv
from nunif.modules.psnr import PSNR
from nunif.training.trainer import Trainer
from .dataset import DA3MonoDataset
from ... import models # noqa


def normalize(x):
    B, C, H, W = x.shape
    x_flat = x.reshape(B, -1)
    min_value = x_flat.min(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
    max_value = x_flat.max(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
    return (x - min_value) / ((max_value - min_value) + 1e-6)


class NormalizedMSELoss(nn.Module):
    # NOTE: SSI loss was not very effective for this task,
    # so simple minmax_normalize + MSE loss will be used.
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = F.mse_loss(normalize(input), normalize(target))
        return loss


class NormalizedPSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.psnr = PSNR()

    def forward(self, input, target):
        input = normalize(input)
        target = normalize(target)
        return self.psnr(input, target)


class DA3MonoEnv(I2IEnv):
    def __init__(self, model, criterion, eval_criterion=None):
        super().__init__(model, criterion, eval_criterion=eval_criterion)

    def print_eval_result(self, psnr_loss, file=sys.stdout):
        psnr = -psnr_loss
        print(f"Batch PSNR: {psnr}", file=file)


class DA3MonoTrainer(Trainer):
    def create_model(self):
        kwargs = {}
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = DA3MonoDataset(path.join(self.args.data_dir, "train"),
                                     size=self.args.size,
                                     training=True)
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=torch.utils.data.RandomSampler(dataset, num_samples=self.args.num_samples),
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            return loader
        else:
            dataset = DA3MonoDataset(path.join(self.args.data_dir, "eval"),
                                     size=self.args.size,
                                     training=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            return loader

    def create_env(self):
        eval_criterion = NormalizedPSNR().to(self.device)
        criterion = NormalizedMSELoss().to(self.device)
        return DA3MonoEnv(self.model, criterion, eval_criterion=eval_criterion)


def train(args):
    trainer = DA3MonoTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "iw3.da3mono",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="iw3.da3mono_disparity", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")
    parser.add_argument("--size", type=int, default=320, help="input size. other than 256, it only works with mlbw model")

    parser.set_defaults(
        batch_size=8,
        backward_step=8,
        optimizer="adamw_schedulefree",
        learning_rate=0.0003,
        max_epoch=400,
    )
    parser.set_defaults(handler=train)

    return parser
