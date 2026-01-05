from os import path
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import create_model
from nunif.training.env import I2IEnv
from nunif.modules.psnr import PSNRPerImage
from nunif.training.trainer import Trainer
from .dataset import DSODDataset
from ... import models # noqa


def normalize(x):
    B, C, H, W = x.shape
    x_flat = x.reshape(B, -1)
    min_value = x_flat.min(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
    max_value = x_flat.max(dim=1, keepdim=True)[0].reshape(B, 1, 1, 1)
    return (x - min_value) / ((max_value - min_value) + 1e-6)


class DSODEnv(I2IEnv):
    def __init__(self, model, criterion, eval_criterion=None):
        super().__init__(model, criterion, eval_criterion=eval_criterion)

    def print_eval_result(self, psnr_loss, file=sys.stdout):
        psnr = -psnr_loss
        print(f"PSNR: {psnr}", file=file)


class MultiBCEWithLogitsLoss(nn.Module):
    def __call__(self, input, target):
        loss = 0
        for i, x in enumerate(input):
            if x.shape[-2:] != target.shape[-2:]:
                x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", antialias=False, align_corners=False)
            weight = 1 / len(input)  # 1 / (i + 1)
            loss = loss + F.binary_cross_entropy_with_logits(x, target) * weight
        return loss


class DSODTrainer(Trainer):
    def create_model(self):
        kwargs = {}
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            depth_root_dir = path.join(self.args.data_dir, "DUTS-TR", "depth")
            mask_dir = path.join(self.args.data_dir, "DUTS-TR", "DUTS-TR-Mask")
            dataset = DSODDataset(depth_root_dir,
                                  mask_dir,
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
            depth_root_dir = path.join(self.args.data_dir, "DUTS-TE", "depth")
            mask_dir = path.join(self.args.data_dir, "DUTS-TE", "DUTS-TE-Mask")

            dataset = DSODDataset(depth_root_dir,
                                  mask_dir,
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
        eval_criterion = PSNRPerImage().to(self.device)
        criterion = MultiBCEWithLogitsLoss()
        return DSODEnv(self.model, criterion, eval_criterion=eval_criterion)


def train(args):
    trainer = DSODTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "dsod",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="iw3.dsod_v1", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")
    parser.add_argument("--size", type=int, default=192, help="input size")

    parser.set_defaults(
        batch_size=16,
        optimizer="adam",
        scheduler="step",
        learning_rate=1e-4,
        max_epoch=200,
        eval_step=4,
    )
    parser.set_defaults(handler=train)

    return parser
