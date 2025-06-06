from os import path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import create_model
from nunif.training.env import RGBPSNREnv
from nunif.training.trainer import Trainer
from nunif.modules.auxiliary_loss import AuxiliaryLoss
from nunif.modules.clamp_loss import ClampLoss
from .dataset import SBSDataset
from ... import models # noqa


class DeltaPenalty(nn.Module):
    def forward(self, input, dummy):
        # warp points(grid + delta) should be monotonically increasing
        N = 3
        penalty = 0
        for i in range(1, N):
            penalty = penalty + F.relu(input[:, :, :, :-i] - input[:, :, :, i:], inplace=True).mean()
        return penalty / N


class SBSEnv(RGBPSNREnv):
    def __init__(self, model, criterion, sampler):
        super().__init__(model, criterion)
        self.sampler = sampler

    def train_loss_hook(self, data, loss):
        super().train_loss_hook(data, loss)
        index = data[-1]
        self.sampler.update_losses(index, loss.item())

    def train_end(self):
        self.sampler.update_weights()
        return super().train_end()


class SBSTrainer(Trainer):
    def create_model(self):
        kwargs = {}
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        if self.args.symmetric:
            model.symmetric = True
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = self.model.i2i_offset
        if type == "train":
            dataset = SBSDataset(path.join(self.args.data_dir, "train"), model_offset,
                                 symmetric=self.args.symmetric, training=True)
            self.sampler = dataset.create_sampler(self.args.num_samples)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=self.sampler,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            return loader
        else:
            dataset = SBSDataset(path.join(self.args.data_dir, "eval"), model_offset,
                                 symmetric=self.args.symmetric, training=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            return loader

    def create_best_model_filename(self):
        return path.join(self.args.model_dir, f"{self.model.name}.left.pth")

    def create_checkpoint_filename(self):
        return path.join(self.args.model_dir, f"{self.model.name}.left.checkpoint.pth")

    def create_env(self):
        if self.args.loss == "l1":
            criterion = ClampLoss(nn.L1Loss()).to(self.device)
        elif self.args.loss == "l1_delta":
            criterion = AuxiliaryLoss(
                (ClampLoss(nn.L1Loss()), DeltaPenalty()),
                (1.0, 1.0)).to(self.device)
        elif self.args.loss == "aux_l1":
            criterion = AuxiliaryLoss(
                (ClampLoss(nn.L1Loss()), ClampLoss(nn.L1Loss()), DeltaPenalty()),
                (1.0, 0.5, 1.0)).to(self.device)
        return SBSEnv(self.model, criterion, self.sampler)


def train(args):
    if args.loss is None:
        if args.arch == "sbs.row_flow":
            args.loss = "l1"
        elif args.arch == "sbs.row_flow_v2":
            args.loss = "aux_l1"
        else:
            args.loss = "l1_delta"

    trainer = SBSTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "sbs",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="sbs.row_flow_v3", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")
    parser.add_argument("--loss", type=str, nargs="?", default=None, const="l1",
                        choices=["l1", "aux_l1", "l1_delta"], help="loss")
    parser.add_argument("--symmetric", action="store_true",
                        help="use symmetric warp training. only for `--arch sbs.row_flow_v3`")

    parser.set_defaults(
        batch_size=16,
        optimizer="adam",
        learning_rate=0.0001,
        scheduler="cosine",
        learning_rate_cycles=4,
        max_epoch=200,
        learning_rate_decay=0.98,
        learning_rate_decay_step=[1],
        momentum=0.9,
        weight_decay=0,
    )
    parser.set_defaults(handler=train)

    return parser
