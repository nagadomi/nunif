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
from nunif.modules.auxiliary_loss import AuxiliaryLoss
from nunif.modules.clamp_loss import ClampLoss, clamp_loss
from nunif.modules.dct_loss import window_dct_loss, dct_loss
from nunif.modules.gaussian_filter import GaussianFilter2d
from .dataset import SBSDataset
from ... import models # noqa


class DeltaPenalty(nn.Module):
    def forward(self, grid, dummy):
        # warp points(grid + delta) should be monotonically increasing
        N = 3
        penalty = 0
        for i in range(1, N):
            penalty = penalty + F.relu(grid[:, :, :, :-i] - grid[:, :, :, i:], inplace=False).mean()
        return penalty / N


def delta_eval(grid):
    non_inc_count = int((grid[:, :, :, :-1] > grid[:, :, :, 1:]).sum())
    all_count = grid[:, :, :, 1:].numel()
    print(f"Non monotonically increasing = {non_inc_count} / {all_count}, ({non_inc_count/all_count})")


def l1_none(input, target):
    return F.l1_loss(input, target, reduction="none")


class MLBWLoss(nn.Module):
    def __init__(self, mask_weight):
        super().__init__()
        self.mask_weight = mask_weight
        self.blur = GaussianFilter2d(1, 3, padding=1)
        self.delta_penalty = DeltaPenalty()

    def forward(self, input, target):
        z, grid, *_ = input
        y, mask = target

        delta_penalty = self.delta_penalty(grid, None)
        if self.mask_weight > 0:
            mask = 1.0 - torch.clamp(mask + self.blur(mask), 0, 1) * self.mask_weight
            z = z * mask
            y = y * mask
            loss = (window_dct_loss(z, y, window_size=24) +
                    window_dct_loss(z, y, window_size=4) +
                    dct_loss(z, y)) * 0.3
        else:
            loss = (window_dct_loss(z, y, window_size=24) +
                    window_dct_loss(z, y, window_size=4) +
                    dct_loss(z, y)) * 0.3

        return loss + delta_penalty


class RowFlowV3Loss(nn.Module):
    def __init__(self, mask_weight):
        super().__init__()
        self.mask_weight = mask_weight
        self.blur = GaussianFilter2d(1, 3, padding=1)
        self.delta_penalty = DeltaPenalty()

    def _masked_loss(self, input, target, mask):
        mask = 1.0 - torch.clamp(mask + self.blur(mask), 0, 1) * self.mask_weight
        loss = (clamp_loss(input, target, loss_function=l1_none, min_value=0, max_value=1) * mask).mean()
        return loss

    def forward(self, input, target):
        z, grid = input
        y, mask = target

        delta_penalty = self.delta_penalty(grid, None)
        if self.mask_weight > 0:
            if mask.shape[1] == 1:
                loss = self._masked_loss(z, y, mask)
            else:
                # symmetric
                mask_l, mask_r = mask.chunk(2, dim=1)
                z_l, z_r = z.chunk(2, dim=1)
                y_l, y_r = y.chunk(2, dim=1)
                loss = (self._masked_loss(z_l, y_l, mask_l) + self._masked_loss(z_r, y_r, mask_r)) * 0.5
        else:
            loss = clamp_loss(z, y, loss_function=l1_none, min_value=0, max_value=1).mean()

        return loss + delta_penalty


class MaskedPSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.psnr = PSNR()

    def forward(self, input, target):
        y, mask = target
        mask = 1 - mask

        if mask.shape[1] == 1:
            return self.psnr(input * mask, y * mask)
        else:
            # symmetric
            z = input
            mask_l, mask_r = mask.chunk(2, dim=1)
            z_l, z_r = z.chunk(2, dim=1)
            y_l, y_r = y.chunk(2, dim=1)
            input = torch.cat((z_l * mask_l, z_r * mask_r), dim=1)
            target = torch.cat((y_l * mask_l, y_r * mask_r), dim=1)
            return self.psnr(input, target)


class SBSEnv(I2IEnv):
    def __init__(self, model, criterion, sampler, eval_criterion=None):
        super().__init__(model, criterion, eval_criterion=eval_criterion)
        self.sampler = sampler

    def train_loss_hook(self, data, loss):
        super().train_loss_hook(data, loss)
        if not self.trainer.args.disable_hard_example:
            index = data[-1]
            self.sampler.update_losses(index, loss.item())

    def train_end(self):
        if not self.trainer.args.disable_hard_example:
            self.sampler.update_weights()
        return super().train_end()

    def print_eval_result(self, psnr_loss, file=sys.stdout):
        psnr = -psnr_loss
        print(f"Batch RGB-PSNR: {psnr}", file=file)


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
            dataset = SBSDataset(path.join(self.args.data_dir, "train"),
                                 size=self.args.size, model_offset=model_offset,
                                 symmetric=self.args.symmetric, training=True,
                                 weak_convergence=self.args.weak_convergence)
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
            dataset = SBSDataset(path.join(self.args.data_dir, "eval"),
                                 size=self.args.size, model_offset=model_offset,
                                 symmetric=self.args.symmetric, training=False,
                                 weak_convergence=self.args.weak_convergence)
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
        eval_criterion = MaskedPSNR().to(self.device)
        if self.args.loss == "l1":
            criterion = ClampLoss(nn.L1Loss()).to(self.device)
        elif self.args.loss == "row_flow_v3":
            criterion = RowFlowV3Loss(mask_weight=self.args.mask_weight).to(self.device)
        elif self.args.loss == "mlbw":
            criterion = MLBWLoss(mask_weight=self.args.mask_weight).to(self.device)
        elif self.args.loss == "aux_l1":
            criterion = AuxiliaryLoss(
                (ClampLoss(nn.L1Loss()), ClampLoss(nn.L1Loss()), DeltaPenalty()),
                (1.0, 0.5, 1.0)).to(self.device)
        return SBSEnv(self.model, criterion, self.sampler, eval_criterion=eval_criterion)


def train(args):
    if args.arch == "sbs.row_flow":
        args.loss = "l1"
    elif args.arch == "sbs.row_flow_v2":
        args.loss = "aux_l1"
    elif args.arch == "sbs.row_flow_v3":
        args.loss = "row_flow_v3"
    elif args.arch.startswith("sbs.mlbw"):
        args.loss = "mlbw"

    trainer = SBSTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "sbs",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="sbs.mlbw_l2", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")
    parser.add_argument("--size", type=int, default=256, help="input size. other than 256, it only works with mlbw model")
    parser.add_argument("--mask-weight", type=float, default=0.75,
                        help="hole mask weight. 1 means completely excluded from the loss")
    parser.add_argument("--symmetric", action="store_true",
                        help="use symmetric warp training. only for `--arch sbs.row_flow_v3`")
    parser.add_argument("--disable-hard-example", action="store_true", help="Disable hard example mining")
    parser.add_argument("--weak-convergence", action="store_true", help="Use 0.375 <= convergence <= 0.625 only ")

    parser.set_defaults(
        batch_size=16,
        # optimizer="adamw_schedulefree",
        optimizer="adam",
        scheduler="cosine",
        num_samples=10000,
        max_epoch=200,
        learning_rate=0.0001,
        learning_rate_cosine_min=1e-8,
        learning_rate_cycles=5,
        weight_decay=0.001,
        weight_decay_end=0.01,
        eval_step=2,
    )
    parser.set_defaults(handler=train)

    return parser
