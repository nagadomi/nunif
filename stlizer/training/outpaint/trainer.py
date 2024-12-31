# python train.py stlizer.outpaint -i ./data/sr_dataset --model-dir models/light_outpaint
# python train.py stlizer.outpaint -i ./data/sr_dataset --model-dir models/light_outpaint --resume --reset-state --learning-rate 3e-5 --max-epoch 40 --learning-rate-cycles 1 --ema-model
import os
from os import path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from nunif.models import create_model
from nunif.training.env import RGBPSNREnv
from nunif.training.trainer import Trainer
from nunif.modules.dct_loss import DCTLoss
from nunif.modules.auxiliary_loss import AuxiliaryLoss
from nunif.modules.multiscale_loss import MultiscaleLoss
from .dataset import OutpaintDataset
from ... import models # noqa


class OutpaintEnv(RGBPSNREnv):
    def __init__(self, model, criterion):
        super().__init__(model, criterion)

    def train_step(self, data):
        x, mask, y, *_ = data
        x, mask, y = self.to_device(x), self.to_device(mask), self.to_device(y)
        with self.autocast():
            z = self.model(x, mask)
            loss = self.criterion(z, y)
        if not torch.isnan(loss):
            self.sum_loss += loss.item()
            self.sum_step += 1
        return loss

    def eval_begin(self):
        super().eval_begin()
        self.eval_count = 0

    def eval_step(self, data):
        x, mask, y, *_ = data
        x, mask, y = self.to_device(x), self.to_device(mask), self.to_device(y)
        model = self.get_eval_model()
        with self.autocast():
            z = model(x, mask)
            loss = self.eval_criterion(z, y)
            self.eval_count += 1
            if self.eval_count % 20 == 0:
                self.save_eval(x, y, z, self.eval_count // 20)

        self.sum_loss += loss.item()
        self.sum_step += 1

    def save_eval(self, x, y, z, i):
        offset = (x.shape[2] - z.shape[2]) // 2
        x = F.pad(x, (-offset, ) * 4)
        x = torch.cat([x, z, y], dim=3)
        eval_output_dir = path.join(self.trainer.args.model_dir, "eval")
        os.makedirs(eval_output_dir, exist_ok=True)
        output_file = path.join(eval_output_dir, f"{i}.png")
        TF.to_pil_image(make_grid(x, nrow=1)).save(output_file)


class OutpaintTrainer(Trainer):
    def create_model(self):
        kwargs = {}
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = self.model.i2i_offset
        if type == "train":
            dataset = OutpaintDataset(
                path.join(self.args.data_dir, "train"),
                model_offset=model_offset,
                tile_size=self.args.size,
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
            dataset = OutpaintDataset(
                path.join(self.args.data_dir, "eval"),
                tile_size=self.args.size,
                model_offset=model_offset,
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
        if self.args.loss == "dct":
            criterion = AuxiliaryLoss([DCTLoss(clamp=True), nn.L1Loss()], weight=[1.0, 0.1])
        elif self.args.loss == "dctm":
            dct = DCTLoss(clamp=True)
            # 1x scale + 1/4 scale
            dctm = MultiscaleLoss(dct, scale_factors=(1, 4), weights=(0.5, 0.5), mode="avg")
            criterion = AuxiliaryLoss([dctm, dctm], weight=[1.0, 0.1])
        elif self.args.loss == "l1":
            criterion = AuxiliaryLoss([nn.L1Loss(), nn.L1Loss()], weight=[1.0, 0.1])

        return OutpaintEnv(self.model, criterion)


def train(args):
    trainer = OutpaintTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "stlizer.outpaint",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="stlizer.light_outpaint_v1", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")
    parser.add_argument("--loss", type=str, default="dctm",
                        choices=["dct", "dctm", "l1"], help="loss")
    parser.add_argument("--size", type=int, default=320, help="model input size")
    parser.set_defaults(
        batch_size=4,
        optimizer="adamw",
        learning_rate=0.0001,
        learning_rate_cosine_min=1e-6,
        scheduler="cosine",
        max_epoch=120,
        learning_rate_cycles=3,
        weight_decay=0.001,
        weight_decay_end=0.01,
        seed=71,  # fixed seed for fixed eval dataset
    )
    parser.set_defaults(handler=train)

    return parser
