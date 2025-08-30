# python train.py inpaint -i ./data/dataset --model-dir models/light_inpaint
import os
from os import path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from nunif.models import create_model
from nunif.training.env import I2IEnv
from nunif.training.trainer import Trainer
from nunif.modules.transforms import DiffPairRandomTranslate
from nunif.modules.weighted_loss import WeightedLoss
from nunif.modules.clamp_loss import ClampLoss
from nunif.modules.dct_loss import DCTLoss
from nunif.modules.lpips import LPIPSWith
from nunif.modules.dinov2 import DINOv2PoolWith
from .dataset import InpaintDataset
from ... import models # noqa


class InpaintEnv(I2IEnv):
    def __init__(self, model, criterion):
        super().__init__(model, criterion=criterion, eval_criterion=criterion)

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


class InpaintTrainer(Trainer):
    def create_model(self):
        kwargs = {}
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = self.model.i2i_offset
        if type == "train":
            dataset = InpaintDataset(path.join(self.args.data_dir, "train"), model_offset,
                                     training=True)
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=torch.utils.data.RandomSampler(dataset, num_samples=self.args.num_samples),
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader
        else:
            dataset = InpaintDataset(path.join(self.args.data_dir, "eval"), model_offset,
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
            criterion = WeightedLoss(
                (DCTLoss(window_size=4, clamp=True),
                 DCTLoss(window_size=24, clamp=True, random_instance_rotate=True),
                 DCTLoss(clamp=True, random_instance_rotate=True)),
                weights=(0.2, 0.2, 0.6),
                preprocess_pair=DiffPairRandomTranslate(size=12, padding_mode="zeros", expand=True, instance_random=True))
        elif self.args.loss == "l1lpips":
            criterion = LPIPSWith(ClampLoss(torch.nn.L1Loss()), weight=0.4)
        elif self.args.loss == "l1dinov2":
            criterion = DINOv2PoolWith(ClampLoss(torch.nn.L1Loss()), weight=1.0)
        else:
            raise ValueError(f"{self.args.loss}")

        return InpaintEnv(self.model, criterion)


def train(args):
    trainer = InpaintTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "inpaint",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="inpaint.light_inpaint_v1", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")
    parser.add_argument("--loss", type=str, default="l1dinov2", choices=["dct", "l1lpips", "l1dinov2"], help="loss")

    parser.set_defaults(
        batch_size=16,
        optimizer="adam",
        learning_rate=0.0001,
        learning_rate_cosine_min=1e-8,
        scheduler="cosine",
        learning_rate_cycles=5,
        max_epoch=200,
        learning_rate_decay=0.99,
        learning_rate_decay_step=[1],
        momentum=0.9,
        weight_decay=0.001,
    )
    parser.set_defaults(handler=train)

    return parser
