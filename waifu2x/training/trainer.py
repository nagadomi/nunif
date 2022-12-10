import argparse
from os import path
import torch
from torch import nn
from .. import models  # noqa: F401
from . dataset import Waifu2xScale2xDataset
from nunif.training.trainer import Trainer
from nunif.training.env import LuminancePSNREnv
from nunif.models import create_model, get_model_config
from nunif.modules import ClampLoss, LuminanceWeightedLoss, AuxiliaryLoss, LBPLoss


class Waifu2xEnv(LuminancePSNREnv):
    pass


class Waifu2xTrainer(Trainer):
    def create_model(self):
        kwargs = {"in_channels": 3, "out_channels": 3}
        if self.args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            kwargs["no_clip"] = True
        model = create_model(self.args.arch, **kwargs)
        if len(self.args.gpu) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = get_model_config(self.model, "i2i_offset")
        if self.args.method == "scale":
            if type == "train":
                dataset = Waifu2xScale2xDataset(
                    input_dir=path.join(self.args.data_dir, "train"),
                    model_offset=model_offset,
                    tile_size=self.args.size,
                    num_samples=self.args.num_samples
                )
                return torch.utils.data.DataLoader(
                    dataset, batch_size=self.args.minibatch_size,
                    worker_init_fn=dataset.worker_init,
                    shuffle=False,
                    pin_memory=True,
                    sampler=dataset.sampler(),
                    num_workers=self.args.num_workers,
                    drop_last=True)
            elif type == "eval":
                dataset = Waifu2xScale2xDataset(
                    input_dir=path.join(self.args.data_dir, "validation"),
                    model_offset=model_offset,
                    tile_size=self.args.size, validation=True)
                return torch.utils.data.DataLoader(
                    dataset, batch_size=self.args.minibatch_size,
                    worker_init_fn=dataset.worker_init,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    drop_last=False)
        else:
            raise NotImplementedError()

    def create_env(self):
        if self.args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            criterion = AuxiliaryLoss([
                ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1))),
                ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1)))],
                weight=(1.0, 0.5)).to(self.device)
        else:
            criterion = ClampLoss(LuminanceWeightedLoss(nn.HuberLoss(delta=0.3))).to(self.device)

        return Waifu2xEnv(self.model, criterion=criterion)

    def create_best_model_filename(self):
        if self.args.method == "scale":
            return path.join(self.args.model_dir, "scale2x.pth")
        else:
            raise NotImplementedError()

    def create_checkpoint_filename(self):
        if self.args.method == "scale":
            return path.join(self.args.model_dir, "scale2x.checkpoint.pth")
        else:
            raise NotImplementedError()


def train(args):
    if args.size % 4 != 0:
        raise ValueError("--size must be a multiple of 4")

    trainer = Waifu2xTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "waifu2x",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str, choices=["scale"], required=True,
                        help="waifu2x method")
    parser.add_argument("--arch", type=str,
                        choices=["waifu2x.cunet", "waifu2x.upcunet", "waifu2x.upconv_7", "waifu2x.vgg_7"],
                        required=True,
                        help="network arch")
    parser.add_argument("--size", type=int, default=104,
                        help="input size")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="number of samples for each epoch")

    parser.set_defaults(
        minibatch_size=8,
        optimizer="adam",
        learning_rate=0.0002,
        learning_rate_decay=0.995,
        learning_rate_decay_step=[1]
    )
    parser.set_defaults(handler=train)

    return parser
