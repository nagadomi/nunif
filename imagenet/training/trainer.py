import argparse
import torch
from torch import nn
from nunif.training.env import SoftmaxEnv
from nunif.training.trainer import Trainer
from nunif.models import create_model as nunif_create_model
from .dataset import ImageNetDataset


class ImageNetTrainer(Trainer):
    def create_model(self):
        kwargs = {}
        if self.args.pretrained:
            kwargs = {"pretrained": True}
        model = nunif_create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = ImageNetDataset(
                self.args.data_dir, split="train",
                resize=self.args.resize, size=self.args.size,
                norm=self.args.norm,
                resize_mode=self.args.resize_mode)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                persistent_workers=True,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor,
                drop_last=True)
            return loader
        else:
            dataset = ImageNetDataset(
                self.args.data_dir, split="val",
                resize=self.args.resize, size=self.args.size,
                norm=self.args.norm,
                resize_mode=self.args.resize_mode)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor,
                drop_last=False)
            return loader

    def create_env(self):
        return SoftmaxEnv(self.model, eval_tta=False, max_print_class=-1)


def train(args):
    trainer = ImageNetTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "imagenet",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str,
                        required=True,
                        help="network arch")
    parser.add_argument("--resize", type=int, default=256,
                        help="resize size")
    parser.add_argument("--resize-mode", type=str, choices=["resize", "reflect"], default="reflect",
                        help=("resize mode. "
                              "`resize`: just resize, "
                              "`reflect`: resize with preserving aspect ratio, reflection pad the smaller side"))
    parser.add_argument("--size", type=int, default=224,
                        help="input size")
    parser.add_argument("--num-samples", type=int, default=1_281_167,
                        help="number of samples for each epoch")
    parser.add_argument("--norm", type=str, default="imagenet",
                        choices=["none", "imagenet", "gcn", "center"],
                        help="input normalization mode")
    parser.add_argument("--pretrained", action="store_true",
                        help="load pretrained weights for torchvision models")

    parser.set_defaults(
        batch_size=128,
        optimizer="adam",
        learning_rate=0.01,
        scheduler="step",
        learning_rate_decay=0.1,
        learning_rate_decay_step=[30],
        max_epoch=90,
        # for sgd
        momentum=0.9,
        weight_decay=0,
    )
    parser.set_defaults(handler=train)

    return parser
