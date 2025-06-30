# python train.py inpaint -i ./data/sr_dataset --model-dir models/light_inpaint
# python train.py inpaint -i ./data/sr_dataset --model-dir models/light_inpaint --resume --reset-state --learning-rate 3e-5 --ema-model
from os import path
import argparse
import torch
from nunif.models import create_model
from nunif.training.env import LuminancePSNREnv
from nunif.training.trainer import Trainer
from nunif.modules.transforms import DiffPairRandomTranslate
from nunif.modules.weighted_loss import WeightedLoss
from nunif.modules.dct_loss import DCTLoss
from nunif.modules.lbp_loss import LBPLoss
from .dataset import DepthAADataset
from ... import models # noqa


class DepthAATrainer(Trainer):
    def create_model(self):
        kwargs = {}
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = self.model.i2i_offset
        if type == "train":
            dataset = DepthAADataset(path.join(self.args.data_dir, "train"), model_offset,
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
            dataset = DepthAADataset(path.join(self.args.data_dir, "eval"), model_offset,
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
        elif self.args.loss == "l1":
            criterion = torch.nn.L1Loss()
        elif self.args.loss == "lbp":
            criterion = LBPLoss(in_channels=1, out_channels=64, kernel_size=3, num_kernels=1)

        return LuminancePSNREnv(self.model, criterion)


def train(args):
    trainer = DepthAATrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "iw3.depth_aa",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--arch", type=str, default="iw3.depth_aa", help="network arch")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="number of samples for each epoch")
    parser.add_argument("--loss", type=str, default="dct",
                        choices=["dct", "l1", "lbp"], help="loss")

    parser.set_defaults(
        batch_size=16,
        optimizer="adam",
        learning_rate=0.0001,
        learning_rate_cosine_min=1e-8,
        scheduler="cosine",
        learning_rate_cycles=5,
        max_epoch=200,
        eval_step=2,
        learning_rate_decay=0.99,
        learning_rate_decay_step=[1],
        momentum=0.9,
        weight_decay=0.001,
        disable_backup=True,
    )
    parser.set_defaults(handler=train)

    return parser
