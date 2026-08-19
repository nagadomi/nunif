import argparse

import torch

from nunif.training.trainer import Trainer

from .twoafc_dataset import TwoAFCDataset
from .twoafc_env import TwoAFCEnv


class LPIPSTrainer(Trainer):
    def create_dataloader(self, type):
        assert type in {"train", "eval"}
        if type == "train":
            dataset = TwoAFCDataset(self.args.data_dir, training=True)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.create_sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True,
            )
            return loader
        else:
            dataset = TwoAFCDataset(self.args.data_dir, training=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False,
            )
            return loader

    def create_env(self):
        return TwoAFCEnv(self.model)


def train(args):
    trainer = LPIPSTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "cliqa.lpips", parents=[default_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--arch", type=str, default="cliqa.lpips_repvgg", help="network arch")
    parser.add_argument("--num-samples", type=int, default=100000, help="number of samples for each epoch")

    parser.set_defaults(
        batch_size=8,
        warmup_epoch=1,
        optimizer="adam",
        learning_rate=1e-4,
        learning_rate_cosine_min=1e-5,
        scheduler="cosine",
        learning_rate_cycles=1,
        learning_rate_decay=0.95,
        learning_rate_decay_step=[1],
        max_epoch=10,
        momentum=0.9,
        weight_decay=0.0001,
        weight_decay_end=0.01,
        eval_step=1,
        disable_amp=False,
        seed=-1,
        save_epoch=True,
    )
    parser.set_defaults(handler=train)

    return parser
