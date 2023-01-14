import argparse
from os import path
import torch
from torch import nn
from .. import models  # noqa: F401
from . dataset import Waifu2xDataset
from nunif.training.trainer import Trainer
from nunif.training.env import LuminancePSNREnv
from nunif.models import create_model, get_model_config, get_model_names
from nunif.modules import ClampLoss, LuminanceWeightedLoss, AuxiliaryLoss, LBPLoss, CharbonnierLoss


class Waifu2xEnv(LuminancePSNREnv):
    def train_loss_hook(self, data, loss):
        super().train_loss_hook(data, loss)
        if self.trainer.args.hard_example == "none":
            return
        dataset = self.trainer.train_loader.dataset
        index = data[-1]
        dataset.update_hard_example_losses(index, loss.item())

    def train_end(self):
        super().train_end()
        if self.trainer.args.hard_example != "none":
            dataset = self.trainer.train_loader.dataset
            dataset.update_hard_example_weights()


class Waifu2xTrainer(Trainer):
    def setup(self):
        dataset = self.train_loader.dataset
        dataset.set_hard_example(self.args.hard_example)

    def create_model(self):
        kwargs = {"in_channels": 3, "out_channels": 3}
        if self.args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            kwargs["no_clip"] = True
        model = create_model(self.args.arch, **kwargs)
        if len(self.args.gpu) > 1:
            model = nn.DataParallel(model, device_ids=self.args.gpu)
        model = model.to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = get_model_config(self.model, "i2i_offset")
        if self.args.method in {"scale", "noise_scale"}:
            scale_factor = 2
        elif self.args.method in {"scale4x", "noise_scale4x"}:
            scale_factor = 4
        elif self.args.method in {"noise"}:
            scale_factor = 1
        else:
            raise NotImplementedError()

        if type == "train":
            dataset = Waifu2xDataset(
                input_dir=path.join(self.args.data_dir, "train"),
                model_offset=model_offset,
                scale_factor=scale_factor,
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                num_samples=self.args.num_samples,
                da_jpeg_p=self.args.da_jpeg_p,
                da_scale_p=self.args.da_scale_p,
                da_chshuf_p=self.args.da_chshuf_p,
                training=True,
            )
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size,
                worker_init_fn=dataset.worker_init,
                shuffle=False,
                pin_memory=True,
                sampler=dataset.sampler(),
                num_workers=self.args.num_workers,
                drop_last=True)
        elif type == "eval":
            dataset = Waifu2xDataset(
                input_dir=path.join(self.args.data_dir, "eval"),
                model_offset=model_offset,
                scale_factor=scale_factor,
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                training=False)
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size,
                worker_init_fn=dataset.worker_init,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False)

    def create_env(self):
        if self.args.loss == "lbp":
            criterion = ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1)))
        elif self.args.loss == "lbp5":
            criterion = ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1, kernel_size=5)))
        elif self.args.loss == "y_charbonnier":
            criterion = ClampLoss(LuminanceWeightedLoss(CharbonnierLoss())).to(self.device)
        elif self.args.loss == "charbonnier":
            criterion = ClampLoss(CharbonnierLoss()).to(self.device)
        elif self.args.loss == "aux_lbp":
            criterion = AuxiliaryLoss([
                ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1))),
                ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1))),
            ], weight=(1.0, 0.5)).to(self.device)
        elif self.args.loss == "aux_y_charbonnier":
            criterion = AuxiliaryLoss([
                ClampLoss(LuminanceWeightedLoss(CharbonnierLoss())),
                ClampLoss(LuminanceWeightedLoss(CharbonnierLoss()))],
                weight=(1.0, 0.5)).to(self.device)
        elif self.args.loss == "aux_charbonnier":
            criterion = AuxiliaryLoss([
                ClampLoss(CharbonnierLoss()),
                ClampLoss(CharbonnierLoss())],
                weight=(1.0, 0.5)).to(self.device)
        else:
            raise NotImplementedError()

        return Waifu2xEnv(self.model, criterion=criterion)

    def create_best_model_filename(self):
        if self.args.method == "scale":
            return path.join(self.args.model_dir, "scale2x.pth")
        elif self.args.method == "scale4x":
            return path.join(self.args.model_dir, "scale4x.pth")
        elif self.args.method == "noise_scale":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}_scale2x.pth")
        elif self.args.method == "noise_scale4x":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}_scale4x.pth")
        elif self.args.method == "noise":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}.pth")
        else:
            raise NotImplementedError()

    def create_checkpoint_filename(self):
        if self.args.method == "scale":
            return path.join(self.args.model_dir, "scale2x.checkpoint.pth")
        elif self.args.method == "scale4x":
            return path.join(self.args.model_dir, "scale4x.checkpoint.pth")
        elif self.args.method == "noise_scale":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}_scale2x.checkpoint.pth")
        elif self.args.method == "noise_scale4x":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}_scale4x.checkpoint.pth")
        elif self.args.method == "noise":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}.checkpoint.pth")
        else:
            raise NotImplementedError()


def train(args):
    ARCH_SWIN_UNET = {"waifu2x.swin_unet_1x",
                      "waifu2x.swin_unet_2x",
                      "waifu2x.swin_unet_4x",
                      "waifu2x.swinunet",
                      "waifu2x.upswinunet"}
    if args.size % 4 != 0:
        raise ValueError("--size must be a multiple of 4")
    if args.arch in ARCH_SWIN_UNET and args.size % 64 != 0:
        raise ValueError("--size must be a multiple of 64 for SWinUNet models")
    if args.method in {"noise", "noise_scale", "noise_scale4x"} and args.noise_level is None:
        raise ValueError("--noise-level is required for noise/noise_scale")

    if args.method in {"scale", "scale4x"}:
        # disable
        args.noise_level = -1

    if args.loss is None:
        if args.arch in {"waifu2x.vgg_7", "waifu2x.upconv_7"}:
            args.loss = "y_charbonnier"
        elif args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            args.loss = "aux_lbp"
        elif args.arch in {"waifu2x.swin_unet_1x", "waifu2x.swin_unet_2x",
                           "waifu2x.swinunet", "waifu2x.upswinunet"}:
            args.loss = "lbp"
        elif args.arch in {"waifu2x.swin_unet_4x"}:
            args.loss = "lbp5"
        else:
            args.loss = "y_charbonnier"

    trainer = Waifu2xTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "waifu2x",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    waifu2x_models = sorted([name for name in get_model_names() if name.startswith("waifu2x.")])

    parser.add_argument("--method", type=str,
                        choices=["scale", "noise_scale", "scale4x", "noise_scale4x", "noise"],
                        required=True,
                        help="waifu2x method")
    parser.add_argument("--arch", type=str,
                        choices=waifu2x_models,
                        required=True,
                        help="network arch")
    parser.add_argument("--style", type=str,
                        choices=["art", "photo"],
                        default="art",
                        help="image style used for jpeg noise level")
    parser.add_argument("--noise-level", type=int,
                        choices=[0, 1, 2, 3],
                        help="jpeg noise level for noise/noise_scale")
    parser.add_argument("--size", type=int, default=104,
                        help="input size")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="number of samples for each epoch")
    parser.add_argument("--loss", type=str,
                        choices=["lbp", "lbp5", "y_charbonnier", "charbonnier",
                                 "aux_lbp", "aux_y_charbonnier", "aux_charbonnier"],
                        help="loss function")
    parser.add_argument("--da-jpeg-p", type=float, default=0.0,
                        help="HQ JPEG(quality=92-99) data argumentation for gt image")
    parser.add_argument("--da-scale-p", type=float, default=0.25,
                        help="random downscale data argumentation for gt image")
    parser.add_argument("--da-chshuf-p", type=float, default=0.0,
                        help="random channel shuffle data argumentation for gt image")
    parser.add_argument("--hard-example", type=str, default="linear",
                        choices=["none", "linear", "top10", "top20"],
                        help="hard example mining for training data sampleing")

    parser.set_defaults(
        batch_size=8,
        optimizer="adam",
        learning_rate=0.0002,
        scheduler="cosine",
        learning_rate_cycles=5,
        learning_rate_decay=0.995,
        learning_rate_decay_step=[1],
        # for adamw
        weight_decay=0.001,
    )
    parser.set_defaults(handler=train)

    return parser
