from os import path
import torch
from .. import models  # noqa: F401
from . dataset import Waifu2xDataset
from nunif.training.trainer import Trainer
from nunif.training.env import LuminancePSNREnv
from nunif.models import create_model, get_model_config, call_model_method
from nunif.modules import (
    ClampLoss, LuminanceWeightedLoss, AuxiliaryLoss, LBPLoss, CharbonnierLoss,
    Alex11Loss
)
from nunif.logger import logger


def create_criterion(loss):
    if loss == "lbp":
        criterion = ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1)))
    elif loss == "lbp5":
        criterion = ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1, kernel_size=5)))
    elif loss == "alex11":
        criterion = ClampLoss(LuminanceWeightedLoss(Alex11Loss(in_channels=1)))
    elif loss == "y_charbonnier":
        criterion = ClampLoss(LuminanceWeightedLoss(CharbonnierLoss()))
    elif loss == "charbonnier":
        criterion = ClampLoss(CharbonnierLoss())
    elif loss == "aux_lbp":
        criterion = AuxiliaryLoss([
            ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1))),
            ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1))),
        ], weight=(1.0, 0.5))
    elif loss == "aux_alex11":
        criterion = AuxiliaryLoss([
            ClampLoss(LuminanceWeightedLoss(Alex11Loss(in_channels=1))),
            ClampLoss(LuminanceWeightedLoss(Alex11Loss(in_channels=1))),
        ], weights=(1.0, 0.5))
    elif loss == "aux_y_charbonnier":
        criterion = AuxiliaryLoss([
            ClampLoss(LuminanceWeightedLoss(CharbonnierLoss())),
            ClampLoss(LuminanceWeightedLoss(CharbonnierLoss()))],
            weight=(1.0, 0.5))
    elif loss == "aux_charbonnier":
        criterion = AuxiliaryLoss([
            ClampLoss(CharbonnierLoss()),
            ClampLoss(CharbonnierLoss())],
            weight=(1.0, 0.5))
    else:
        raise NotImplementedError()

    return criterion


# basic training


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
        dataset.set_hard_example(self.args.hard_example, self.args.hard_example_scale)

    def setup_model(self):
        if self.args.freeze and hasattr(self.model, "freeze"):
            call_model_method(self.model, "freeze")
            logger.debug("call model.freeze()")

    def create_model(self):
        kwargs = {"in_channels": 3, "out_channels": 3}
        if self.args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            kwargs["no_clip"] = True
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
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
                bicubic_only=self.args.b4b,
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                num_samples=self.args.num_samples,
                da_jpeg_p=self.args.da_jpeg_p,
                da_scale_p=self.args.da_scale_p,
                da_chshuf_p=self.args.da_chshuf_p,
                da_unsharpmask_p=self.args.da_unsharpmask_p,
                da_grayscale_p=self.args.da_grayscale_p,
                deblur=self.args.deblur,
                resize_blur_p=self.args.resize_blur_p,
                training=True,
            )
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size,
                worker_init_fn=dataset.worker_init,
                shuffle=False,
                pin_memory=True,
                sampler=dataset.sampler(),
                persistent_workers=True,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor,
                drop_last=True)
        elif type == "eval":
            dataset = Waifu2xDataset(
                input_dir=path.join(self.args.data_dir, "eval"),
                model_offset=model_offset,
                scale_factor=scale_factor,
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                deblur=self.args.deblur,
                training=False)
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size,
                worker_init_fn=dataset.worker_init,
                shuffle=False,
                persistent_workers=True,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor,
                drop_last=False)

    def create_env(self):
        criterion = create_criterion(self.args.loss).to(self.device)
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
