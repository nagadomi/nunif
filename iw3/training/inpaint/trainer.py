# python train.py inpaint -i ./data/dataset --model-dir models/light_inpaint
import sys
import os
from os import path
import math
import time
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from nunif.models import create_model
from nunif.training.env import I2IEnv
from nunif.training.trainer import Trainer
from nunif.modules.weighted_loss import WeightedLoss
from nunif.modules.clamp_loss import ClampLoss
from nunif.modules.dct_loss import DCTLoss
from nunif.modules.lpips import LPIPSWith
from nunif.modules.dinov2 import DINOv2CosineWith
from nunif.modules.transforms import DiffPairRandomTranslate, DiffPairRandomRotate
from nunif.transforms import pair as TP
from nunif.modules.gan_loss import GANFakeMaskHingeLoss
from nunif.logger import logger
from .dataset import InpaintDataset
from ... import models as _m # noqa
from ...models.discriminator import L3ConditionalDiscriminator


def create_discriminator(discriminator, device_ids, device):
    if discriminator is None:
        return None
    elif discriminator == "l3v1c":
        model = create_model(L3ConditionalDiscriminator.name, device_ids=device_ids)
    else:
        model = create_model(discriminator, device_ids=device_ids)
    return model.to(device)


def to_dtype(x, dtype):
    if isinstance(x, (tuple, list)):
        return [xx.to(dtype) for xx in x]
    else:
        return x.to(dtype)


def diff_dequant_noise(inputs):
    # Remove 8bit conversion marks
    scale = (1.0 / 255.0) * 0.5
    noise = (torch.randn_like(inputs[0]) * 0.5).clamp(-1, 1) * scale
    return [input + noise for input in inputs]


def get_last_layer(model):
    if model.name == "inpaint.light_inpaint_v1":
        return model.to_image[-1].weight
    else:
        raise NotImplementedError()


def inf_loss():
    return float(-time.time() / 1000000000)


def ste_clamp(x, overshoot_scale=0.1):
    x_clamp = x.clamp(0, 1)
    x = x_clamp + (x - x_clamp) * overshoot_scale
    return x + x.clamp(0, 1).detach() - x.detach()


class InpaintEnv(I2IEnv):
    def __init__(self, model, criterion, discriminator=None):
        super().__init__(model, criterion=criterion, eval_criterion=criterion)
        self.discriminator = discriminator
        if discriminator:
            loss_weights = getattr(self.discriminator, "loss_weights", (1.0,))
            self.discriminator_criterion = GANFakeMaskHingeLoss(loss_weights=loss_weights).to(self.device)
        else:
            self.discriminator_criterion = None

        self.diff_aug = TP.RandomChoice([
            DiffPairRandomTranslate(size=8, padding_mode="reflection", expand=False, instance_random=False),
            DiffPairRandomRotate(angle=15, padding_mode="reflection", expand=False, instance_random=False),
            TP.Identity()], p=[0.33, 0.33, 0.33])

        self.adaptive_weight_ema = None
        self.epoch_iteration = 0

    def clear_loss(self):
        super().clear_loss()
        self.sum_loss = 0
        self.sum_p_loss = 0
        self.sum_d_weight = 0
        self.sum_g_loss = 0
        self.sum_d_loss = 0
        self.sum_step = 0

    def get_current_iteration(self):
        batch_iteration = (self.trainer.epoch - 1) * self.trainer.args.num_samples
        batch_iteration = batch_iteration // (self.trainer.args.batch_size * self.trainer.args.backward_step)
        iteration = batch_iteration + self.epoch_iteration // self.trainer.args.backward_step
        return iteration

    def get_generator_warmup_weight(self, k=4):
        t = self.get_current_iteration()
        n = self.trainer.args.generator_warmup_iteration
        if n == 0 or t > n:
            return 1.0

        x = t / n
        return (math.exp(k * x) - 1) / (math.exp(k) - 1)

    def train_begin(self):
        super().train_begin()
        self.epoch_iteration = 0
        if self.discriminator is not None:
            self.discriminator.train()

    def train_step(self, data):
        self.epoch_iteration += 1
        x, mask, y, *_ = data
        x, mask, y = self.to_device(x), self.to_device(mask), self.to_device(y)
        with self.autocast():
            if self.discriminator is None:
                x, mask = self.model.preprocess(x, mask)
                z = to_dtype(self.model(x, mask), x.dtype)
                if self.trainer.args.diff_aug:
                    z, y = self.diff_aug(z, y)
                loss = self.criterion(z, y)
                if not torch.isnan(loss):
                    self.sum_loss += loss.item()
                    self.sum_step += 1
            else:
                self.discriminator.requires_grad_(False)

                x, mask = self.model.preprocess(x, mask)
                z = to_dtype(self.model(x, mask), x.dtype)
                if self.trainer.args.diff_aug:
                    z, y = self.diff_aug(z, y)
                cond = y
                fake = z

                z_real = to_dtype(self.discriminator(ste_clamp(fake), cond, mask=None), fake.dtype)
                recon_loss = self.criterion(z, y)
                generator_loss = self.discriminator_criterion(z_real)

                self.sum_p_loss += recon_loss.item()
                self.sum_g_loss += generator_loss.item()

                self.discriminator.requires_grad_(True)
                fake, y = diff_dequant_noise((torch.clamp(fake.detach(), 0, 1), y))
                real = y
                z_fake, z_mask = to_dtype(self.discriminator(fake, cond, mask=mask), fake.dtype)
                z_real = to_dtype(self.discriminator(real, cond, mask=None), real.dtype)
                discriminator_loss = self.discriminator_criterion(z_real, z_fake, z_mask > 0)

                self.sum_d_loss += discriminator_loss.item()
                loss = (recon_loss, generator_loss, discriminator_loss)
                self.sum_step += 1

        return loss

    def calc_weight(self, recon_loss, generator_loss, grad_scaler):
        last_layer = get_last_layer(self.model)
        weight = self.calculate_adaptive_weight(
            recon_loss, generator_loss, last_layer, grad_scaler,
            min=1e-5,
            max=1.0,
            mode="norm",
            adaptive_weight=1.0 if self.adaptive_weight_ema is None else self.adaptive_weight_ema
        )
        weight_is_nan = math.isnan(weight)
        if not weight_is_nan:
            if self.adaptive_weight_ema is None:
                self.adaptive_weight_ema = weight
            else:
                alpha = 0.99
                self.adaptive_weight_ema = self.adaptive_weight_ema * alpha + weight * (1 - alpha)
            weight = self.adaptive_weight_ema
        elif self.adaptive_weight_ema is not None:
            weight = self.adaptive_weight_ema
        else:
            weight = 1.0  # inf

        if weight_is_nan:
            use_disc_loss = False
        else:
            use_disc_loss = True

        return weight, use_disc_loss

    def train_backward_step(self, loss, optimizers, grad_scalers, update):
        if self.discriminator is None:
            super().train_backward_step(loss, optimizers, grad_scalers, update)
        else:
            backward_step = self.trainer.args.backward_step
            recon_loss, generator_loss, d_loss = loss
            g_opt, d_opt = optimizers
            optimizers = []

            # update generator
            weight, use_disc_loss = self.calc_weight(recon_loss, generator_loss, grad_scalers[0])
            warmup_weight = self.get_generator_warmup_weight()
            if warmup_weight < 0.1:
                warmup_weight = 0.0
            if use_disc_loss:
                g_loss = (recon_loss + generator_loss * weight * self.trainer.args.discriminator_weight * warmup_weight)
            else:
                g_loss = recon_loss
            self.sum_d_weight += weight
            self.backward(g_loss, grad_scalers[0])
            optimizers.append((g_opt, grad_scalers[0]))

            logger.debug(
                (f"iteration: {self.get_current_iteration()}, "
                 f"recon: {round(recon_loss.item() * backward_step, 4)}, "
                 f"gen: {round(generator_loss.item() * backward_step, 4)}, "
                 f"disc: {round(d_loss.item() * backward_step, 4)}, "
                 f"weight: {round(weight, 6)}"
                 ) + (f", warmup weight: {round(warmup_weight, 4)}" if warmup_weight < 1 else "")
            )
            # update discriminator
            self.backward(d_loss, grad_scalers[1])
            optimizers.append((d_opt, grad_scalers[1]))

            if optimizers and update:
                for optimizer, grad_scaler in optimizers:
                    self.optimizer_step(optimizer, grad_scaler)

    def train_end(self):
        # show loss
        mean_loss = self.sum_loss / self.sum_step
        if self.discriminator is not None:
            mean_p_loss = self.sum_p_loss / self.sum_step
            mean_d_loss = self.sum_d_loss / self.sum_step
            mean_g_loss = self.sum_g_loss / self.sum_step
            mean_d_weight = self.sum_d_weight / self.sum_step
            print(f"loss: {round(mean_loss, 6)}, "
                  f"reconstruction loss: {round(mean_p_loss, 6)}, "
                  f"generator loss: {round(mean_g_loss, 6)}, "
                  f"discriminator loss: {round(mean_d_loss, 6)}, "
                  f"discriminator weight: {round(mean_d_weight, 6)}")
            mean_loss = mean_loss + mean_d_loss
        else:
            print(f"loss: {round(mean_loss, 6)}")

        return mean_loss

    def eval_begin(self):
        super().eval_begin()
        self.eval_count = 0

    def eval_step(self, data):
        x, mask, y, *_ = data
        x, mask, y = self.to_device(x), self.to_device(mask), self.to_device(y)
        model = self.get_eval_model()
        with self.autocast():
            x, mask = self.model.preprocess(x, mask)
            z = model(x, mask)
            loss = self.eval_criterion(z, y)
            self.eval_count += 1
            if self.eval_count % 10 == 0:
                self.save_eval(x, y, z, self.eval_count // 10)

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

    def eval_end(self, file=sys.stdout):
        if self.discriminator is not None:
            return inf_loss()

        mean_loss = self.sum_loss / self.sum_step
        self.print_eval_result(mean_loss, file=file)
        return mean_loss


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
                (
                    DCTLoss(window_size=4, clamp=True),
                    DCTLoss(window_size=24, clamp=True, random_instance_rotate=True),
                    DCTLoss(clamp=True, random_instance_rotate=True)
                ),
                weights=(0.2, 0.2, 0.6),
            )
        elif self.args.loss == "l1lpips":
            criterion = LPIPSWith(ClampLoss(torch.nn.L1Loss()), weight=0.4)
        elif self.args.loss == "l1dinov2":
            criterion = DINOv2CosineWith(ClampLoss(torch.nn.L1Loss()), weight=0.1)
        else:
            raise ValueError(f"{self.args.loss}")

        return InpaintEnv(self.model, criterion=criterion, discriminator=self.discriminator)

    def setup_model(self):
        self.discriminator = create_discriminator(self.args.discriminator, self.args.gpu, self.device)

    def create_optimizers(self):
        if self.discriminator is not None:
            g_opt = self.create_optimizer(self.model)
            d_opt = self.create_optimizer(self.discriminator)
            return g_opt, d_opt
        else:
            return super().create_optimizers()

    def create_grad_scalers(self):
        if self.discriminator is not None:
            return [self.create_grad_scaler(), self.create_grad_scaler()]
        else:
            return super().create_grad_scalers()


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
    parser.add_argument("--loss", type=str, default="dct", choices=["dct", "l1lpips", "l1dinov2"], help="loss")
    parser.add_argument("--discriminator", type=str, help="discriminator")
    parser.add_argument("--generator-warmup-iteration", type=int, default=500,
                        help=("warm-up iterations for the discriminator loss affecting the generator."))
    parser.add_argument("--discriminator-weight", type=float, default=1.0,
                        help="discriminator loss weight")
    parser.add_argument("--diff-aug", action="store_true",
                        help="Use differentiable transforms for reconstruction loss and discriminator")

    parser.set_defaults(
        batch_size=16,
        optimizer="adamw",
        learning_rate=0.0001,
        learning_rate_cosine_min=1e-8,
        scheduler="cosine_wd",
        learning_rate_cycles=5,
        max_epoch=200,
        learning_rate_decay=0.99,
        learning_rate_decay_step=[1],
        momentum=0.9,
        weight_decay=0.001,
        weight_decay_end=0.01,
        eval_step=4,
    )
    parser.set_defaults(handler=train)

    return parser
