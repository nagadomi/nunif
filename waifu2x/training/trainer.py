from os import path
import sys
from time import time
import argparse
import torch
import torch.nn.functional as F
from .dataset import Waifu2xDataset
from ..models import discriminator as _discriminator  # noqa
from nunif.training.sampler import MiningMethod
from nunif.training.trainer import Trainer
from nunif.training.env import LuminancePSNREnv
from nunif.models import (
    create_model,
    load_model, save_model,
    get_model_names
)
from nunif.modules import (
    ClampLoss, LuminanceWeightedLoss,
    AuxiliaryLoss,
    CharbonnierLoss,
    Alex11Loss,
    MultiscaleLoss,
)
from nunif.modules.gan_loss import GANHingeLoss, GANBCELoss, GANSoftplusLoss, GANHingeClampLoss
from nunif.modules.local_std_mask import local_std_mask
from nunif.modules.lbp_loss import YLBP, YRGBL1LBP, YRGBLBP, YRGBFlatLBP
from nunif.modules.fft_loss import YRGBL1FFTGradientLoss
from nunif.modules.lpips import LPIPSWith
from nunif.modules.weighted_loss import WeightedLoss
from nunif.modules.dct_loss import DCTLoss
from nunif.modules.dinov2 import DINOv2CosineWith
from nunif.modules.identity_loss import IdentityLoss
from nunif.modules.transforms import DiffPairRandomTranslate, DiffPairRandomRotate, DiffPairRandomDownsample
from nunif.transforms import pair as TP
from nunif.logger import logger
import random
import math


# basic training


# loss


def _dctirm():
    return WeightedLoss(
        (DCTLoss(window_size=4, clamp=True),
         DCTLoss(window_size=24, clamp=True, random_instance_rotate=True),
         DCTLoss(clamp=True, random_instance_rotate=True)),
        weights=(0.2, 0.2, 0.6),
        preprocess_pair=DiffPairRandomTranslate(size=12, padding_mode="zeros", expand=True, instance_random=True))


LOSS_FUNCTIONS = {
    "l1": lambda: ClampLoss(torch.nn.L1Loss()),
    "y_l1": lambda: ClampLoss(LuminanceWeightedLoss(torch.nn.L1Loss())),
    "charbonnier": lambda: ClampLoss(CharbonnierLoss()),
    "y_charbonnier": lambda: ClampLoss(LuminanceWeightedLoss(CharbonnierLoss())),
    "lbp": lambda: YLBP(),
    "lbpm": lambda: MultiscaleLoss(YLBP(), mode="avg"),
    "lbp5": lambda: YLBP(kernel_size=5),
    "lbp5m": lambda: MultiscaleLoss(YLBP(kernel_size=5), mode="avg"),

    "yrgb_l1lbp5": lambda: YRGBL1LBP(kernel_size=5, weight=0.4),
    "yrgb_flatlbp5": lambda: YRGBFlatLBP(kernel_size=5, weight=0.4),
    "yrgb_lbp5": lambda: YRGBLBP(kernel_size=5),
    "yrgb_lbp": lambda: YRGBLBP(kernel_size=3),

    "alex11": lambda: ClampLoss(LuminanceWeightedLoss(Alex11Loss(in_channels=1))),
    "y_l1fftgrad": lambda: YRGBL1FFTGradientLoss(fft_weight=0.1, grad_weight=0.1, diag=False),

    "dct": lambda: DCTLoss(clamp=True),
    "dctirm": lambda: _dctirm(),
    "dctir24": lambda: WeightedLoss(
        (DCTLoss(window_size=24, clamp=True, random_rotate=True, overlap=True),),
        weights=(1.0,),
        preprocess_pair=DiffPairRandomTranslate(size=12, padding_mode="zeros", expand=True, instance_random=True)),

    "aux_lbp": lambda: AuxiliaryLoss((YLBP(), YLBP()), weight=(1.0, 0.5)),
    "aux_alex11": lambda: AuxiliaryLoss((
        ClampLoss(LuminanceWeightedLoss(Alex11Loss(in_channels=1))),
        ClampLoss(LuminanceWeightedLoss(Alex11Loss(in_channels=1)))), weights=(1.0, 0.5)),
    "aux_charbonnier": lambda: AuxiliaryLoss((ClampLoss(CharbonnierLoss()), ClampLoss(CharbonnierLoss())), weight=(1.0, 0.5)),
    "aux_y_charbonnier": lambda: AuxiliaryLoss((
        ClampLoss(LuminanceWeightedLoss(CharbonnierLoss())),
        ClampLoss(LuminanceWeightedLoss(CharbonnierLoss()))), weight=(1.0, 0.5)),

    # weight=0.1, gradient norm is about the same as L1Loss.
    "l1lpips": lambda: LPIPSWith(ClampLoss(torch.nn.L1Loss()), weight=0.4),
    "l1lpips_std_mask": lambda: LPIPSWith(ClampLoss(torch.nn.L1Loss()), weight=0.4, std_mask=True),

    "yrgb_lbp_dinov2": lambda: DINOv2CosineWith(YRGBLBP(kernel_size=3), weight=2.0),
    "yrgb_lbp_dinov2_01": lambda: DINOv2CosineWith(YRGBLBP(kernel_size=3), weight=0.2),

    "aux_lbp_ident": lambda: AuxiliaryLoss((YLBP(), IdentityLoss()), weight=(1.0, 1.0)),
    # loss is computed in model.forward()
    "ident": lambda: IdentityLoss(),
}


def diff_dequant_noise(inputs):
    # Remove 8bit conversion marks
    scale = (1.0 / 255.0) * 0.5
    noise = (torch.randn_like(inputs[0]) * 0.5).clamp(-1, 1) * scale
    return [input + noise for input in inputs]


def diff_random_noise(inputs, p=0.1, strength=0.01):
    # NOTE: only applies to the discriminator input.
    if random.uniform(0.0, 1.0) < p:
        base = inputs[0]
        B, C, H, W = base.shape
        noise1x = torch.randn_like(base)
        if random.uniform(0., 1.) < 0.5:
            noise2x = torch.randn((B, C, H // 2, W // 2), dtype=base.dtype, device=base.device)
            noise2x = torch.nn.functional.interpolate(noise2x, size=(H, W), mode="nearest")
            noise = ((noise1x + noise2x) * (strength / 2.0)).detach()
        else:
            noise = (noise1x * strength).detach()
        return [input + noise for input in inputs]
    else:
        return diff_dequant_noise(inputs)


def create_criterion(loss):
    if loss in LOSS_FUNCTIONS:
        criterion = LOSS_FUNCTIONS[loss]()
    else:
        raise NotImplementedError(loss)

    return criterion


def create_discriminator(discriminator, device_ids, device):
    if discriminator is None:
        return None
    elif discriminator == "l3v1c":
        model = create_model("waifu2x.l3v1_conditional_discriminator", device_ids=device_ids)
    elif discriminator == "l3v1ec":
        model = create_model("waifu2x.l3v1_ensemble_conditional_discriminator", device_ids=device_ids)
    elif discriminator == "u3c":
        model = create_model("waifu2x.u3_conditional_discriminator", device_ids=device_ids)
    elif path.exists(discriminator):
        model, _ = load_model(discriminator, device_ids=device_ids)
    else:
        model = create_model(discriminator, device_ids=device_ids)
    return model.to(device)


def get_last_layer(model):
    if model.name in {"waifu2x.swin_unet_1x",
                      "waifu2x.swin_unet_2x",
                      "waifu2x.swin_unet_4x",
                      "waifu2x.swin_unet_8x",
                      }:
        return model.unet.to_image.proj.weight
    elif model.name in {"waifu2x.swin_unet_v2_4x",
                        "waifu2x.swin_unet_v2_1x",
                        "waifu2x.swin_unet_v2_2x",
                        "waifu2x.swin_unet_v2_1xs",
                        }:
        return model.unet.to_residual_image.proj.weight
    elif model.name in {"waifu2x.cunet", "waifu2x.upcunet"}:
        return model.unet2.conv_bottom.weight
    elif model.name in {"waifu2x.upconv_7", "waifu2x.vgg_7"}:
        return model.net[-1].weight
    else:
        raise NotImplementedError()


def inf_loss():
    return float(-time() / 1000000000)


def fit_size(z, y):
    if isinstance(z, (tuple, list)):
        if z[0].shape[2] != y.shape[2] or z[0].shape[3] != y.shape[3]:
            pad_h = (y.shape[2] - z[0].shape[2]) // 2
            pad_w = (y.shape[3] - z[0].shape[3]) // 2
            assert pad_h >= 0 or pad_w >= 0
            y = torch.nn.functional.pad(y, (-pad_w, -pad_w, -pad_h, -pad_h))
    else:
        if z.shape[2] != y.shape[2] or z.shape[3] != y.shape[3]:
            pad_h = (y.shape[2] - z.shape[2]) // 2
            pad_w = (y.shape[3] - z.shape[3]) // 2
            assert pad_h >= 0 or pad_w >= 0
            y = torch.nn.functional.pad(y, (-pad_w, -pad_w, -pad_h, -pad_h))

    return z, y


def to_dtype(x, dtype):
    if isinstance(x, (tuple, list)):
        return [xx.to(dtype) for xx in x]
    else:
        return x.to(dtype)


def ste_clamp(x, overshoot_scale=0.1):
    # scale down the overshoot value
    # this prevents NaN loss
    x_clamp = x.clamp(0, 1)
    x = x_clamp + (x - x_clamp) * overshoot_scale
    # straight through estimator
    return x + x.clamp(0, 1).detach() - x.detach()


class Waifu2xEnv(LuminancePSNREnv):
    def __init__(self, model, criterion,
                 discriminator,
                 discriminator_criterion,
                 sampler, use_diff_aug=False, use_diff_aug_downsample=False, use_diff_aug_noise=False,
                 adaptive_weight_ema=None):
        super().__init__(model, criterion)
        self.discriminator = discriminator
        self.discriminator_criterion = discriminator_criterion
        self.adaptive_weight_ema = adaptive_weight_ema
        self.sampler = sampler
        self.use_diff_aug = use_diff_aug
        self.use_diff_aug_noise = use_diff_aug_noise
        self.epoch_iteration = 0
        if use_diff_aug:
            if use_diff_aug_downsample:
                self.diff_aug = TP.RandomChoice([
                    DiffPairRandomTranslate(size=8, padding_mode="reflection", expand=False, instance_random=False),
                    DiffPairRandomRotate(angle=15, padding_mode="reflection", expand=False, instance_random=False),
                    DiffPairRandomDownsample(scale_factor_min=0.5, scale_factor_max=0.5),
                    TP.Identity()], p=[0.25, 0.25, 0.25, 0.25])
            else:
                self.diff_aug = TP.RandomChoice([
                    DiffPairRandomTranslate(size=8, padding_mode="reflection", expand=False, instance_random=False),
                    DiffPairRandomRotate(angle=15, padding_mode="reflection", expand=False, instance_random=False),
                    TP.Identity()], p=[0.25, 0.25, 0.5])

        else:
            self.diff_aug = TP.Identity()

    def train_loss_hook(self, data, loss):
        super().train_loss_hook(data, loss)
        if self.trainer.args.hard_example == "none":
            return
        if isinstance(loss, (list, tuple)):
            if any(math.isnan(val) for val in loss):
                return
        else:
            if math.isnan(loss):
                return

        index = data[-1]
        if self.discriminator is None:
            self.sampler.update_losses(index, loss.item())
        else:
            recon_loss, generator_loss, d_loss = loss
            if not self.trainer.args.discriminator_only:
                self.sampler.update_losses(index, recon_loss.item())

    def get_scale_factor(self):
        scale_factor = self.model.i2i_scale
        return scale_factor

    @staticmethod
    def gen_fake_input(fake, real, cond, p):
        inputs = []
        conditions = []
        false_count = 0
        for i in range(cond.shape[0]):
            use_fake_condition = random.uniform(0, 1) < p
            if not use_fake_condition:
                # fake + true condition
                inputs.append(fake[i: i + 1])
                conditions.append(cond[i: i + 1])
                continue

            # real + false condition
            false_count += 1
            method = random.choice([0, 1, 2, 3])
            if method == 0:
                # flip
                axis = random.choice([-2, -1] + ([0, 0] if cond.shape[0] > 1 else []))
                if axis == 0:
                    batch_index = random.choice([j for j in range(cond.shape[0]) if j != i])
                    false_condition = cond[batch_index:batch_index + 1]
                else:
                    false_condition = torch.flip(cond[i:i + 1], dims=(axis,))
            elif method == 1:
                # shift
                shift_w = random.randint(4, 18) * random.choice([-1, 1])
                shift_h = random.randint(4, 18) * random.choice([-1, 1])
                false_condition = F.pad(cond[i:i + 1], (shift_w, -shift_w, shift_h, -shift_h), mode="reflect")
            elif method == 2:
                # color
                scale_r = random.uniform(0.6, 0.9)
                scale_g = random.uniform(0.6, 0.9)
                scale_b = random.uniform(0.6, 0.9)
                bias_r = 1 - scale_r
                bias_g = 1 - scale_g
                bias_b = 1 - scale_b
                scale = torch.tensor([scale_r, scale_g, scale_b], dtype=cond.dtype, device=cond.device).view(1, 3, 1, 1)
                bias = torch.tensor([bias_r, bias_g, bias_b], dtype=cond.dtype, device=cond.device).view(1, 3, 1, 1)
                if random.choice([True, False]):
                    false_condition = cond[i:i + 1] * scale + bias
                else:
                    false_condition = (cond[i:i + 1] + bias) * scale
            elif method == 3:
                # more blur (too high res)
                scale_factor = random.uniform(0.2, 0.6)
                false_condition = F.interpolate(cond[i:i + 1], scale_factor=scale_factor,
                                                mode="bilinear", align_corners=True, antialias=True)
                false_condition = F.interpolate(false_condition, size=cond.shape[-2:],
                                                mode="bilinear", align_corners=True)

            inputs.append(real[i: i + 1])
            conditions.append(false_condition)

        condition = torch.cat(conditions, dim=0)
        input = torch.cat(inputs, dim=0)

        assert condition.shape == cond.shape
        assert input.shape == fake.shape == real.shape
        # print(f"false_condition {false_count}/{cond.shape[0]}")

        return input, condition

    @staticmethod
    def gen_cond(x, y, scale_factor, request_x_condition):
        if request_x_condition:
            unpad = (x.shape[-1] - (y.shape[-1] // scale_factor)) // 2
            cond = F.pad(x, (-unpad,) * 4)
            cond = F.interpolate(cond, scale_factor=scale_factor, mode="bilinear", align_corners=True)
        else:
            cond = y
        return cond

    def clear_loss(self):
        super().clear_loss()
        self.sum_p_loss = 0
        self.sum_g_loss = 0
        self.sum_d_loss = 0
        self.sum_d_weight = 0
        self.sum_psnr = 0

    def train_begin(self):
        super().train_begin()
        self.epoch_iteration = 0
        if self.discriminator is not None:
            self.discriminator.train()
            if self.trainer.args.discriminator_only:
                self.model.eval()

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

    def train_step(self, data):
        self.epoch_iteration += 1

        if not self.trainer.args.privilege:
            x, y, *_ = data
            privilege = None
        else:
            x, y, privilege, *_ = data

        x, y = self.to_device(x), self.to_device(y)
        scale_factor = self.get_scale_factor()

        with self.autocast():
            if self.discriminator is None:
                if not self.trainer.args.privilege:
                    z = to_dtype(self.model(x), x.dtype)
                    z, y = fit_size(z, y)
                else:
                    z = to_dtype(self.model(x, self.to_device(privilege)), x.dtype)
                    z, y = fit_size(z, y)
                if isinstance(z, (list, tuple)) and self.use_diff_aug:
                    raise ValueError(f"--diff-aug does not support {self.model.name}")
                z, y = self.diff_aug(z, y)
                loss = self.criterion(z, y)
                self.sum_loss += loss.item()
            else:
                if not self.trainer.args.discriminator_only:
                    # generator (sr) step
                    if hasattr(self.discriminator, "round"):
                        self.discriminator.round()
                    self.discriminator.requires_grad_(False)
                    if not self.trainer.args.privilege:
                        z = to_dtype(self.model(x), x.dtype)
                        z, y = fit_size(z, y)
                    else:
                        z = to_dtype(self.model(x, self.to_device(privilege)), x.dtype)
                        z, y = fit_size(z, y)

                    cond = self.gen_cond(
                        x, y,
                        scale_factor=scale_factor,
                        request_x_condition=getattr(self.discriminator, "request_x_condition", False))
                    if isinstance(z, (list, tuple)):
                        # NOTE: models using auxiliary loss return tuple.
                        #       first element is SR result.
                        if self.use_diff_aug:
                            raise ValueError(f"--diff-aug does not support {self.model.name}")
                        fake = z[0]
                    else:
                        z, y = self.diff_aug(z, torch.cat([y, cond], dim=1))
                        y, cond = y.chunk(2, dim=1)
                        fake = z

                    z_real = to_dtype(self.discriminator(ste_clamp(fake), cond, scale_factor), fake.dtype)
                    recon_loss = self.criterion(z, y)
                    generator_loss = self.discriminator_criterion(z_real)
                    self.sum_p_loss += recon_loss.item()
                    self.sum_g_loss += generator_loss.item()

                    # loss weight will be recalculated later,
                    # but multiplied by 10 here to reduce the gap.
                    # (gradient norm of generator_loss is 10-100x larger than recon_loss)
                    recon_loss = recon_loss * self.trainer.args.reconstruction_loss_scale
                else:
                    recon_loss = generator_loss = torch.zeros(1, dtype=x.dtype, device=x.device)
                    with torch.inference_mode():
                        z = to_dtype(self.model(x), x.dtype)
                        z, y = fit_size(z, y)
                        fake = z[0] if isinstance(z, (list, tuple)) else z
                    cond = self.gen_cond(
                        x, y,
                        scale_factor=scale_factor,
                        request_x_condition=getattr(self.discriminator, "request_x_condition", False))
                    z, y = self.diff_aug(z, torch.cat([y, cond], dim=1))
                    y, cond = y.chunk(2, dim=1)

                # discriminator step
                self.discriminator.requires_grad_(True)
                fake, y, cond = diff_random_noise((torch.clamp(fake.detach(), 0, 1), y, cond),
                                                  p=0.1 if self.use_diff_aug_noise else 0)
                real = y
                real_cond = fake_cond = cond
                request_false_condition = getattr(self.discriminator, "request_false_condition", False)
                fake_input, fake_cond = self.gen_fake_input(
                    fake, real, cond,
                    p=0.25 if request_false_condition else 0
                )
                z_fake = self.discriminator(fake_input, fake_cond, scale_factor)
                z_real = self.discriminator(real, real_cond, scale_factor)
                z_fake = to_dtype(z_fake, fake.dtype)
                z_real = to_dtype(z_real, y.dtype)
                discriminator_loss = self.discriminator_criterion(z_real, z_fake)

                self.sum_d_loss += discriminator_loss.item()
                loss = (recon_loss, generator_loss, discriminator_loss)

        self.sum_step += 1
        return loss

    def train_backward_step(self, loss, optimizers, grad_scalers, update):
        if self.discriminator is None:
            super().train_backward_step(loss, optimizers, grad_scalers, update)
        else:
            backward_step = self.trainer.args.backward_step
            recon_loss, generator_loss, d_loss = loss
            g_opt, d_opt = optimizers
            optimizers = []

            # update generator
            if not self.trainer.args.discriminator_only:
                last_layer = get_last_layer(self.model)
                weight = self.calculate_adaptive_weight(
                    recon_loss, generator_loss, last_layer, grad_scalers[0],
                    min=self.trainer.args.discriminator_adaptive_weight_min,
                    max=self.trainer.args.discriminator_adaptive_weight_max,
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

                warmup_weight = self.get_generator_warmup_weight()
                if use_disc_loss:
                    g_loss = (recon_loss + generator_loss * weight * self.trainer.args.discriminator_weight * warmup_weight)
                else:
                    g_loss = recon_loss
                self.sum_loss += g_loss.item()
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
        # update sampler
        if self.trainer.args.hard_example != "none":
            self.sampler.update_weights()

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
        if self.discriminator is not None:
            self.discriminator.eval()

    def eval_step(self, data):
        if self.trainer.args.discriminator_only:
            return

        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        model = self.get_eval_model()
        scale_factor = self.get_scale_factor()

        psnr = 0
        with self.autocast():
            if self.trainer.args.update_criterion in {"psnr", "all"}:
                z = model(x)
                z, y = fit_size(z, y)
                psnr = self.eval_criterion(z, y)
                if self.trainer.args.update_criterion == "psnr":
                    loss = psnr
                else:
                    loss = torch.tensor(inf_loss())
            elif self.trainer.args.update_criterion == "loss":
                z = model(x)
                z, y = fit_size(z, y)
                # TODO: AuxiliaryLoss does not work
                psnr = self.eval_criterion(z, y)
                loss = self.criterion(z, y)

        self.sum_psnr += psnr.item()
        self.sum_loss += loss.item()
        self.sum_step += 1

    def eval_end(self, file=sys.stdout):
        if self.trainer.args.discriminator_only:
            return inf_loss()

        mean_psnr = self.sum_psnr / self.sum_step
        mean_loss = self.sum_loss / self.sum_step

        if self.trainer.args.update_criterion == "psnr":
            print(f"Batch Y-PSNR: {round(-mean_psnr, 4)}", file=file)
            return mean_psnr
        else:
            print(f"Batch Y-PSNR: {round(-mean_psnr, 4)}, loss: {round(mean_loss, 6)}", file=file)
            return mean_loss


class Waifu2xTrainer(Trainer):
    def create_env(self):
        criterion = create_criterion(self.args.loss).to(self.device)
        if self.discriminator is not None:
            loss_weights = getattr(self.discriminator, "loss_weights", (1.0,))
            if self.args.gan_loss == "hinge":
                discriminator_criterion = GANHingeLoss(loss_weights=loss_weights).to(self.device)
            elif self.args.gan_loss == "hinge_clamp":
                # old behavior
                discriminator_criterion = GANHingeClampLoss(loss_weights=loss_weights).to(self.device)
            elif self.args.gan_loss == "bce":
                discriminator_criterion = GANBCELoss(loss_weights=loss_weights).to(self.device)
            elif self.args.gan_loss == "softplus":
                discriminator_criterion = GANSoftplusLoss(loss_weights=loss_weights).to(self.device)
            else:
                raise ValueError(self.args.gan_loss)
        else:
            discriminator_criterion = None

        return Waifu2xEnv(
            self.model, criterion=criterion,
            discriminator=self.discriminator,
            discriminator_criterion=discriminator_criterion,
            sampler=self.sampler,
            use_diff_aug=self.args.diff_aug,
            use_diff_aug_downsample=self.args.diff_aug_downsample,
            use_diff_aug_noise=self.args.diff_aug_noise,
            adaptive_weight_ema=self.adaptive_weight_ema
        )

    def setup(self):
        method = self.args.hard_example
        if method == "top10":
            self.sampler.method = MiningMethod.TOP10
        elif method == "top20":
            self.sampler.method = MiningMethod.TOP20
        elif method == "linear":
            self.sampler.method = MiningMethod.LINEAR
        self.sampler.scale_factor = self.args.hard_example_scale

    def setup_model(self):
        self.adaptive_weight_ema = None
        self.discriminator = create_discriminator(self.args.discriminator, self.args.gpu, self.device)
        if self.args.freeze and hasattr(self.model, "freeze"):
            self.model.freeze()
            logger.debug("call model.freeze()")
        if self.args.tile_mode:
            self.model.set_tile_mode()
        if self.args.tile_2x2_mode:
            self.model.set_tile_2x2_mode()

    def create_model(self):
        kwargs = {"in_channels": 3, "out_channels": 3}
        if self.args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            kwargs["no_clip"] = True
        if self.args.pre_antialias and self.args.arch == "waifu2x.swin_unet_4x":
            kwargs["pre_antialias"] = True
        model = create_model(self.args.arch, device_ids=self.args.gpu, **kwargs)
        model = model.to(self.device)
        return model

    def create_optimizers(self):
        if self.discriminator is not None:
            g_opt = self.create_optimizer(self.model)

            lr = self.args.discriminator_learning_rate or self.args.learning_rate
            d_opt = self.create_optimizer(self.discriminator, lr=lr)
            return g_opt, d_opt
        else:
            return super().create_optimizers()

    def create_grad_scalers(self):
        if self.discriminator is not None:
            return [self.create_grad_scaler(), self.create_grad_scaler()]
        else:
            return super().create_grad_scalers()

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = self.model.i2i_offset
        return_no_offset_y = self.args.privilege
        if self.args.method in {"scale", "noise_scale"}:
            scale_factor = 2
        elif self.args.method in {"scale4x", "noise_scale4x"}:
            scale_factor = 4
        elif self.args.method in {"scale8x", "noise_scale8x"}:
            scale_factor = 8
        elif self.args.method in {"noise", "ae"}:
            scale_factor = 1
        else:
            raise NotImplementedError()

        dataloader_extra_options = {}
        if self.args.num_workers > 0:
            dataloader_extra_options.update({
                "prefetch_factor": self.args.prefetch_factor,
                "persistent_workers": True
            })

        if type == "train":
            dataset = Waifu2xDataset(
                input_dir=path.join(self.args.data_dir, "train"),
                additional_data_dir=self.args.additional_data_dir,
                additional_data_dir_p=self.args.additional_data_dir_p,
                model_offset=model_offset,
                scale_factor=scale_factor,
                bicubic_only=self.args.b4b,
                skip_screentone=self.args.skip_screentone,
                skip_dot=self.args.skip_dot,
                crop_samples=self.args.crop_samples,
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                num_samples=self.args.num_samples,
                da_jpeg_p=self.args.da_jpeg_p,
                da_scale_p=self.args.da_scale_p,
                da_chshuf_p=self.args.da_chshuf_p,
                da_unsharpmask_p=self.args.da_unsharpmask_p,
                da_grayscale_p=self.args.da_grayscale_p,
                da_color_p=self.args.da_color_p,
                da_antialias_p=self.args.da_antialias_p,
                da_hflip_only=self.args.da_hflip_only,
                da_cutmix_p=self.args.da_cutmix_p,
                da_mixup_p=self.args.da_mixup_p,
                deblur=self.args.deblur,
                resize_blur_range=self.args.resize_blur_range,
                resize_blur_p=self.args.resize_blur_p,
                resize_step_p=self.args.resize_step_p,
                resize_no_antialias_p=self.args.resize_no_antialias_p,
                return_no_offset_y=return_no_offset_y,
                training=True,
            )
            self.sampler = dataset.create_sampler()
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size,
                worker_init_fn=dataset.worker_init,
                shuffle=False,
                pin_memory=True,
                sampler=self.sampler,
                num_workers=self.args.num_workers,
                drop_last=True,
                **dataloader_extra_options)
            return dataloader
        elif type == "eval":
            dataset = Waifu2xDataset(
                input_dir=path.join(self.args.data_dir, "eval"),
                model_offset=model_offset,
                scale_factor=scale_factor,
                skip_screentone=self.args.skip_screentone,
                skip_dot=self.args.skip_dot,
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                deblur=self.args.deblur,
                resize_blur_range=self.args.resize_blur_range,
                return_no_offset_y=False,
                training=False)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size,
                worker_init_fn=dataset.worker_init,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=bool(self.args.drop_last),
                **dataloader_extra_options)
            return dataloader

    def create_filename_prefix(self):
        if self.args.method == "scale":
            return "scale2x"
        elif self.args.method == "noise_scale":
            return f"noise{self.args.noise_level}_scale2x"
        elif self.args.method == "scale4x":
            return "scale4x"
        elif self.args.method == "noise_scale4x":
            return f"noise{self.args.noise_level}_scale4x"
        elif self.args.method == "scale8x":
            return "scale8x"
        elif self.args.method == "noise_scale8x":
            return f"noise{self.args.noise_level}_scale8x"
        elif self.args.method == "noise":
            return f"noise{self.args.noise_level}"
        elif self.args.method == "ae":
            return "ae"
        else:
            raise NotImplementedError()

    def save_best_model(self):
        super().save_best_model()
        if self.discriminator is not None:
            discriminator_filename = self.create_discriminator_model_filename()
            save_model(self.discriminator, discriminator_filename)
            if not self.args.disable_backup:
                backup_file = f"{path.splitext(discriminator_filename)[0]}.{self.runtime_id}.pth.bk"
                save_model(self.discriminator, backup_file)

    def save_checkpoint(self, **kwargs):
        if self.discriminator is not None:
            kwargs.update({"discriminator_state_dict": self.discriminator.state_dict()})
            kwargs.update({"adaptive_weight_ema": self.env.adaptive_weight_ema})
        super().save_checkpoint(**kwargs)

    def resume(self):
        meta = super().resume()
        if self.discriminator is not None and "discriminator_state_dict" in meta:
            self.discriminator.load_state_dict(meta["discriminator_state_dict"])
            if "adaptive_weight_ema" in meta:
                self.adaptive_weight_ema = meta["adaptive_weight_ema"]

    def load_initial_parameters(self, checkpoint_filename):
        meta = super().load_initial_parameters(checkpoint_filename)
        if self.discriminator is not None and "discriminator_state_dict" in meta:
            self.discriminator.load_state_dict(meta["discriminator_state_dict"])
            if "adaptive_weight_ema" in meta:
                self.adaptive_weight_ema = meta["adaptive_weight_ema"]

    def create_discriminator_model_filename(self):
        return path.join(
            self.args.model_dir,
            f"{self.create_filename_prefix()}_discriminator.pth")

    def create_best_model_filename(self):
        return path.join(
            self.args.model_dir,
            self.create_filename_prefix() + ".pth")

    def create_checkpoint_filename(self):
        return path.join(
            self.args.model_dir,
            self.create_filename_prefix() + ".checkpoint.pth")


def train(args):
    torch._functorch.config.donated_buffer = False
    ARCH_SWIN_UNET = {"waifu2x.swin_unet_1x",
                      "waifu2x.swin_unet_2x",
                      "waifu2x.swin_unet_4x"}
    # if args.size % 4 != 0:
    #     raise ValueError("--size must be a multiple of 4")
    if args.arch in ARCH_SWIN_UNET and ((args.size - 16) % 12 != 0 or (args.size - 16) % 16 != 0):
        raise ValueError("--size must be `(SIZE - 16) % 12 == 0 and (SIZE - 16) % 16 == 0` for SwinUNet models")
    if args.method in {"noise", "noise_scale", "noise_scale4x"} and args.noise_level is None:
        raise ValueError("--noise-level is required for noise/noise_scale")
    if args.pre_antialias and args.arch != "waifu2x.swin_unet_4x":
        raise ValueError("--pre-antialias is only supported for waifu2x.swin_unet_4x")

    if args.method in {"scale", "scale4x", "scale8x", "ae"}:
        # disable
        args.noise_level = -1

    if args.loss is None:
        if args.arch in {"waifu2x.vgg_7", "waifu2x.upconv_7"}:
            args.loss = "y_charbonnier"
        elif args.arch in {"waifu2x.cunet", "waifu2x.upcunet"}:
            args.loss = "aux_lbp"
        elif args.arch in {"waifu2x.swin_unet_1x", "waifu2x.swin_unet_2x"}:
            args.loss = "lbp"
        elif args.arch in {"waifu2x.swin_unet_4x"}:
            args.loss = "lbp5"
        elif args.arch in {"waifu2x.swin_unet_8x"}:
            args.loss = "y_charbonnier"
        elif args.arch.startswith("waifu2x.winc_unet"):
            args.loss = "dctirm"
        else:
            args.loss = "y_charbonnier"

    if args.b4b:
        # disable random resize blur
        args.resize_blur_p = 0.0
        args.deblur = 0.0
        # disable random rescale
        args.da_scape_p = 0.0

    trainer = Waifu2xTrainer(args)
    trainer.fit()


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "waifu2x",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    waifu2x_models = sorted([name for name in get_model_names() if name.startswith("waifu2x.")])

    parser.add_argument("--method", type=str,
                        choices=["noise", "scale", "noise_scale", "ae",
                                 "scale4x", "noise_scale4x",
                                 "scale8x", "noise_scale8x"],
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
    parser.add_argument("--size", type=int, default=112,
                        help="input size")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="number of samples for each epoch")
    parser.add_argument("--drop-last", action="store_true",
                        help="force drop_last=True for DataLoader")

    parser.add_argument("--loss", type=str,
                        choices=list(LOSS_FUNCTIONS.keys()),
                        help="loss function")
    parser.add_argument("--additional-data-dir", type=str, help="additional data dir for training")
    parser.add_argument("--additional-data-dir-p", type=float, default=0.01,
                        help="probability that --additional-data-dir should be used")
    parser.add_argument("--da-jpeg-p", type=float, default=0.0,
                        help="HQ JPEG(quality=92-99) data augmentation for gt image")
    parser.add_argument("--da-scale-p", type=float, default=0.25,
                        help="random downscale data augmentation for gt image")
    parser.add_argument("--da-chshuf-p", type=float, default=0.0,
                        help="random channel shuffle data augmentation for gt image")
    parser.add_argument("--da-unsharpmask-p", type=float, default=0.0,
                        help="random unsharp mask data augmentation for gt image")
    parser.add_argument("--da-grayscale-p", type=float, default=0.0,
                        help="random grayscale data augmentation for gt image")
    parser.add_argument("--da-color-p", type=float, default=0.0,
                        help="random color jitter data augmentation for gt image")
    parser.add_argument("--da-antialias-p", type=float, default=0.0,
                        help="random antialias input degradation")
    parser.add_argument("--da-hflip-only", action="store_true",
                        help="restrict random flip to horizontal flip only")
    parser.add_argument("--da-cutmix-p", type=float, default=0.0,
                        help="random cutmix data augmentation for gt image")
    parser.add_argument("--da-mixup-p", type=float, default=0.0,
                        help="random mixup(overlay) data augmentation for gt image")

    parser.add_argument("--deblur", type=float, default=0.0,
                        help=("shift parameter of random resize blur."
                              " 0.0-0.05 is a reasonable value. "
                              "see --resize-blur-range for details"))
    parser.add_argument("--resize-blur-range", type=float, nargs="+", default=[0.05],
                        help=("max shift of random resize blur."
                              " blur = 1 + uniform(-resize_blur_range + deblur, resize_blur_range + deblur)."
                              " or "
                              " blur = 1 + uniform(resize_blur_range[0] + deblur, resize_blur_range[1] + deblur)."
                              " blur >= 1 is blur, blur <= 1 is sharpen. mean 1 by default"))
    parser.add_argument("--resize-blur-p", type=float, default=0.1,
                        help=("probability that resize blur should be used"))
    parser.add_argument("--resize-step-p", type=float, default=0.,
                        help=("probability that 2 step downscaling should be used"))
    parser.add_argument("--resize-no-antialias-p", type=float, default=0.,
                        help="probability that no antialias(jagged edge) downscaling should be used")

    parser.add_argument("--hard-example", type=str, default="linear",
                        choices=["none", "linear", "top10", "top20"],
                        help="hard example mining for training data sampleing")
    parser.add_argument("--hard-example-scale", type=float, default=4.,
                        help="max weight scaling factor of hard example sampler")
    parser.add_argument("--b4b", action="store_true",
                        help="use only bicubic downsampling for bicubic downsampling restoration (classic super-resolution)")
    parser.add_argument("--freeze", action="store_true",
                        help="call model.freeze() if avaliable")
    parser.add_argument("--tile-mode", action="store_true",
                        help="call model.set_tile_mode()")
    parser.add_argument("--tile-2x2-mode", action="store_true",
                        help="call model.set_tile_2x2_mode()")
    parser.add_argument("--pre-antialias", action="store_true",
                        help=("Set `pre_antialias=True` for SwinUNet4x."))
    parser.add_argument("--privilege", action="store_true",
                        help=("Use model.forward(LR_image, HR_image)"))
    parser.add_argument("--skip-screentone", action="store_true",
                        help=("Skip files containing '__SCREENTONE_' in the filename"))
    parser.add_argument("--skip-dot", action="store_true",
                        help=("Skip files containing '__DOT_' in the filename"))
    parser.add_argument("--crop-samples", type=int, default=4,
                        help=("number of samples for hard example cropping"))

    # GAN related options
    parser.add_argument("--discriminator", type=str,
                        help="discriminator name or .pth or [`l3`, `l3c`, `l3v1`, `l3v1`].")
    parser.add_argument("--discriminator-weight", type=float, default=1.0,
                        help="discriminator loss weight")
    parser.add_argument("--discriminator-adaptive-weight-min", type=float, default=1e-4,
                        help="minimum adaptive loss weight")
    parser.add_argument("--discriminator-adaptive-weight-max", type=float, default=1e4,
                        help="maxmum adaptive loss weight")
    parser.add_argument("--update-criterion", type=str, choices=["psnr", "loss", "all"], default="psnr",
                        help=("criterion for updating the best model file. "
                              "`all` forced to saves the best model each epoch."))
    parser.add_argument("--discriminator-only", action="store_true",
                        help="training discriminator only")
    parser.add_argument("--generator-warmup-iteration", type=int, default=500,
                        help=("warm-up iterations for the discriminator loss affecting the generator."))
    parser.add_argument("--discriminator-learning-rate", type=float,
                        help=("learning-rate for discriminator. --learning-rate by default."))
    parser.add_argument("--reconstruction-loss-scale", type=float, default=1.0,
                        help=("pre scaling factor for reconstruction loss."))
    parser.add_argument("--diff-aug", action="store_true",
                        help="Use differentiable transforms for reconstruction loss and discriminator")
    parser.add_argument("--diff-aug-downsample", action="store_true",
                        help="Use addtional 2x downsample transforms")
    parser.add_argument("--diff-aug-noise", action="store_true",
                        help="Use addtional random noise transforms")
    parser.add_argument("--gan-loss", type=str, choices=["hinge", "bce", "softplus", "hinge_clamp"], default="hinge",
                        help="GAN loss function")

    parser.set_defaults(
        batch_size=16,
        optimizer="adamw",
        learning_rate=0.0002,
        scheduler="cosine",
        learning_rate_cosine_min=1e-8,
        learning_rate_cycles=5,
        learning_rate_decay=0.995,
        learning_rate_decay_step=[1],
        # for adamw cosine_wd
        weight_decay=0.001,
        weight_decay_end=0.01,
    )
    parser.set_defaults(handler=train)

    return parser
