from os import path
import sys
from time import time
import argparse
import torch
from . dataset import Waifu2xDataset
from .. models.discriminator import SelfSupervisedDiscriminator
from nunif.training.sampler import MiningMethod
from nunif.training.trainer import Trainer
from nunif.training.env import LuminancePSNREnv
from nunif.models import (
    create_model, get_model_config, call_model_method,
    load_model, save_model,
    get_model_names
)
from nunif.modules import (
    ClampLoss, LuminanceWeightedLoss, AverageWeightedLoss,
    AuxiliaryLoss,
    LBPLoss, CharbonnierLoss,
    Alex11Loss,
    DiscriminatorHingeLoss,
    MultiscaleLoss,
)
from nunif.modules.lbp_loss import L1LBP, YL1LBP, YLBP, RGBLBP
from nunif.logger import logger
import random


# basic training


def create_criterion(loss):
    if loss == "l1":
        criterion = ClampLoss(torch.nn.L1Loss())
    elif loss == "y_l1":
        criterion = ClampLoss(LuminanceWeightedLoss(torch.nn.L1Loss()))
    elif loss == "lbp":
        criterion = YLBP()
    elif loss == "lbpm":
        criterion = MultiscaleLoss(YLBP())
    elif loss == "lbp5":
        criterion = YLBP(kernel_size=5)
    elif loss == "lbp5m":
        criterion = MultiscaleLoss(YLBP(kernel_size=5))
    elif loss == "rgb_lbp":
        criterion = RGBLBP()
    elif loss == "rgb_lbp5":
        criterion = RGBLBP(kernel_size=5)
    elif loss == "alex11":
        criterion = ClampLoss(LuminanceWeightedLoss(Alex11Loss(in_channels=1)))
    elif loss == "charbonnier":
        criterion = ClampLoss(CharbonnierLoss())
    elif loss == "y_charbonnier":
        criterion = ClampLoss(LuminanceWeightedLoss(CharbonnierLoss()))
    elif loss == "aux_lbp":
        criterion = AuxiliaryLoss([
            YLBP(),
            YLBP(),
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
    elif loss == "l1lpips":
        from nunif.modules.lpips import LPIPSWith
        # weight=0.1, gradient norm is about the same as L1Loss.
        criterion = LPIPSWith(ClampLoss(AverageWeightedLoss(torch.nn.L1Loss(), in_channels=3)), weight=0.4)
    elif loss == "l1lbp5":
        criterion = YL1LBP(kernel_size=5, weight=0.4)
    elif loss == "rgb_l1lbp5":
        criterion = L1LBP(kernel_size=5, weight=0.4)
    elif loss == "rgb_l1lbp":
        criterion = L1LBP(kernel_size=3, weight=0.4)
    else:
        raise NotImplementedError(loss)

    return criterion


def create_discriminator(discriminator, device_ids, device):
    if discriminator is None:
        return None
    elif discriminator == "l3":
        model = create_model("waifu2x.l3_discriminator", device_ids=device_ids)
    elif discriminator == "l3c":
        model = create_model("waifu2x.l3_conditional_discriminator", device_ids=device_ids)
    elif discriminator == "l3v1":
        model = create_model("waifu2x.l3v1_discriminator", device_ids=device_ids)
    elif discriminator == "l3v1c":
        model = create_model("waifu2x.l3v1_conditional_discriminator", device_ids=device_ids)
    elif discriminator == "u3c":
        model = create_model("waifu2x.u3_conditional_discriminator", device_ids=device_ids)
    elif discriminator == "u3fftc":
        model = create_model("waifu2x.u3fft_conditional_discriminator", device_ids=device_ids)
    elif path.exists(discriminator):
        model, _ = load_model(discriminator, device_ids=device_ids)
    else:
        model = create_model(discriminator, device_ids=device_ids)
    return model.to(device)


def get_last_layer(model):
    if model.name in {"waifu2x.swin_unet_1x",
                      "waifu2x.swin_unet_2x",
                      "waifu2x.swin_unet_4x",
                      "waifu2x.swin_unet_8x"}:
        return model.unet.to_image.proj.weight
    elif model.name in {"waifu2x.cunet", "waifu2x.upcunet"}:
        return model.unet2.conv_bottom.weight
    elif model.name in {"waifu2x.upconv_7", "waifu2x.vgg_7"}:
        return model.net[-1].weight
    else:
        raise NotImplementedError()


def inf_loss():
    return float(-time() / 1000000000)


class Waifu2xEnv(LuminancePSNREnv):
    def __init__(self, model, criterion,
                 discriminator,
                 discriminator_criterion,
                 sampler):
        super().__init__(model, criterion)
        self.discriminator = discriminator
        self.discriminator_criterion = discriminator_criterion
        self.sampler = sampler

    def train_loss_hook(self, data, loss):
        super().train_loss_hook(data, loss)
        if self.trainer.args.hard_example == "none":
            return

        index = data[-1]
        if self.discriminator is None:
            self.sampler.update_losses(index, loss.item())
        else:
            recon_loss, generator_loss, d_loss = loss
            if not self.trainer.args.discriminator_only:
                self.sampler.update_losses(index, recon_loss.item())

    def get_scale_factor(self):
        scale_factor = get_model_config(self.model, "i2i_scale")
        return scale_factor

    def calc_discriminator_skip_prob(self, d_loss):
        start = self.trainer.args.generator_start_criteria
        stop = self.trainer.args.discriminator_stop_criteria
        cur = d_loss.item()
        if cur > start:
            return 0.
        elif cur < stop:
            return 1.
        else:
            p = (start - cur) / (start - stop)
            return p

    def clear_loss(self):
        super().clear_loss()
        self.sum_p_loss = 0
        self.sum_g_loss = 0
        self.sum_d_loss = 0
        self.sum_d_weight = 0
        self.sum_psnr = 0

    def train_begin(self):
        super().train_begin()
        if self.discriminator is not None:
            self.discriminator.train()
            if self.trainer.args.discriminator_only:
                self.model.eval()

    def train_step(self, data):
        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        scale_factor = self.get_scale_factor()

        with self.autocast():
            if self.discriminator is None:
                z = self.model(x)
                loss = self.criterion(z, y)
                self.sum_loss += loss.item()
            else:
                if not self.trainer.args.discriminator_only:
                    # generator (sr) step
                    self.discriminator.requires_grad_(False)
                    z = self.model(x)
                    if isinstance(z, (list, tuple)):
                        # NOTE: models using auxiliary loss return tuple.
                        #       first element is SR result.
                        fake = z[0]
                    else:
                        fake = z
                    if isinstance(self.discriminator, SelfSupervisedDiscriminator):
                        *z_real, _ = self.discriminator(torch.clamp(fake, 0, 1), y, scale_factor)
                        if len(z_real) == 1:
                            z_real = z_real[0]
                    else:
                        z_real = self.discriminator(torch.clamp(fake, 0, 1), y, scale_factor)
                    recon_loss = self.criterion(z, y)
                    generator_loss = self.discriminator_criterion(z_real)
                    self.sum_p_loss += recon_loss.item()
                    self.sum_g_loss += generator_loss.item()

                    # loss weight will be recalculated later,
                    # but multiplied by 10 here to reduce the gap.
                    # (gradient norm of generator_loss is 10-100x larger than recon_loss)
                    recon_loss = recon_loss * 10
                else:
                    with torch.no_grad():
                        z = self.model(x)
                        fake = z[0] if isinstance(z, (list, tuple)) else z
                    recon_loss = generator_loss = torch.zeros(1, dtype=x.dtype, device=x.device)

                # discriminator step
                self.discriminator.requires_grad_(True)
                if isinstance(self.discriminator, SelfSupervisedDiscriminator):
                    *z_fake, fake_ss_loss = self.discriminator(torch.clamp(fake.detach(), 0, 1),
                                                               y, scale_factor)
                    *z_real, real_ss_loss = self.discriminator(y, y, scale_factor)
                    if len(z_fake) == 1:
                        z_fake = z_fake[0]
                        z_real = z_real[0]
                else:
                    z_fake = self.discriminator(torch.clamp(fake.detach(), 0, 1), y, scale_factor)
                    z_real = self.discriminator(y, y, scale_factor)
                    fake_ss_loss = real_ss_loss = 0

                discriminator_loss = (self.discriminator_criterion(z_real, z_fake) +
                                      (real_ss_loss + fake_ss_loss) * 0.5)

                self.sum_d_loss += discriminator_loss.item()
                loss = (recon_loss, generator_loss, discriminator_loss)

        self.sum_step += 1
        return loss

    def train_backward_step(self, loss, optimizers, grad_scaler, update):
        if self.discriminator is None:
            super().train_backward_step(loss, optimizers, grad_scaler, update)
        else:
            # NOTE: Ignore `update` flag,
            #       gradient accumulation does not work with Discriminator.

            recon_loss, generator_loss, d_loss = loss
            g_opt, d_opt = optimizers
            optimizers = []

            # update generator
            disc_skip_prob = self.calc_discriminator_skip_prob(d_loss)
            if not self.trainer.args.discriminator_only:
                g_opt.zero_grad()
                last_layer = get_last_layer(self.model)
                weight = self.calculate_adaptive_weight(
                    recon_loss, generator_loss, last_layer, grad_scaler,
                    min=1e-3, max=10, mode="norm") * self.trainer.args.discriminator_weight
                recon_weight = 1.0 / weight
                if generator_loss > 0.0 and (d_loss < self.trainer.args.generator_start_criteria or
                                             generator_loss > 0.95):
                    g_loss = (recon_loss * recon_weight + generator_loss) * 0.5
                else:
                    g_loss = recon_loss * recon_weight * 0.5
                self.sum_loss += g_loss.item()
                self.sum_d_weight += weight
                self.backward(g_loss, grad_scaler)
                optimizers.append(g_opt)

                logger.debug(f"recon: {round(recon_loss.item(), 4)}, gen: {round(generator_loss.item(), 4)}, "
                             f"disc: {round(d_loss.item(), 4)}, weight: {round(weight, 6)}, "
                             f"disc skip: {round(disc_skip_prob, 3)}")

            # update discriminator
            d_opt.zero_grad()
            if not (random.uniform(0., 1.) < disc_skip_prob):
                self.backward(d_loss, grad_scaler)
                optimizers.append(d_opt)

            if optimizers:
                self.optimizer_step(optimizers, grad_scaler)

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
        scale_factor = self.get_scale_factor()

        psnr = 0
        with self.autocast():
            if self.trainer.args.update_criterion in {"psnr", "all"}:
                z = self.model(x)
                psnr = self.eval_criterion(z, y)
                if self.trainer.args.update_criterion == "psnr":
                    loss = psnr
                else:
                    loss = torch.tensor(inf_loss())
            elif self.trainer.args.update_criterion == "loss":
                z = self.model(x)
                # TODO: AuxiliaryLoss does not work
                psnr = self.eval_criterion(z, y)
                loss = self.criterion(z, y)
                if self.discriminator is not None:
                    z_real = self.discriminator(z, y, scale_factor)
                    loss = loss + self.discriminator_criterion(z_real)

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
            conf = get_model_config(self.discriminator)
            loss_weights = conf.get("loss_weights", (1.0,))
            discriminator_criterion = DiscriminatorHingeLoss(loss_weights=loss_weights).to(self.device)
        else:
            discriminator_criterion = None
        return Waifu2xEnv(self.model, criterion=criterion,
                          discriminator=self.discriminator,
                          discriminator_criterion=discriminator_criterion,
                          sampler=self.sampler)

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
        self.discriminator = create_discriminator(self.args.discriminator, self.args.gpu, self.device)
        if self.discriminator is not None:
            # initialize lazy modules
            model_offset = get_model_config(self.model, "i2i_offset")
            scale_factor = get_model_config(self.model, "i2i_scale")
            output_size = self.args.size * scale_factor - model_offset * 2
            y = torch.zeros((1, 3, output_size, output_size)).to(self.device)
            _ = self.discriminator(y, y, scale_factor)

        if self.args.freeze and hasattr(self.model, "freeze"):
            call_model_method(self.model, "freeze")
            logger.debug("call model.freeze()")

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

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        model_offset = get_model_config(self.model, "i2i_offset")
        if self.args.method in {"scale", "noise_scale"}:
            scale_factor = 2
        elif self.args.method in {"scale4x", "noise_scale4x"}:
            scale_factor = 4
        elif self.args.method in {"scale8x", "noise_scale8x"}:
            scale_factor = 8
        elif self.args.method in {"noise"}:
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
                da_color_p=self.args.da_color_p,
                da_antialias_p=self.args.da_antialias_p,
                deblur=self.args.deblur,
                resize_blur_p=self.args.resize_blur_p,
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
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                deblur=self.args.deblur,
                training=False)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size,
                worker_init_fn=dataset.worker_init,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False,
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
        super().save_checkpoint(**kwargs)

    def resume(self):
        meta = super().resume()
        if self.discriminator is not None and "discriminator_state_dict" in meta:
            self.discriminator.load_state_dict(meta["discriminator_state_dict"])

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
    ARCH_SWIN_UNET = {"waifu2x.swin_unet_1x",
                      "waifu2x.swin_unet_2x",
                      "waifu2x.swin_unet_4x"}
    assert args.discriminator_stop_criteria < args.generator_start_criteria
    if args.size % 4 != 0:
        raise ValueError("--size must be a multiple of 4")
    if args.arch in ARCH_SWIN_UNET and ((args.size - 16) % 12 != 0 or (args.size - 16) % 16 != 0):
        raise ValueError("--size must be `(SIZE - 16) % 12 == 0 and (SIZE - 16) % 16 == 0` for SwinUNet models")
    if args.method in {"noise", "noise_scale", "noise_scale4x"} and args.noise_level is None:
        raise ValueError("--noise-level is required for noise/noise_scale")
    if args.pre_antialias and args.arch != "waifu2x.swin_unet_4x":
        raise ValueError("--pre-antialias is only supported for waifu2x.swin_unet_4x")

    if args.method in {"scale", "scale4x", "scale8x"}:
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
                        choices=["noise", "scale", "noise_scale",
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
    parser.add_argument("--loss", type=str,
                        choices=["lbp", "lbp5", "lbpm", "lbp5m", "rgb_lbp", "rgb_lbp5",
                                 "y_charbonnier", "charbonnier",
                                 "aux_lbp", "aux_y_charbonnier", "aux_charbonnier",
                                 "alex11", "aux_alex11", "l1", "y_l1", "l1lpips",
                                 "l1lbp5", "rgb_l1lbp5", "rgb_l1lbp"],
                        help="loss function")
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
    parser.add_argument("--deblur", type=float, default=0.0,
                        help=("shift parameter of resize blur."
                              " 0.0-0.1 is a reasonable value."
                              " blur = uniform(0.95 + deblur, 1.05 + deblur)."
                              " blur >= 1 is blur, blur <= 1 is sharpen. mean 1 by default"))
    parser.add_argument("--resize-blur-p", type=float, default=0.1,
                        help=("probability that resize blur should be used"))
    parser.add_argument("--hard-example", type=str, default="linear",
                        choices=["none", "linear", "top10", "top20"],
                        help="hard example mining for training data sampleing")
    parser.add_argument("--hard-example-scale", type=float, default=4.,
                        help="max weight scaling factor of hard example sampler")
    parser.add_argument("--b4b", action="store_true",
                        help="use only bicubic downsampling for bicubic downsampling restoration")
    parser.add_argument("--freeze", action="store_true",
                        help="call model.freeze() if avaliable")
    # GAN related options
    parser.add_argument("--discriminator", type=str,
                        help="discriminator name or .pth or [`l3`, `l3c`, `l3v1`, `l3v1`].")
    parser.add_argument("--discriminator-weight", type=float, default=1.0,
                        help="discriminator loss weight")
    parser.add_argument("--update-criterion", type=str, choices=["psnr", "loss", "all"], default="psnr",
                        help=("criterion for updating the best model file. "
                              "`all` forced to saves the best model each epoch."))
    parser.add_argument("--discriminator-only", action="store_true",
                        help="training discriminator only")
    parser.add_argument("--discriminator-stop-criteria", type=float, default=0.5,
                        help=("When the loss of the discriminator is less than the specified value,"
                              " stops training of the discriminator."
                              " This is the limit to prevent too strong discriminator."
                              " Also, the discriminator skip probability is interpolated between --generator-start-criteria and --discriminator-stop-criteria."))
    parser.add_argument("--generator-start-criteria", type=float, default=0.9,
                        help=("When the loss of the discriminator is greater than the specified value,"
                              " stops training of the generator."
                              " This is the limit to prevent too strong generator."
                              " Also do not hit the newbie discriminator."))
    parser.add_argument("--discriminator-learning-rate", type=float,
                        help=("learning-rate for discriminator. --learning-rate by default."))
    parser.add_argument("--pre-antialias", action="store_true",
                        help=("Set `pre_antialias=True` for SwinUNet4x."))

    parser.set_defaults(
        batch_size=16,
        optimizer="adamw",
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
