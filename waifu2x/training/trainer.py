from os import path
import sys
from time import time
import torch
from .. models import UNet2Discriminator, UNet1Discriminator, UNet2ConditionalDiscriminator
from . dataset import Waifu2xDataset
from nunif.training.trainer import Trainer
from nunif.training.env import LuminancePSNREnv
from nunif.models import (
    create_model, get_model_config, call_model_method,
    load_model, save_model
)
from nunif.modules import (
    ClampLoss, LuminanceWeightedLoss, AuxiliaryLoss, LBPLoss, CharbonnierLoss,
    Alex11Loss,
    DiscriminatorBCELoss,
    DiscriminatorHingeLoss,
)
from nunif.logger import logger


# basic training


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


def create_discriminator(discriminator, device):
    if discriminator is None:
        return None
    elif discriminator == "unet1":
        model = UNet1Discriminator()
    elif discriminator == "unet2":
        model = UNet2Discriminator()
    elif discriminator == "unet2_cond":
        model = UNet2ConditionalDiscriminator()
    elif path.exists(discriminator):
        model, _ = load_model(discriminator)
        if model.name in {"waifu2x.cunet", "waifu2x.upcunet"}:
            model = UNet2Discriminator.from_cunet(model)
    else:
        raise NotImplementedError()
    return model.to(device)


class Waifu2xEnv(LuminancePSNREnv):
    def __init__(self, model, criterion,
                 discriminator,
                 discriminator_criterion,
                 discriminator_weight,
                 update_criterion,
                 discriminator_only):
        super().__init__(model, criterion)
        self.discriminator = discriminator
        self.discriminator_criterion = discriminator_criterion
        self.discriminator_weight = discriminator_weight
        self.discriminator_only = discriminator_only
        self.update_criterion = update_criterion

    def train_loss_hook(self, data, loss):
        super().train_loss_hook(data, loss)
        if self.trainer.args.hard_example == "none":
            return
        dataset = self.trainer.train_loader.dataset
        index = data[-1]
        if self.discriminator is None:
            dataset.update_hard_example_losses(index, loss.item())
        else:
            g_loss, d_loss = loss
            if not self.discriminator_only:
                dataset.update_hard_example_losses(index, g_loss.item())

    def get_scale_factor(self):
        scale_factor = get_model_config(self.model, "i2i_scale")
        return scale_factor

    def clear_loss(self):
        super().clear_loss()
        self.sum_p_loss = 0
        self.sum_g_loss = 0
        self.sum_d_loss = 0
        self.sum_psnr = 0

    def train_begin(self):
        super().train_begin()
        if self.discriminator is not None:
            self.discriminator.train()
            if self.discriminator_only:
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
                if not self.discriminator_only:
                    # generator (sr) step
                    z = self.model(x)
                    self.discriminator.requires_grad_(False)
                    if isinstance(z, (list, tuple)):
                        # NOTE: models using auxiliary loss return tuple.
                        #       first element is SR result.
                        fake = z[0]
                    else:
                        fake = z
                    fake = fake # torch.clamp(fake, 0., 1.) * 0.99 + fake * 0.01
                    z_real = self.discriminator(fake, x, scale_factor)
                    recon_loss = self.criterion(z, y)
                    generator_loss = self.discriminator_criterion(z_real)
                    g_loss = recon_loss + self.discriminator_weight * generator_loss

                    self.sum_loss += g_loss.item()
                    self.sum_p_loss += recon_loss.item()
                    self.sum_g_loss += generator_loss.item()
                else:
                    with torch.no_grad():
                        z = self.model(x)
                        fake = z[0] if isinstance(z, (list, tuple)) else z
                    g_loss = torch.zeros(1, dtype=x.dtype, device=x.device)

                # discriminator step
                self.discriminator.requires_grad_(True)
                z_fake = self.discriminator(fake.detach().clone(), x, scale_factor)
                z_real = self.discriminator(y, x, scale_factor)
                d_loss = self.discriminator_criterion(z_real, z_fake)

                self.sum_d_loss += d_loss.item()
                loss = (g_loss, d_loss)

        self.sum_step += 1
        return loss

    def train_backward_step(self, loss, optimizers, grad_scaler, update):
        if self.discriminator is None:
            super().train_backward_step(loss, optimizers, grad_scaler, update)
        else:
            # NOTE: Ignore `update` flag,
            #       gradient accumulation does not work with Discriminator.

            g_loss, d_loss = loss
            g_opt, d_opt = optimizers

            # update generator
            if not self.discriminator_only:
                g_opt.zero_grad()
                self.backward(g_loss, grad_scaler)
                self.optimizer_step(g_opt, grad_scaler)

            # update discriminator
            d_opt.zero_grad()
            self.backward(d_loss, grad_scaler)
            self.optimizer_step(d_opt, grad_scaler)

    def train_end(self):
        # update sampler
        if self.trainer.args.hard_example != "none":
            dataset = self.trainer.train_loader.dataset
            dataset.update_hard_example_weights()

        # show loss
        mean_loss = self.sum_loss / self.sum_step
        if self.discriminator is not None:
            mean_p_loss = self.sum_p_loss / self.sum_step
            mean_d_loss = self.sum_d_loss / self.sum_step
            mean_g_loss = self.sum_g_loss / self.sum_step
            print(f"loss: {round(mean_loss, 6)}, "
                  f"reconstruction loss: {round(mean_p_loss, 6)}, "
                  f"realistic loss: {round(mean_g_loss, 6)}, "
                  f"discriminator loss: {round(mean_d_loss, 6)}")
            mean_loss = mean_loss + mean_d_loss
        else:
            print(f"loss: {round(mean_loss, 6)}")

        return mean_loss

    def eval_begin(self):
        super().eval_begin()
        if self.discriminator is not None:
            self.discriminator.eval()

    def eval_step(self, data):
        if self.discriminator_only:
            return

        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        scale_factor = self.get_scale_factor()

        psnr = 0
        with self.autocast():
            if self.update_criterion == "psnr":
                z = self.model(x)
                psnr = self.eval_criterion(z, y)
                loss = psnr
            else:
                z = self.model(x)
                # TODO: AuxiliaryLoss does not work
                psnr = self.eval_criterion(z, y)
                loss = self.criterion(z, y)
                if self.discriminator is not None:
                    z_real = self.discriminator(z, x, scale_factor)
                    loss = loss + self.discriminator_criterion(z_real) * self.discriminator_weight

        self.sum_psnr += psnr.item()
        self.sum_loss += loss.item()
        self.sum_step += 1

    def eval_end(self, file=sys.stdout):
        if self.discriminator_only:
            return float(-time() / 1000000000)

        mean_psnr = self.sum_psnr / self.sum_step
        mean_loss = self.sum_loss / self.sum_step

        if self.update_criterion == "psnr":
            print(f"Batch Y-PSNR: {round(-mean_psnr, 4)}", file=file)
            return mean_psnr
        else:
            print(f"Batch Y-PSNR: {round(-mean_psnr, 4)}, loss: {round(mean_loss, 6)}", file=file)
            return mean_loss


class Waifu2xTrainer(Trainer):
    def create_env(self):
        criterion = create_criterion(self.args.loss).to(self.device)
        if self.discriminator is not None:
            discriminator_criterion = DiscriminatorHingeLoss().to(self.device)
        else:
            discriminator_criterion = None
        return Waifu2xEnv(self.model, criterion=criterion,
                          discriminator=self.discriminator,
                          discriminator_criterion=discriminator_criterion,
                          discriminator_weight=self.args.discriminator_weight,
                          discriminator_only=bool(self.args.discriminator_only),
                          update_criterion=self.args.update_criterion)

    def setup(self):
        dataset = self.train_loader.dataset
        dataset.set_hard_example(self.args.hard_example, self.args.hard_example_scale)

    def setup_model(self):
        self.discriminator = create_discriminator(self.args.discriminator, self.device)

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

    def create_optimizers(self):
        if self.discriminator is not None:
            g_opt = self.create_optimizer(self.model)
            d_opt = self.create_optimizer(self.discriminator)
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

    def create_filename_prefix(self):
        if self.args.method == "scale":
            return "scale2x"
        elif self.args.method == "scale4x":
            return "scale4x"
        elif self.args.method == "noise_scale":
            return f"noise{self.args.noise_level}_scale2x"
        elif self.args.method == "noise_scale4x":
            return f"noise{self.args.noise_level}_scale4x"
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
