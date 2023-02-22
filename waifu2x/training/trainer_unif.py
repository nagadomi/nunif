from os import path
import torch
from . dataset_unif import Waifu2xUnifiedDataset
from nunif.models import get_model_config
from .trainer import create_criterion, Waifu2xEnv, Waifu2xTrainer


# 1x 2x 4x unified training


class Waifu2xUnifiedEnv(Waifu2xEnv):
    def train_step(self, data):
        x, y, scale_factors, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        # scale_factor in {1, 2, 4}
        scale_factor = scale_factors[0].item()
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            z = self.model(x, scale_factor)
            loss = self.criterion(z, y)
        if scale_factor == 1:
            loss = loss * 0.25
        elif scale_factor == 2:
            loss = loss * 0.5
        self.sum_loss += loss.item()
        self.sum_step += 1
        return loss

    def eval_step(self, data):
        x, y, scale_factors, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        scale_factor = scale_factors[0].item()
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            z = self.model(x, scale_factor)
            loss = self.eval_criterion(z, y)
        self.sum_loss += loss.item()
        self.sum_step += 1


class Waifu2xUnifiedTrainer(Waifu2xTrainer):
    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if self.args.method in {"unif4x", "noise_unif4x"}:
            scale_factors = get_model_config(self.model, "i2i_unif_scale_factors")
            model_offsets = get_model_config(self.model, "i2i_unif_model_offsets")
            if self.args.method == "unif4x" and scale_factors[0] == 1:
                # remove 1x for no denoising models
                scale_factors = scale_factors[1:]
                model_offsets = model_offsets[1:]
        else:
            raise NotImplementedError()
        assert len(scale_factors) == len(model_offsets)

        if type == "train":
            dataset = Waifu2xUnifiedDataset(
                input_dir=path.join(self.args.data_dir, "train"),
                scale_factors=scale_factors,
                model_offsets=model_offsets,
                batch_size=self.args.batch_size,
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
                dataset,
                worker_init_fn=dataset.worker_init,
                pin_memory=True,
                batch_sampler=dataset.batch_sampler(),
                persistent_workers=True,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor)
        elif type == "eval":
            dataset = Waifu2xUnifiedDataset(
                input_dir=path.join(self.args.data_dir, "eval"),
                model_offsets=model_offsets,
                scale_factors=scale_factors,
                batch_size=self.args.batch_size,
                style=self.args.style,
                noise_level=self.args.noise_level,
                tile_size=self.args.size,
                deblur=self.args.deblur,
                training=False)
            return torch.utils.data.DataLoader(
                dataset,
                worker_init_fn=dataset.worker_init,
                batch_sampler=dataset.batch_sampler(),
                persistent_workers=True,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor)

    def create_env(self):
        criterion = create_criterion(self.args.loss).to(self.device)
        return Waifu2xUnifiedEnv(self.model, criterion=criterion)

    def create_best_model_filename(self):
        if self.args.method == "unif4x":
            return path.join(self.args.model_dir, "unif4x.pth")
        elif self.args.method == "noise_unif4x":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}_unif4x.pth")
        else:
            raise NotImplementedError()

    def create_checkpoint_filename(self):
        if self.args.method == "unif4x":
            return path.join(self.args.model_dir, "unif4x.checkpoint.pth")
        elif self.args.method == "noise_unif4x":
            return path.join(self.args.model_dir, f"noise{self.args.noise_level}_unif4x.checkpoint.pth")
        else:
            raise NotImplementedError()
