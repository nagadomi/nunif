# A rough implementation of Masked Autoencoders https://arxiv.org/abs/2111.06377
# for CIFAR10
# python -m playground.mae.train_cifar10 --data-dir ./data/cifar10 --model-dir ./models/mae
# reconstruction result will be stored in ./models/mae/eval
# https://github.com/user-attachments/assets/18c38401-4c44-4f93-ac91-e497c6ec0e7a
from os import path
import os
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode)
import torch
from torch import nn
from torch.nn import functional as F
from nunif.models import Model
from nunif.modules.init import basic_module_init
from nunif.modules.attention import MHA
from nunif.training.env import RGBPSNREnv
from nunif.training.trainer import Trainer, create_trainer_default_parser
from torchvision.utils import make_grid


IMG_SIZE = 64
EVAL_N = 32
PATCH_SIZE = 4
MASK_RATIO = 0.7


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train):
        super().__init__()
        self.train = train
        if train:
            transform = T.Compose([
                T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            self.cifar10 = CIFAR10(root, train=train, transform=transform, download=True)
        else:
            if IMG_SIZE == 32:
                transform = T.ToTensor()
            else:
                transform = T.Compose([
                    T.Resize(IMG_SIZE),
                    T.ToTensor()
                ])
            self.cifar10 = CIFAR10(root, train=train, transform=transform, download=True)
            self.eval_index = torch.randperm(len(self.cifar10))[:EVAL_N]

    def __len__(self):
        if self.train:
            return len(self.cifar10)
        else:
            return len(self.eval_index)

    def sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def __getitem__(self, i):
        if self.train:
            x, y = self.cifar10[i]
            return x, x
        else:
            x, y = self.cifar10[self.eval_index[i]]
            return x, x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        num_heads = max(embed_dim // 32, 1)
        self.mha = MHA(embed_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim))
        self.norm_mha = nn.LayerNorm(embed_dim, bias=False)
        self.norm_mlp = nn.LayerNorm(embed_dim, bias=False)
        basic_module_init(self.mlp)

    def forward(self, x):
        x = x + self.mha(self.norm_mha(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, mask_ratio, num_blocks, image_size, patch_size):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.masked_num_patches = self.num_patches - int(self.num_patches * mask_ratio)
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size,
                               stride=patch_size, padding=0, bias=False)
        # NOTE: No cls token
        self.pos_bias = nn.Parameter(torch.randn((1, self.num_patches, embed_dim)) * 0.01)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim=embed_dim) for _ in range(num_blocks)])

        basic_module_init(self.patch)

    def forward(self, x, index_shuffle=None):
        # patch embed
        x = self.patch(x)
        B, C, H, W = x.shape
        N = H * W
        assert self.num_patches == N

        x = x.permute(0, 2, 3, 1).reshape(B, N, C)
        x = x + self.pos_bias
        # generate mask and shuffle index
        if index_shuffle is None:
            noise = torch.rand((B, N), dtype=x.dtype, device=x.device)
            index_shuffle = torch.argsort(noise, dim=1).reshape(B, N, 1)
        else:
            assert index_shuffle.shape == (B, N, 1)
        index_restore = torch.argsort(index_shuffle, dim=1).reshape(B, N, 1)
        # drop mask
        x = x.take_along_dim(index_shuffle, dim=1)[:, :self.masked_num_patches, :]
        # encoder
        for block in self.blocks:
            x = block(x)

        return x, index_restore


class ToImage(nn.Module):
    def __init__(self, in_channels, scale_factor, out_channels=3):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels * scale_factor ** 2)
        self.scale_factor = scale_factor
        basic_module_init(self.proj)

    def forward(self, x):
        x = self.proj(x)
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # expect only square
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_blocks, image_size, patch_size):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.pos_bias = nn.Parameter(torch.randn((1, self.num_patches, embed_dim)) * 0.01)
        self.mask_bias = nn.Parameter(torch.zeros((1, 1, embed_dim)))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim=embed_dim) for _ in range(num_blocks)])
        self.to_image = ToImage(embed_dim, scale_factor=patch_size)

    def forward(self, x, index_restore):
        B, N, C = x.shape
        # append mask token
        x = torch.cat([x, self.mask_bias.expand(B, self.num_patches - N, -1)], dim=1)
        # restore index shuffle
        x = x.take_along_dim(index_restore, dim=1)
        # pos embed
        x = x + self.pos_bias
        # decoder
        for block in self.blocks:
            x = block(x)
        # restore to image
        x = self.to_image(x)
        return x


class MaskedAutoencoder(Model):
    def __init__(self, embed_dim=128, num_blocks=4, image_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO):
        super().__init__({})
        self.encoder = Encoder(embed_dim=embed_dim, num_blocks=num_blocks, image_size=image_size, patch_size=patch_size,
                               mask_ratio=mask_ratio)
        self.decoder = Decoder(embed_dim=embed_dim, num_blocks=num_blocks, image_size=image_size, patch_size=patch_size)

    def forward(self, x, index_shuffle=None):
        features, index = self.encoder(x, index_shuffle)
        recon = self.decoder(features, index)
        if not self.training:
            recon = recon.clamp(0, 1)

        return recon


class MAEEnv(RGBPSNREnv):
    def eval_step(self, data):
        x, y, *_ = data
        x, y = self.to_device(x), self.to_device(y)
        model = self.get_eval_model()

        # pre mask fill for visualization. mask=black
        B, C, H, W = x.shape
        x = F.pixel_unshuffle(x, PATCH_SIZE).permute(0, 2, 3, 1).reshape(B, (H // PATCH_SIZE) ** 2, -1)
        B, N, _ = x.shape
        noise = torch.rand((B, N), dtype=x.dtype, device=x.device)
        index_shuffle = torch.argsort(noise, dim=1).reshape(B, N, 1)
        index_restore = torch.argsort(index_shuffle, dim=1).reshape(B, N, 1)
        masked_num_patches = N - int(N * MASK_RATIO)
        x = x.take_along_dim(index_shuffle, dim=1)[:, :masked_num_patches, :]
        x = torch.cat([x, torch.zeros((1, 1, x.shape[2]), dtype=x.dtype, device=x.device).expand(B, N, -1)], dim=1)
        x = x.take_along_dim(index_restore, dim=1).permute(0, 2, 1).reshape(B, -1, (H // PATCH_SIZE), (H // PATCH_SIZE))
        x = F.pixel_shuffle(x, PATCH_SIZE)

        with self.autocast():
            z = model(x, index_shuffle)
            self.save_eval(x, z, y, self.sum_step)
            loss = self.eval_criterion(z, y)

        self.sum_loss += loss.item()
        self.sum_step += 1

    def save_eval(self, x, z, y, i):
        x = torch.cat([x, z, y], dim=3)
        eval_output_dir = path.join(self.trainer.args.model_dir, "eval")
        os.makedirs(eval_output_dir, exist_ok=True)
        output_file = path.join(eval_output_dir, f"{i}.png")
        x = TF.to_pil_image(make_grid(x, nrow=1))
        # x = TF.resize(x, (x.height * 2, x.width * 2), interpolation=InterpolationMode.NEAREST)
        x.save(output_file)


class CIFAR10Trainer(Trainer):
    def create_model(self):
        model = MaskedAutoencoder().to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = CIFAR10Dataset(self.args.data_dir, train=True)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            return loader
        else:
            dataset = CIFAR10Dataset(self.args.data_dir, train=False)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader

    def create_env(self):
        criterion = nn.L1Loss().to(self.device)
        return MAEEnv(self.model, criterion)


def main():
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=60000)
    parser.set_defaults(
        batch_size=128,
        num_workers=8,
        optimizer="adamw",
        learning_rate=1e-4,
        scheduler="cosine",
        learning_rate_cycles=5,
        max_epoch=200,
        disable_amp=False,
    )
    args = parser.parse_args()
    trainer = CIFAR10Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
