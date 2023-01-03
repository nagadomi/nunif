# Conv VAE
# python3 -m playground.vae.train_celeba --data-dir ./tmp/vae --model-dir ./tmp/vae --latent-dim 2
from torchvision.datasets import CelebA
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
import torch
from torch import nn
from torch.nn import functional as F
from os import path
from nunif.models import Model, get_model_device
from nunif.modules import functional as NF
from nunif.training.env import UnsupervisedEnv
from nunif.training.trainer import Trainer, create_trainer_default_parser
from nunif.modules.res_block import ResBlock as _ResBlock, ResGroup as _ResGroup
from nunif.modules.attention import SEBlock
from nunif.modules.embedding import PositionalSeeding
from nunif.modules import ClampLoss, LBPLoss, CharbonnierLoss, LuminanceWeightedLoss


IMAGE_SIZE = 64


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        super().__init__()
        transform = T.Compose([
            T.CenterCrop((160, 160)),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomAutocontrast(1),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        self.celeba = CelebA(root, split, target_type="bbox",
                             transform=transform, download=True)

    def __len__(self):
        return len(self.celeba)

    def sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def show(self, im):
        from nunif.utils.pil_io import to_cv2
        import cv2
        cv2.imshow("debug", to_cv2(im))
        cv2.waitKey(0)

    def __getitem__(self, i):
        x, y = self.celeba[i]
        return x


class ResBlock(_ResBlock):
    def __init__(self, in_channels, out_channels, stride, se):
        self.se = se
        super().__init__(in_channels, out_channels, stride)

    def bias_enabled(self):
        return True

    def padding_mode(self):
        return "replicate"

    def create_activate_function(self):
        return nn.LeakyReLU(0.2, inplace=True)

    def create_norm_layer(self, in_channels):
        return nn.Identity()

    def create_attention_layer(self, in_channels):
        if self.se:
            return SEBlock(in_channels, bias=True)
        else:
            return nn.Identity()


class ResGroup(_ResGroup):
    def __init__(self, in_channels, out_channels, num_layers, stride, se):
        self.se = se
        super().__init__(in_channels, out_channels, num_layers, stride)

    def create_layer(self, in_channels, out_channels, stride):
        return ResBlock(in_channels, out_channels, stride, se=self.se)


class Encoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            # 64x64
            ResGroup(3, 32, num_layers=1, stride=1, se=False),
            ResGroup(32, 64, num_layers=2, stride=2, se=True),
            # 32x32
            ResGroup(64, 128, num_layers=2, stride=2, se=True),
            # 16x16
            ResGroup(128, 256, num_layers=2, stride=2, se=True),
            # 8x8
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim))

    def forward(self, x):
        return self.net(x)


class Up2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                      padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.seeding = PositionalSeeding(in_channels=latent_dim, out_channels=256, upscale_factor=8)
        self.up1 = nn.Sequential(
            # 8x8
            ResGroup(256, 256, num_layers=2, stride=1, se=True),
            Up2x(256))
        self.up2 = nn.Sequential(
            # 16x16
            ResGroup(256, 128, num_layers=2, stride=1, se=True),
            Up2x(128))
        self.up3 = nn.Sequential(
            # 32x32
            ResGroup(128, 64, num_layers=2, stride=1, se=False),
            Up2x(64))
        # 64x64
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode="replicate")

    def forward(self, x):
        B, C = x.shape
        seed = self.seeding(x)
        x = self.up1(seed)
        x = self.up2(x)
        x = self.up3(x)
        z = self.final_conv(x)
        return z


class ConvVAE(Model):
    name = "conv_vae"

    def __init__(self, latent_dim):
        super().__init__(locals())
        self.latent_dim = latent_dim
        self.encoder = Encoder(512)
        self.to_mean = nn.Linear(512, latent_dim)
        self.to_log_var = nn.Linear(512, latent_dim)
        self.decoder = Decoder(latent_dim)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        feat = self.encoder(x)
        mean = self.to_mean(feat)
        log_var = self.to_log_var(feat)
        return mean, log_var

    def decode(self, x):
        z = self.decoder(x)
        if not self.training:
            z = torch.clamp(z, 0, 1)
        return z

    def reparameterize(self, mean, log_var):
        return NF.gaussian_noise(mean, log_var)

    def forward(self, x):
        B, C, H, W = x.shape
        mean, log_var = self.encode(x)
        sample = self.reparameterize(mean, log_var)
        recon = self.decode(sample)
        return recon, mean, log_var, x


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_loss = ClampLoss(LuminanceWeightedLoss(LBPLoss(in_channels=1)))
        #  self.pixel_loss = ClampLoss(LuminanceWeightedLoss(CharbonnierLoss()))

    def forward(self, z):
        recon, mean, log_var, x = z
        recon_loss = self.pixel_loss(recon, x)
        kl_loss = NF.gaussian_kl_divergence_loss(mean, log_var)
        beta = mean.shape[1] / sum(recon.shape[1:])
        loss = beta * kl_loss + recon_loss
        return loss


class VAEEnv(UnsupervisedEnv):
    def __init__(self, model, criterion):
        super().__init__(model, criterion)
        device = get_model_device(model)
        self.validation_data = torch.randn((15, 15, model.latent_dim), device=device)

    def draw_map(self, size=25):
        # draw 2d topological map from 2d latent vector
        assert self.model.latent_dim == 2
        lin = torch.linspace(-2, 2, size, dtype=torch.float32)
        cols = lin.view(1, size).expand(size, size).view(size, size, 1)
        rows = lin.view(size, 1).expand(size, size).view(size, size, 1)
        loc = torch.cat([rows, cols], dim=2)
        loc[size // 2, size // 2].fill_(0)  # center = all zero latent vector
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.amp):
            x = loc.view(size, size, 2).to(get_model_device(self.model))
            images = []
            # split forward due to VRAM usage
            for col in x:
                images.append(self.model.decode(col))
        images = torch.cat(images, dim=0)
        im = TF.to_pil_image(make_grid(images, nrow=size, padding=0))
        return im

    def eval_end(self):
        model_dir = path.relpath(self.trainer.args.model_dir)
        if self.model.latent_dim == 2:
            # draw 2d topological map
            output_file = path.join(model_dir, f"vae_{self.trainer.epoch}.png")
            im = self.draw_map()
            im.save(output_file)
            print(f"save map to `{output_file}`")
        else:
            # draw random
            output_file = path.join(model_dir, f"vae_{self.trainer.epoch}.png")
            images = []
            for col in self.validation_data:
                images.append(self.model.decode(col))
            images = torch.cat(images, dim=0)
            im = TF.to_pil_image(make_grid(images, nrow=15, padding=0))
            im.save(output_file)
            print(f"save random images to `{output_file}`")


class VAETrainer(Trainer):
    def create_model(self):
        model = ConvVAE(self.args.latent_dim).to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            dataset = CelebADataset(self.args.data_dir, split="train")
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=dataset.sampler(self.args.num_samples),
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader
        else:
            return None

    def create_env(self):
        criterion = VAELoss().to(self.device)
        return VAEEnv(self.model, criterion=criterion)


def main():
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.set_defaults(
        batch_size=32,
        num_workers=4,
        max_epoch=200,
        learning_rate=0.00025,
        learning_rate_decay=0.985,
        optimizer="adam",
        disable_amp=False
    )
    args = parser.parse_args()
    trainer = VAETrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
