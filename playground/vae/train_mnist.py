# VAE
# python3 -m playground.vae.train_mnist --data-dir ./tmp/vae --model-dir ./tmp/vae
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
import torch
from torch import nn
from os import path
from PIL import ImageOps
from nunif.models import Model, get_model_device
from nunif.modules import functional as NF
from nunif.training.env import UnsupervisedEnv
from nunif.training.trainer import Trainer, create_trainer_default_parser


IMAGE_SCALE = 0.5
MNIST_SIZE = 28
IMAGE_SIZE = int(MNIST_SIZE * IMAGE_SCALE)


class VAE(Model):
    name = "vae"

    def __init__(self, input_dim, feat_dim, latent_dim):
        super().__init__(locals())
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True))
        self.to_mean = nn.Linear(feat_dim, latent_dim)
        self.to_log_var = nn.Linear(feat_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, input_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
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
            z = torch.sigmoid(z)
        return z

    def reparameterize(self, mean, log_var):
        return NF.gaussian_noise(mean, log_var)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C * H * W)
        mean, log_var = self.encode(x)
        sample = self.reparameterize(mean, log_var)
        recon = self.decode(sample)
        return recon, mean, log_var, x


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, z):
        recon, mean, log_var, x = z
        recon_loss = self.bce(recon, x)
        kl_loss = NF.gaussian_kl_divergence_loss(mean, log_var)
        beta = mean.shape[1] / recon.shape[1]
        loss = beta * kl_loss + recon_loss
        return loss


class MNISTImageOnly(MNIST):
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x


class MinMaxNormalize():
    def __call__(self, x):
        min_v, max_v = x.min(), x.max()
        return (x - min_v) / (max_v - min_v)


class VAEEnv(UnsupervisedEnv):
    def __init__(self, model, criterion):
        super().__init__(model, criterion)
        device = get_model_device(model)
        self.validation_data = torch.randn((8 * 8, model.latent_dim), device=device)

    def draw_map(self, size=33):
        # draw 2d topological map from 2d latent vector
        assert self.model.latent_dim == 2
        lin = torch.linspace(-2, 2, size, dtype=torch.float32)
        cols = lin.view(1, size).expand(size, size).view(size, size, 1)
        rows = lin.view(size, 1).expand(size, size).view(size, size, 1)
        loc = torch.cat([rows, cols], dim=2)
        loc[size // 2, size // 2].fill_(0)  # center = all zero latent vector
        with torch.no_grad():
            x = loc.view(size * size, 2).to(get_model_device(self.model))
            images = self.model.decode(x)
            images = images.view(images.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)
        im = ImageOps.invert(TF.to_pil_image(make_grid(images, nrow=33, padding=0)))
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
            images = self.model.decode(self.validation_data)
            images = images.view(images.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)
            im = TF.to_pil_image(make_grid(images, nrow=8, padding=2))
            im.save(output_file)
            print(f"save random images to `{output_file}`")


class VAETrainer(Trainer):
    def create_model(self):
        model = VAE(IMAGE_SIZE * IMAGE_SIZE, 512, 2).to(self.device)
        return model

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                MinMaxNormalize()
            ])
            dataset = MNISTImageOnly(self.args.data_dir, train=True, download=True, transform=transform)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
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
    parser.set_defaults(
        batch_size=32,
        num_workers=2,
        max_epoch=100,
        learning_rate=0.00025,
        learning_rate_decay=0.98,
        disable_amp=True
    )
    args = parser.parse_args()
    trainer = VAETrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
