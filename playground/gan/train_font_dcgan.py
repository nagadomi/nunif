# Generate JIS Level-1 Kanji Fonts with DCGAN
# python -m font_resource.download_google_fonts
# python -m playground.gan.train_font_dcgan --data-dir ./tmp/dcgan --model-dir ./tmp/dcgan
from os import path
from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
import torch
from torch import nn
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from nunif.models import Model, get_model_device, save_model
from nunif.training.env import BaseEnv
from nunif.training.trainer import Trainer, create_trainer_default_parser
from font_resource.utils import load_font
from text_resource.char import Char


IMAGE_SIZE = 32
FONT_NAMES = [
    "Yuji Boku Regular",
    "Yuji Mai Regular",
    "Yuji Syuku Regular",
    "BIZ UDGothic",
    "M PLUS 1 Code Regular",
    "Noto Sans JP",
    "Noto Serif JP",
    "Shippori Mincho B1 Regular",
]


class FontDataset(torch.utils.data.Dataset):
    def __init__(self, fonts, size, chars):
        super().__init__()
        assert all([font_info.drawable("".join(chars)) for font_info in fonts])
        self.size = size
        self.margin = int(size * 0.2)
        self.font_size = size - self.margin * 2
        self.fonts = [ImageFont.truetype(font_info.file_path, size=self.font_size) for font_info in fonts]
        self.chars = chars
        self.font_images = defaultdict(lambda: {})

    def sampler(self, num_samples):
        return torch.utils.data.sampler.RandomSampler(
            self,
            num_samples=num_samples,
            replacement=True)

    def generate_font_image(self, font, char):
        im = Image.new("L", (self.size, self.size), (255,))
        gc = ImageDraw.Draw(im, mode="L")
        gc.text((self.margin, self.margin), char + "　", font=font, fill="black",
                direction="ttb", anchor=None, language="ja")
        return im

    def __getitem__(self, i):
        font_index = i // len(self.chars)
        char_index = i % len(self.chars)
        font = self.fonts[font_index]
        char = self.chars[char_index]
        if char_index not in self.font_images[font_index]:
            self.font_images[font_index][char_index] = self.generate_font_image(font, char)
        return TF.to_tensor(self.font_images[font_index][char_index])

    def __len__(self):
        return len(self.chars) * len(self.fonts)


def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, 0.0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d,)):
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)


class Generator(Model):
    name = "playground.gan.dcgan_font_generator"

    def __init__(self, seed_dim=64):
        super().__init__(locals())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(seed_dim, 256,
                               kernel_size=4, stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128,
                               kernel_size=4, stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64,
                               kernel_size=4, stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64,
                               kernel_size=4, stride=2,
                               padding=1,
                               bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1, padding_mode="replicate",
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1,
                      padding=1, padding_mode="replicate",
                      bias=False),

            nn.Tanh()
        )
        reset_parameters(self)

    def forward(self, x):
        z = self.net(x)
        return z


class DiscriminatorHighLevel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 1x1
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        reset_parameters(self)

    def forward(self, x):
        z = self.net(x)
        return z


class DiscriminatorLowLevel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            # 32x32
        )
        reset_parameters(self)

    def forward(self, x):
        z = self.net(x)
        return z


class GANWrapper(Model):
    name = "playground.gan.dcgan_font_wrapper"

    def __init__(self, generator, discriminator_high, discriminator_low):
        super().__init__({})
        self.generator = generator
        self.discriminator_high = discriminator_high
        self.discriminator_low = discriminator_low

    def forward(self, **kwargs):
        raise NotImplementedError()


class GANEnv(BaseEnv):
    def __init__(self, model, seed_dim):
        super().__init__()
        self.model = model
        self.seed_dim = seed_dim
        self.device = get_model_device(self.model)
        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_data = torch.randn((64, seed_dim, 1, 1), dtype=torch.float32, device=self.device)

    def clear_loss(self):
        self.sum_g_loss = 0
        self.sum_d_loss = 0
        self.sum_step = 0

    def train_begin(self):
        self.model.train()
        self.clear_loss()

    def train_step(self, data):
        real = self.to_device(data)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            # generator
            noise = torch.randn((real.shape[0], self.seed_dim, 1, 1),
                                dtype=real.dtype,
                                device=real.device)
            fake = self.model.generator(noise)

            yl_fake = self.model.discriminator_low(fake)
            tl_fake = torch.zeros(yl_fake.shape, dtype=yl_fake.dtype,
                                  device=yl_fake.device, requires_grad=False)
            tl_real = torch.ones(yl_fake.shape, dtype=yl_fake.dtype,
                                 device=yl_fake.device, requires_grad=False)

            yh_fake = self.model.discriminator_high(fake)
            th_fake = torch.zeros(yh_fake.shape, dtype=yh_fake.dtype,
                                  device=yh_fake.device, requires_grad=False)
            th_real = torch.ones(yh_fake.shape, dtype=yh_fake.dtype,
                                 device=yh_fake.device, requires_grad=False)

            g_loss = sum([self.criterion(yl_fake, tl_real),
                          self.criterion(yh_fake, th_real)]) * 0.5

            # discriminator
            yl_fake = self.model.discriminator_low(fake.detach())
            yl_real = self.model.discriminator_low(real)
            yh_fake = self.model.discriminator_high(fake.detach())
            yh_real = self.model.discriminator_high(real)
            d_loss = sum([self.criterion(yl_fake, tl_fake),
                          self.criterion(yl_real, tl_real),
                          self.criterion(yh_fake, th_fake),
                          self.criterion(yh_real, th_real),
                          ]) * 0.25

        self.sum_g_loss += g_loss.item()
        self.sum_d_loss += d_loss.item()
        self.sum_step += 1

        return g_loss, d_loss

    def train_backward_step(self, loss, optimizers, grad_scaler, update):
        g_loss, d_loss = loss
        g_opt, dl_opt, dh_opt = optimizers

        # update generator
        g_opt.zero_grad()
        self.backward(g_loss, grad_scaler)
        self.optimizer_step(g_opt, grad_scaler)

        # update discriminator
        dl_opt.zero_grad()
        dh_opt.zero_grad()
        self.backward(d_loss, grad_scaler)
        self.optimizer_step(dl_opt, grad_scaler)
        self.optimizer_step(dh_opt, grad_scaler)

    def train_end(self):
        mean_g_loss = self.sum_g_loss / self.sum_step
        mean_d_loss = self.sum_d_loss / self.sum_step
        print(f"generator loss: {mean_g_loss}, discriminator loss: {mean_d_loss}")
        return mean_g_loss + mean_d_loss

    def eval_begin(self):
        self.model.eval()

    def eval_step(self, data):
        pass

    def eval_end(self):
        model_dir = path.relpath(self.trainer.args.model_dir)
        output_file = path.join(model_dir, f"dcgan_{self.trainer.epoch}.png")
        images = []
        with torch.no_grad():
            images = self.model.generator(self.validation_data)
        im = TF.to_pil_image(make_grid(images, nrow=8, padding=2, normalize=True, scale_each=True))
        im.save(output_file)
        print(f"save random images to `{output_file}`")

        return None


class GANTrainer(Trainer):
    def create_model(self):
        model = GANWrapper(
            generator=Generator(self.args.seed_dim),
            discriminator_low=DiscriminatorLowLevel(),
            discriminator_high=DiscriminatorHighLevel(),
        ).to(self.device)
        return model

    def save_best_model(self):
        generator = self.model.generator
        model_path = path.join(self.args.model_dir, f"{generator.name}.pth")
        save_model(generator, model_path)

    def create_optimizers(self):
        g = self.create_optimizer(self.model.generator)
        dl = self.create_optimizer(self.model.discriminator_low)
        dh = self.create_optimizer(self.model.discriminator_high)
        return g, dl, dh

    def create_dataloader(self, type):
        assert (type in {"train", "eval"})
        if type == "train":
            fonts = []
            for font_name in FONT_NAMES:
                font_info = load_font(font_name)
                if font_info is None:
                    raise ValueError(f"{font_name} not found")
                fonts.append(font_info)
            dataset = FontDataset(fonts, size=IMAGE_SIZE, chars=list(Char.JIS1))
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
        return GANEnv(self.model, self.args.seed_dim)


def _test_model():
    g = Generator(128)
    dl = DiscriminatorLowLevel()
    dh = DiscriminatorHighLevel()

    x = torch.zeros((4, 128, 1, 1))
    z = g(x)
    print("generator", z.shape)

    x = torch.zeros((4, 1, IMAGE_SIZE, IMAGE_SIZE))
    z = dl(x)
    print("discriminator_low", z.shape)
    z = dh(x)
    print("discriminator_high", z.shape)


def _test_dataset():
    import random

    def show_image(im):
        from nunif.utils.pil_io import to_cv2
        import cv2
        cv2.namedWindow("debug", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("debug", to_cv2(im))
        return cv2.waitKey(0)
    print("`q` key to exit")

    fonts = [load_font(font_name) for font_name in FONT_NAMES]
    dataset = FontDataset(fonts, size=IMAGE_SIZE, chars=list(Char.JIS1))
    for _ in range(100):
        im = dataset[random.randint(0, len(dataset) - 1)]
        if show_image(TF.to_pil_image(im)) in {ord("q"), ord("x")}:
            break


def main():
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.add_argument("--seed-dim", type=int, default=128)
    parser.set_defaults(
        batch_size=128,
        num_workers=4,
        max_epoch=100,
        learning_rate=0.0002,
        learning_rate_decay=0.96,
        optimizer="adam",
        adam_beta1=0.5,
    )
    args = parser.parse_args()
    trainer = GANTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    # _test_dataset()
    # _test_model()
    main()
