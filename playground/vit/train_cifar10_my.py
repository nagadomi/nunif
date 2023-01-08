# A ViT designed from what I heard
# 92% accuracy on CIFAR10 using only CIFAR10 training data,
# python3 -m playground.vit.train_cifar10_my --data-dir ./tmp/vit --model-dir ./tmp/vit
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import torch
from torch import nn
from torch.nn import functional as F
from nunif.models import SoftmaxBaseModel
from nunif.training.env import SoftmaxEnv
from nunif.training.trainer import Trainer, create_trainer_default_parser
import nunif.transforms as NT
import math


def normalize(x):
    return (x - 0.5) * 2


class Normalize():
    def __call__(self, x):
        return normalize(x)


IMG_SIZE = 64


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train):
        super().__init__()
        self.train = train
        if train:
            t1 = T.Compose([
                T.RandomCrop((24, 24)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            t2 = T.Compose([
                T.RandomCrop((26, 26)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            t3 = T.Compose([
                T.RandomCrop((28, 28)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            t4 = T.Compose([
                T.RandomCrop((30, 30)),
                T.Resize((IMG_SIZE, IMG_SIZE))])
            transform = T.Compose([
                T.RandomChoice([NT.Identity(),
                                T.Compose([
                                    NT.ReflectionResize((38, 38)),
                                    T.RandomPerspective(distortion_scale=0.15, p=1),
                                    T.CenterCrop((32, 32))])]),
                T.RandomChoice([t1, t2, t3, t4]),
                T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                Normalize(),
            ])
        else:
            # TTA at __getitem__
            transform = None
        self.cifar10 = CIFAR10(root, train, transform=transform, download=True)

    def __len__(self):
        return len(self.cifar10)

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
        x, y = self.cifar10[i]
        if self.train:
            return x, y
        else:
            tta = []
            tta += [TF.resize(crop, (IMG_SIZE, IMG_SIZE)) for crop in TF.five_crop(x, (24, 24))]
            tta += [TF.resize(crop, (IMG_SIZE, IMG_SIZE)) for crop in TF.five_crop(x, (28, 28))]
            tta += [TF.resize(x, (IMG_SIZE, IMG_SIZE))]
            tta = tta + [TF.hflip(crop) for crop in tta]
            x = torch.stack([normalize(TF.to_tensor(crop)) for crop in tta], dim=0)
            return x, y


CIFAR10_CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class DropPath(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.register_buffer("mask", None)

    def reset_parameters(self, x):
        self.mask = x.new_empty(x.shape[0], *((1,) * (x.ndim - 1))).fill_(1)

    def forward(self, x):
        if self.training:
            if self.mask is None or self.mask.shape[0] != x.shape[0]:
                self.reset_parameters(x)
            return x * F.dropout(self.mask, p=self.p, training=self.training)
        else:
            return x


class Residual(nn.Module):
    def __init__(self, layer, shortcut=None, drop_path=0.2):
        super().__init__()
        self.layer = layer
        self.shortcut = shortcut or nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.layer(x)) + self.shortcut(x)


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mask_size, mask_kernel_size, drop_path=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim,
            dropout=0,
            activation=F.gelu,
            norm_first=True, batch_first=True)
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim))
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.register_buffer("mask", self.generate_window_mask(mask_size, mask_kernel_size))

    @staticmethod
    def generate_window_mask(size, kernel_size):
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        cols = ([-math.inf for _ in range(pad)] +
                [i for i in range(size)] +
                [-math.inf for _ in range(pad)])
        pad_cols = [-math.inf for _ in range(len(cols))]
        rows = [pad_cols]
        for i in range(size):
            rows.append([i * size + v for v in cols])
        rows.append(pad_cols)
        rows = [[i if i >= 0 else -1 for i in row] for row in rows]
        grid = torch.tensor(rows, dtype=torch.long)
        #  print("***", size, grid)
        mask_indices = []
        for y in range(size):
            for x in range(size):
                indices = grid[y:y + kernel_size,
                               x:x + kernel_size].flatten()
                valid_indices = [i.item() for i in indices if i >= 0]
                mask_indices.append(valid_indices)
        #  print("***", size, mask_indices)
        mask = torch.full((size ** 2, size ** 2), -math.inf,
                          dtype=torch.float, requires_grad=False)
        for i, indices in enumerate(mask_indices):
            for j in indices:
                mask[i][j] = 0.0
        return mask

    def forward(self, x):
        x = x + self.drop_path1(self.attn(x, mask=self.mask))
        x = x + self.drop_path2(self.mlp(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, N, C = x.shape
        assert int(math.sqrt(N)) ** 2 == N
        size = int(math.sqrt(N))
        x = x.permute(0, 2, 1).view(B, C, size, size)
        x = self.conv(x)
        x = x.view(B, x.shape[1], size // 2 * size // 2).permute(0, 2, 1).contiguous()
        x = self.norm(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        w = self.attn(x)
        w = F.softmax(w.view(B, N, 1), dim=1).expand(x.shape)
        z = (x * w).sum(dim=1)
        return z


class VIT(SoftmaxBaseModel):
    name = "myvit"

    def __init__(self):
        super().__init__(locals(), CIFAR10_CLASS_NAMES)
        dim = 64
        self.patch = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, dim, kernel_size=4, stride=4, padding=0),
        )
        #self.pos_embed = nn.Parameter(torch.zeros((1, dim, 16, 16)))
        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = nn.Sequential(
            Transformer(dim, 8, num_layers=2, mask_size=16, mask_kernel_size=5),
            PatchMerging(dim, dim * 2),
            Transformer(dim * 2, 8, num_layers=6, mask_size=8, mask_kernel_size=5),
            PatchMerging(dim * 2, dim * 4),
            Transformer(dim * 4, 16, num_layers=2, mask_size=4, mask_kernel_size=3),
        )
        self.fc = nn.Sequential(
            AttentionPooling(dim * 4, reduction=8),
            nn.Linear(dim * 4, len(CIFAR10_CLASS_NAMES))
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        # 24x24
        x = self.patch(x) # + self.pos_embed
        # 12x12
        x = x.view(B, x.shape[1], -1).permute(0, 2, 1).contiguous()
        x = self.transformer(x)
        z = self.fc(x)
        if self.training:
            return F.log_softmax(z, dim=1)
        else:
            return F.softmax(z, dim=1)


class CIFAR10Trainer(Trainer):
    def create_model(self):
        model = VIT().to(self.device)
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
                batch_size=self.args.batch_size // 4,
                shuffle=False,
                pin_memory=True,
                num_workers=self.args.num_workers,
                drop_last=False)
            return loader

    def create_env(self):
        return SoftmaxEnv(self.model, eval_tta=True)


def main():
    parser = create_trainer_default_parser()
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.set_defaults(
        batch_size=128,
        num_workers=4,
        scheduler="cosine",
        optimizer="adamw",
        learning_rate=0.00015,
        learning_rate_decay=0.99,
        warmup_epoch=2,
        warmup_learning_rate=1e-7,
        max_epoch=200,
        disable_amp=False
    )
    args = parser.parse_args()
    trainer = CIFAR10Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
