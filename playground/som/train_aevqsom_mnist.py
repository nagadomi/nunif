# AE-VQ-SOM
# python3 -m playground.som.train_aevqsom_mnist
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
import torch
from torch import nn
from torch.nn import functional as F
from os import path
import math
from tqdm import tqdm
from PIL import ImageOps


DATA_DIR = path.join(path.dirname(__file__), "..", "..", "tmp", "aevqsom")


class SOMVectorQuantizer(nn.Module):
    def __init__(self, grid_size, input_size, max_t, step_t, warmup_t=0, max_distance_factor=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.max_kernel_size = grid_size
        self.t = 1
        self.max_t = max_t
        self.step_t = step_t
        self.warmup_t = -warmup_t
        self.units = nn.Parameter(torch.rand(size=(grid_size, grid_size, input_size), requires_grad=True))
        self.deno = 1.0 / (0.9 * max_t / math.log(self.max_kernel_size))
        coord = torch.empty((grid_size * grid_size, 2), dtype=torch.float32)
        for y in range(grid_size):
            for x in range(grid_size):
                coord[y * grid_size + x] = torch.tensor([y, x], dtype=torch.float32)
        self.register_buffer("coord", coord)

    def to_image(self, image_shape):
        images = self.units.view(self.grid_size * self.grid_size, *image_shape)
        return TF.to_pil_image(make_grid(images, nrow=self.grid_size, padding=0))

    def calc_kernel_size(self, t):
        return self.max_kernel_size * math.exp(-t * self.deno)

    def set_center(self, centroid):
        self.units.data[self.grid_size // 2, self.grid_size // 2] = centroid.flatten()

    def forward(self, x):
        # NOTE: some `expand` is not necessary, but it is used for clarification.
        x = x.detach()
        B, C, H, W = x.shape

        if self.warmup_t < 0:
            self.warmup_t += self.step_t
        else:
            self.t = min(self.t + self.step_t, self.max_t)

        # find BMU(best match unit)
        units = self.units.expand(B, *self.units.shape)
        x = x.view(B, 1, 1, -1).expand(units.shape)
        feat_diff = (units - x)
        feat_distance = torch.sum((feat_diff.detach() ** 2), dim=3, keepdims=True)
        bmu_index = torch.argmin(feat_distance.view(B, -1), dim=1)

        # calculate the distance on topological map from BMU to each unit
        bmu_pos = self.coord[bmu_index].view(B, 1, self.coord.shape[1]).expand((B, *self.coord.shape))
        coord = self.coord.view(1, *self.coord.shape).expand((B, *self.coord.shape))
        pos_distance = ((coord - bmu_pos) ** 2).sum(dim=2).view(B, self.grid_size, self.grid_size, 1)

        # generate a gaussian kernel centered on BMU and shrinking with t
        ksize = self.calc_kernel_size(self.t)
        sigma = (0.3 * ((ksize - 1) * 0.5 - 1) + 0.8)
        gaussian = torch.exp(-pos_distance / (2.0 * sigma ** 2))
        gaussian[gaussian < 0.001] = 0

        # update units with calculated weights
        temperature = math.exp(-(self.t * 2) / self.max_t)
        delta = (temperature * gaussian.expand(feat_diff.shape) * feat_diff)
        loss = (delta ** 2).view(B, self.units.shape[0] ** 2, self.units.shape[2]).mean()

        return bmu_index.view(B, 1), loss


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        assert (stride in {1, 2})
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, padding_mode="replicate")
        )
        if stride == 2:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        elif in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x) + self.identity(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
        return x


def Encoder(feat_dim):
    return nn.Sequential(
        # 14x14
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(0.1, inplace=True),
        # 12x12
        ResBlock(16, 32, stride=2),
        # 6x6
        ResBlock(32, 64, stride=1),
        ResBlock(64, 32, stride=2),
        # 3x3
        # fc
        nn.Conv2d(32, feat_dim, kernel_size=3, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0)
    )


class Decoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.conv1 = ResBlock(feat_dim, 64, stride=1)
        self.conv2 = ResBlock(64, 32, stride=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate")

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2(x)
        x = self.conv3(x)
        if self.training:
            return x
        else:
            return torch.clamp(x, 0, 1)


class AEVQSOM(nn.Module):
    def __init__(self, grid_size, feat_dim, max_t, step_t, warmup_t=10):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = Encoder(feat_dim)
        self.vq = SOMVectorQuantizer(grid_size, feat_dim, max_t=max_t, step_t=step_t, warmup_t=warmup_t)
        self.embed = nn.Embedding(grid_size * grid_size, 7 * 7 * feat_dim)
        self.vq_decoder = Decoder(feat_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, feat_dim, 7, 7))
        self.feat_decoder = Decoder(feat_dim)

    def forward(self, x):
        B = x.shape[0]
        # Feature Encoder
        feat = self.encoder(x)
        # VQ-Decoder path (train for SOM and VQ-Decoder)
        bmu, vq_loss = self.vq(feat)
        seed = self.embed(bmu).view(B, -1, 1, 1)
        seed = F.pixel_shuffle(seed, 7)
        recon1 = self.vq_decoder(seed)
        # Auto Encoder-Decoder path (train for Feature Encoder)
        feat_seed = feat.expand(seed.shape) + self.positional_embedding.expand(seed.shape)
        recon2 = self.feat_decoder(feat_seed)
        return recon1, recon2, x, vq_loss

    def t(self):
        return self.vq.t

    def current_kernel_size(self):
        return self.vq.calc_kernel_size(self.vq.t)

    def to_image(self):
        vq_index = torch.tensor([list(range(0, self.grid_size * self.grid_size))],
                                dtype=torch.long)
        vq_index = vq_index.view(self.grid_size * self.grid_size, 1)
        with torch.no_grad():
            device = next(self.parameters()).device
            seed = self.embed(vq_index.to(device)).view(vq_index.shape[0], -1, 1, 1)
            seed = F.pixel_shuffle(seed, 7)
            images = self.vq_decoder(seed).cpu()
        return TF.to_pil_image(make_grid(images, nrow=self.grid_size, padding=0))


class AEVQSOMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon1, recon2, x, vq_loss):
        # charbonnier loss for reconstruction
        recon1_loss = torch.sqrt((recon1 - x) ** 2 + 1e-6).mean()
        recon2_loss = torch.sqrt((recon2 - x) ** 2 + 1e-6).mean()
        # composite
        beta = 1
        return beta * vq_loss + recon1_loss + recon2_loss


class MinMaxNormalize():
    def __call__(self, x):
        min_v, max_v = x.min(), x.max()
        return (x - min_v) / (max_v - min_v)


def main():
    GRID_SIZE = 33
    IMAGE_SIZE = 14
    MAX_T = 70  # max epoch
    device = "cuda:0"

    torch.manual_seed(71)

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        MinMaxNormalize()
    ])
    dataset = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    BATCH_SIZE = 32
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=False)

    warmup_t = 10
    step_t = 1 / (len(dataset) // BATCH_SIZE)
    som = AEVQSOM(GRID_SIZE, feat_dim=16, max_t=MAX_T - warmup_t, step_t=step_t, warmup_t=warmup_t).to(device)
    criterion = AEVQSOMLoss()
    optimizer = torch.optim.Adam(som.parameters(), lr=0.0002)
    image_list = []
    for t in range(MAX_T):
        som.train()
        print(f"t={som.t()}, kernel_size={som.current_kernel_size()}")
        for x, y in tqdm(loader, ncols=80):
            optimizer.zero_grad()
            z = som(x.to(device))
            loss = criterion(*z)
            loss.backward()
            optimizer.step()
        som.eval()
        with torch.no_grad():
            map_image = ImageOps.invert(som.to_image())
        map_image.save(path.join(DATA_DIR, f"aevqsom_{t}.png"))
        image_list.append(map_image)

    # save animated gif
    image_list[0].save(
        path.join(DATA_DIR, "aevqsom.gif"), format="gif",
        append_images=image_list, save_all=True,
        duration=100, loop=0)
    print(f"save images in `{path.abspath(DATA_DIR)}`")


if __name__ == "__main__":
    main()
