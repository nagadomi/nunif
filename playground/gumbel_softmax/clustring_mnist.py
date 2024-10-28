# python3 -m playground.gumbel_softmax.clustring_mnist
#
# GumbelCustering(soft): https://github.com/user-attachments/assets/236f18d8-8a21-4e32-920d-b5844d09563a
# GumbelCusteringOnlyBMU(hard): https://github.com/user-attachments/assets/6a6ebe6c-81d4-42a8-b9b3-700fd165c939

from torchvision.datasets import MNIST
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from os import path
import math
from tqdm import tqdm
from PIL import ImageOps


DATA_DIR = path.join(path.dirname(__file__), "..", "..", "tmp", "gumbel_softmax")


def cosine_annealing(min_v, max_v, t, max_t):
    if max_t > t:
        return min_v + 0.5 * (max_v - min_v) * (1.0 + math.cos((t / max_t) * math.pi))
    else:
        return min_v


class GumbelCustering(nn.Module):
    def __init__(self, input_size, codebook_size, max_t, min_tau=1e-8, max_tau=10.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.max_t = max_tau
        self.codebook = nn.Parameter(torch.zeros(size=(codebook_size, input_size), dtype=torch.float32))
        self.bmu_net = nn.Sequential(
            nn.Linear(input_size, int(codebook_size ** 0.5)),
            nn.ReLU(True),
            nn.Linear(int(codebook_size ** 0.5), codebook_size))

    def forward(self, x, t):
        B = x.shape[0]
        x = x.reshape(B, -1)
        temperature = cosine_annealing(self.min_tau, self.max_tau, t, self.max_t)
        logits = self.bmu_net(x)
        z = F.gumbel_softmax(logits, tau=temperature, dim=-1, hard=False)

        codebook = self.codebook.expand(B, *self.codebook.shape)
        feat_diff = codebook - x.view(B, 1, -1)
        feat_distance = torch.sum((feat_diff ** 2), dim=-1)
        bmu_index = torch.argmin(feat_distance.view(B, -1), dim=-1)
        delta = (feat_diff.abs().mean(dim=-1) * z.detach()).mean()

        return logits, bmu_index, delta


class GumbelCusteringBMUOnly(nn.Module):
    def __init__(self, input_size, codebook_size, max_t, min_tau=1e-8, max_tau=10.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.max_t = max_tau
        self.codebook_emb = nn.Embedding(codebook_size, input_size)
        self.bmu_net = nn.Sequential(
            nn.Linear(input_size, int(codebook_size ** 0.5)),
            nn.ReLU(True),
            nn.Linear(int(codebook_size ** 0.5), codebook_size))

        torch.nn.init.constant_(self.codebook_emb.weight, 0)

    @property
    def codebook(self):
        return self.codebook_emb.weight

    def forward(self, x, t):
        B = x.shape[0]
        x = x.reshape(B, -1)
        temperature = cosine_annealing(self.min_tau, self.max_tau, t, self.max_t)
        logits = self.bmu_net(x)
        z = F.gumbel_softmax(logits, tau=temperature, dim=-1, hard=True)

        feat_distance = (x.pow(2).sum(1, keepdim=True) - 2 * x @ self.codebook.t() +
                         self.codebook.pow(2).sum(1, keepdim=True).t())
        bmu_index = torch.argmin(feat_distance.view(B, -1), dim=-1)
        delta = (self.codebook_emb(torch.argmax(z.detach(), dim=-1)) - x).abs().mean()

        if t == 0:
            # epoch 1, random init
            uniform_bmu = torch.randint(0, high=self.codebook.shape[0], size=(B,), device=x.device, dtype=torch.long)
            delta = delta * 0.1 + (self.codebook_emb(uniform_bmu) - x).abs().mean()

        return logits, bmu_index, delta


class MinMaxNormalize():
    def __call__(self, x):
        min_v, max_v = x.min(), x.max()
        return (x - min_v) / (max_v - min_v)


def main():
    if True:
        MODEL_FACTORY = GumbelCusteringBMUOnly
        VQ_LOSS_WEIGHT = 1.0
    else:
        MODEL_FACTORY = GumbelCustering
        VQ_LOSS_WEIGHT = 10.0

    GRID_SIZE = 24
    BATCH_SIZE = 64
    IMAGE_SCALE = 0.5
    MNIST_SIZE = 28
    IMAGE_SIZE = int(MNIST_SIZE * IMAGE_SCALE)
    MAX_T = 100  # max epoch
    EPOCH_MULT = 2
    LR = 1e-3
    device = "cuda"

    torch.manual_seed(72)

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        MinMaxNormalize()
    ])
    dataset = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    model = MODEL_FACTORY(input_size=1 * IMAGE_SIZE * IMAGE_SIZE,
                          codebook_size=GRID_SIZE ** 2, max_t=MAX_T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=False)

    image_list = []
    for t in range(MAX_T):
        print(f"t={t}/{MAX_T}")
        model.train()
        for _ in range(EPOCH_MULT):
            for x, y in tqdm(loader, ncols=80):
                optimizer.zero_grad()
                x = x.to(device)
                z, y, vq_loss = model(x, t)
                loss = vq_loss * VQ_LOSS_WEIGHT + F.cross_entropy(z, y).mean()
                loss.backward()
                optimizer.step()

        model.eval()
        images = model.codebook.detach().view(model.codebook.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)
        images = torch.clamp(images, 0, 1)
        map_image = TF.to_pil_image(make_grid(images, nrow=GRID_SIZE, padding=0))
        map_image = ImageOps.invert(map_image)
        map_image.save(path.join(DATA_DIR, f"gumbel_softmax_{t}.png"))
        image_list.append(map_image)

    # save animated gif
    image_list[0].save(
        path.join(DATA_DIR, "gumbel_softmax.gif"), format="gif",
        append_images=image_list, save_all=True,
        duration=66, loop=0)
    print(f"save images in `{path.abspath(DATA_DIR)}`")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
