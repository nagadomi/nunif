# SOM
# python3 -m playground.som.train_mnist
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import make_grid
import torch
from torch import nn
from os import path
import math
from tqdm import tqdm
from PIL import ImageOps


DATA_DIR = path.join(path.dirname(__file__), "..", "..", "tmp", "som")


class SOM(nn.Module):
    def __init__(self, grid_size, input_size, max_t, max_distance_factor=0.6, learning_rate=1e-2):
        super().__init__()
        self.grid_size = grid_size
        self.max_kernel_size = grid_size
        self.max_t = max_t
        self.learning_rate = learning_rate

        self.register_buffer("units", torch.rand(size=(grid_size, grid_size, input_size), dtype=torch.float32))
        self.deno = 1.0 / (0.9 * max_t / math.log(self.max_kernel_size))
        coord = torch.empty((grid_size * grid_size, 2), dtype=torch.float32)
        for y in range(grid_size):
            for x in range(grid_size):
                coord[y * grid_size + x] = torch.tensor([y, x], dtype=torch.float32)
        self.register_buffer("coord", coord)

    def to_image(self, image_shape):
        images = self.units.view(self.grid_size * self.grid_size, *image_shape)
        return F.to_pil_image(make_grid(images, nrow=self.grid_size, padding=0))

    def calc_kernel_size(self, t):
        return self.max_kernel_size * math.exp(-t * self.deno)

    def set_center(self, centroid):
        self.units[self.grid_size // 2, self.grid_size // 2] = centroid.flatten()

    def update(self, x, t):
        # NOTE: some `expand` is not necessary, but it is used for clarification.
        B, C, H, W = x.shape

        # find BMU(best match unit)
        units = self.units.expand(B, *self.units.shape)
        x = x.view(B, 1, 1, -1).expand(units.shape)
        feat_diff = (units - x)
        feat_distance = torch.sum((feat_diff ** 2), dim=3, keepdims=True)
        bmu_index = torch.argmin(feat_distance.view(B, -1), dim=1)

        # calculate the distance on topological map from BMU to each unit
        bmu_pos = self.coord[bmu_index].view(B, 1, self.coord.shape[1]).expand((B, *self.coord.shape))
        coord = self.coord.view(1, *self.coord.shape).expand((B, *self.coord.shape))
        pos_distance = ((coord - bmu_pos) ** 2).sum(dim=2).view(B, self.grid_size, self.grid_size, 1)

        # generate a gaussian kernel centered on BMU and shrinking with t
        ksize = self.calc_kernel_size(t)
        sigma = (0.3 * ((ksize - 1) * 0.5 - 1) + 0.8)
        gaussian = torch.exp(-pos_distance / (2.0 * sigma ** 2))
        gaussian[gaussian < 0.001] = 0

        # update units with calculated weights
        temperature = math.exp(-(t * 2) / self.max_t)
        delta = (temperature * gaussian.expand(feat_diff.shape) * feat_diff).mean(dim=0)
        self.units -= self.learning_rate * delta


class MinMaxNormalize():
    def __call__(self, x):
        min_v, max_v = x.min(), x.max()
        return (x - min_v) / (max_v - min_v)


def main():
    GRID_SIZE = 33
    IMAGE_SCALE = 0.5
    MNIST_SIZE = 28
    IMAGE_SIZE = int(MNIST_SIZE * IMAGE_SCALE)
    MAX_T = 100  # max epoch
    INNER_EPOCH = 2
    device = "cuda:0"

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        MinMaxNormalize()
    ])
    dataset = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    som = SOM(GRID_SIZE, input_size=1 * IMAGE_SIZE * IMAGE_SIZE, max_t=MAX_T).to(device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=False)

    # initialize the topological center of the units with the centroid of the data.
    print("initialize")
    centroid = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    scale = 1.0 / len(dataset)
    for x, _ in tqdm(dataset, ncols=80):
        centroid += scale * x
    som.set_center(centroid)

    # update
    image_list = []
    for t in range(MAX_T):
        for _ in range(INNER_EPOCH):
            print(f"t={t}/{MAX_T}, kernel_size={som.calc_kernel_size(t)}")
            for x, y in tqdm(loader, ncols=80):
                x = x.to(device)
                som.update(x, t)
        map_image = ImageOps.invert(som.to_image((1, IMAGE_SIZE, IMAGE_SIZE)))
        map_image.save(path.join(DATA_DIR, f"som_{t}.png"))
        image_list.append(map_image)

    # save animated gif
    image_list[0].save(
        path.join(DATA_DIR, "som.gif"), format="gif",
        append_images=image_list, save_all=True,
        duration=66, loop=0)
    print(f"save images in `{path.abspath(DATA_DIR)}`")


def scheduler_test():
    # test code for annealing scheduler
    max_t = 100
    max_kernel_size = 32
    deno = 1.0 / (0.9 * max_t / math.log(max_kernel_size))
    for t in range(max_t):
        print(f"---- t={t}")
        ksize = max_kernel_size * math.exp(-t * deno)
        sigma = (0.3 * ((ksize - 1) * 0.5 - 1) + 0.8)
        print(f"ksize={ksize}, sigma={sigma}")
        for dist in range(32):
            gaussian = math.exp(-dist**2 / (2 * sigma ** 2))
            print("gaussian", f"distance={dist}", f"value={gaussian}")


if __name__ == "__main__":
    if True:
        main()
    else:
        scheduler_test()
