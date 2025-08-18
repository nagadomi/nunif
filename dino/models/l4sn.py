import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init
from nunif.modules.pad import get_pad_size
from nunif.modules.reflection_pad2d import reflection_pad2d_naive


C1 = 32
C2 = 64
C3 = 128
C4 = 256
C5 = 512
FEAT_DIMS = [C2, C3, C4, C5]
RANDOM_PROJECTION_DIM = 64
# TODO: uplaod
CHECKPOINT_URL = "../dino/models/l4sn_v2/l4sn.pth"


def normalize(x):
    return (x - 0.5) / 0.5


class L4SNFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            spectral_norm(nn.Conv2d(3, C1, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(C1, C2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(C2, C2, kernel_size=3, stride=1, padding=1, bias=False)),
        )
        self.l2 = nn.Sequential(
            spectral_norm(nn.Conv2d(C2, C3, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(C3, C3, kernel_size=3, stride=1, padding=1, bias=False)),
        )
        self.l3 = nn.Sequential(
            spectral_norm(nn.Conv2d(C3, C4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(C4, C4, kernel_size=3, stride=1, padding=1, bias=False))
        )
        self.l4 = nn.Sequential(
            spectral_norm(nn.Conv2d(C4, C5, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(C5, C5, kernel_size=3, stride=1, padding=1, bias=False))
        )
        basic_module_init(self)

    @conditional_compile(["DINO_TRAIN", "NUNIF_TRAIN"])
    def forward_features(self, x, activation=True):
        x = normalize(x)
        x1 = self.l1(x)
        x1a = F.leaky_relu(x1, 0.2, inplace=True)
        x2 = self.l2(x1a)
        x2a = F.leaky_relu(x2, 0.2, inplace=True)
        x3 = self.l3(x2a)
        x3a = F.leaky_relu(x3, 0.2, inplace=True)
        x4 = self.l4(x3a)
        x4a = F.leaky_relu(x4, 0.2, inplace=True)
        if activation:
            return [x1a, x2a, x3a, x4a]
        else:
            return [x1, x2, x3, x4]

    @conditional_compile(["DINO_TRAIN", "NUNIF_TRAIN"])
    def forward(self, x):
        assert x.shape[2] % 16 == 0
        # x is normalized
        x = self.l1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.l2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.l3(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.l4(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return x


class L4SN(nn.Module):
    def __init__(self, activation=True):
        super().__init__()
        self.activation = activation  # for loss
        self.feature = L4SNFeature()
        # self.fc is a dummy. It will be replaced with nn.Identity by DINO train.
        self.fc = nn.Linear(C5, 1)

    def forward(self, x):
        B = x.shape[0]
        x = self.feature(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(B, C5)
        x = self.fc(x)
        return x


class L4SNLoss(nn.Module):
    def __init__(self, activation=True, loss_weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.feature = L4SNFeature()
        self.activation = activation
        self.loss_weights = loss_weights
        self.init_random_projection()
        self.load_pth(CHECKPOINT_URL)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def init_random_projection(self):
        with torch.no_grad():
            rng_state = torch.random.get_rng_state()
            try:
                torch.manual_seed(0)
                dims = [RANDOM_PROJECTION_DIM] * len(FEAT_DIMS)
                feat_dims = FEAT_DIMS
                for i, (dim, feat_dim) in enumerate(zip(dims, feat_dims)):
                    scale = 1.0  # already normalized by SN
                    proj = torch.randn((dim, feat_dim, 1, 1)) * scale
                    self.register_buffer(f"random_projection_{i}", proj)
            finally:
                torch.random.set_rng_state(rng_state)

    def load_pth(self, model_path):
        if model_path.startswith("http://") or model_path.startswith("https://"):
            state_dict = torch.hub.load_state_dict_from_url(model_path, weights_only=True, map_location="cpu")
        else:
            state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
        self.feature.load_state_dict(state_dict)

    def forward_loss(self, input, target):
        f1s = self.feature.forward_features(input, activation=self.activation)
        f2s = self.feature.forward_features(target, activation=self.activation)
        loss = 0
        for i, (f1, f2) in enumerate(zip(f1s, f2s)):
            weight = getattr(self, f"random_projection_{i}")
            f1 = F.conv2d(f1, weight=weight, bias=None, stride=1)
            f2 = F.conv2d(f2, weight=weight, bias=None, stride=1)
            f1 = f1 + F.avg_pool2d(f1, kernel_size=3, stride=1, padding=1, count_include_pad=False) * 0.1
            f2 = f2 + F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1, count_include_pad=False) * 0.1
            loss = loss + self.loss_fn(f1, f2) * self.loss_weights[i]
        return loss / len(f1s)

    def forward(self, input, target):
        pad = get_pad_size(input, 16)
        input = reflection_pad2d_naive(input, pad, detach=True)
        target = reflection_pad2d_naive(target, pad, detach=True)
        loss = self.forward_loss(input, target)
        loss = loss + F.l1_loss(input, target) * 0.4
        return loss


def _test_grad():
    import torchvision.io as io

    y = io.read_image("cc0/320/dog.png") / 255.0
    y = y.unsqueeze(0)
    x = y + (torch.rand_like(y) * 0.1)
    x.requires_grad_(True)

    x.grad = None
    l1_loss = nn.L1Loss()
    l1_loss(x, y).backward()
    l1_norm, l1_max = x.grad.norm(), x.grad.abs().max()

    x.grad = None
    l4sn_loss = L4SNLoss()
    l4sn_loss(x, y.detach()).backward()
    l4sn_norm, l4sn_max = x.grad.norm(), x.grad.abs().max()

    print("l4sn loss norm", l4sn_norm, "max", l4sn_max)
    print("l1 loss norm", l1_norm, "max", l1_max, "weight", l1_norm / l4sn_norm)
    # l1 weight=0.37


def _test():
    import torch
    device = "cuda:0"
    model = L4SN().to(device)
    x = torch.zeros((1, 3, 256, 256)).to(device)
    with torch.no_grad():
        z = model(x)
        print(z.shape)


if __name__ == "__main__":
    _test_grad()
    # for more test,
    # python -m playground.fr_iqa.recover -i tmp/art_320x320.png -o tmp/l4sn_art --random-shif --model l4sn --init shift
    # python -m playground.fr_iqa.recover -i tmp/art_320x320.png -o tmp/l4sn_art --random-shif --model l4sn --init noise
