import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import xformers  # noqa
except ModuleNotFoundError:
    import os
    import warnings
    os.environ["XFORMERS_DISABLED"] = "1"
    warnings.filterwarnings(action="ignore", category=UserWarning, message="xFormers is disabled*")
    warnings.filterwarnings(action="ignore", category=UserWarning, message="xFormers is not available*")


DINO_PATCH_SIZE = 14


def dinov2_normalize(x):
    return x * 2.0 - 1.0


def dinov2_pad(x):
    B, C, H, W = x.shape
    pad_h = DINO_PATCH_SIZE - H % DINO_PATCH_SIZE if H % DINO_PATCH_SIZE != 0 else 0
    pad_w = DINO_PATCH_SIZE - W % DINO_PATCH_SIZE if W % DINO_PATCH_SIZE != 0 else 0
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x


def dinov2_crop(x):
    B, C, H, W = x.shape
    pad_h = H % DINO_PATCH_SIZE
    pad_w = W % DINO_PATCH_SIZE

    pad_h1 = pad_h // 2
    pad_h2 = pad_h - pad_h1
    pad_w1 = pad_w // 2
    pad_w2 = pad_w - pad_w1
    x = F.pad(x, (-pad_w1, -pad_w2, -pad_h1, -pad_h2))
    return x


class DINOEmbedding(nn.Module):
    def __init__(self, model_type="dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_type,
                                    verbose=False, trust_repo=True).eval()
        self.model.requires_grad_(False)

    def train(self, mode=True):
        self.model.train(False)
        self.model.requires_grad_(False)
        return self

    def forward(self, x):
        assert x.ndim == 4
        assert x.shape[2] % DINO_PATCH_SIZE == 0 and x.shape[3] % DINO_PATCH_SIZE == 0

        x = self.model.forward(x)
        return x

    def forward_patch_raw(self, x):
        assert x.ndim == 4
        assert x.shape[2] % DINO_PATCH_SIZE == 0 and x.shape[3] % DINO_PATCH_SIZE == 0
        assert x.shape[2] == x.shape[3]

        x = self.model.forward_features(x)["x_norm_patchtokens"]
        return x

    def forward_patch(self, x):
        z = self.forward_patch_raw(x)
        z = z.permute(0, 2, 1).reshape(
            z.shape[0], z.shape[-1],
            x.shape[2] // DINO_PATCH_SIZE, x.shape[3] // DINO_PATCH_SIZE)
        return z


class DINOLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.dino = DINOEmbedding()

    def forward(self, input, target):
        input = dinov2_crop(input)
        target = dinov2_crop(target)
        return self.loss(self.dino.forward_patch_raw(dinov2_normalize(input)),
                         self.dino.forward_patch_raw(dinov2_normalize(target)).detach())


def DINOL1Loss():
    return DINOLoss(nn.L1Loss())


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, input, target):
        return (1.0 - self.cosine(input, target)).mean()


def DINOCosineLoss():
    return DINOLoss(CosineLoss())


if __name__ == "__main__":
    model = DINOEmbedding().cuda()
    x = torch.rand((4, 3, 112, 112)).cuda()
    x = dinov2_normalize(x)
    with torch.autocast(device_type="cuda"):
        z = model(x)
        print(z.shape)
        z = model.forward_patch(x)
        print(z.shape)

    loss = DINOCosineLoss().cuda()
    y = torch.rand((4, 3, 112, 112)).cuda()
    print(loss(x, y))
