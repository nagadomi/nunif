import torch
import torch.nn as nn
import torch.nn.functional as F
from .weighted_loss import WeightedLoss
from .compile_wrapper import conditional_compile
from .pad import get_crop_size, get_pad_size
from .reflection_pad2d import reflection_pad2d_naive


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
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    return (x - mean) / stdv


def dinov2_pad(x):
    pad = get_pad_size(x, DINO_PATCH_SIZE)
    x = reflection_pad2d_naive(x, pad, detach=True)
    return x


def dinov2_crop(x):
    pad = get_crop_size(x, DINO_PATCH_SIZE)
    x = F.pad(x, pad)
    return x


def dinov2_crop_pair(x, y, training=False):
    assert x.shape == y.shape
    pad = get_crop_size(x, DINO_PATCH_SIZE, random_shift=training)
    x = F.pad(x, pad)
    y = F.pad(y, pad)
    return x, y


def dinov2_pad_pair(x, y, training=False):
    assert x.shape == y.shape
    pad = get_pad_size(x, DINO_PATCH_SIZE, random_shift=training)
    x = reflection_pad2d_naive(x, pad, detach=True)
    y = reflection_pad2d_naive(y, pad, detach=True)
    return x, y


def dinov2_dim(model_type):
    return {"vits": 384, "vitb": 768, "vitl": 1024}[model_type]


class DINOv2IntermediateFeatures(nn.Module):
    def __init__(self, model_type="vits", index=None, random_projection=None):
        super().__init__()
        self.model_type = model_type
        if index is None:
            self.intermediate_layer_index = {
                "vits": [2, 5, 8, 11],
                "vitb": [2, 5, 8, 11],
                "vitl": [4, 11, 17, 23]
            }[model_type]
        else:
            self.intermediate_layer_index = index
        self.model = torch.hub.load('facebookresearch/dinov2', f"dinov2_{model_type}14_reg",
                                    verbose=False, trust_repo=True).eval()
        self.model.requires_grad_(False)

        if random_projection is not None:
            self.init_random_projection(random_projection)
        else:
            self.random_projection = None

    def init_random_projection(self, dim):
        feat_dim = dinov2_dim(self.model_type)
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(0)
            adj = 1.0 / 14.0
            proj = torch.randn((dim, feat_dim, 1, 1)) * (feat_dim ** -0.5) * adj
            self.register_buffer("random_projection", proj)
        finally:
            torch.random.set_rng_state(rng_state)

    def train(self, mode=True):
        self.model.train(False)
        self.model.requires_grad_(False)
        return self

    # @conditional_compile("NUNIF_TRAIN")  # error when backward
    def forward(self, x):
        assert x.ndim == 4
        assert x.shape[2] % DINO_PATCH_SIZE == 0 and x.shape[3] % DINO_PATCH_SIZE == 0

        features = self.model.get_intermediate_layers(
            x, self.intermediate_layer_index,
            reshape=True,
            return_class_token=False,
        )
        if self.random_projection is not None:
            features = [
                F.conv2d(feat, weight=self.random_projection, bias=None, stride=1)
                for feat in features
            ]

        return features


class DINOv2Loss(nn.Module):
    def __init__(self, loss, model_type="vits", index=None, normalize=True, random_projection=None):
        super().__init__()
        self.loss = loss
        self.normalize = normalize
        self.dino = DINOv2IntermediateFeatures(model_type=model_type, index=None, random_projection=random_projection)

    def forward(self, input, target):
        input, target = dinov2_pad_pair(input, target, self.training)
        if self.normalize:
            input = dinov2_normalize(input)
            target = dinov2_normalize(target)
        input_features = self.dino(input)
        target_features = self.dino(target)
        loss = 0.0
        for input_feat, target_feat in zip(input_features, target_features):
            target_feat = target_feat.detach()
            loss = loss + self.loss(input_feat, target_feat)

        return loss / len(input_features)


class CosineLoss(nn.Module):
    def forward(self, input, target):
        return (1.0 - F.cosine_similarity(input, target, dim=1)).mean()


class Pool(nn.Module):
    def __init__(self, base_loss, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_loss = base_loss

    def pool(self, x):
        x = x + F.avg_pool2d(
            x, kernel_size=self.kernel_size, stride=1,
            padding=(self.kernel_size - 1) // 2,
            count_include_pad=False
        )
        return x

    def forward(self, input, target):
        input = self.pool(input)
        target = self.pool(target)
        return self.base_loss(input, target)


def DINOv2CosineLoss(model_type="vits", index=None, normalize=True):
    return DINOv2Loss(CosineLoss(), model_type=model_type, index=index, normalize=normalize, random_projection=None)


def DINOv2PoolLoss(model_type="vits", index=None, normalize=True):
    return DINOv2Loss(
        Pool(nn.L1Loss()),
        model_type=model_type, index=index,
        normalize=normalize,
        random_projection=64
    )


def DINOv2CosineWith(base_loss, weight=1.0, model_type="vits", index=None, normalize=True):
    return WeightedLoss((
        base_loss,
        DINOv2CosineLoss(model_type=model_type, index=index, normalize=normalize)
    ), weights=(1.0, weight))


def DINOv2PoolWith(base_loss, weight=1.0, model_type="vits", index=None, normalize=True):
    return WeightedLoss((
        base_loss,
        DINOv2PoolLoss(model_type=model_type, index=index, normalize=normalize)
    ), weights=(1.0, weight))


def _test_feat():
    model = DINOv2IntermediateFeatures(model_type="vits").cuda()
    x = torch.zeros((4, 3, 518, 518), dtype=torch.float32).cuda()
    features = model(x)
    for feat in features:
        print(feat.shape)

    loss1 = DINOv2CosineLoss("vitb").cuda().eval()
    loss2 = DINOv2CosineWith(nn.L1Loss()).cuda().eval()
    x = torch.rand((4, 3, 112, 112)).cuda()
    x = dinov2_normalize(x)
    y = torch.rand((4, 3, 112, 112)).cuda()
    y = dinov2_normalize(y)
    print(loss1(x, y))
    print(loss1(x, x))
    print(loss2(x, y))
    print(loss2(x, x))


def _test_grad():
    import torchvision.io as io
    from .lbp_loss import YRGBLBP

    y = io.read_image("cc0/320/dog.png") / 255.0
    y = y.unsqueeze(0)
    x = y + (torch.rand_like(y) * 0.5)
    x.requires_grad_(True)

    x.grad = None
    l1_loss = nn.L1Loss()
    l1_loss(x, y).backward()
    l1_norm, l1_max = x.grad.norm(), x.grad.abs().max()

    x.grad = None
    lbp_loss = YRGBLBP()
    lbp_loss(x, y).backward()
    lbp_norm, lbp_max = x.grad.norm(), x.grad.abs().max()

    x.grad = None
    dinov2_loss = DINOv2PoolLoss()  # DINOv2CosineLoss()
    (dinov2_loss(x, y.detach())).backward()
    dinov2_norm, dinov2_max = x.grad.norm(), x.grad.abs().max()

    print("dinov2 loss norm", dinov2_norm, "max", dinov2_max)
    print("l1 loss norm", l1_norm, "max", l1_max, "weight", l1_norm / dinov2_norm)
    print("lbp loss norm", lbp_norm, "max", lbp_max, "weight", lbp_norm / dinov2_norm)
    # DINOv2L2PoolLoss, weight=0.078
    # DINOv2CosineLoss, weight=0.056


if __name__ == "__main__":
    _test_feat()
    _test_grad()
