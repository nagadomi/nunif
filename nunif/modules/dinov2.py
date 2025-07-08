import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .weighted_loss import WeightedLoss
from .compile_wrapper import conditional_compile


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


def dinov2_crop_pair(x, y, training=False):
    assert x.shape == y.shape
    B, C, H, W = x.shape
    pad_h = H % DINO_PATCH_SIZE
    pad_w = W % DINO_PATCH_SIZE
    if training:
        pad_h += DINO_PATCH_SIZE
        pad_w += DINO_PATCH_SIZE
        pad_h1 = random.randint(0, pad_h)
        pad_h2 = pad_h - pad_h1
        pad_w1 = random.randint(0, pad_w)
        pad_w2 = pad_w - pad_w1
    else:
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
    x = F.pad(x, (-pad_w1, -pad_w2, -pad_h1, -pad_h2))
    y = F.pad(y, (-pad_w1, -pad_w2, -pad_h1, -pad_h2))
    return x, y


class DINOv2IntermediateFeatures(nn.Module):
    def __init__(self, model_type="vits", index=None):
        super().__init__()
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
        return features


class DINOv2Loss(nn.Module):
    def __init__(self, loss, model_type="vits", index=None, normalize=True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize
        self.dino = DINOv2IntermediateFeatures(model_type=model_type, index=None)

    def forward(self, input, target):
        input, target = dinov2_crop_pair(input, target, self.training)
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


def cosine_loss(input, target):
    return (1.0 - F.cosine_similarity(input, target, dim=1)).mean()

class CosineLoss(nn.Module):
    def forward(self, input, target):
        return cosine_loss(input, target)


def DINOv2CosineLoss(model_type="vits", index=None, normalize=True):
    return DINOv2Loss(CosineLoss(), model_type=model_type, index=index, normalize=normalize)


def DINOv2CosineWith(base_loss, weight=1.0, model_type="vits", index=None, normalize=True):
    return WeightedLoss((base_loss, DINOv2Loss(CosineLoss(), model_type=model_type, index=index, normalize=normalize)),
                        weights=(1.0, weight))


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


if __name__ == "__main__":
    _test_feat()
