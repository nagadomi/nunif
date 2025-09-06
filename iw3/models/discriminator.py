import torch
import torch.nn.functional as F
from nunif.models import Model
from nunif.modules.pad import get_crop_size, get_fit_pad_size
from nunif.modules.reflection_pad2d import reflection_pad2d_naive

# l3v1c discriminator
import torch.nn as nn
from nunif.models import register_model
from nunif.modules.attention import SEBlock
from nunif.modules.res_block import ResBlockSNLReLU
from torch.nn.utils.parametrizations import spectral_norm
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init


def normalize(x):
    return x * 2. - 1.


def modcrop(x, n):
    unpad = get_crop_size(x, n)
    x = F.pad(x, unpad)
    return x


def fit_to_size(x, cond):
    pad = get_fit_pad_size(cond, x)
    cond = reflection_pad2d_naive(cond, pad, detach=True)
    return cond


class Discriminator(Model):
    def __init__(self, kwargs, loss_weights=(1.0,)):
        super().__init__(kwargs)
        self.loss_weights = loss_weights


class ImageToCondition(nn.Module):
    # 1x1 vector
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d((4, 4)),
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.aggregate = nn.Linear(embed_dim * 16, embed_dim, bias=True)
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=True),
                nn.ReLU(inplace=True),
                spectral_norm(nn.Linear(embed_dim, out_channels, bias=True)))
            for out_channels in outputs])
        basic_module_init(self)

    @conditional_compile("NUNIF_TRAIN")
    def forward(self, x):
        B = x.shape[0]
        x = normalize(x)
        x = self.features(x)
        x = self.aggregate(x.view(B, -1))
        outputs = []
        for fc in self.fc:
            enc = fc(x)
            enc = enc.view(B, enc.shape[1], 1, 1)
            outputs.append(enc)
        return outputs


class L3Discriminator(Discriminator):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(locals())
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(128, bias=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
        )
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256, bias=True),
            ResBlockSNLReLU(256, 512, bias=False),
            SEBlock(512, bias=True),
            spectral_norm(nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=1)))
        basic_module_init(self)

    @conditional_compile("NUNIF_DISC_COMPILE")
    def forward(self, x, c=None):
        x = modcrop(x, 8)
        x = normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        x = F.pad(x, (-8,) * 4)
        return x


@register_model
class L3ConditionalDiscriminator(L3Discriminator):
    name = "inpaint.l3_conditional_discriminator"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.to_cond = ImageToCondition(32, [256])

    @conditional_compile("NUNIF_DISC_COMPILE")
    def forward(self, x, c=None, mask=None):
        x = modcrop(x, 8)
        c = fit_to_size(x, c)
        if mask is not None:
            mask = fit_to_size(x, mask)
        cond = self.to_cond(c)
        x = normalize(x)
        x = self.features(x)
        x = self.classifier(x + cond[0])

        mask = F.pixel_unshuffle(mask, 8).mean(dim=1, keepdim=True)
        assert mask.shape[-2:] == x.shape[-2:]
        mask = F.pad(mask, (-2,) * 4)
        x = F.pad(x, (-2,) * 4)

        return x, mask


def _test():
    from nunif.models import create_model
    device = "cuda"
    model = create_model("inpaint.l3_conditional_discriminator").to(device).eval()
    x = torch.zeros((4, 3, 128, 128)).to(device)
    c = torch.zeros((4, 3, 128, 128)).to(device)
    mask = torch.zeros((4, 1, 128, 128)).to(device)

    z = model(x, c, mask=mask)
    print(z.shape)


if __name__ == "__main__":
    _test()
