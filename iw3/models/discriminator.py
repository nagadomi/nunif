import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from nunif.models import Model
from nunif.modules.pad import get_crop_size, get_fit_pad_size
from nunif.modules.reflection_pad2d import reflection_pad2d_naive
from nunif.models import register_model
from nunif.modules.attention import SEBlock
from nunif.modules.res_block import ResBlockGNLReLU
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init
from nunif.modules.fourier_unit import FourierUnit
from iw3.dilation import dilate


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


def mask_dilate(mask, n_iter=None):
    if n_iter is None:
        n_iter = mask.shape[-1] // 8 + 1
    for i in range(n_iter):
        mask = dilate(mask)
    return mask


class Discriminator(Model):
    def __init__(self, kwargs, loss_weights=(1.0,)):
        super().__init__(kwargs)
        self.loss_weights = loss_weights


class ImageToCondition(nn.Module):
    # 1x1 vector
    def __init__(self, embed_dim, outputs):
        super().__init__()
        self.features = nn.Sequential(
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
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(128, bias=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256, bias=True),
            ResBlockGNLReLU(256, 512, bias=False),
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
        x = F.pad(x, (-2,) * 4)

        if mask is not None:
            mask = F.pixel_unshuffle(mask, 8).amax(dim=1, keepdim=True)
            mask = mask_dilate(mask.float()) > 0
            mask = F.pad(mask, (-2,) * 4)
            assert mask.shape[-2:] == x.shape[-2:]

            return x, mask
        else:
            return x


class FFCBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ffc = FourierUnit(in_channels, in_channels,
                               activation_layer=lambda dim: nn.LeakyReLU(0.2, inplace=True),
                               residual=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        )

    def forward(self, x):
        x = x + self.fusion(torch.cat((x, self.ffc(x)), dim=1))
        return x


@register_model
class FFCDiscriminator(Discriminator):
    name = "inpaint.ffc_discriminator"

    def __init__(self):
        super().__init__(locals())
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            FFCBlock(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            FFCBlock(128),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            FFCBlock(256),
        )
        self.classifier = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

    @conditional_compile("NUNIF_DISC_COMPILE")
    def forward(self, x, c=None, mask=None):
        x = modcrop(x, 8)
        c = fit_to_size(x, c)
        if mask is not None:
            mask = fit_to_size(x, mask)
        x = normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        x = F.pad(x, (-2,) * 4)

        if mask is not None:
            mask = F.pixel_unshuffle(mask, 8).amax(dim=1, keepdim=True)
            mask = mask_dilate(mask.float())
            mask = F.pad(mask, (-2,) * 4)
            assert mask.shape[-2:] == x.shape[-2:]

            return x, mask
        else:
            return x


def _test(name):
    from nunif.models import create_model
    device = "cuda"
    model = create_model(name).to(device).eval()
    x = torch.zeros((4, 3, 256, 256)).to(device)
    c = torch.zeros((4, 3, 256, 256)).to(device)
    mask = torch.zeros((4, 1, 256, 256)).to(device)

    z, mask = model(x, c, mask=mask)
    print(z.shape)


if __name__ == "__main__":
    _test("inpaint.l3_conditional_discriminator")
    _test("inpaint.ffc_discriminator")
