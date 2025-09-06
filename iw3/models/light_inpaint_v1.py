import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.permute import pixel_shuffle, pixel_unshuffle
from nunif.modules.replication_pad2d import replication_pad2d_naive, ReplicationPad2dNaive
from nunif.modules.init import basic_module_init, icnr_init
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import FastLayerNorm
from nunif.modules.attention import WindowGMLP2d
from nunif.modules.gaussian_filter import SeparableGaussianFilter2d
from iw3.dilation import mask_closing, dilate


class GLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mlp_ratio=2, padding=True):
        super().__init__()
        mid = int(out_channels * mlp_ratio)
        # assert kernel_size % 2 == 1
        if padding:
            self.pad = ReplicationPad2dNaive(((kernel_size - 1) // 2,) * 4, detach=True)
        else:
            self.pad = nn.Identity()
        self.w1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid // 2, out_channels, kernel_size=kernel_size, stride=1, padding=0)
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.pad(x)
        x = self.w2(x)
        return x


class GMLPBlock(nn.Module):
    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False, kernel_size=3):
        super().__init__()
        self.gmlp = WindowGMLP2d(in_channels, window_size=window_size, shift=shift, mlp_ratio=mlp_ratio)
        self.norm1 = FastLayerNorm(in_channels, bias=False)
        self.norm2 = FastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = GLUConvMLP(in_channels, in_channels, mlp_ratio=1, kernel_size=kernel_size)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = x + self.gmlp(x, self.norm1, self.norm2)
        x = x + self.glu_conv(x)
        return x


@register_model
class LightInpaintV1(I2IBaseModel):
    name = "inpaint.light_inpaint_v1"

    def __init__(self):
        super(LightInpaintV1, self).__init__(locals(), scale=1, offset=16, in_channels=3, blend_size=8)
        self.downscaling_factor = 4
        self.mod = 16
        pack = self.downscaling_factor ** 2
        C = 96
        C2 = C * 2
        self.mask_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.patch = nn.Sequential(
            nn.Conv2d(3 * pack, C, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = GMLPBlock(C, window_size=16, mlp_ratio=2, shift=True)
        self.down = nn.Conv2d(C, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = nn.Sequential(
            GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=False),
            GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=True),
            GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=False),
            GMLPBlock(C2, window_size=8, mlp_ratio=2, shift=True),
        )
        self.up = nn.Conv2d(C2, C * 4, kernel_size=1, stride=1, padding=0)
        self.dec1 = GMLPBlock(C, window_size=16, mlp_ratio=2, shift=False)
        self.to_image = nn.Sequential(
            ReplicationPad2dNaive((1,) * 4, detach=True),
            nn.Conv2d(C, 3 * pack, kernel_size=3, stride=1, padding=0),
        )
        basic_module_init(self.patch)
        icnr_init(self.to_image[-1], scale_factor=4)
        nn.init.trunc_normal_(self.mask_bias, 0, 0.01)

        self.mask_blur = SeparableGaussianFilter2d(1, kernel_size=15, padding=15 // 2)

    def preprocess(self, x, mask, closing=False, dilation=0):
        if closing:
            mask = mask_closing(mask)
        else:
            mask = mask.float()

        for _ in range(dilation):
            mask = dilate(mask, kernel_size=(1, 3))

        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    def infer(self, x, mask, closing=False, dilation=0):
        x, mask = self.preprocess(x, mask, closing=closing, dilation=dilation)
        return self.forward(x, mask, skip_i2i_offset=True)

    def _forward(self, x, mask):
        x = pixel_unshuffle(x, self.downscaling_factor)
        x = self.patch(x)

        mask = pixel_unshuffle(mask, self.downscaling_factor).amax(dim=1, keepdim=True) > 0.99
        x = torch.where(mask, self.mask_bias.to(x.dtype), x)

        x1 = self.enc1(x)
        x2 = self.down(x1)
        x2 = self.enc2(x2)
        x2 = self.up(x2)
        x2 = F.pixel_shuffle(x2, 2)
        x = self.dec1(x1 + x2)
        x = self.to_image(x)
        x = pixel_shuffle(x, self.downscaling_factor)

        return x

    def forward(self, x, mask, skip_i2i_offset=False):
        src = x

        x = (x - 0.5) / 0.5

        input_height, input_width = x.shape[2:]
        pad1 = (self.mod * self.downscaling_factor) - input_width % (self.mod * self.downscaling_factor)
        pad2 = (self.mod * self.downscaling_factor) - input_height % (self.mod * self.downscaling_factor)
        padding = (0, pad1, 0, pad2)
        x = replication_pad2d_naive(x, padding, detach=True)
        mask = replication_pad2d_naive(mask, padding, detach=True)

        # forward
        x = self._forward(x, mask)
        x = F.pad(x, (0, -pad1, 0, -pad2))
        mask = F.pad(mask, (0, -pad1, 0, -pad2))

        # post process
        if not skip_i2i_offset:
            src = F.pad(src.to(x.dtype), (-self.i2i_offset,) * 4)
            mask = F.pad(mask, (-self.i2i_offset,) * 4)
            x = F.pad(x, (-self.i2i_offset,) * 4)

        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask

        if not self.training:
            src = src.clamp(0, 1)

        return src


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = False  # compiled model is about 2x faster but no windows support
    N = 20
    B = 4
    # S = (4320, 7680)  # 8K, 4.7FPS, 4.5GB VRAM
    # S = (2160, 3840)  # 4K, 18.3FPS, 900MB VRAM
    S = (1080, 1920)  # HD, 70FPS, 240MB VRAM
    # S = (320, 320)  # tile, 714FPS, 28MB VRAM

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    x = torch.zeros((B, 3, *S)).to(device)
    mask = torch.zeros((B, 1, *S), dtype=torch.bool).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(*model.preprocess(x, mask))
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(*model.preprocess(x, mask))
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    _bench("inpaint.light_inpaint_v1")
