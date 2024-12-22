import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.permute import pixel_shuffle, pixel_unshuffle
from nunif.modules.replication_pad2d import replication_pad2d_naive
from nunif.modules.init import basic_module_init, icnr_init
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import RMSNorm1


class PoolBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5, mlp_ratio=2, layer_norm=False):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)
        self.mlp = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels * mlp_ratio, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels * mlp_ratio, in_channels * mlp_ratio,
                      kernel_size=3, stride=1, padding=0, groups=in_channels * mlp_ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels * mlp_ratio, in_channels, kernel_size=1, stride=1, padding=0),
        ])
        if layer_norm:
            self.norm = RMSNorm1((1, in_channels, 1, 1), dim=1)
        else:
            self.norm = nn.Identity()

    @conditional_compile(["NUNIF_TRAIN", "IW3_MAIN"])
    def forward(self, x):
        x1 = self.norm(x)
        x1 = self.pooling(x1) - x1
        for block in self.mlp:
            x1 = block(x1)
        x = F.pad(x, (-1,) * 4) + x1
        return x


@register_model
class LightInpaintV1(I2IBaseModel):
    name = "inpaint.light_inpaint_v1"

    def __init__(self):
        super(LightInpaintV1, self).__init__(locals(), scale=1, offset=12, in_channels=3, blend_size=8)
        self.downscaling_factor = 4
        self.mod = 4
        pack = self.downscaling_factor ** 2
        C = 64
        self.blocks = nn.ModuleList([
            nn.Conv2d(5 * pack, C, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            PoolBlock(C),
            PoolBlock(C),
            PoolBlock(C),
            nn.Conv2d(C, 3 * pack, kernel_size=1, stride=1, padding=0),
        ])
        basic_module_init(self)
        icnr_init(self.blocks[-1], scale_factor=4)

    def _forward(self, x, mask_f):
        ones = torch.ones_like(mask_f)
        x = torch.cat([x, ones, ones * mask_f], dim=1)
        x = pixel_unshuffle(x, self.downscaling_factor)
        for block in self.blocks:
            x = block(x)
        x = pixel_shuffle(x, self.downscaling_factor)
        return x

    def forward(self, x, mask):
        src = x
        input_height, input_width = x.shape[2:]
        pad1 = (self.mod * self.downscaling_factor) - input_width % (self.mod * self.downscaling_factor)
        pad2 = (self.mod * self.downscaling_factor) - input_height % (self.mod * self.downscaling_factor)
        padding = (0, pad1, 0, pad2)
        x = replication_pad2d_naive(x, padding, detach=True)

        mask_f = mask.to(x.dtype)
        mask_f = replication_pad2d_naive(mask_f, padding, detach=True)
        x = self._forward(x, mask_f)
        x = F.pad(x, (0, -pad1, 0, -pad2))

        if not self.training:
            x.clamp_(0, 1)

        src = F.pad(src.to(x.dtype), (-self.i2i_offset,) * 4)
        mask = F.pad(mask, (-self.i2i_offset,) * 4)
        mask = mask.expand_as(src)
        src[mask] = x[mask]

        return src

    @torch.inference_mode()
    def infer(self, x, mask):
        src = x
        input_height, input_width = x.shape[2:]
        pad1 = (self.mod * self.downscaling_factor) - input_width % (self.mod * self.downscaling_factor)
        pad2 = (self.mod * self.downscaling_factor) - input_height % (self.mod * self.downscaling_factor)
        padding = (self.i2i_offset, pad1 + self.i2i_offset, self.i2i_offset, pad2 + self.i2i_offset)
        x = replication_pad2d_naive(x, padding, detach=True)

        mask_f = mask.to(x.dtype)
        mask_f = replication_pad2d_naive(mask_f, padding, detach=True)
        x = self._forward(x, mask_f)
        x = F.pad(x, (0, -pad1, 0, -pad2))

        mask = mask.expand_as(src)
        src[mask] = x[mask].clamp(0, 1).to(src.dtype)

        return src


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = False  # compiled model is about 2x faster but no windows support
    N = 20
    B = 1
    # S = (4320, 7680)  # 8K, 9FPS, 2.3GB VRAM
    S = (2160, 3840)  # 4K, 35FPS, 590MB VRAM
    # S = (1080, 1920)  # HD, 130FPS, 150MB VRAM
    # S = (320, 320) # tile, 1350FPS, 9MB VRAM

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    x = torch.zeros((B, 3, *S)).to(device)
    mask = torch.zeros((B, 1, *S), dtype=torch.bool).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model.infer(x, mask)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model.infer(x, mask)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    _bench("inpaint.light_inpaint_v1")
