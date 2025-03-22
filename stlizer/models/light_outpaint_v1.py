# coarse blurred outpaint
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.replication_pad2d import replication_pad2d_naive, ReplicationPad2dNaive
from nunif.modules.init import basic_module_init
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.attention import WindowMHA2d, WindowScoreBias


class PoolBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            ReplicationPad2dNaive((1, 1, 1, 1)),
            nn.Conv2d(in_channels * 2, in_channels * 2,
                      kernel_size=3, stride=1, padding=0, groups=in_channels * 2),
            nn.GLU(dim=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )
        basic_module_init(self.mlp)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x1 = x
        x1 = self.pooling(x1) - x1
        x = x + self.mlp(x1)
        return x


class MHABlock(nn.Module):
    def __init__(self, in_channels, window_size=4, num_heads=4):
        super().__init__()
        self.mha = WindowMHA2d(in_channels, num_heads=num_heads, window_size=window_size)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=1, padding=0),
            nn.GLU(dim=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )
        self.bias = WindowScoreBias(window_size=window_size)
        basic_module_init(self.mlp)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.mlp(x)
        return x


class Downsampling(nn.Module):
    def __init__(self, in_channels, dims):
        super().__init__()
        blocks = []
        in_ch = in_channels
        for dim in dims:
            blocks.append(ReplicationPad2dNaive((1, 1, 1, 1), detach=True))
            blocks.append(nn.Conv2d(in_ch, dim, kernel_size=3, stride=2, padding=0))
            blocks.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = dim

        self.blocks = nn.ModuleList(blocks)
        basic_module_init(self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ToImageBilinaer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.scale_factor = scale_factor
        basic_module_init(self.proj)

    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return x


class OutpaintBase(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.mod = 8
        self.downscaling_factor = window_size
        C = dim
        C2 = dim // 2
        self.dct = Downsampling(4, dims=[C // 8, C // 4, C])  # 1/2, 1/4, 1/8
        self.proj_mid = nn.Conv2d(C, C2, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(C2, C, kernel_size=1, stride=1, padding=0)
        self.enc_block = nn.Sequential(
            MHABlock(C, window_size=8, num_heads=C // 32),
            PoolBlock(C),
        )
        self.mid_block = nn.Sequential(
            MHABlock(C2, window_size=8, num_heads=C2 // 32),
            PoolBlock(C2),
            MHABlock(C2, window_size=8, num_heads=C2 // 32),
            PoolBlock(C2),
        )
        self.dec_block = nn.Sequential(
            MHABlock(C, window_size=8, num_heads=C // 32),
            PoolBlock(C),
        )
        self.to_image_biliner = ToImageBilinaer(C, 3, scale_factor=self.downscaling_factor)
        basic_module_init(self.proj_mid)
        basic_module_init(self.proj_out)

    def _forward(self, x, mask_f):
        x = torch.cat([x, mask_f], dim=1)
        x = self.dct(x)
        x = self.enc_block(x)
        x = x + self.proj_out(self.mid_block(self.proj_mid(x)))
        x = self.dec_block(x)
        x = self.to_image_biliner(x)
        return x

    def forward(self, x, mask):
        input_height, input_width = x.shape[2:]
        if input_width % (self.mod * self.downscaling_factor) == 0:
            if self.training:
                # random padding with/without
                pad1 = 0 if torch.rand((1,)).item() < 0.5 else self.mod * self.downscaling_factor
            else:
                pad1 = 0
        else:
            pad1 = (self.mod * self.downscaling_factor) - input_width % (self.mod * self.downscaling_factor)
        if input_height % (self.mod * self.downscaling_factor) == 0:
            if self.training:
                pad2 = 0 if torch.rand((1,)).item() < 0.5 else self.mod * self.downscaling_factor
            else:
                pad2 = 0
        else:
            pad2 = (self.mod * self.downscaling_factor) - input_height % (self.mod * self.downscaling_factor)

        if pad1 != 0 or pad2 != 0:
            x = replication_pad2d_naive(x, (0, pad1, 0, pad2), detach=True)
        mask_f = mask.to(x.dtype)
        if pad1 != 0 or pad2 != 0:
            # mask the padding area
            mask_f = F.pad(mask_f, (0, pad1, 0, pad2), mode="constant", value=1.0)
            x = x * (1 - mask_f)

        x = self._forward(x, mask_f)
        x = F.pad(x, (0, -pad1, 0, -pad2))

        return x


@register_model
class LightOutpaintV1(I2IBaseModel):
    name = "stlizer.light_outpaint_v1"

    def __init__(self):
        super(LightOutpaintV1, self).__init__(locals(), scale=1, offset=0, in_channels=3, blend_size=0)
        self.net = OutpaintBase(64, window_size=8)

    def forward(self, x, mask):
        mask3 = mask.expand_as(x).to(x.dtype)
        z = self.net(x, mask)
        x = x.to(z.dtype).clone()
        x = x * (1 - mask3) + z * mask3

        if self.training:
            return x, z
        else:
            return x.clamp(0, 1)

    def infer(self, x, mask, max_size=640, composite=True):
        src = x
        src_mask3 = mask.expand_as(x)

        height, width = x.shape[-2:]
        if max(height, width) > max_size:
            if height > width:
                new_height = max_size
                new_width = round(width * (new_height / height))
            else:
                new_width = max_size
                new_height = round(height * (new_width / width))
            x = F.interpolate(x, (new_height, new_width), mode="bilinear", align_corners=False)
            mask = F.interpolate(mask.to(x.dtype), (new_height, new_width), mode="bilinear", align_corners=False)
            # dilate mask border
            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
            mask = (mask > 0.5)
            mask3 = mask.expand_as(x)
            x[mask3] = 0
        else:
            mask3 = src_mask3

        z = self.net(x, mask)
        if z.shape[2] != height or z.shape[3] != width:
            z = F.interpolate(z, (height, width), mode="bilinear", align_corners=False)

        if composite:
            src = src.clone()
            src[src_mask3] = z[src_mask3].clamp(0, 1).to(src.dtype)
            return src
        else:
            return z


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = False
    N = 20
    B = 8
    S = (640, 640)  # 800FPS on RTX3070ti

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    x = torch.zeros((B, 3, *S)).to(device)
    mask = torch.zeros((B, 1, *S), dtype=torch.bool).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z = model(x, mask)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(x, mask)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    _bench("stlizer.light_outpaint_v1")
