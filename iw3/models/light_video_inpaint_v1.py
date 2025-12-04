import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.permute import pixel_shuffle, pixel_unshuffle
from nunif.modules.replication_pad2d import replication_pad2d_naive, ReplicationPad2dNaive
from nunif.modules.init import basic_module_init, icnr_init
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import FastLayerNorm
from nunif.modules.attention import WindowGMLP2d, WindowGMLP3d
from nunif.modules.gaussian_filter import SeparableGaussianFilter2d
from iw3.dilation import mask_closing, dilate_inner, dilate_outer


def linear_blur(x, n=4):
    z = x / (n + 1)
    for i in range(n):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        z = z + x / (n + 1)

    return z.clamp(0, 1)


class GLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=1, padding=True):
        super().__init__()
        kernel_size = 3
        if padding:
            self.pad = ReplicationPad2dNaive(((kernel_size - 1) // 2,) * 4, detach=True)
        else:
            self.pad = nn.Identity()
        self.w1 = nn.Conv2d(in_channels, in_channels * mlp_ratio, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(in_channels * mlp_ratio // 2, in_channels, kernel_size=kernel_size, stride=1, padding=0)
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.pad(x)
        x = self.w2(x)
        return x


class GMLPBlock(nn.Module):
    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False):
        super().__init__()
        self.gmlp = WindowGMLP2d(in_channels, window_size=window_size, mlp_ratio=mlp_ratio, shift=shift)
        self.norm1 = FastLayerNorm(in_channels, bias=False)
        self.norm2 = FastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = GLUConvMLP(in_channels, in_channels, mlp_ratio=1)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = x + self.gmlp(x, self.norm1, self.norm2)
        x = x + self.glu_conv(x)
        return x


class GMLP3DBlock(nn.Module):
    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False):
        super().__init__()
        self.gmlp = WindowGMLP3d(in_channels, window_size=window_size, mlp_ratio=mlp_ratio, shift=shift)
        self.norm1 = FastLayerNorm(in_channels, bias=False)
        self.norm2 = FastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = GLUConvMLP(in_channels, in_channels, mlp_ratio=1)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        # BCHW -> 1CDHW
        B, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3).reshape(1, C, B, H, W)
        x = x + self.gmlp(x, self.norm1, self.norm2)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C, H, W)
        x = x + self.glu_conv(x)
        return x


def chunked_forward(module, x, chunk):
    assert x.shape[0] % chunk == 0
    ret = []
    for i in range(0, x.shape[0], chunk):
        ret.append(module(x[i:i + chunk]))
    return torch.cat(ret, dim=0)


SEQ_LEN = 12


@register_model
class LightVideoInpaintV1(I2IBaseModel):
    name = "inpaint.light_video_inpaint_v1"
    name_alias = ("inpaint.light_video_inpaint_v1_small",)

    def __init__(self, base_dim=96, lv2_mlp_ratio=1):
        super(LightVideoInpaintV1, self).__init__(locals(), scale=1, offset=16, in_channels=3, blend_size=8)
        self.sequence_offset = 0
        self.downscaling_factor = 4
        self.mod = 16
        pack = self.downscaling_factor ** 2
        C = base_dim
        C2 = C * 2
        self.mask_bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.patch = nn.Conv2d(3, C, kernel_size=self.downscaling_factor, stride=self.downscaling_factor, padding=0)
        self.enc1 = GMLPBlock(C, mlp_ratio=2, window_size=16, shift=False)
        self.down = nn.Conv2d(C, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = nn.ModuleList([
            GMLPBlock(C2, window_size=(8, 8), mlp_ratio=lv2_mlp_ratio, shift=True),
            GMLP3DBlock(C2, window_size=(SEQ_LEN, 1, 1), mlp_ratio=2, shift=False),
            GMLPBlock(C2, window_size=(8, 8), mlp_ratio=lv2_mlp_ratio, shift=False),
            GMLP3DBlock(C2, window_size=(SEQ_LEN, 1, 1), mlp_ratio=2, shift=False),
            GMLPBlock(C2, window_size=(8, 8), mlp_ratio=lv2_mlp_ratio, shift=True),
        ])
        self.up = nn.Conv2d(C2, C * 4, kernel_size=1, stride=1, padding=0)
        self.dec1 = GMLPBlock(C, mlp_ratio=2, window_size=16, shift=False)
        self.to_image = nn.Conv2d(C, 3 * pack, kernel_size=1, stride=1, padding=0)
        basic_module_init(self.patch)
        icnr_init(self.to_image, scale_factor=4)
        nn.init.trunc_normal_(self.mask_bias, 0, 0.01)

        self.mask_blur = SeparableGaussianFilter2d(1, kernel_size=15, padding=15 // 2)

    def preprocess(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None):
        if closing:
            mask = mask_closing(mask)
        else:
            mask = mask.float()

        mask = dilate_inner(mask, n_iter=inner_dilation, base_width=base_width)
        mask = dilate_outer(mask, n_iter=outer_dilation, base_width=base_width)

        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    def infer(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None):
        if x.shape[0] % SEQ_LEN != 0:
            # TODO: refactor
            pad_b = (SEQ_LEN - x.shape[0] % SEQ_LEN)
            pad_b1 = pad_b // 2
            pad_b2 = pad_b - pad_b1
            x_pad = [x[0:1].detach()] * pad_b1 + [x] + [x[-1:].detach()] * pad_b2
            x = torch.cat(x_pad, dim=0)
            mask_pad = [mask[0:1].detach()] * pad_b1 + [mask] + [mask[-1:].detach()] * pad_b2
            mask = torch.cat(mask_pad, dim=0)
        else:
            pad_b1 = pad_b2 = 0

        x, mask = self.preprocess(x, mask, closing=closing,
                                  inner_dilation=inner_dilation, outer_dilation=outer_dilation,
                                  base_width=base_width)

        out = self.forward(x, mask, skip_i2i_offset=True, micro_batch_size=2)
        if pad_b1 > 0:
            out = out[pad_b1:]
        if pad_b2 > 0:
            out = out[:-pad_b2]

        return out

    def _forward(self, x, mask, micro_batch_size=SEQ_LEN):
        assert x.shape[0] == SEQ_LEN

        mask = pixel_unshuffle(mask, self.downscaling_factor).amax(dim=1, keepdim=True) > 0.99

        x1s = []
        x2s = []
        for i in range(0, x.shape[0], micro_batch_size):
            x0 = F.leaky_relu(self.patch(x[i:i + micro_batch_size]), 0.1, inplace=True)
            x0 = torch.where(mask[i:i + micro_batch_size], self.mask_bias.to(x0.dtype), x0)
            x1 = self.enc1(x0)
            x2 = self.down(x1)
            x1s.append(x1)
            x2s.append(x2)

        x2 = torch.cat(x2s, dim=0)
        del x2s

        for mod in self.enc2:
            if isinstance(mod, (GMLP3DBlock,)):
                x2 = mod(x2)
            else:
                x2 = chunked_forward(mod, x2, micro_batch_size)

        outs = []
        for i in range(0, x.shape[0], micro_batch_size):
            x3 = self.up(x2[i:i + micro_batch_size])
            x3 = F.pixel_shuffle(x3, 2)
            out = self.dec1(x1s[i // micro_batch_size] + x3)
            out = self.to_image(out)
            outs.append(out)
        x = torch.cat(outs, dim=0)
        x = pixel_shuffle(x, self.downscaling_factor)
        return x

    def forward(self, x, mask, skip_i2i_offset=False, micro_batch_size=SEQ_LEN):
        src = x
        x = (x - 0.5) / 0.5

        input_height, input_width = x.shape[2:]
        pad1 = (self.mod * self.downscaling_factor) - input_width % (self.mod * self.downscaling_factor)
        pad2 = (self.mod * self.downscaling_factor) - input_height % (self.mod * self.downscaling_factor)
        padding = (0, pad1, 0, pad2)
        x = replication_pad2d_naive(x, padding, detach=True)
        mask = replication_pad2d_naive(mask, padding, detach=True)

        # forward
        x = self._forward(x, mask, micro_batch_size=micro_batch_size)
        x = F.pad(x, (0, -pad1, 0, -pad2))
        mask = F.pad(mask, (0, -pad1, 0, -pad2))

        # post process
        if not skip_i2i_offset:
            src = F.pad(src.to(x.dtype), (-self.i2i_offset,) * 4)
            mask = F.pad(mask, (-self.i2i_offset,) * 4)
            x = F.pad(x, (-self.i2i_offset,) * 4)
            if self.sequence_offset > 0:
                src = src[self.sequence_offset:-self.sequence_offset]
                x = x[self.sequence_offset:-self.sequence_offset]
                mask = mask[self.sequence_offset:-self.sequence_offset]
        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask

        if not self.training:
            src = src.clamp(0, 1)

        return src


@register_model
class LightVideoInpaintV1Medium(LightVideoInpaintV1):
    name = "inpaint.light_video_inpaint_v1_medium"
    name_alias = ()

    def __init__(self, base_dim=128, lv2_mlp_ratio=2):
        super(LightVideoInpaintV1Medium, self).__init__(base_dim=base_dim, lv2_mlp_ratio=lv2_mlp_ratio)


@register_model
class LightVideoInpaintV1Large(LightVideoInpaintV1):
    name = "inpaint.light_video_inpaint_v1_large"
    name_alias = ()

    def __init__(self, base_dim=192, lv2_mlp_ratio=2):
        super(LightVideoInpaintV1Large, self).__init__(base_dim=base_dim, lv2_mlp_ratio=lv2_mlp_ratio)


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = False  # compiled model is about 2x faster but no windows support
    N = 20
    B = SEQ_LEN
    # S = (4320, 7680)
    # S = (2160, 3840)
    S = (1080, 1920)  # HD, 36FPS
    # S = (320, 320)

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    x = torch.zeros((B, 3, *S)).to(device)
    mask = torch.zeros((B, 1, *S), dtype=torch.bool).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z = model(*model.preprocess(x, mask), micro_batch_size=2)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(*model.preprocess(x, mask), micro_batch_size=2)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    _bench("inpaint.light_video_inpaint_v1_small")  # same as `inpaint.light_video_inpaint_v1`
    _bench("inpaint.light_video_inpaint_v1_medium")
    _bench("inpaint.light_video_inpaint_v1_large")
