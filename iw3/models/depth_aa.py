import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.replication_pad2d import replication_pad2d_naive, ReplicationPad2dNaive
from nunif.modules.init import basic_module_init
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.attention import WindowMHA2d, WindowScoreBias


class WABlock(nn.Module):
    def __init__(self, in_channels, window_size, shift, layer_norm=False):
        super(WABlock, self).__init__()
        self.mha = WindowMHA2d(in_channels, num_heads=2, window_size=window_size, shift=shift)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.GELU(),
            ReplicationPad2dNaive((1, 1, 1, 1), detach=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1, inplace=True))
        self.bias = WindowScoreBias(window_size)

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x


@register_model
class DepthAA(I2IBaseModel):
    name = "iw3.depth_aa"

    def __init__(self):
        super(DepthAA, self).__init__(locals(), scale=1, offset=0, in_channels=1, blend_size=0)
        C = 32
        self.proj_in = nn.Conv2d(4, C, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([
            WABlock(C, window_size=8, shift=True),
            WABlock(C, window_size=8, shift=False),
            WABlock(C, window_size=8, shift=True),
        ])
        self.proj_out = nn.Conv2d(C, 4, kernel_size=1, stride=1, padding=0)
        basic_module_init(self.proj_in)
        basic_module_init(self.proj_out)

    @torch.inference_mode()
    def infer(self, x):
        # TODO: not tested
        min_value, max_value = x.amin(), x.amax()
        scale = (max_value - min_value)
        x = (x - min_value) / scale
        x = torch.nan_to_num(x)
        x = self.forward(x, clamp=False)
        x = x * scale
        x = x + min_value
        return x

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x, clamp=None):
        src = x
        input_height, input_width = x.shape[2:]
        pad_w = (16 - input_width % 16)
        pad_h = (16 - input_height % 16)
        pad_w1 = pad_w // 2
        pad_w2 = pad_w - pad_w1
        pad_h1 = pad_h // 2
        pad_h2 = pad_h - pad_h1
        x = replication_pad2d_naive(x, (pad_w1, pad_w2, pad_h1, pad_h2), detach=True)

        x = F.pixel_unshuffle(x, 2)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (-pad_w1, -pad_w2, -pad_h1, -pad_h2), mode="constant")
        x = src + x

        if clamp is None:
            if not self.training:
                x = torch.clamp(x, 0, 1)
        else:
            if clamp:
                x = torch.clamp(x, 0, 1)

        return x


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = False  # compiled model is about 2x faster but no windows support
    N = 100
    B = 4
    S = (518, 518)

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    x = torch.zeros((B, 1, *S)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z = model(x)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model(x)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    # 520 FPS on RTX3070ti
    _bench("iw3.depth_aa")
