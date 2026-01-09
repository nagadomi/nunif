import torch
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.utils.u2netp import U2NETP


@register_model
class DSODV1(I2IBaseModel):
    name = "iw3.dsod_v1"

    def __init__(self):
        super(DSODV1, self).__init__(locals(), scale=1, offset=0, in_channels=4, blend_size=0, in_size=192)
        self.u2netp = U2NETP(in_ch=6)

    @staticmethod
    def to_feature(depth):
        depth_sqrt = depth ** 0.5
        depth_pow = depth ** 2
        x = torch.cat([depth, depth_sqrt, depth_pow], dim=1)
        return x

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        H, W = x.shape[-2:]
        rgb, depth = x[:, 0:3], x[:, 3:4]
        x = torch.cat((rgb, self.to_feature(depth)), dim=1)
        outputs = self.u2netp(x)
        return outputs

    @torch.inference_mode()
    def infer(self, rgb, depth, point):
        s = (self.i2i_in_size, self.i2i_in_size)
        rgb = F.interpolate(rgb, s, mode="bilinear", antialias=False, align_corners=False)
        depth = F.interpolate(depth, s, mode="bilinear", antialias=False, align_corners=False)

        x = torch.cat((rgb, depth), dim=1)
        w = self.forward(x)
        return w, depth


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = True
    N = 100
    B = 8
    S_DEPTH = (512, 910)
    S_RGB = (2160, 4384)

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    rgb = torch.zeros((B, 3, *S_RGB)).to(device)
    depth = torch.zeros((B, 1, *S_DEPTH)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z = model.infer(rgb, depth)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model.infer(rgb, depth)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    # 485 FPS on RTX3070ti
    _bench("iw3.dsod_v1")
