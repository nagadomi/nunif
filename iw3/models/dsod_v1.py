import torch
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.utils.u2netp import U2NETP


@register_model
class DSODV1(I2IBaseModel):
    name = "iw3.dsod_v1"

    def __init__(self):
        super(DSODV1, self).__init__(locals(), scale=1, offset=0, in_channels=3, blend_size=0)
        self.u2netp = U2NETP()

    @staticmethod
    def to_feature(depth):
        depth_sqrt = depth ** 0.5
        depth_pow = depth ** 2
        x = torch.cat([depth, depth_sqrt, depth_pow], dim=1)
        return x

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.to_feature(x)
        outputs = self.u2netp(x)
        return outputs

    @torch.inference_mode()
    def infer(self, x):
        H, W = x.shape[-2:]
        B = x.shape[0]

        if not (H == 192 and W == 192):
            x = F.interpolate(x, (192, 192), mode="bilinear", antialias=True, align_corners=False)

        w = self.forward(x)
        w = w.float()
        x = x.float()
        dim = tuple(range(1, x.ndim))
        x_flat = x.flatten(start_dim=1)
        low_q = x_flat.quantile(0.1, dim=1, keepdim=True).view(B, 1, 1, 1)
        high_q = x_flat.quantile(0.9, dim=1, keepdim=True).view(B, 1, 1, 1)
        mask = (x >= low_q) & (x <= high_q)
        mean_sod_pos = (w * x * mask).sum(dim=dim, keepdim=True) / ((w * mask).sum(dim=dim, keepdim=True) + 1e-6)

        return mean_sod_pos


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = True
    N = 100
    B = 8
    S = (512, 910)  # this reiszed to 192x192

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    x = torch.zeros((B, 1, *S)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z = model.infer(x)
        print(z.shape)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            z = model.infer(x)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    # 500 FPS on RTX3070ti
    _bench("iw3.dsod_v1")
