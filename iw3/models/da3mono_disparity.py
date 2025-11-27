import torch
import torch.nn as nn
from nunif.models import I2IBaseModel, register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.init import basic_module_init


FEAT_DIM = 64


@register_model
class DA3MonoDisparity(I2IBaseModel):
    name = "iw3.da3mono_disparity"

    def __init__(self):
        super(DA3MonoDisparity, self).__init__(locals(), scale=1, offset=0, in_channels=1, blend_size=0)
        C = 128
        self.mlp = nn.Sequential(
            nn.Linear(FEAT_DIM, C),
            nn.SiLU(),
            nn.Linear(C, C),
            nn.SiLU(),
            nn.Linear(C, 2),
            nn.ReLU(),  # clip(min=0)
        )
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, depth):
        x = self.extract_features(depth)

        x = self.mlp(x)

        if depth.ndim == 3:
            shift = x[0:1].reshape(1, 1, 1)
            sky_shift = x[1:2].reshape(1, 1, 1)
            sky_mask = depth == depth.max()
            depth = torch.where(sky_mask, depth + sky_shift, depth)
        else:
            shift = x[:, 0:1].reshape(-1, 1, 1, 1)
            sky = x[:, 1:2].reshape(-1, 1, 1, 1)
            depths = []
            for i in range(depth.shape[0]):
                sky_mask = depth[i] == depth[i].max()
                depths.append(torch.where(sky_mask, sky[i] + depth[i], depth[i]))
            depth = torch.stack(depths)

        disparity = 1.0 / (depth + shift)

        return disparity

    @staticmethod
    def extract_features(x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
            batch = False
        else:
            batch = True
        B, C, H, W = x.shape
        N = FEAT_DIM
        assert C == 1, "C must be 1"

        x_flat = x.view(B, -1)
        x_sorted, _ = torch.sort(x_flat, dim=-1)
        indices = torch.linspace(1, x_sorted.size(-1) - 2, N - 2, device=x.device).long()
        quantile_values = torch.gather(x_sorted, -1, indices[None, :].expand(B, -1))
        # (min_value, quantile_values, max_value)
        features = torch.cat([x_sorted[:, :1], quantile_values, x_sorted[:, -1:]], dim=-1)

        if not batch:
            features = features.squeeze(0)

        return features

    def load(self):
        pass


def _bench(name):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    do_compile = False  # compiled model is about 2x faster but no windows support
    N = 1000
    B = 8
    S = (920, 518)

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
    # 4700 FPS on RTX3070ti
    _bench("iw3.da3mono_disparity")
