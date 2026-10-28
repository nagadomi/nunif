import torch
import torch.nn.functional as F
from torch import nn

from nunif.models import Model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import RMSNorm2d

from .dit import DOWNSAMPLING_FACTOR, Blocks, LastLayer


class DiTSelfCond(Model):
    name = "playground.rectified_flow.self_cond_dit"

    def __init__(self, dim=384, time_dim=32, use_norm_path=True):
        assert dim % 64 == 0
        super().__init__(locals())
        C = dim
        self.time_dim = time_dim
        self.proj_dim = C
        self.in_proj = nn.Conv2d(3 * DOWNSAMPLING_FACTOR**2, C, kernel_size=1, stride=1, padding=0)
        self.blocks = Blocks(C, num_heads=C // 64, num_layers=12, time_dim=self.time_dim)
        self.last_layer = LastLayer(C, time_dim, 3 * DOWNSAMPLING_FACTOR**2, use_norm_path=use_norm_path)
        self.self_cond_mlp = nn.Sequential(
            nn.Conv2d(3 * DOWNSAMPLING_FACTOR**2, C, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            RMSNorm2d(C),
            nn.Conv2d(C, time_dim, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(time_dim, time_dim, kernel_size=1, stride=1, padding=0),
        )

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x, t=None):
        # v-pred
        x = F.pixel_unshuffle(x, DOWNSAMPLING_FACTOR)
        # use self cond instead of timestep cond
        self_emb = self.self_cond_mlp(x)
        x = self.in_proj(x)
        x = self.blocks(x, self_emb)
        x = self.last_layer(x, self_emb)
        x = F.pixel_shuffle(x, DOWNSAMPLING_FACTOR)

        return x


def _test_model():
    g = DiTSelfCond()
    x = torch.zeros((4, 3, 112, 112))
    t = torch.zeros((4,))
    z = g(x, t)
    assert z.shape == (4, 3, 112, 112)


if __name__ == "__main__":
    _test_model()
