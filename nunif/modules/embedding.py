import torch
import torch.nn as nn


class SoftEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.k = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels))
        self.v = nn.Parameter(torch.zeros((out_channels, out_channels),
                                          dtype=torch.float32))
        self.proj = nn.Linear(out_channels, out_channels)
        nn.init.normal_(self.v, 0, out_channels ** -0.5)

    def forward(self, x):
        B, _ = x.shape
        C = self.v.shape[0]
        v = self.v.view(1, C, C).expand((B, C, C))
        k = self.k(x)
        w = torch.sigmoid(k) * (C ** -0.5)
        w = w.view(k.shape[0], C, 1).expand(v.shape)
        return self.proj((v * w).sum(dim=2))


class PositionalSeeding(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        assert isinstance(upscale_factor, int)
        super().__init__()
        self.upscaler = nn.PixelShuffle(upscale_factor)
        self.embeds = nn.ModuleList([SoftEmbedding(in_channels, out_channels)
                                     for _ in range(upscale_factor * upscale_factor)])

    def forward(self, x):
        assert x.ndim == 2
        B, C = x.shape
        x = torch.cat([embed(x) for embed in self.embeds], dim=1).view(B, -1, 1, 1)
        z = self.upscaler(x)
        return z


def _spec():
    s = PositionalSeeding(2, 16, 4)
    x = torch.rand(4, 2)
    z = s(x)
    print(z.shape)


if __name__ == "__main__":
    _spec()
