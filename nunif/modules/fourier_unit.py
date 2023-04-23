import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm


# NOTE: This module does not support export to ONNX (at 2023-04, rfftn and irfftn)


class FourierUnit(nn.Module):
    """ From LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions
             https://github.com/advimman/lama
             Fast Fourier Convolution
             https://github.com/pkumivision/FFC
    """
    def __init__(self, in_channels, out_channels,
                 norm_layer=lambda dim: nn.BatchNorm2d(dim),
                 activation_layer=lambda dim: nn.ReLU(inplace=True),
                 spectral_norm=False, bias=False, residual=True):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels * 2, out_channels * 2,
                                    kernel_size=1, stride=1, padding=0, bias=bias)
        if spectral_norm:
            self.conv = _spectral_norm(self.conv)
        self.act = activation_layer(out_channels * 2)
        self.norm = norm_layer(out_channels * 2)
        if residual:
            if in_channels == out_channels:
                self.identity = nn.Identity()
            else:
                self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.identity = None

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, H, W/2+1, 2)
        if x.dtype == torch.float16:
            ffted = torch.fft.rfftn(x.to(torch.float32), dim=(-2, -1), norm="ortho")
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1).to(torch.float16)
        else:
            ffted = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # (B, C, 2, H, W/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        # (B, C*2, H, W/2+1)
        ffted = ffted.view((B, -1) + ffted.shape[3:])

        # (B, OUT_C*2, H, W/2+1)
        ffted = self.act(self.norm(self.conv(ffted)))
        # (B, OUT_C, H, W/2+1, 2)
        ffted = ffted.view((B, -1, 2) + ffted.shape[2:]).permute(0, 1, 3, 4, 2).contiguous()
        if x.dtype == torch.float16:
            # (B, OUT_C, H, W/2+1)
            ffted = ffted.to(torch.float32)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])
            # (B, OUT_C, H, W)
            output = torch.fft.irfftn(ffted, s=(H, W), dim=(-2, -1), norm="ortho").to(torch.float16)
        else:
            # (B, OUT_C, H, W/2+1)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])
            # (B, OUT_C, H, W)
            output = torch.fft.irfftn(ffted, s=(H, W), dim=(-2, -1), norm="ortho")

        if self.identity is not None:
            output = output + self.identity(x)

        return output


def FourierUnitSNLReLU(in_channels, out_channels, residual=True):
    return FourierUnit(in_channels, out_channels,
                       norm_layer=lambda dim: nn.Identity(),
                       activation_layer=lambda dim: nn.LeakyReLU(0.2, inplace=True),
                       spectral_norm=True, bias=True, residual=residual)


if __name__ == "__main__":
    x = torch.zeros((4, 8, 32, 32)).cuda()
    fourier_unit = FourierUnitSNLReLU(8, 64).cuda()
    z = fourier_unit(x)
    print(z.shape)
