import torch
import torch.nn as nn
import torch.nn.functional as F


# ref: Refining activation downsampling with SoftPool
#      https://arxiv.org/abs/2101.00440


def soft_pool2d(x, kernel_size=2, stride=None, eps=1e-6, fp16_max=6e4, fp32_max=3e38):
    # Note: input x should be small values.
    #       value greater than 8.8, the result may be incorrect by clipping.
    # Note: When used for image downscaling, modcrop/padding is needed.
    #       Also corner align is different from general downscaling filers.
    fp16 = (x.dtype == torch.float16)
    if fp16:
        x = x.to(torch.float32)

    e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
    e_x = torch.clamp(e_x, 0., fp32_max)
    x = F.avg_pool2d(x * e_x, kernel_size, stride=stride)
    x = torch.clamp(x, -fp32_max, fp32_max)
    w = F.avg_pool2d(e_x, kernel_size, stride=stride)
    # x / w == (x * kernel_size * 2) / (w * kernel_size * 2)
    x = x / (w + eps)
    if fp16:
        x = x.to(torch.float16)
        x = torch.clamp(x, -fp16_max, fp16_max)

    return x


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, eps=1e-6, fp16_max=6e4, fp32_max=3e38):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.fp16_max = fp16_max
        self.fp32_max = fp32_max

    def forward(self, x):
        return soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride,
                           eps=self.eps, fp16_max=self.fp16_max, fp32_max=self.fp32_max)


def _test():
    scale = 2
    torch.manual_seed(1)
    x = torch.randn((16, 3, 32, 32)).cuda() * scale
    pool = SoftPool2d(2).cuda()
    z = pool(x)
    print(z.shape, z.min(), z.mean(), z.max())

    z = pool(x.half())
    print(z.shape, z.min(), z.mean(), z.max())


def _test_downscaling():
    import argparse
    import torchvision.transforms.functional as TF
    import torchvision.io as io

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input file")
    args = parser.parse_args()
    x = io.read_image(args.input, io.ImageReadMode.RGB)
    x = (x / 255.0)
    pad_h = x.shape[1] % 2
    pad_w = x.shape[2] % 2
    if pad_h != 0 or pad_w != 0:
        x = TF.pad(x, (0, 0, pad_w, pad_h), padding_mode="edge")
    z = soft_pool2d(x.unsqueeze(0), 2)
    z = torch.clamp(z, 0, 1).squeeze(0)
    TF.to_pil_image(z.squeeze(0)).show()


if __name__ == "__main__":
    # _test()
    _test_downscaling()
