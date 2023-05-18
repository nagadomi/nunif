import torch
import torch.nn as nn


class ChannelWiseSum(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                              stride=1, padding=1, padding_mode="zeros", groups=in_channels,
                              bias=False)
        self.conv.weight.data.fill_(1.)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        if x.ndim == 3:
            # CHW
            x = x.unsqueeze(0)
            x = self.conv(x)
            x = x.squeeze(0)
        elif x.ndim == 2:
            # HW
            x = x.unsqueeze(0).unsqueeze(0)
            x = self.conv(x)
            x = x.squeeze(0).squeeze(0)
        else:
            # BCHW
            x = self.conv(x)

        return x


class AlphaBorderPadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.sum_alpha = ChannelWiseSum(1, 3)
        self.sum_rgb = ChannelWiseSum(3, 3)
        self.eval()

    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor, offset: int):
        # rgb: CHW, alpha: CHW
        assert rgb.ndim == 3 and alpha.ndim == 3 and rgb.shape[0] == 3 and alpha.shape[0] == 1
        rgb = rgb.clone()
        alpha = alpha.squeeze(0)
        mask = alpha.new_zeros(alpha.shape)
        mask[alpha > 0] = 1.
        mask_nega = mask < 1.
        rgb[:, mask_nega] = 0.
        for i in range(offset):
            mask_weight = self.sum_alpha(mask)
            border = self.sum_rgb(rgb)
            border /= mask_weight + 1e-7
            rgb[:, mask_nega] = border[:, mask_nega]
            mask.zero_()
            mask[mask_weight > 0] = 1.
            mask_nega = mask < 1.

        return rgb.clamp_(0., 1.)


if __name__ == "__main__":
    from nunif.utils import pil_io
    import argparse
    import cv2

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input file")
    parser.add_argument("--output", "-o", type=str, help="output file")
    parser.add_argument("--output-rgb", type=str, help="output rgb file")
    args = parser.parse_args()
    im, _ = pil_io.load_image(args.input, color="rgb", keep_alpha=True)
    rgb, alpha = pil_io.to_tensor(im, return_alpha=True)
    alpha_pad = AlphaBorderPadding().eval()
    with torch.no_grad():
        padded_rgb = alpha_pad(rgb, alpha, 8)
        padded_rgb = pil_io.to_image(padded_rgb)
        rgb = pil_io.to_image(rgb)
        if args.output is not None:
            padded_rgb.save(args.output)
        if args.output_rgb is not None:
            rgb.save(args.output_rgb)
        cv2.imshow("rgb", pil_io.to_cv2(rgb))
        cv2.imshow("padded_rgb", pil_io.to_cv2(padded_rgb))
        cv2.waitKey(0)
