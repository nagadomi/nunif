import torch
from torch import nn
import lpips
from os import path
from .compile_wrapper import conditional_compile
from .pad import get_pad_size
from .local_std_mask import local_std_mask
from .reflection_pad2d import reflection_pad2d_naive


# Patch `lpips.normalize_tensor`
# ref: https://github.com/facebookresearch/NeuralCompression/blob/main/neuralcompression/loss_fn/_normfix_lpips.py
def _normalize_tensor_fix(in_feat, eps=1e-8):
    return in_feat * torch.rsqrt(torch.sum(in_feat.to(torch.float32) ** 2 + eps, dim=1, keepdim=True)).to(in_feat.dtype)


def _spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def _upsample(in_tens, out_HW=(64, 64)):
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIPSFix(lpips.LPIPS):
    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        # from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            # nagadomi: use _normalize_tensor_fix
            feats0[kk], feats1[kk] = _normalize_tensor_fix(outs0[kk]), _normalize_tensor_fix(outs1[kk])
            # nagadomi: detach target grad
            diffs[kk] = (feats0[kk] - feats1[kk].detach()) ** 2

        if self.lpips:
            if self.spatial:
                res = [_upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [_spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [_upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [_spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for i in range(self.L):
            val += res[i]

        if retPerLayer:
            return (val, res)
        else:
            return val


# MODEL_PATH = None
# MODEL_PATH = path.join(path.dirname(__file__), "_lpips_1.pth")
MODEL_PATH = path.join(path.dirname(__file__), "_lpips_2.pth")


class LPIPSWith(nn.Module):
    def __init__(self, base_loss, weight=1.0, std_mask=False):
        super().__init__()
        self.base_loss = base_loss
        self.weight = weight
        self.std_mask = std_mask
        self.lpips = lpips.LPIPS(net='vgg', model_path=MODEL_PATH).eval()
        # This needed because LPIPS has duplicate parameter references problem
        self.lpips.requires_grad_(False)
        # Override foward method
        self.lpips.__class__ = LPIPSFix

    def train(self, mode=True):
        super().train(mode)
        self.lpips.train(False)
        self.lpips.requires_grad_(False)

    def forward(self, input, target):
        pad = get_pad_size(input, 16, random_shift=False)
        input = reflection_pad2d_naive(input, pad, detach=True)
        target = reflection_pad2d_naive(target, pad, detach=True)
        base_loss = self.base_loss(input, target)
        if self.std_mask:
            lpips_loss = self.lpips(local_std_mask(input, target), target, normalize=True).mean()
        else:
            lpips_loss = self.lpips(input, target, normalize=True).mean()
        return base_loss + lpips_loss * self.weight


def _test():
    loss = LPIPSWith(nn.L1Loss())
    print(loss.lpips)
    print(loss.lpips.training)
    print([p.requires_grad for p in loss.lpips.parameters()])

    x = torch.randn((4, 3, 64, 64))
    t = torch.randn((4, 3, 64, 64))
    print(loss(x, t))


def _check_patch():
    loss = LPIPSWith(nn.L1Loss()).cuda()
    x = torch.randn((4, 3, 64, 64)).cuda()
    t = torch.randn((4, 3, 64, 64)).cuda()
    with torch.autocast(device_type="cuda"):
        print(loss(x, t))


if __name__ == "__main__":
    # _check_patch()
    _test()
