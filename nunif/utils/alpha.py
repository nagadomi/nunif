import torch
import torch.nn.functional as F


EPS = 1e-7


def _sum2d(x, kernel):
    if x.dim() == 2:
        return F.conv2d(x.unsqueeze(0).unsqueeze(0), weight=kernel, stride=1, padding=1).squeeze(0).squeeze(0)
    else:
        return F.conv2d(x.unsqueeze(0), weight=kernel, stride=1, groups=3, padding=1).squeeze(0)


@torch.no_grad()
def make_alpha_border(rgb, alpha, offset):
    # Note: this function is faster on CPU than on GPU
    if alpha is None:
        return rgb
    rgb = rgb.clone()
    alpha = alpha[0]
    kernel = torch.ones((1, 1, 3, 3)).to(rgb.device)
    kernel3 = torch.ones((3, 1, 3, 3)).to(rgb.device)
    mask = torch.zeros(alpha.shape).to(rgb.device)
    mask[alpha > 0] = 1
    mask_nega = (mask - 1).abs_().byte()

    rgb[:, mask_nega > 0] = 0

    for i in range(offset):
        mask_weight = _sum2d(mask, kernel)
        border = _sum2d(rgb, kernel3)
        border /= mask_weight + EPS
        rgb[:, mask_nega > 0] = border[:, mask_nega > 0]
        mask.copy_(mask_weight)
        mask[mask_weight > 0] = 1
        mask_nega = (mask - 1).abs_().byte()

    return rgb.clamp_(0, 1)


def fill_alpha(fg, alpha, val=0):
    if alpha is None:
        return fg
    assert (alpha.shape[1] == fg.shape[1] and alpha.shape[2] == fg.shape[2])
    assert (isinstance(fg, (torch.FloatTensor, torch.cuda.FloatTensor)) and
            isinstance(alpha, (torch.FloatTensor, torch.cuda.FloatTensor)))

    alpha = alpha.squeeze(0)
    fg = fg.clone()
    bg = torch.full(fg.shape, fill_value=val)
    alpha_inv = 1.0 - alpha
    for ch in range(fg.shape[0]):
        bg[ch].mul_(alpha_inv)
        fg[ch].mul_(alpha)
    fg.add_(bg).clamp_(0, 1)
    return fg
