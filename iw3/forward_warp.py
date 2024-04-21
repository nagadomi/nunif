import torch
import torch.nn.functional as F


def box_blur(x, kernel_size=7):
    padding = kernel_size // 2
    x = F.avg_pool2d(x, kernel_size=kernel_size, padding=padding, stride=1, count_include_pad=False)
    return x


def blur_blend(x, mask):
    mask = torch.clamp(box_blur(mask.to(x.dtype)), 0, 1)
    x_blur = box_blur(x)
    return x * (1.0 - mask) + x_blur * mask


def shift_fill(x, max_tries=100):
    # TODO: If holes exist between different depth layers, they are not masked and may cause artifact
    mask = x < 0
    shift = 1
    while mask.sum() > 0 and max_tries > 0:
        if shift > 0:
            x[mask] = F.pad(x[:, :, :, 1:], (0, 1, 0, 0))[mask]
        else:
            x[mask] = F.pad(x[:, :, :, :-1], (1, 0, 0, 0))[mask]
        mask = x < 0
        shift = 0 if shift == 1 else 1
        max_tries = max_tries - 1


def forward_warp(c, depth, divergence, convergence, fill=True):
    if c.shape[2] != depth.shape[2] or c.shape[3] != depth.shape[3]:
        depth = F.interpolate(depth, size=c.shape[-2:],
                              mode="bilinear", align_corners=True, antialias=False)

    def make_forward_warp_index(batch, width, height, index, index_shift, device):
        # NOTE: For small images, this `round().long()` causes disparity banding artifacts.
        index = torch.clamp((index + index_shift).round().long(), 0, width - 1)
        index = index + torch.arange(0, height, device=device).view(1, height, 1) * width
        index = index + torch.arange(0, batch, device=device).view(batch, 1, 1) * height * width
        return index.view(-1)

    def ordered_index_copy(c, src_index, dest_index, index_order):
        B, _, H, W = c.shape
        c = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
        out = torch.empty_like(c).fill_(-1)

        # index_copy must run deterministically (depth order orverride)
        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            # NOTE: `torch.use_deterministic_algorithms(True)` is a global setting.
            #        This may cause an error if other threads run non-deterministic method while in this block.
            #        Need to exclusive lock to prevent other threads running.
            out.index_copy_(0, dest_index[index_order], c[src_index[index_order]])
        finally:
            torch.use_deterministic_algorithms(deterministic)

        return out.view(B, H, W, -1).permute(0, 3, 1, 2)

    # forward warping

    B, _, H, W = depth.shape
    shift_size = divergence * 0.01 * W * 0.5
    index_shift = depth * shift_size - (shift_size * convergence)
    x_index = torch.arange(0, W, device=c.device).view(1, 1, W).expand(B, H, W)
    src_index = make_forward_warp_index(B, W, H, x_index, 0, c.device)
    index_order = torch.argsort(depth.view(-1), dim=0)

    left_dest_index = make_forward_warp_index(B, W, H, x_index, index_shift, c.device)
    right_dest_index = make_forward_warp_index(B, W, H, x_index, -index_shift, c.device)
    left_eye = ordered_index_copy(c, src_index, left_dest_index, index_order)
    right_eye = ordered_index_copy(c, src_index, right_dest_index, index_order)

    if fill:
        # super simple inpainting
        shift_fill(left_eye)
        shift_fill(right_eye)
    else:
        # drop undefined values
        left_eye = torch.clamp(left_eye, 0, 1)
        right_eye = torch.clamp(right_eye, 0, 1)

    return left_eye, right_eye


def forward_warp2x(c, depth, divergence, convergence):
    # This reduces disparity banding artifacts but result is blurred
    out_size = c.shape[2:]
    c = F.interpolate(c, (c.shape[2] * 2, c.shape[3] * 2), mode="bilinear", align_corners=True)
    left_eye, right_eye = forward_warp(c, depth, divergence, convergence, fill=True)
    left_eye = F.interpolate(left_eye, out_size, mode="bilinear")
    right_eye = F.interpolate(right_eye, out_size, mode="bilinear")
    return left_eye, right_eye


def apply_divergence_forward_warp(c, depth, divergence, convergence, method=None):
    fill = (method == "forward_fill")
    return forward_warp(c, depth, divergence, convergence, fill=fill)
