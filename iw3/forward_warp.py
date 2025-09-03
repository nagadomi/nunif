import torch
import torch.nn.functional as F
from nunif.modules.replication_pad2d import ReplicationPad2d
from nunif.device import autocast
from .dilation import mask_closing


def box_blur(x, kernel_size=7):
    padding = kernel_size // 2
    x = F.avg_pool2d(x, kernel_size=kernel_size, padding=padding, stride=1, count_include_pad=False)
    return x


def blur_blend(x, mask):
    mask = torch.clamp(box_blur(mask.to(x.dtype)), 0, 1)
    x_blur = box_blur(x)
    return x * (1.0 - mask) + x_blur * mask


def shift_fill(x, sign, flip_sign=False, max_tries=100):
    mask = x < 0
    while mask.any().item() and max_tries > 0:
        if sign > 0:
            x[mask] = F.pad(x[:, :, :, 1:], (0, 1, 0, 0))[mask]
        else:
            x[mask] = F.pad(x[:, :, :, :-1], (1, 0, 0, 0))[mask]
        mask = x < 0
        max_tries = max_tries - 1
        if flip_sign:
            sign = -1 if sign > 0 else 1

    return x


def shift_fill_pack(left_eye, right_eye, inconsistent_shift=False):
    if inconsistent_shift:
        pack = torch.cat([left_eye, right_eye], dim=1)
        left_eye, right_eye = shift_fill(pack, 1, flip_sign=True).chunk(2, dim=1)
        return left_eye, right_eye
    else:
        pack = torch.cat([left_eye, torch.flip(right_eye, dims=(-1,))], dim=1)
        left_eye, right_eye = shift_fill(pack, -1).chunk(2, dim=1)
        right_eye = torch.flip(right_eye, dims=(-1,))
        return left_eye, right_eye


def fix_layered_holes(side_image, index_image, sign, max_tries=100):
    if sign > 0:
        mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (0, 1, 0, 0))
        while mask.any().item() and max_tries > 0:
            side_image[mask.expand_as(side_image)] = -2  # set undefined value
            index_image[mask] = F.pad(index_image[:, :, :, 1:], (0, 1, 0, 0))[mask]
            mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (0, 1, 0, 0))
            max_tries -= 1
    else:
        mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (1, 0, 0, 0))
        while mask.any().item() and max_tries > 0:
            side_image[mask.expand_as(side_image)] = -2
            index_image[mask] = F.pad(index_image[:, :, :, :-1], (1, 0, 0, 0))[mask]
            mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (1, 0, 0, 0))
            max_tries -= 1


def __detect_overlap_mask(index_image, mask):
    overlap_mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]).abs() > 2.1, (0, 1, 0, 0))
    overlap_mask[mask] = False
    return overlap_mask


def to_flat_index(batch, width, height, index):
    index = index + torch.arange(0, height, device=index.device).view(1, height, 1) * width
    index = index + torch.arange(0, batch, device=index.device).view(batch, 1, 1) * height * width
    index = index.view(-1)
    return index


def make_bilinear_data(batch, width, height, index, index_shift):
    float_index = torch.clamp(index + index_shift, 0, width - 1)
    floor_index = torch.clamp(float_index.floor(), 0, width - 1)
    ceil_index = torch.clamp(float_index.ceil(), 0, width - 1)
    ceil_weight = (float_index - floor_index).reshape(batch, 1, height, width)
    ceil_weight = torch.clamp(ceil_weight, min=1e-5, max=1.0 - 1e-5)
    floor_weight = 1.0 - ceil_weight
    floor_index = to_flat_index(batch, width, height, floor_index.long())
    ceil_index = to_flat_index(batch, width, height, ceil_index.long())

    return floor_index, ceil_index, floor_weight, ceil_weight


def ordered_index_copy(c, src_index, dest_index, index_order, undefined_value=-1):
    B, _, H, W = c.shape
    c = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
    if torch.is_tensor(undefined_value):
        out = undefined_value.view(1, -1).repeat(c.shape[0], 1)
    else:
        out = torch.empty_like(c).fill_(undefined_value)

    # index_copy must run deterministically (depth order orverride)
    # NOTE: `torch.use_deterministic_algorithms(True)` is a global setting.
    #        This may cause an error if other threads run non-deterministic method while in this block.
    #        Need to exclusive lock to prevent other threads running.
    # TODO: Need to remove the complicated conditions of this very simple operation.
    # for i in index_order:
    #   out[dest_index[i]] = c[src_index[i]]
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        out.index_copy_(0, dest_index[index_order], c[src_index[index_order]])
    finally:
        torch.use_deterministic_algorithms(deterministic)

    return out.view(B, H, W, -1).permute(0, 3, 1, 2)


def warp(batch, width, height, c, x_index, index_shift, src_index, index_order):
    floor_index, ceil_index, floor_weight, ceil_weight = make_bilinear_data(batch, width, height, x_index, index_shift)

    # pack for optimization
    floor_data = torch.cat([floor_weight, c], dim=1)
    ceil_data = torch.cat([ceil_weight, c], dim=1)

    # 0 for weight, -1 for pixel
    undefined_value = torch.tensor([0] + [-1] * c.shape[1], dtype=c.dtype, device=c.device)
    floor_warp = ordered_index_copy(floor_data, src_index, floor_index, index_order, undefined_value=undefined_value)
    ceil_warp = ordered_index_copy(ceil_data, src_index, ceil_index, index_order, undefined_value=undefined_value)

    # unpack
    floor_weight_warp, floor_warp = floor_warp[:, 0:1, :, :], floor_warp[:, 1:, :, :]
    ceil_weight_warp, ceil_warp = ceil_warp[:, 0:1, :, :], ceil_warp[:, 1:, :, :]

    out = (floor_warp * floor_weight_warp + ceil_warp * ceil_weight_warp) / (floor_weight_warp + ceil_weight_warp)
    out = torch.nan_to_num(out, -1)

    return out


def gen_mask2(mask):
    mask = mask[:, 0:1]
    return torch.clamp((mask == -1).float() + (mask == -2).float() * 0.5, 0, 1)


def depth_order_bilinear_forward_warp(c, depth, divergence, convergence, fill=True,
                                      synthetic_view="both", inpaint_model=None,
                                      return_mask=False, inconsistent_shift=False):
    src_image = c
    assert synthetic_view in {"both", "right", "left"}
    if c.shape[2] != depth.shape[2] or c.shape[3] != depth.shape[3]:
        depth = F.interpolate(depth, size=c.shape[-2:],
                              mode="bilinear", align_corners=True, antialias=True)
    if synthetic_view != "both":
        divergence *= 2

    # pad
    org_width = c.shape[3]
    padding_size = int(org_width * divergence * 0.01 + 2)
    pad = ReplicationPad2d((padding_size, padding_size, 0, 0))
    unpad = ReplicationPad2d((-padding_size, -padding_size, 0, 0))
    c = pad(c)
    depth = pad(depth)

    # forward warping
    B, _, H, W = depth.shape
    shift_size = divergence * 0.01 * org_width * 0.5
    index_shift = depth * shift_size - (shift_size * convergence)
    index_shift = index_shift.view(B, H, W)
    x_index = torch.arange(0, W, device=c.device).view(1, 1, W).expand(B, H, W)
    src_index = to_flat_index(B, W, H, x_index)
    index_order = torch.argsort(depth.view(-1), dim=0)

    c = torch.cat([c, x_index.view(B, 1, H, W).to(c.dtype)], dim=1)  # warp width index together

    if synthetic_view == "both":
        left_eye = warp(B, W, H, c, x_index, index_shift, src_index, index_order)
        right_eye = warp(B, W, H, c, x_index, -index_shift, src_index, index_order)

        # unpad
        left_eye = unpad(left_eye)
        right_eye = unpad(right_eye)
        left_eye, left_eye_index = left_eye[:, :-1, :, :], left_eye[:, -1:, :, :]
        right_eye, right_eye_index = right_eye[:, :-1, :, :], right_eye[:, -1:, :, :]

        # Fix layered holes
        # inspired by @math-artist patch: https://github.com/nagadomi/nunif/discussions/274
        left_eye_index, right_eye_index = shift_fill_pack(left_eye_index, right_eye_index,
                                                          inconsistent_shift=inconsistent_shift)
        fix_layered_holes(left_eye, left_eye_index, 1)
        fix_layered_holes(right_eye, right_eye_index, -1)

        if return_mask:
            left_mask, right_mask = gen_mask2(left_eye), gen_mask2(right_eye)

        if fill:
            if inpaint_model is None:
                # super simple inpainting
                left_eye, right_eye = shift_fill_pack(left_eye, right_eye, inconsistent_shift=inconsistent_shift)
            else:
                with autocast(device=left_eye.device, enabled=True):
                    left_eye = left_eye.flip(-1)
                    left_eye = inpaint_model(left_eye, (left_eye == -1)[:, 0:1, :, :], (left_eye == -2)[:, 0:1, :, :]).flip(-1)
                    right_eye = inpaint_model(right_eye, (right_eye == -1)[:, 0:1, :, :], (right_eye == -2)[:, 0:1, :, :])
        else:
            # drop undefined values
            left_eye = torch.clamp(left_eye, 0, 1)
            right_eye = torch.clamp(right_eye, 0, 1)

        if return_mask:
            return left_eye.contiguous(), right_eye.contiguous(), left_mask, right_mask
        else:
            return left_eye.contiguous(), right_eye.contiguous()

    elif synthetic_view == "right":
        right_eye = warp(B, W, H, c, x_index, -index_shift, src_index, index_order)
        right_eye = unpad(right_eye)
        right_eye, right_eye_index = right_eye[:, :-1, :, ], right_eye[:, -1:, :, ]
        right_eye_index = shift_fill(right_eye_index, 1)
        fix_layered_holes(right_eye, right_eye_index, -1)
        if return_mask:
            right_mask = gen_mask2(right_eye)
        if fill:
            if inpaint_model is None:
                right_eye = shift_fill(right_eye, 1)
            else:
                with autocast(device=right_eye.device, enabled=True):
                    right_eye = inpaint_model(right_eye, (right_eye == -1)[:, 0:1, :, :], (right_eye == -2)[:, 0:1, :, :])
        else:
            right_eye = torch.clamp(right_eye, 0, 1)

        if return_mask:
            return src_image, right_eye.contiguous(), None, right_mask
        else:
            return src_image, right_eye.contiguous()

    elif synthetic_view == "left":
        left_eye = warp(B, W, H, c, x_index, index_shift, src_index, index_order)
        left_eye = unpad(left_eye)
        left_eye, left_eye_index = left_eye[:, :-1, :, ], left_eye[:, -1:, :, ]
        left_eye_index = shift_fill(left_eye_index, -1)
        fix_layered_holes(left_eye, left_eye_index, 1)
        if return_mask:
            left_mask = gen_mask2(left_eye)

        if fill:
            if inpaint_model is None:
                left_eye = shift_fill(left_eye, -1)
            else:
                with autocast(device=left_eye.device, enabled=True):
                    left_eye = left_eye.flip(-1)
                    left_eye = inpaint_model(left_eye, (left_eye == -1)[:, 0:1, :, :], (left_eye == -2)[:, 0:1, :, :]).flip(-1)
        else:
            left_eye = torch.clamp(left_eye, 0, 1)

        if return_mask:
            return left_eye.contiguous(), src_image, left_mask, None
        else:
            return left_eye.contiguous(), src_image


def apply_divergence_forward_warp(c, depth, divergence, convergence, method=None,
                                  synthetic_view="both", inpaint_model=None,
                                  return_mask=False, inconsistent_shift=False):
    fill = (method == "forward_fill")
    with torch.inference_mode():
        return depth_order_bilinear_forward_warp(c, depth, divergence, convergence,
                                                 fill=fill, synthetic_view=synthetic_view,
                                                 inpaint_model=inpaint_model,
                                                 return_mask=return_mask,
                                                 inconsistent_shift=inconsistent_shift)


def nonwarp_mask(c, depth, divergence, convergence):
    divergence = divergence * 0.5  # cancels out 2x multiplier for synthetic_view = right|left

    if c.shape[2] != depth.shape[2] or c.shape[3] != depth.shape[3]:
        depth = F.interpolate(depth, size=c.shape[-2:],
                              mode="bilinear", align_corners=True, antialias=True)

    # warp depth to the left
    depth3 = depth.repeat(1, 3, 1, 1)
    warped_depth, _ = depth_order_bilinear_forward_warp(
        depth3, depth, divergence, convergence,
        synthetic_view="left",
        fill=True, inconsistent_shift=False, return_mask=False)
    warped_depth = warped_depth.mean(dim=1, keepdim=True)
    # warp warped_depth to the right and back to original position
    dummy = torch.zeros_like(c)
    _, _, _, mask = depth_order_bilinear_forward_warp(
        dummy, warped_depth, divergence, convergence,
        fill=False, synthetic_view="right", inconsistent_shift=False,
        return_mask=True)

    return c, mask


def _bench():
    import time
    from nunif.modules.gaussian_filter import GaussianFilter2d

    synthetic_view = "both"  # both, right, left
    device = "cuda:0"
    B = 4
    N = 100
    S = (512, 512)    # 230 FPS on RTX3070Ti
    # S = (1080, 1920)  # HD, 22FPS and 600MB*batch_size VRAM

    rgb = torch.zeros((B, 3, *S)).to(device)
    # This test depth should call shift_fill and fix_layered_holes 5-10 times
    smooth = GaussianFilter2d(1, kernel_size=7, padding=1).to(device)
    depth = torch.zeros((B, 1, *S)).to(device)
    depth[:, :, 128:-128, 128:-128] = 1.0
    depth[:, :, :, 250:-250] = 0
    depth[:, :, :, 280:-230] = 0
    depth = smooth(depth)

    # TF.to_pil_image(depth[0]).show()
    divergence = 10.0
    convergence = 0.5

    # benchmark
    apply_divergence_forward_warp(rgb, depth, divergence, convergence,
                                  method="forward_fill", synthetic_view=synthetic_view)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(N):
        apply_divergence_forward_warp(rgb, depth, divergence, convergence,
                                      method="forward_fill", synthetic_view=synthetic_view)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


def _test_nonwarp_mask():
    # https://github.com/user-attachments/assets/69ea87ff-4f01-40d2-abd7-477bfe368df6
    import torchvision.transforms.functional as TF
    import torchvision.io as io

    x = io.read_image("cc0/320/dog.png") / 255.0
    depth = io.read_image("cc0/depth/dog.png") / 65536.0
    x = x.unsqueeze(0).cuda()
    depth = depth.unsqueeze(0).cuda()

    x, mask = nonwarp_mask(x, depth, divergence=4 * 2, convergence=0)
    mask = mask_closing(mask, kernel_size=3, n_iter=2)

    x = x.mean(dim=1, keepdim=True)
    x = torch.cat([x, mask, torch.zeros_like(mask)], dim=1)[0]
    TF.to_pil_image(x).show()


if __name__ == "__main__":
    _bench()
    _test_nonwarp_mask()
