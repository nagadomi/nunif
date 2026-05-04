# This is a non-ML, heuristic-based method that generates stereo images from depth
# with results very similar to iw3's row_flow_v3, but faster.
# This algorithm describes two things that row_flow_v3 acquires during training:
#   - Approximating forward warping through backward warping.
#   - Enforcing monotonicity constraints on the grid.
#     (Prevent folding artifacts and overly sharp edges)
import torch
import torch.nn as nn
import torch.nn.functional as F

from nunif.models import I2IBaseModel, register_model
from nunif.modules.gaussian_filter import GaussianFilter2d


@register_model
class MonoBW(I2IBaseModel):
    name = "sbs.monobw"
    smooth_filter: None | nn.Module

    def __init__(self, smooth_kernel=9) -> None:
        super(MonoBW, self).__init__(locals(), scale=1, offset=0, in_channels=3, blend_size=0)
        if smooth_kernel > 0:
            pad = (smooth_kernel - 1) // 2
            self.smooth_filter = GaussianFilter2d(
                in_channels=1,
                kernel_size=(1, smooth_kernel),
                padding=(pad, pad, 0, 0),
            )
        else:
            self.smooth_filter = None

    def smoothing(self, dest_index_fix: torch.Tensor, dest_index: torch.Tensor) -> torch.Tensor:
        if self.smooth_filter is None:
            return dest_index_fix

        dest_index_mask = dest_index != dest_index_fix
        dest_index_mask = F.max_pool2d(dest_index_mask.float(), kernel_size=(1, 5), stride=1, padding=(0, 2)) > 0
        dest_index_fix[dest_index_mask] = self.smooth_filter(dest_index_fix)[dest_index_mask]

        return dest_index_fix

    def get_device(self) -> torch.device:
        if self.smooth_filter is not None:
            device = self.smooth_filter.kernel.device
            assert isinstance(device, torch.device)
            return device

        raise NotImplementedError()

    @staticmethod
    def interpolate_1d(dest_index: torch.Tensor, src_index: torch.Tensor) -> torch.Tensor:
        # 1D interpolation for monotone mapping
        dest_index = dest_index.contiguous()
        src_index = src_index.contiguous()
        BH, W = dest_index.shape

        idx = torch.searchsorted(dest_index, src_index, right=False)
        idx0 = (idx - 1).clamp(0, W - 1)
        idx1 = idx.clamp(0, W - 1)

        d0 = torch.gather(dest_index, 1, idx0)
        d1 = torch.gather(dest_index, 1, idx1)
        s0 = torch.gather(src_index, 1, idx0)
        s1 = torch.gather(src_index, 1, idx1)

        denom = d1 - d0
        t = (src_index - d0) / (denom + 1e-5)
        interp_src = s0 + t * (s1 - s0)

        return interp_src

    def compute_backward_grid(
        self,
        depth: torch.Tensor,
        divergence: float | torch.Tensor,
        convergence: float | torch.Tensor,
        border_pix=0,
    ):
        B, _, H, W = depth.shape
        dtype = depth.dtype
        device = depth.device

        if isinstance(convergence, float):
            convergence = torch.tensor(convergence, dtype=dtype, device=device).expand(B, 1).view(B, 1, 1, 1)
        else:
            if convergence.ndim != 4:
                convergence = convergence.view(B, 1, 1, 1)

        # Base indices for low-res grid
        src_index = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)

        # Calculate pixel shift
        base_size = max(H, W)
        delta_scale = base_size / W
        shift_size_px = divergence * (0.01 * delta_scale * (W - 1) * 0.5)
        index_shift = (depth - convergence) * shift_size_px

        if border_pix > 0:
            view_shape = [1] * (index_shift.ndim - 1) + [-1]
            border_weight_l = torch.linspace(0.0, 1.0, border_pix, dtype=dtype, device=device).view(view_shape)
            border_weight_r = torch.linspace(1.0, 0.0, border_pix, dtype=dtype, device=device).view(view_shape)
            index_shift[..., :border_pix] *= border_weight_l
            index_shift[..., -border_pix:] *= border_weight_r

        # Destination mapping and Monotonization
        dest_index = src_index + index_shift
        dest_index_fix = torch.cummax(dest_index, dim=-1)[0]

        # Optional smoothing
        dest_index = self.smoothing(dest_index_fix, dest_index)

        # Inverse mapping
        src_index_flat = src_index.reshape(B * H, W)
        dest_index_flat = dest_index.view(B * H, W)
        index_back = self.interpolate_1d(dest_index_flat, src_index_flat)
        index_back = index_back.reshape(B, 1, H, W)

        # Final sampling grid construction
        grid_x = (index_back / (W - 1)) * 2.0 - 1.0
        mesh_y = torch.linspace(-1, 1, H, device=depth.device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        grid_bchw = torch.cat([grid_x, mesh_y], dim=1)

        return grid_bchw

    def compute_stretch_mask(
        self,
        grid: torch.Tensor,
        threshold: float = 0.5,
    ):
        W = grid.shape[-1]
        grid_x = grid[:, 0:1]
        diff_x = grid_x[..., 1:] - grid_x[..., :-1]
        threshold = (2.0 / (W - 1)) * threshold
        is_stretched = diff_x < threshold
        mask = torch.zeros_like(grid_x, dtype=torch.bool, device=grid.device)
        mask[..., :-1] |= is_stretched
        mask[..., 1:] |= is_stretched
        return mask

    def warp(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
    ):
        if grid.shape[-2] != x.shape[-2:]:
            grid = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=True)
        grid = grid.permute(0, 2, 3, 1).to(x.dtype)
        warped_x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return warped_x

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        divergence: float,
        convergence: float | torch.Tensor,
        preserve_screen_border: bool = False,
        fix_screen_border_mask: int = 1,  # 0: No fix, 1: Fix the uninpaintable side, 2: Fix both sides
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert fix_screen_border_mask in {0, 1, 2}
        if preserve_screen_border:
            image_width = rgb.shape[-1]
            depth_width = depth.shape[-1]
            border_pix = round(divergence * 0.75 * 0.01 * image_width * (depth_width / image_width))
        else:
            border_pix = 0

        # Disable autocast to reduce precision loss in coordinate calculations
        with torch.autocast(device_type=depth.device.type, enabled=False):
            # Compute grid at low resolution
            grid_bchw = self.compute_backward_grid(depth, divergence, convergence, border_pix=border_pix)
            # Warp at high resolution
            if grid_bchw.shape[-2] != rgb.shape[-2:]:
                grid_bchw = F.interpolate(grid_bchw, size=rgb.shape[-2:], mode="bilinear", align_corners=True)
            warped_rgb = self.warp(rgb, grid_bchw)

            mask: None | torch.Tensor
            if return_mask:
                mask = self.compute_stretch_mask(grid_bchw)
                if not preserve_screen_border and fix_screen_border_mask > 0:
                    # Force disable the mask caused by screen border stretching.
                    image_width = rgb.shape[-1]
                    depth_width = depth.shape[-1]
                    border_pix = round(divergence * 0.01 * image_width * (depth_width / image_width)) + 1
                    mask[..., :border_pix] = False
                    if fix_screen_border_mask == 2:
                        mask[..., -border_pix:] = False
            else:
                mask = None

        if return_mask:
            assert mask is not None
            return warped_rgb, mask

        return warped_rgb


def _bench(name, preserve_screen_border=False, do_compile=False):
    import time

    from nunif.models import create_model

    device = "cuda:0"
    B = 4
    N = 100

    model = create_model(name).to(device).eval()
    if do_compile:
        model = torch.compile(model)
    rgb = torch.zeros((B, 3, 1080, 1920)).to(device)
    depth = torch.zeros((B, 1, 518, 910)).to(device)
    convergence = torch.tensor([0.5]).expand(B, 1).to(device).clone()

    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z = model(rgb, depth, divergence=2.0, convergence=convergence, preserve_screen_border=preserve_screen_border)
        params = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{params}", z.shape)

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            model(rgb, depth, divergence=2.0, convergence=convergence, preserve_screen_border=preserve_screen_border)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")


def _test_mask():
    import time

    import torchvision.io as io
    import torchvision.transforms.functional as TF

    model = MonoBW().cuda()
    depth = io.read_image("cc0/depth/dog.png") / 65535.0
    depth = depth.cuda().unsqueeze(0)
    depth_low_res = F.interpolate(
        depth, size=(depth.shape[-2] // 2, depth.shape[-1] // 2), mode="bilinear", align_corners=False
    )
    warped_depth, mask = model(depth, depth_low_res, divergence=5.0, convergence=0.0, return_mask=True)

    rgb = torch.cat((mask.float() * 0.5, warped_depth, warped_depth), dim=1)
    TF.to_pil_image(rgb[0]).show()
    time.sleep(1)

    warped_depth, mask = model(depth, depth_low_res, divergence=5.0, convergence=1.0, return_mask=True)
    rgb = torch.cat((mask.float() * 0.5, warped_depth, warped_depth), dim=1)
    TF.to_pil_image(rgb[0]).show()


if __name__ == "__main__":
    # at FHD (1920x1080)
    # 1800 FPS on RTX3070Ti
    _bench("sbs.monobw", do_compile=False)
    # 2950 FPS on RTX3070Ti
    _bench("sbs.monobw", do_compile=True)

    # _test_mask()
