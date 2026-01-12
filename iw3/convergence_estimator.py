import torch
from nunif.utils.ui import TorchHubDir
from nunif.models import load_model
from nunif.device import create_device, autocast
from .hub_dir import HUB_MODEL_DIR


SOD_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_sod_v1_20260109.pth"


class ConvergenceEstimator():
    def __init__(self, convergence, device_id, enable_ema=False, decay=0.9, compile=False):
        with TorchHubDir(HUB_MODEL_DIR):
            self.model, _ = load_model(SOD_URL, device_ids=[device_id], weights_only=True)
            self.model = self.model.eval().fuse()
            # SOD input is resized to 192×192, recompilation does not occur due to image size differences.
            # However, recompilation does occur when the batch size changes.
            self.model = self.model.compile(mode=compile)
            self.convergence = convergence

        self.device = create_device(device_id)
        self.enable_ema = enable_ema
        self.decay = decay
        self.convergence_ema = None

    def reset(self, enable_ema=None, decay=None):
        if enable_ema is not None:
            self.enable_ema = enable_ema
        if decay is not None:
            self.decay = decay
        self.convergence_ema = None

    @staticmethod
    def depth_position_from_ratio(saliency_map, depth, pos):
        B = depth.shape[0]
        result = []
        for i in range(B):
            d = depth[i].flatten()
            mask = saliency_map[i].flatten() > 0.5
            d = d[mask]
            if d.numel() == 0:
                # Depth values are normalized to the [0, 1] range,
                # with 0.5 corresponding to the mid depth.
                result.append(torch.tensor(0.5, device=saliency_map.device, dtype=saliency_map.dtype))
                continue

            q01 = d.quantile(0.1)
            q09 = d.quantile(0.9)
            q_range = (q09 - q01)
            if q_range < 1e-6:
                q_pos = q01
            else:
                # Convert pos from [0, 1] into an internal range of [-1, 2],
                # effectively 3x the usable depth range around the central region.
                center = (q01 + q09) / 2
                expanded_range = q_range * 3.0
                q_pos = (center + (pos - 0.5) * expanded_range)
            result.append(q_pos)
        return torch.stack(result, dim=0).reshape(B, 1, 1, 1).clamp(0, 1)

    def __call__(self, rgb, depth, reset_pts=None):
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)

        with torch.inference_mode(), autocast(self.device):
            saliency_map, depth_scaled = self.model.infer(rgb, depth, self.convergence)
            z_pos = self.depth_position_from_ratio(saliency_map, depth_scaled, self.convergence)

        if self.enable_ema:
            reset_pts = reset_pts if reset_pts is not None else [False] * depth.shape[0]
            results = []
            for i in range(z_pos.shape[0]):
                p = z_pos[i]
                if self.convergence_ema is None:
                    self.convergence_ema = p.clone()
                else:
                    self.convergence_ema = self.decay * self.convergence_ema + (1. - self.decay) * p
                results.append(self.convergence_ema.clone())
                if reset_pts[i]:
                    self.reset()

            z_pos = torch.stack(results, dim=0)

        return z_pos
