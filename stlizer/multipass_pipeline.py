from os import path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import ImageDraw
import nunif.utils.video as VU
import nunif.utils.superpoint as KU
from nunif.models import load_model
from nunif.modules.gaussian_filter import get_gaussian_kernel1d
from nunif.modules.replication_pad2d import replication_pad1d_naive
from nunif.utils.ui import TorchHubDir
from . import models  # noqa
from tqdm import tqdm
import scipy


DEFAULT_RESOLUTION = 320


SUPERPOINT_CONF = {
    "nms_radius": 4,
    "max_num_keypoints": None,
    "detection_threshold": 0.01,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
}


ANGLE_MAX_HARD = 90.0
KEYPOINT_COSINE_THRESHOLD = 0.3

TORCH_HUB_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")
OUTPAINT_MODEL_URL = "https://github.com/nagadomi/nunif/releases/download/torchhub/stlizer_light_outpaint_v1_20241230.pth"


def resize(x, size):
    B, C, H, W = x.shape
    if H < W:
        new_height = size
        new_width = int(W / (H / size))
    else:
        new_width = size
        new_height = int(H / (W / size))

    height_scale = H / new_height
    width_scale = W / new_width
    x = F.interpolate(x, (new_height, new_width), mode="bilinear", align_corners=False, antialias=False)

    return x, (height_scale + width_scale) * 0.5


def plot_keypoints(x, kp):
    img = TF.to_pil_image(x.cpu())
    gc = ImageDraw.Draw(img)
    for xy in kp:
        xx, yy = int(xy[0].item()), int(xy[1].item())
        gc.circle((xx, yy), radius=2, fill="red")

    return img


def gen_savgol_kernel(kernel_size, device):
    kernel = scipy.signal.savgol_coeffs(kernel_size, polyorder=2)
    return torch.from_numpy(kernel).to(device).reshape(1, 1, -1)


def gen_smoothing_kernel(name, kernel_size, device):
    if name == "gaussian":
        return get_gaussian_kernel1d(kernel_size, dtype=torch.float64, device=device).reshape(1, 1, -1)
    elif name == "savgol":
        return gen_savgol_kernel(kernel_size, device)
    else:
        raise NotImplementedError(f"--filter {name}")


def smoothing(x, weight):
    kernel_size = weight.shape[2]
    padding = (kernel_size - 1) // 2
    x = replication_pad1d_naive(x, (padding, padding))
    x = F.conv1d(x, weight=weight, stride=1, padding=0)
    return x


def calc_scene_weight(mean_match_scores, device=None):
    # mean_match_scores: mean of best match keypoint's cosine similarity
    # when score < 0.5, it is highly likely that scene change has occurred
    # when score < 0.65, scene change possibly has occurred
    # when score > 0.75, it is probably safe range
    if torch.is_tensor(mean_match_scores):
        score = mean_match_scores
    else:
        score = torch.tensor(mean_match_scores, dtype=torch.float32, device=device)

    max_score = 0.75
    min_score = 0.5
    weight = ((score - min_score) / (max_score - min_score)).clamp(0, 1)
    weight[weight < 0.65] = weight[weight < 0.65] ** 2

    # Set start and end frames to 0
    weight[0] = 0.0
    weight[-1] = 0.0

    return weight


def make_video_codec_option(args):
    if args.video_codec in {"libx264", "libx265", "hevc_nvenc", "h264_nvenc"}:
        options = {"preset": args.preset, "crf": str(args.crf)}

        if args.tune:
            options["tune"] = ",".join(set(args.tune))

        if args.profile_level:
            options["level"] = str(int(float(args.profile_level) * 10))

        if args.video_codec == "libx265":
            x265_params = ["log-level=warning", "high-tier=enabled"]
            if args.profile_level:
                x265_params.append(f"level-idc={int(float(args.profile_level) * 10)}")
            options["x265-params"] = ":".join(x265_params)
        elif args.video_codec in {"hevc_nvenc", "h264_nvenc"}:
            options["rc"] = "constqp"
            options["qp"] = str(args.crf)
            if torch.cuda.is_available() and args.gpu >= 0:
                options["gpu"] = str(args.gpu)
    else:
        options = {}

    return options


def video_config_callback(args, fps_hook=None):
    def callback(stream):
        fps = VU.get_fps(stream)
        if float(fps) > args.max_fps:
            fps = args.max_fps
        if fps_hook is not None:
            fps_hook(fps)

        return VU.VideoOutputConfig(
            fps=fps,
            container_format=args.video_format,
            video_codec=args.video_codec,
            pix_fmt=args.pix_fmt,
            colorspace=args.colorspace,
            options=make_video_codec_option(args),
            container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        )
    return callback


def list_chunk(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def pass1(args):
    device = args.state["device"]

    keypoint_model = KU.SuperPoint(**SUPERPOINT_CONF).load().to(device)
    mean_match_scores = []
    points1 = []
    points2 = []
    kp_seam = [None]
    center = [[0, 0]]
    resize_scale = [1.0]

    def keypoint_callback(x, pts):
        with torch.inference_mode(), torch.autocast(device_type=device.type):
            x, resize_scale[0] = resize(x, args.resolution)
            center[0] = [x.shape[3] / 2, x.shape[2] / 2]
            kp_batch = keypoint_model.infer(x)

        kp_batch.insert(0, kp_seam[0])
        kp_seam[0] = kp_batch[-1]

        for i in range(1, len(kp_batch)):
            kp1 = kp_batch[i - 1]
            kp2 = kp_batch[i]
            if kp1 is None or (kp1["keypoints"].shape[0] == 0 or kp2["keypoints"].shape[0] == 0):
                zero_points = torch.zeros((0, 2), dtype=x.dtype, device=torch.device("cpu"))
                mean_match_scores.append(0.0)
                points1.append(zero_points)
                points2.append(zero_points)
                continue

            index1, index2, match_score = KU.find_match_index(
                kp1, kp2,
                threshold=KEYPOINT_COSINE_THRESHOLD, return_score_all=True)
            kp1 = kp1["keypoints"][index1]
            kp2 = kp2["keypoints"][index2]

            mean_match_scores.append(match_score.mean().item())
            points1.append(kp1.cpu())
            points2.append(kp2.cpu())

    fps_value = [0]

    def fps_hook(fps):
        fps_value[0] = fps

    keypoint_callback_pool = VU.FrameCallbackPool(
        keypoint_callback,
        require_pts=True,
        batch_size=args.batch_size,
        device=device,
        max_workers=0,  # must be sequential
    )

    VU.hook_frame(args.input, keypoint_callback_pool,
                  config_callback=video_config_callback(args, fps_hook),
                  vf=args.vf,
                  stop_event=args.state["stop_event"],
                  suspend_event=args.state["suspend_event"],
                  tqdm_fn=args.state["tqdm_fn"],
                  title="pass 1/4")

    return points1, points2, mean_match_scores, center[0], resize_scale[0], fps_value[0]


def pack_points(batch1, batch2):
    fixed_size = max(max(pts.shape[0] for pts in batch1), max(pts.shape[0] for pts in batch2))
    batch1_fixed = []
    batch2_fixed = []
    batch_mask = []
    for pts1, pts2 in zip(batch1, batch2):
        assert pts1.shape[0] == pts2.shape[0]
        pack1 = torch.zeros((fixed_size, pts1.shape[1]), dtype=pts1.dtype, device=pts1.device)
        pack2 = torch.zeros((fixed_size, pts1.shape[1]), dtype=pts1.dtype, device=pts1.device)
        mask = torch.zeros((fixed_size, pts1.shape[1]), dtype=torch.bool, device=pts1.device)

        pack1[:pts1.shape[0]] = pts1
        pack2[:pts2.shape[0]] = pts2
        mask[:pts1.shape[0]] = True

        batch1_fixed.append(pack1)
        batch2_fixed.append(pack2)
        batch_mask.append(mask)

    return torch.stack(batch1_fixed), torch.stack(batch2_fixed), torch.stack(batch_mask)


def pass2(points1, points2, center, resize_scale, args):
    device = args.state["device"]

    if len(points1) == 0:
        return []

    transforms = []

    points1, points2, masks = pack_points(points1, points2)
    batch_size = args.batch_size * 32

    pbar = tqdm(total=len(points1), ncols=80, desc="pass 2/4")
    for kp1, kp2, mask in zip(points1.split(batch_size), points2.split(batch_size), masks.split(batch_size)):
        kp1, kp2, mask = kp1.to(device), kp2.to(device), mask.to(device)
        center_batch = torch.tensor(center, dtype=torch.float32, device=device).view(1, 2).expand(kp1.shape[0], 1, 2)
        shift, scale, angle, center_batch = KU.find_transform(
            kp1, kp2, center=center_batch, mask=mask,
            iteration=args.iteration, sigma=2.0,
            disable_scale=True)
        for i in range(kp1.shape[0]):
            transforms.append((shift[i].tolist(), scale[i].item(), angle[i].item(), center, resize_scale))
            pbar.update(1)
    pbar.close()
    return transforms


def conv1d_smoothing(shift_x, shift_y, angle, method, smoothing_seconds, fps, device):
    shift_x_smooth = shift_x
    shift_y_smooth = shift_y
    angle_smooth = angle
    kernel_sec = smoothing_seconds
    kernel_size = int(kernel_sec * float(fps))
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    kernel = gen_smoothing_kernel(name=method, kernel_size=kernel_size, device=device)
    shift_x_smooth = smoothing(shift_x_smooth, kernel)
    shift_y_smooth = smoothing(shift_y_smooth, kernel)
    angle_smooth = smoothing(angle_smooth, kernel)

    shift_x_fix = (shift_x_smooth - shift_x).flatten()
    shift_y_fix = (shift_y_smooth - shift_y).flatten()
    angle_fix = (angle_smooth - angle).flatten()

    return shift_x_fix, shift_y_fix, angle_fix


def grad_opt(tx, ty, ta, scene_weight, resolution, iteration=100, penalty_weight=1e-3):
    """
    The basic idea is from: "Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths"
    But not L1 or Exact solution.
    """
    resolution_weight = resolution / DEFAULT_RESOLUTION

    tx = replication_pad1d_naive(tx, (0, 3)).flatten() * resolution_weight
    ty = replication_pad1d_naive(ty, (0, 3)).flatten() * resolution_weight
    ta = replication_pad1d_naive(ta, (0, 3)).flatten()
    sw = F.pad(scene_weight, (0, 3), mode="constant", value=0).flatten()

    px = tx.clone().requires_grad_(True)
    py = ty.clone().requires_grad_(True)
    pa = ta.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS([px, py, pa], history_size=10, max_iter=4)
    grad_weight = 1.0 / 9.0  # 3grad * 3axis

    def f():
        optimizer.zero_grad()
        loss = 0.0
        for x, t in zip((px, py, pa), (tx, ty, ta)):
            fx1 = x[1:] - x[:-1]
            fx2 = fx1[1:] - fx1[:-1]
            fx3 = fx2[1:] - fx2[:-1]
            grad_loss = (fx1.pow(2).mul(sw[:fx1.shape[0]]).mean() +
                         fx2.pow(2).mul(sw[:fx2.shape[0]]).mean() +
                         fx3.pow(2).mul(sw[:fx3.shape[0]]).mean())
            penalty = (x - t).pow(2).mean()
            loss = loss + grad_loss * grad_weight + penalty * penalty_weight
        # print(i, loss.item())
        loss.backward()
        return loss

    for i in tqdm(range(iteration), ncols=80, desc="pass 3/4"):
        optimizer.step(f)

    px = (px[:-3].detach() - tx[:-3]) / resolution_weight
    py = (py[:-3].detach() - ty[:-3]) / resolution_weight
    pa = (pa[:-3].detach() - ta[:-3])

    return px, py, pa


def pass3_smoothing(shift_x, shift_y, angle, scene_weight, method, smoothing_seconds, fps, resolution, device):
    shift_x = shift_x * scene_weight  # + 0 * (1 - scene_weight)
    shift_y = shift_y * scene_weight
    angle = angle * scene_weight

    shift_x = shift_x.cumsum(dim=0).reshape(1, 1, -1)
    shift_y = shift_y.cumsum(dim=0).reshape(1, 1, -1)
    angle = angle.cumsum(dim=0).reshape(1, 1, -1)

    if method in {"gaussian", "savgol"}:
        return conv1d_smoothing(shift_x, shift_y, angle, method, smoothing_seconds, fps, device)
    elif method == "grad_opt":
        return grad_opt(shift_x, shift_y, angle, scene_weight, resolution, penalty_weight=2e-3 / smoothing_seconds)


def pass3(transforms, scene_weight, fps, args):
    device = args.state["device"]

    shift_x = torch.tensor([rec[0][0] for rec in transforms], dtype=torch.float64, device=device)
    shift_y = torch.tensor([rec[0][1] for rec in transforms], dtype=torch.float64, device=device)
    angle = torch.tensor([rec[2] for rec in transforms], dtype=torch.float64, device=device)
    # limit angle
    angle = angle.clamp(-ANGLE_MAX_HARD, ANGLE_MAX_HARD)

    shift_x_fix, shift_y_fix, angle_fix = pass3_smoothing(
        shift_x, shift_y, angle, scene_weight, resolution=args.resolution,
        method=args.filter, smoothing_seconds=args.smoothing, fps=fps,
        device=device)

    return shift_x_fix, shift_y_fix, angle_fix


def outpaint(x, mask, model, device, composite):
    with torch.inference_mode(), torch.autocast(device_type=device.type):
        return model.infer(x, mask, max_size=640, composite=composite)


def pass4(output_path, shift_x_fix, shift_y_fix, angle_fix, transforms, scene_weight, fps, args):
    device = args.state["device"]
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)

    if args.border in {"outpaint", "expand_outpaint"}:
        with TorchHubDir(TORCH_HUB_DIR):
            outpaint_model, _ = load_model(OUTPAINT_MODEL_URL, device_ids=[-1])
        outpaint_model = outpaint_model.eval().to(device)
    else:
        outpaint_model = None

    def test_callback(frame):
        if frame is None:
            return None
        x = VU.to_tensor(frame, device=device)

        if args.border in {"expand", "expand_outpaint"}:
            padding = int(max(x.shape[1], x.shape[2]) * args.padding)
            x = F.pad(x, (padding,) * 4, mode="constant", value=0)
        elif args.border == "crop":
            padding = int(max(x.shape[1], x.shape[2]) * args.padding)
            x = F.pad(x, (-padding,) * 4)

        if args.debug:
            z = torch.cat([x, x], dim=2)
            return VU.to_frame(z, use_16bit=use_16bit)
        else:
            return VU.to_frame(x, use_16bit=use_16bit)

    index = [0]
    buffer = [None]

    def stabilizer_callback(x):
        B = x.shape[0]
        i = index[0]
        index[0] += x.shape[0]

        # assume all values are the same
        center = transforms[i][3]
        resize_scale = transforms[i][4]
        center = [center[0] * resize_scale, center[1] * resize_scale]

        if args.border == "black":
            padding = 0
            x_input = x
            padding_mode = "zeros"
        elif args.border in {"outpaint", "expand_outpaint"}:
            padding = int(max(x.shape[2], x.shape[3]) * args.padding)
            x_input = F.pad(x, (padding,) * 4, mode="constant", value=torch.nan)
            center = [center[0] + padding, center[1] + padding]
            padding_mode = "border"
        elif args.border == "crop":
            padding = 0
            x_input = x
            padding_mode = "zeros"
        elif args.border == "expand":
            padding = int(max(x.shape[2], x.shape[3]) * args.padding)
            x_input = F.pad(x, (padding,) * 4, mode="constant", value=0)
            center = [center[0] + padding, center[1] + padding]
            padding_mode = "zeros"
        else:
            raise ValueError(f"Unknown --border mode {args.border}")

        shifts = torch.tensor([[shift_x_fix[i + j].item() * resize_scale,
                                shift_y_fix[i + j].item() * resize_scale] for j in range(B)],
                              dtype=x.dtype, device=x.device)
        centers = torch.tensor([center for _ in range(B)], dtype=x.dtype, device=x.device)
        angles = torch.tensor([angle_fix[i + j] for j in range(B)],
                              dtype=x.dtype, device=x.device)
        scales = torch.ones((B,), dtype=x.dtype, device=x.device)

        z = KU.apply_transform(x_input, shifts, scales, angles, centers, padding_mode=padding_mode)

        if args.border in {"outpaint", "expand_outpaint"}:
            if args.border == "outpaint":
                z = F.pad(z, (-padding,) * 4)
            else:
                z = z.clone()
            masks = torch.isnan(z)
            z[masks] = 0

            if args.buffer_decay > 0.0:
                buffer_decay = (1.0 - args.buffer_decay) * (29.97 / float(fps))
                buffer_decay = min(max(0.5, buffer_decay), 1.0)
                buffer_decay = 1.0 - buffer_decay

                coarse_view = outpaint(z, masks[:, 0:1, :, :], outpaint_model, device, composite=False)
                z = z.clone()
                # Update EMA frame buffer
                for j in range(z.shape[0]):
                    if buffer[0] is None or scene_weight[i + j] < 0.01:
                        # reset buffer
                        buffer[0] = coarse_view[j].clone()
                    mask = masks[j]
                    buffer[0].mul_(buffer_decay)
                    buffer[0].add_(coarse_view[j], alpha=(1.0 - buffer_decay))
                    z[j][mask] = buffer[0][mask]
                z.clamp_(0, 1)
            else:
                z = outpaint(z, masks[:, 0:1, :, :], outpaint_model, device, composite=True)

        elif args.border == "crop":
            padding = int(max(z.shape[2], z.shape[3]) * args.padding)
            z = F.pad(z, (-padding,) * 4).clamp_(0, 1)
        else:
            z = z.clamp(0, 1)

        if args.debug:
            if args.border in {"expand", "expand_outpaint"}:
                x = x_input
            elif args.border == "crop":
                x = F.pad(x_input, (-padding,) * 4)
            z = torch.cat([x, z], dim=3)

        return z

    stabilizer_callback_pool = VU.FrameCallbackPool(
        stabilizer_callback,
        batch_size=args.batch_size,
        device=device,
        max_workers=0,
        use_16bit=use_16bit,
    )
    VU.process_video(args.input, output_path,
                     stabilizer_callback_pool,
                     config_callback=video_config_callback(args),
                     test_callback=test_callback,
                     vf=args.vf,
                     stop_event=args.state["stop_event"],
                     suspend_event=args.state["suspend_event"],
                     tqdm_fn=args.state["tqdm_fn"],
                     title="pass 4/4")
