import os
import argparse
import torch
import torch.nn.functional as F
from os import path
import random
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from multiprocessing import cpu_count
import hashlib
from torchvision import transforms as T
from torchvision.transforms import (
    functional as TF,
    InterpolationMode
)
import nunif.utils.video as VU
from iw3.utils import get_mapper
from iw3.stereo_model_factory import create_stereo_model
from iw3.dilation import mask_closing
from iw3.backward_warp import nonwarp_mask as backward_nonwarp_mask
from iw3.forward_warp import nonwarp_mask as forward_nonwarp_mask
from .create_training_data import (
    gen_divergence,
    gen_convergence,
    gen_mapper,
    gen_edge_dilation,
)


def md5(s, nbytes=8):
    MD5_SALT = "iw3"
    return hashlib.md5((s + MD5_SALT).encode()).hexdigest()[:nbytes]


def crop_resize(frames, min_size):
    if random.choice([True, False]):
        crop_size = int(min(frames[0][0].shape[-2:]) * (1 / 2 ** 0.5))
        angle = random.uniform(-45, 45)
        new_frames = []
        for frame in frames:
            frame = TF.rotate(frame, angle=angle, interpolation=InterpolationMode.BILINEAR)
            frame = TF.center_crop(frame, (crop_size, crop_size))
            new_frames.append(frame)
        frames = torch.stack(new_frames)

    h, w = frames[0].shape[-2:]
    if min(h, w) != min_size:
        if w > h:
            new_h = min_size
            new_w = int(w * (new_h / h))
        else:
            new_w = min_size
            new_h = int(h * (new_w / w))

        mode = random.choice(["bicubic", "bilinear"])
        frames = F.interpolate(frames, size=(new_h, new_w), mode=mode, align_corners=False, antialias=True)

    return frames


def gen_data(frames, depth_model, mask_mlbw, args):
    mapper = gen_mapper(depth_model.is_metric())
    convergence = gen_convergence()
    edge_dilation = gen_edge_dilation(args.model_type, args.resolution)
    image_size = random.choices([720, 1080], weights=[0.25, 0.5], k=1)[0]
    forward_base_view = random.choice(["right", "left"])
    # TODO: Maybe need to adjust this later.

    width, height = frames[0].size
    frames = torch.stack([TF.to_tensor(frame) for frame in frames])

    with torch.inference_mode():
        frames = crop_resize(frames, image_size).to(depth_model.device)
        divergence = gen_divergence(width, args.divergence_level)
        depth_aa = random.choice([False, False, False, True])
        depths = []
        for c in frames.split(args.batch_size, dim=0):
            depths.append(depth_model.infer(c, edge_dilation=edge_dilation, depth_aa=depth_aa, tta=False, enable_amp=True))
        depths = torch.cat(depths, dim=0)
        depths = torch.stack(depth_model.minmax_normalize(depths))

        c1 = []
        m1 = []
        c2 = []
        m2 = []
        for c, depth in zip(frames.split(args.batch_size, dim=0), depths.split(args.batch_size, dim=0)):
            _, mask = backward_nonwarp_mask(
                mask_mlbw,
                c, depth,
                divergence=divergence,
                convergence=convergence,
                mapper=mapper,
                threshold=0.15,  # constant value
            )

            c_flip = torch.flip(c, (-1,))
            depth_flip = torch.flip(depth, (-1,))
            depth_flip = get_mapper(mapper)(depth_flip)

            _, mask_flip = forward_nonwarp_mask(
                c_flip, depth_flip,
                divergence=divergence,
                convergence=convergence,
                view=forward_base_view,
            )
            mask_flip = mask_closing(mask_flip)

            c1.append(c)
            m1.append(mask.float())
            c2.append(c_flip)
            m2.append(mask_flip.float())

        c1 = torch.cat(c1, dim=0)
        m1 = torch.cat(m1, dim=0)
        c2 = torch.cat(c2, dim=0)
        m2 = torch.cat(m2, dim=0)
        return ((c1, m1), (c2, m2))


def random_crop(size, *images):
    i, j, h, w = T.RandomCrop.get_params(images[0][0], (size, size))
    results = []
    for im in images:
        results.append(im[:, :, i:i + h, j:j + w])

    return tuple(results)


def random_hard_example_crop(size, n, *images):
    assert n > 0
    results = []
    for _ in range(n):
        crop_images = random_crop(size, *images)
        mask = crop_images[-1]
        mask_sum = mask.float().sum().item()
        results.append((mask_sum, crop_images))

    results = sorted(results, key=lambda v: v[0], reverse=True)
    return results[0]


def save_frames(c, mask, base_output_dir, size, num_samples):
    for i in range(num_samples):
        mask_sum, (rgb_frames, mask_frames) = random_hard_example_crop(size, 4, c, mask)
        if mask_sum / mask.shape[0] > 300:
            output_dir = base_output_dir + f"_{i}"
            os.makedirs(output_dir, exist_ok=True)
            seq_no = 0
            rgb_frames = (rgb_frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            mask_frames = (mask_frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            for rgb_rect, mask_rect in zip(rgb_frames, mask_frames):
                TF.to_pil_image(rgb_rect).save(path.join(output_dir, str(seq_no).zfill(4) + "_C.png"))
                TF.to_pil_image(mask_rect).save(path.join(output_dir, str(seq_no).zfill(4) + "_M.png"))
                seq_no += 1


def main(args):
    from ...depth_model_factory import create_depth_model

    assert args.batch_size <= args.seq
    assert args.seq % args.batch_size == 0

    max_workers = cpu_count() // 2 or 1
    filename_prefix = f"{args.prefix}_{args.model_type}_" if args.prefix else args.model_type + "_"
    filename_prefix = filename_prefix + md5(path.basename(args.dataset_dir)) + "_"
    depth_model = create_depth_model(args.model_type)
    depth_model.load(gpu=args.gpu, resolution=args.resolution)
    depth_model.disable_ema()

    # TODO: support for divergence handles when d2 and d3 models become available in the future
    mask_mlbw = create_stereo_model("mask_mlbw_l2", divergence=1, device_id=args.gpu)

    video_file = args.dataset_dir

    def config_callback(stream):
        return VU.VideoOutputConfig(fps=args.fps)

    with PoolExecutor(max_workers=max_workers) as pool:
        futures = []
        seq_no = 0
        skip_counter = args.skip_first
        frames = []

        def frame_callback(frame):
            nonlocal seq_no, skip_counter
            if frame is None:
                return

            if skip_counter > 0:
                skip_counter -= 1
                return

            frames.append(frame.to_image())
            if len(frames) < args.seq:
                return

            data = gen_data(frames, depth_model, mask_mlbw, args)
            for c, mask in data:
                seq_no += 1
                output_dir = path.join(args.data_dir, filename_prefix + str(seq_no))
                f = pool.submit(save_frames, c, mask, output_dir, args.size, args.num_samples)
                futures.append(f)

            if len(futures) > 10:
                for f in futures:
                    f.result()

            frames.clear()
            skip_counter = args.skip_interval

        VU.hook_frame(video_file, frame_callback, config_callback=config_callback)


def register(subparsers, default_parser):
    parser = subparsers.add_parser(
        "video_inpaint",
        parents=[default_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--divergence-level", type=int, default=1, choices=[1, 2, 3],
                        help="divergence level. 1=0-5, 2=3-8, 3=6-11")
    parser.add_argument("--prefix", type=str, default="", help="prefix for output filename")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID. -1 for cpu")
    parser.add_argument("--size", type=int, default=512, help="crop size")
    parser.add_argument("--seq", type=int, default=16, help="frame sequence")
    parser.add_argument("--fps", type=int, default=30, help="sample fps")
    parser.add_argument("--num-samples", type=int, default=1, help="max random crops")
    parser.add_argument("--resolution", type=int, help="input resolution for depth model")
    parser.add_argument("--model-type", type=str, default="ZoeD_Any_L", help="depth model")
    parser.add_argument("--skip-interval", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-first", type=int, default=0)

    parser.set_defaults(handler=main)

    return parser
