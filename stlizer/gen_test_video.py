# test video generator
# python -m playground.video_stabilizer.gen_test_video -i ./cc0/bottle_small.png -o ./tmp/bottle_vidstab.mp4
#
# sample: https://github.com/user-attachments/assets/98e13148-5645-4873-865f-c00db077a0ce
# NOTE: There are no individual movements for each object.

import torch
import argparse
import torch.nn.functional as F
import torchvision.io as IO
import nunif.utils.video as VU
import nunif.utils.superpoint as KU
from nunif.modules.gaussian_filter import get_gaussian_kernel1d


def load_frame(path):
    x = (IO.read_image(path) / 255.0)
    pad_y = x.shape[1] % 8
    pad_x = x.shape[2] % 8

    if pad_x > 0 or pad_y > 0:
        x = F.pad(x, (0, -pad_x, 0, -pad_y))

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="input image path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--noise-scale", type=float, default=10.0)
    parser.add_argument("--disable-shift", action="store_true")
    parser.add_argument("--disable-rotate", action="store_true")

    args = parser.parse_args()
    shift_weight = 0.0 if args.disable_shift else 1.0
    rotate_weight = 0.0 if args.disable_rotate else 1.0

    frames = [load_frame(src) for src in args.input]
    assert all([frame.shape[1] == frames[0].shape[1] and frame.shape[2] == frames[0].shape[2] for frame in frames])

    BASE_FRAMES = 30 * 5
    FRAMES = BASE_FRAMES * len(frames)
    config = VU.VideoOutputConfig(
        fps=30,
        options={"preset": "medium", "crf": "20"},
        output_width=frames[0].shape[2],
        output_height=frames[0].shape[1],
    )
    noise_x1 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.8 * shift_weight
    noise_x2 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.2 * shift_weight
    noise_y1 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.8 * shift_weight
    noise_y2 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.2 * shift_weight
    noise_r1 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.05 * 0.8 * rotate_weight
    noise_r2 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.05 * 0.2 * rotate_weight

    gaussian_kernel3 = get_gaussian_kernel1d(3, device="cpu").reshape(1, 1, -1)
    gaussian_kernel15 = get_gaussian_kernel1d(15, device="cpu").reshape(1, 1, -1)
    gaussian_kernel3 = F.pad(gaussian_kernel3, (6, 6, 0, 0))
    noise_x = (F.conv1d(noise_x1, weight=gaussian_kernel3) + F.conv1d(noise_x2, weight=gaussian_kernel15)).flatten()
    noise_y = (F.conv1d(noise_y1, weight=gaussian_kernel3) + F.conv1d(noise_y2, weight=gaussian_kernel15)).flatten()
    noise_r = (F.conv1d(noise_r1, weight=gaussian_kernel3) + F.conv1d(noise_r2, weight=gaussian_kernel15)).flatten()

    def frame_generator():
        for i, (x, y, r) in enumerate(zip(noise_x, noise_y, noise_r)):
            x = x.item()
            y = y.item()
            r = r.item()

            frame = frames[i // BASE_FRAMES]
            new_frame = KU.apply_transform(
                frame, shift=[x, y], scale=1.0, angle=r,
                center=[frame.shape[2] // 2, frame.shape[1] // 2]
                # when the camera is held in the right hand
                # center=[frame.shape[2] - 1, frame.shape[1] // 2]
            )
            yield VU.to_frame(new_frame)

    VU.generate_video(
        args.output,
        frame_generator,
        config)
