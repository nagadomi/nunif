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


def load_frame(path):
    x = (IO.read_image(path) / 255.0)
    pad_y = x.shape[1] % 8
    pad_x = x.shape[2] % 8

    if pad_x > 0 or pad_y > 0:
        x = F.pad(x, (0, -pad_x, 0, -pad_y))

    return x


def gen_gaussian_kernel(kernel_size, device):
    sigma = kernel_size * 0.15 + 0.35
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=torch.float32, device=device)
    gaussian_kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    return gaussian_kernel.reshape(1, 1, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="input image path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--noise-scale", type=float, default=10.0)
    args = parser.parse_args()

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
    noise_x1 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.8
    noise_x2 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.2
    noise_y1 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.8
    noise_y2 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.2
    noise_r1 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.1 * 0.4
    noise_r2 = torch.randn((1, 1, FRAMES)) * args.noise_scale * 0.1 * 0.1

    gaussian_kernel3 = gen_gaussian_kernel(3, "cpu")
    gaussian_kernel15 = gen_gaussian_kernel(15, "cpu")
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
            new_frame = KU.apply_rigid_transform(
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
