# Roughly, deinterlacing and upscaling/super-resolution are the same task.
# 1. Separate the frame into odd and even scanlines
# 2. 2x upscale only height (Note that either line is shifted by 1px)
# 3. Set the frame order or drop one of the frames
import torch
import os
from os import path
import argparse
import nunif.utils.video as VU
import torch.nn.functional as F


def interleave(a, b):
    B, C, H, W = a.shape
    return torch.stack((a, b)).transpose(0, 1).reshape(B * 2, C, H, W).contiguous()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--bottom-first", action="store_true", help="bottom field first")
    parser.add_argument("--double-framerate", action="store_true", help="double framerate")
    args = parser.parse_args()
    assert path.normpath(args.input) != path.normpath(args.output)
    os.makedirs(path.dirname(args.output), exist_ok=True)

    def deinterlace(frames):
        if frames is None:
            return None
        B, C, H, W = frames.shape
        assert H % 2 == 0
        top = frames[:, :, ::2, :]
        bottom = frames[:, :, 1::2, :]

        if not args.double_framerate:
            # use only top field
            frames = F.interpolate(top, size=(H * 2, W), mode="bilinear")
            return frames
        else:
            top = F.interpolate(top, size=(H * 2, W), mode="bilinear")
            bottom = F.interpolate(bottom, size=(H * 2, W), mode="bilinear")
            bottom = torch.cat((bottom[:, :, 0:1, :], bottom[:, :, :-1, :]), dim=2)
            if args.bottom_first:
                return interleave(top, bottom)
            else:
                return interleave(bottom, top)

    def config_callback(stream):
        fps = VU.get_fps(stream)
        if args.double_framerate:
            output_fps = fps * 2
        else:
            output_fps = None
        return VU.VideoOutputConfig(
            fps=fps, output_fps=output_fps,
        )

    def test_callback(frame):
        if frame is None:
            return None
        return frame

    deinterlace_callback_pool = VU.FrameCallbackPool(
        deinterlace,
        batch_size=4,
        device="cpu",
        max_workers=4,
    )
    VU.process_video(args.input, args.output,
                     deinterlace_callback_pool,
                     config_callback=config_callback,
                     test_callback=test_callback,
                     title="SuperSimpleDeinterlace")


if __name__ == "__main__":
    main()
