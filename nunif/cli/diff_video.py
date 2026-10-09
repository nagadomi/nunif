import argparse
import math
import sys
from collections.abc import Generator
from typing import NamedTuple

import av
import torch
import torchvision.transforms.functional as TF

from nunif.utils.video import to_tensor


class VideoMetadata(NamedTuple):
    width: int
    height: int
    pix_fmt: str | None
    colorspace: int | None
    color_primaries: int | None
    color_trc: int | None
    color_range: int | None
    frames: int


class DiffResult(NamedTuple):
    mean_psnr: float
    min_psnr: float
    total_frames: int
    identical: bool
    meta1: VideoMetadata
    meta2: VideoMetadata


def get_video_metadata(video_path: str) -> VideoMetadata:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        frame_count = stream.frames
        if frame_count <= 0:
            frame_count = sum(1 for _ in container.decode(video=0))

        return VideoMetadata(
            width=ctx.width,
            height=ctx.height,
            pix_fmt=ctx.pix_fmt,
            colorspace=int(ctx.colorspace) if ctx.colorspace is not None else None,
            color_primaries=int(ctx.color_primaries) if ctx.color_primaries is not None else None,
            color_trc=int(ctx.color_trc) if ctx.color_trc is not None else None,
            color_range=int(ctx.color_range) if ctx.color_range is not None else None,
            frames=frame_count,
        )


def _iter_frames(video_path: str) -> Generator[torch.Tensor, None, None]:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield to_tensor(frame)


def compare_videos(
    video1_path: str,
    video2_path: str,
    resize: bool = False,
    verbose: bool = False,
) -> DiffResult:
    meta1 = get_video_metadata(video1_path)
    meta2 = get_video_metadata(video2_path)

    if not resize and (meta1.width != meta2.width or meta1.height != meta2.height):
        raise ValueError(f"Video resolution mismatch: {meta1.width}x{meta1.height} vs {meta2.width}x{meta2.height}")

    psnr_list: list[float] = []
    identical = True

    gen1 = _iter_frames(video1_path)
    gen2 = _iter_frames(video2_path)

    for idx, (t1, t2) in enumerate(zip(gen1, gen2, strict=False)):
        if resize and (t1.shape[2] != t2.shape[2] or t1.shape[1] != t2.shape[1]):
            t1 = TF.resize(t1, size=[t2.shape[1], t2.shape[2]])

        mse = ((t1 - t2) ** 2).mean().item()

        if mse == 0.0:
            psnr = float("inf")
        else:
            identical = False
            psnr = round(float(10.0 * math.log10(1.0 / mse)), 3)

        psnr_list.append(psnr)
        if verbose:
            print(f"  Frame {idx:04d}: PSNR = {psnr}")

    if not psnr_list:
        raise ValueError("No video frames found to compare")

    finite_psnrs = [p for p in psnr_list if not math.isinf(p)]
    if finite_psnrs:
        mean_psnr = round(sum(finite_psnrs) / len(finite_psnrs), 3)
        min_psnr = round(min(finite_psnrs), 3)
    else:
        mean_psnr = float("inf")
        min_psnr = float("inf")

    return DiffResult(
        mean_psnr=mean_psnr,
        min_psnr=min_psnr,
        total_frames=len(psnr_list),
        identical=identical,
        meta1=meta1,
        meta2=meta2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A simple video diff and PSNR tool")
    parser.add_argument("--input", "-i", type=str, nargs=2, required=True, help="Two video files to compare")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Minimum average PSNR threshold for pass")
    parser.add_argument("--resize", action="store_true", help="Allow resizing if resolutions differ")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output result for each frame")
    parser.add_argument("--check-metadata", action="store_true", help="Assert metadata (colorspace, etc.) matches")
    args = parser.parse_args()

    v1, v2 = args.input[0], args.input[1]
    result = compare_videos(v1, v2, resize=args.resize, verbose=args.verbose)

    print(f"Frames: {result.total_frames}")
    if result.identical:
        print("PSNR: inf (Bit-identical)")
    else:
        print(f"Mean PSNR: {result.mean_psnr} dB (Min: {result.min_psnr} dB)")

    if args.check_metadata:
        m1, m2 = result.meta1, result.meta2
        meta_match = (
            m1.pix_fmt == m2.pix_fmt
            and m1.colorspace == m2.colorspace
            and m1.color_primaries == m2.color_primaries
            and m1.color_trc == m2.color_trc
            and m1.color_range == m2.color_range
        )
        if meta_match:
            print("[OK] Metadata match")
        else:
            print(f"[FAIL] Metadata mismatch:\n  Video 1: {m1}\n  Video 2: {m2}")
            sys.exit(1)

    if args.threshold is not None:
        if result.mean_psnr < args.threshold:
            print(f"[FAIL] Mean PSNR ({result.mean_psnr}) < threshold ({args.threshold})")
            sys.exit(1)
        else:
            print(f"[PASS] Mean PSNR ({result.mean_psnr}) >= threshold ({args.threshold})")


if __name__ == "__main__":
    main()
