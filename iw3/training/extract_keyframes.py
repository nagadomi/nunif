# tool to extract keyframe images from video
import argparse
import os
from os import path
from PIL import Image
from nunif.utils.video import process_video_keyframes


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output dir")
    parser.add_argument("--interval", type=float, default=4, help="min interval")
    parser.add_argument("--rotate-left", action="store_true",
                        help="rotate 90 degrees to the left(counterclockwise)")
    parser.add_argument("--rotate-right", action="store_true",
                        help="rotate 90 degrees to the right(clockwise)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    def frame_callback(frame):
        im = frame.to_image()
        if args.rotate_left:
            im = im.transpose(Image.Transpose.ROTATE_90)
        elif args.rotate_right:
            im = im.transpose(Image.Transpose.ROTATE_270)
        output_filename = path.splitext(path.basename(args.input))[0] + f"_{frame.index}.png"
        im.save(path.join(args.output, output_filename))

    process_video_keyframes(args.input, frame_callback, min_interval_sec=args.interval)


if __name__ == "__main__":
    main()
