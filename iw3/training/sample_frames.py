# tool to extract frame images from video
import argparse
import os
from os import path
from PIL import Image
import nunif.utils.video as VU
import hashlib
from concurrent.futures import ThreadPoolExecutor


def md5(s):
    MD5_SALT = "nunif-iw3-training"
    return hashlib.md5((s + MD5_SALT).encode()).hexdigest()


def save_image(im, filename):
    im.save(filename)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output dir")
    parser.add_argument("--samples", type=float, default=1000, help="rough sample frame count")
    parser.add_argument("--rotate-left", action="store_true",
                        help="rotate 90 degrees to the left(counterclockwise)")
    parser.add_argument("--rotate-right", action="store_true",
                        help="rotate 90 degrees to the right(clockwise)")

    args = parser.parse_args()
    output_basename = md5(path.basename(args.input))
    os.makedirs(args.output, exist_ok=True)

    counter = 0
    interval = 1
    futures = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        def config_callback(stream):
            nonlocal interval
            frames = int(VU.get_frames(stream))
            if frames <= 0:
                raise "cannot get frames count"

            if frames < args.samples:
                interval = 1
            else:
                interval = frames // args.samples
            return VU.VideoOutputConfig(fps=None)

        def frame_callback(frame):
            nonlocal counter
            counter += 1

            if frame is None:
                return

            if counter % interval == 0:
                im = frame.to_image()
                if args.rotate_left:
                    im = im.transpose(Image.Transpose.ROTATE_90)
                elif args.rotate_right:
                    im = im.transpose(Image.Transpose.ROTATE_270)
                output_path = path.join(args.output, output_basename + f"_{frame.pts}.png")
                futures.append(pool.submit(save_image, im, output_path))
                if len(futures) > 100:
                    _ = [f.result() for f in futures]
                    futures.clear()

        VU.hook_frame(args.input, frame_callback, config_callback=config_callback)


if __name__ == "__main__":
    main()
