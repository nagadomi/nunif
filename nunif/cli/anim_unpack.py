# see docs/anim_pack_unpack.md
import argparse
import os
import sys

import yaml
from PIL import Image


def unpack(input_path, output_dir):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist", file=sys.stderr)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with Image.open(input_path) as img:
        fmt = img.format
        n_frames = getattr(img, "n_frames", 1)
        loop = img.info.get("loop", 0)
        background = img.info.get("background", (0, 0, 0, 0))
        # Convert tuple to list for safe YAML loading
        if isinstance(background, tuple):
            background = list(background)

        config = {"format": fmt, "loop": loop, "background": background, "frames": []}

        for i in range(n_frames):
            img.seek(i)
            # Standardizing to 6 digits serial number
            frame_filename = f"{i:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save frame. Preserve mode if possible (especially for 16-bit grayscale "I;16")
            # For GIF, we still need to handle transparency/compositing,
            # but for PNG/WebP we should keep the source mode if it's high depth.
            if fmt == "GIF":
                frame_img = img.convert("RGBA")
            else:
                frame_img = img.copy()

            frame_img.save(frame_path)

            frame_info = {
                "file": frame_filename,
                "duration": img.info.get("duration", 100),  # Default to 100ms if not found
            }

            # Additional metadata for GIF
            if fmt == "GIF":
                if "disposal" in img.info:
                    frame_info["disposal"] = img.info["disposal"]
                if "transparency" in img.info:
                    frame_info["transparency"] = img.info["transparency"]

            config["frames"].append(frame_info)

        config_path = os.path.join(output_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

    print(f"Unpacked {n_frames} frames from {input_path} to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unpack animated WebP/GIF into image sequence")
    parser.add_argument("-i", "--input", required=True, help="Input animated image file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    unpack(args.input, args.output)


if __name__ == "__main__":
    main()
