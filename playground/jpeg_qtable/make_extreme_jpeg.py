from PIL import Image
import os
from os import path
import random


ZIGZAG_SCAN_INDEX = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4,
    5, 12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6, 7,
    14, 21, 28, 35, 42, 49, 56,
    57, 50, 43, 36, 29, 22, 15,
    23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53,
    60, 61, 54, 47, 55, 62, 63]


IMAGE_DIR = path.join(path.dirname(__file__), "images")
OUTPUT_DIR = path.join(path.dirname(__file__), "..", "..", "tmp", "jpeg_qtable")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def show(qtable):
    for i in range(8):
        row = [qtable[i * 8 + j] for j in range(8)]
        print(row)


def save_jpeg_with_qtable(im, output_path, qtables):
    im.save(output_path, format="jpeg", qtables=qtables)


def main():
    with Image.open(path.join(IMAGE_DIR, "donut_default_q85.jpg")) as im:
        # Low-frequency: 255 (quality=0)
        # High-frequency:  1 (quality=100)
        qtable_high = [1] * 64
        for i in range(32):
            qtable_high[ZIGZAG_SCAN_INDEX[i]] = 255

        output_path = path.join(OUTPUT_DIR, "donut_extreme_high.jpg")
        save_jpeg_with_qtable(im, output_path,
                              qtables={0: qtable_high, 1: qtable_high})
        show(qtable_high)
        print("save", path.relpath(output_path))

        # Low-frequency:    1 (quality=100)
        # High-frequency: 255 (quality=0)
        qtable_low = [1] * 64
        for i in range(32, 64):
            qtable_low[ZIGZAG_SCAN_INDEX[i]] = 255

        output_path = path.join(OUTPUT_DIR, "donut_extreme_low.jpg")
        save_jpeg_with_qtable(im, output_path,
                              qtables={0: qtable_low, 1: qtable_low})
        show(qtable_low)
        print("save", path.relpath(output_path))

        # random generate
        for i in range(5):
            qtable = [random.randint(1, 255) for _ in range(64)]
            for j in range(6):
                qtable[ZIGZAG_SCAN_INDEX[j]] = 16
            output_path = path.join(OUTPUT_DIR, f"donut_extreme_random_{i}.jpg")
            save_jpeg_with_qtable(im, output_path,
                                  qtables={0: qtable, 1: qtable})
            show(qtable)
            print("save", path.relpath(output_path))


if __name__ == "__main__":
    main()
