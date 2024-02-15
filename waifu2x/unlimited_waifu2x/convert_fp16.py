# Tool to convert model prameters from fp32 to fp16
# input/output keeps fp32, when keep_io_types=True
# because in javascript backend(wasm), fp16 infenrece is not supported
# just reduce file size and data transfer size
import onnx
from onnxconverter_common import float16
import argparse
import os
from os import path
import shutil


def convert(model_in, model_out, keep_io_types=True):
    model = onnx.load(model_in)
    model_fp16 = float16.convert_float_to_float16(
        model, min_positive_val=1e-7, max_finite_val=6e+4,
        keep_io_types=keep_io_types)
    onnx.save(model_fp16, model_out)


def convert_dir(in_dir, out_dir, checkpoint_path):
    in_dir = path.join(in_dir, checkpoint_path)
    out_dir = path.join(out_dir, checkpoint_path)
    os.makedirs(out_dir, exist_ok=True)
    filenames = ["scale1x.onnx", "scale2x.onnx", "scale4x.onnx"]
    for noise_level in (0, 1, 2, 3):
        filenames.append(f"noise{noise_level}.onnx")
        filenames.append(f"noise{noise_level}_scale2x.onnx")
        filenames.append(f"noise{noise_level}_scale4x.onnx")

    for filename in filenames:
        in_file = path.join(in_dir, filename)
        if path.exists(in_file):
            convert(in_file, path.join(out_dir, filename))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="input onnx model dir")
    parser.add_argument("-o", "--output", type=str, required=True, help="output onnx model dir")
    args = parser.parse_args()

    convert_dir(args.input, args.output, path.join("swin_unet", "art"))
    convert_dir(args.input, args.output, path.join("swin_unet", "art_scan"))
    convert_dir(args.input, args.output, path.join("swin_unet", "photo"))
    convert_dir(args.input, args.output, path.join("cunet", "art"))

    shutil.copytree(
        path.join(args.input, "utils"),
        path.join(args.output, "utils"),
        dirs_exist_ok=True)


if __name__ == "__main__":
    main()
