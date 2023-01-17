# convert pytorch models to onnx
# DEBUG=1 python3 -m waifu2x.export_onnx -i ./waifu2x/pretrained_models -o ./waifu2x/onnx_models
# TODO: torchvision's SwinTransformer has bug in Dropout's training flag. Currently I fixed it locally.
import os
import argparse
from nunif.models import save_model, load_model, create_model
from .models import load_state_from_waifu2x_json
from nunif.logger import logger


def convert_cunet(model_dir, output_dir):
    for domain in ("art",):
        in_dir = os.path.join(model_dir, "cunet", domain)
        out_dir = os.path.join(output_dir, "cunet", domain)
        os.makedirs(out_dir, exist_ok=True)
        for noise_level in (0, 1, 2, 3):
            load_path = os.path.join(in_dir, f"noise{noise_level}.pth")
            save_path = os.path.join(out_dir, f"noise{noise_level}.onnx")
            model, _ = load_model(load_path)
            model.export_onnx(save_path)


def convert_upcunet(model_dir, output_dir):
    for domain in ("art",):
        in_dir = os.path.join(model_dir, "cunet", domain)
        out_dir = os.path.join(output_dir, "cunet", domain)
        os.makedirs(out_dir, exist_ok=True)
        for noise_level in (0, 1, 2, 3):
            load_path = os.path.join(in_dir, f"noise{noise_level}_scale2x.pth")
            save_path = os.path.join(out_dir, f"noise{noise_level}_scale2x.onnx")
            model, _ = load_model(load_path)
            model.export_onnx(save_path)

        load_path = os.path.join(in_dir, f"scale2x.pth")
        save_path = os.path.join(out_dir, f"scale2x.onnx")
        model, _ = load_model(load_path)
        model.export_onnx(save_path)


def convert_swin_unet(model_dir, output_dir):
    for domain in ("art",):
        in_dir = os.path.join(model_dir, "swin_unet", domain)
        out_dir = os.path.join(output_dir, "swin_unet", domain)
        os.makedirs(out_dir, exist_ok=True)
        for noise_level in (0, 1, 2, 3):
            load_path = os.path.join(in_dir, f"noise{noise_level}_scale4x.pth")
            save_path = os.path.join(out_dir, f"noise{noise_level}_scale4x.onnx")
            model, _ = load_model(load_path)
            model.export_onnx(save_path)

        load_path = os.path.join(in_dir, f"scale4x.pth")
        save_path = os.path.join(out_dir, f"scale4x.onnx")
        model, _ = load_model(load_path)
        model.export_onnx(save_path)


def convert_utils(output_dir):
    from nunif.models.onnx_helper_models import ONNXReflectionPadding

    utils_dir = os.path.join(output_dir, "utils");
    os.makedirs(utils_dir, exist_ok=True);

    pad = ONNXReflectionPadding()
    pad.export_onnx(os.path.join(utils_dir, "pad.onnx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="input waifu2x json model dir")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="output dir")
    args = parser.parse_args()

    logger.info("cunet")
    convert_cunet(args.input_dir, args.output_dir)
    logger.info("upcunet")
    convert_upcunet(args.input_dir, args.output_dir)

    logger.info("swin_unet")
    convert_swin_unet(args.input_dir, args.output_dir)

    logger.info("utils")
    convert_utils(args.output_dir)
