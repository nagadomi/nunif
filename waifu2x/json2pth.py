# convert old format json model files to pytorch model files
# DEBUG=1 python3 -m waifu2x.json2pth -i ./waifu2x_json_dir -o output_dir
import os
import argparse
from nunif.models import save_model, load_model, create_model
from .models import load_state_from_waifu2x_json
from nunif.logger import logger


def convert_vgg_7(waifu2x_model_dir, output_dir):
    for domain in ("art", "photo"):
        in_dir = os.path.join(waifu2x_model_dir, "vgg_7", domain)
        out_dir = os.path.join(output_dir, "vgg_7", domain)
        os.makedirs(out_dir, exist_ok=True)
        for noise_level in (0, 1, 2, 3):
            model = create_model("waifu2x.vgg_7", in_channels=3, out_channels=3)
            json_path = os.path.join(in_dir, f"noise{noise_level}_model.json")
            model = load_state_from_waifu2x_json(model, json_path)
            save_path = os.path.join(out_dir, f"noise{noise_level}.pth")
            save_model(model, save_path, updated_at=os.path.getmtime(json_path))


def convert_upconv_7(waifu2x_model_dir, output_dir):
    for domain in ("art", "photo"):
        in_dir = os.path.join(waifu2x_model_dir, "upconv_7", domain)
        out_dir = os.path.join(output_dir, "upconv_7", domain)
        os.makedirs(out_dir, exist_ok=True)
        for noise_level in (0, 1, 2, 3):
            model = create_model("waifu2x.upconv_7", in_channels=3, out_channels=3)
            json_path = os.path.join(in_dir, f"noise{noise_level}_scale2.0x_model.json")
            model = load_state_from_waifu2x_json(model, json_path)
            save_path = os.path.join(out_dir, f"noise{noise_level}_scale2x.pth")
            save_model(model, save_path, updated_at=os.path.getmtime(json_path))

        model = create_model("waifu2x.upconv_7", in_channels=3, out_channels=3)
        json_path = os.path.join(in_dir, "scale2.0x_model.json")
        model = load_state_from_waifu2x_json(model, json_path)
        save_path = os.path.join(out_dir, "scale2x.pth")
        save_model(model, save_path, updated_at=os.path.getmtime(json_path))


def convert_cunet(waifu2x_model_dir, output_dir):
    for domain in ("art",):
        in_dir = os.path.join(waifu2x_model_dir, "cunet", domain)
        out_dir = os.path.join(output_dir, "cunet", domain)
        os.makedirs(out_dir, exist_ok=True)
        for noise_level in (0, 1, 2, 3):
            model = create_model("waifu2x.cunet", in_channels=3, out_channels=3)
            json_path = os.path.join(in_dir, f"noise{noise_level}_model.json")
            model = load_state_from_waifu2x_json(model, json_path)
            save_path = os.path.join(out_dir, f"noise{noise_level}.pth")
            save_model(model, save_path, updated_at=os.path.getmtime(json_path))


def convert_upcunet(waifu2x_model_dir, output_dir):
    for domain in ("art",):
        in_dir = os.path.join(waifu2x_model_dir, "cunet", domain)
        out_dir = os.path.join(output_dir, "cunet", domain)
        os.makedirs(out_dir, exist_ok=True)
        for noise_level in (0, 1, 2, 3):
            model = create_model("waifu2x.upcunet", in_channels=3, out_channels=3)
            json_path = os.path.join(in_dir, f"noise{noise_level}_scale2.0x_model.json")
            model = load_state_from_waifu2x_json(model, json_path)
            save_path = os.path.join(out_dir, f"noise{noise_level}_scale2x.pth")
            save_model(model, save_path, updated_at=os.path.getmtime(json_path))

        model = create_model("waifu2x.upcunet", in_channels=3, out_channels=3)
        json_path = os.path.join(in_dir, "scale2.0x_model.json")
        model = load_state_from_waifu2x_json(model, json_path)
        save_path = os.path.join(out_dir, "scale2x.pth")
        save_model(model, save_path, updated_at=os.path.getmtime(json_path))


def _test():
    import PIL
    import torchvision.transforms.functional as TF

    def load_image(filename):
        im = PIL.Image.open(filename).convert("RGB")
        x = TF.to_tensor(im)
        return x.reshape(1, x.shape[0], x.shape[1], x.shape[2])

    def run_model(model, src, dest):
        x = load_image(src)
        z = model(x)
        z = TF.to_pil_image(z[0])
        z.save(dest)
    model = load_model("pretrained_models/waifu2x/upconv_7/art/noise1_scale2x.pth")
    run_model(model, "query/miku_small_noisy.jpg", "out.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="input waifu2x json model dir")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="output dir")
    args = parser.parse_args()

    logger.info("vgg_7")
    convert_vgg_7(args.input_dir, args.output_dir)
    logger.info("upconv_7")
    convert_upconv_7(args.input_dir, args.output_dir)
    logger.info("cunet")
    convert_cunet(args.input_dir, args.output_dir)
    logger.info("upcunet")
    convert_upcunet(args.input_dir, args.output_dir)
