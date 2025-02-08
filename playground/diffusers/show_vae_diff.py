# Show VAE encode -> decode degradation result
#
# pip3 install --upgrade diffusers[torch]
#
# python -m playground.diffusers.show_vae_diff -i path_to_image.png

from diffusers import AutoencoderKL
from diffusers import AutoencoderDC
from nunif.utils import pil_io
import argparse
import torch
import torchvision.transforms.functional as TF


VAE_OPTIONS = {
    "sd-mse": {
        "pretrained_model_name_or_path": "stabilityai/sd-vae-ft-mse"
    },
    "dcae": {
        "pretrained_model_name_or_path": "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
    },
}


def safe_pad(x, mod):
    # make the image size a multiple of 8 to avoid image size changes in encode/decode
    c, h, w = x.shape
    pad_bottom = mod - h % mod
    pad_right = mod - w % mod
    if pad_bottom != 0 or pad_right != 0:
        x = TF.pad(x, (0, 0, pad_right, pad_bottom), padding_mode="edge")
    return x


def main():
    import time

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input image file")
    parser.add_argument("--vae", type=str, default="sd-mse", choices=["sd-mse", "dcae"],
                        help="vae pretrained model")
    args = parser.parse_args()
    im, _ = pil_io.load_image_simple(args.input, color="rgb")

    if args.vae == "dcae":
        vae = AutoencoderDC.from_pretrained(torch_dtype=torch.float32, use_auth_token=False,
                                            **VAE_OPTIONS[args.vae]).cuda()
        mod = 32
    else:
        vae = AutoencoderKL.from_pretrained(torch_dtype=torch.float32,
                                            use_auth_token=False,
                                            **VAE_OPTIONS[args.vae]).cuda()
        mod = 8

    vae.eval()
    with torch.no_grad():
        x = pil_io.to_tensor(im)
        x = safe_pad(x, mod)
        pil_io.to_image(x).show()
        time.sleep(1)

        if isinstance(vae, AutoencoderKL):
            mu = vae.encode(x.unsqueeze(0).cuda()).latent_dist.mode()
            y = vae.decode(mu).sample
        else:
            mu = vae.encode(x.unsqueeze(0).cuda()).latent
            y = vae.decode(mu).sample

        y = y.squeeze(0).cpu()
        pil_io.to_image(y).show()
        time.sleep(1)

        diff = (x - y).abs()
        min_v = diff.min()
        max_v = diff.max()
        diff = (diff - min_v) / (max_v - min_v)

        pil_io.to_image(diff).show()


if __name__ == "__main__":
    main()
