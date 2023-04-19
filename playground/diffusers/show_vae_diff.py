# Show VAE encode -> decode degradation result
#
# pip3 install opencv-python
# pip3 install --upgrade diffusers[torch]
#
# python -m playground.diffusers.show_vae_diff -i path_to_image.png

from diffusers import AutoencoderKL
from nunif.utils import pil_io
import argparse
import cv2
import torch
import torchvision.transforms.functional as TF


VAE_OPTIONS = {
    "anything": {
        "pretrained_model_name_or_path": "Linaqruf/anything-v3.0",
        "subfolder": "vae"
    },
    "sd-mse": {
        "pretrained_model_name_or_path": "stabilityai/sd-vae-ft-mse"
    }
}


def safe_pad(x):
    # make the image size a multiple of 8 to avoid image size changes in encode/decode
    c, h, w = x.shape
    pad_bottom = 8 - h % 8
    pad_right = 8 - w % 8
    if pad_bottom != 0 or pad_right != 0:
        x = TF.pad(x, (0, 0, pad_right, pad_bottom), padding_mode="edge")
    return x


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input image file")
    parser.add_argument("--vae", type=str, default="anything", choices=["anything", "sd-mse"],
                        help="vae pretrained model")
    args = parser.parse_args()
    im, _ = pil_io.load_image_simple(args.input, color="rgb")

    vae = AutoencoderKL.from_pretrained(torch_dtype=torch.float32,
                                        use_auth_token=False,
                                        **VAE_OPTIONS[args.vae]).cuda()
    vae.eval()
    with torch.no_grad():
        x = pil_io.to_tensor(im)
        x = safe_pad(x)
        cv2.imshow("x", pil_io.to_cv2(pil_io.to_image(x)))
        if max(x.shape[2], x.shape[1]) > 512:
            # FIXME: tiled_encode/tiled_decode is broken or usage is incorrect
            #        right or bottom area are repeated
            mu = vae.tiled_encode(x.unsqueeze(0).cuda()).latent_dist.mode()
            y = vae.tiled_decode(mu).sample
        else:
            mu = vae.encode(x.unsqueeze(0).cuda()).latent_dist.mode()
            y = vae.decode(mu).sample

        y = y.squeeze(0).cpu()
        cv2.imshow("y", pil_io.to_cv2(pil_io.to_image(y)))

        diff = (x - y)
        min_v = diff.min()
        max_v = diff.max()
        diff = (diff - min_v) / (max_v - min_v)

        cv2.imshow("diff scaled", pil_io.to_cv2(pil_io.to_image(diff)))
        print("Press any key in windows to exit")
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
