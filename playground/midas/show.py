# Show MiDaS monocular depth estimation result
# from https://github.com/pytorch/hub/blob/master/intelisl_midas_v2.md
# pip3 install opencv-python
# pip3 install timm
# python -m playground.midas.show -i playground/jpeg_qtable/images/donut_q100.jpg
import cv2
import torch
import numpy as np
import argparse


def normalize(x):
    min_v = x.min()
    max_v = x.max()
    return (x - min_v) / ((max_v - min_v) + 1e-6)


def show_depth(args):
    # See https://github.com/isl-org/MiDaS/blob/master/hubconf.py
    model_type = "DPT_BEiT_L_512"
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_BEiT_L_512":
        transform = midas_transforms.beit512_transform
    elif model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    src = img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    print("input", input_batch.shape)
    with torch.no_grad():
        prediction = midas(input_batch)
        print("output", prediction.shape)
        print("min", prediction.min().item(),
              "max", prediction.max().item(),
              "mean", prediction.mean().item())
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        if args.clip is not None:
            prediction = torch.clamp(prediction, min=args.clip)

    # 16bit grayscale output
    output = normalize(prediction).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 0xffff, 0, 0xffff).astype(np.uint16)
    if args.output:
        cv2.imwrite(args.output, output)
    cv2.imshow("src", src)
    cv2.imshow("depth", output)
    print("press key to exit")
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input image file")
    parser.add_argument("--output", "-o", type=str, help="output 16bit depth image.png")
    parser.add_argument("--clip", type=float, help="clip(pred, min=clip)")
    args = parser.parse_args()
    show_depth(args)
