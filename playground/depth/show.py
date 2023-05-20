# Show ZoeDepth monocular depth estimation result as 3d point cloud data
# pip3 install opencv-python timm open3d
# python -m playground.depth.show -i playground/jpeg_qtable/images/donut_q100.jpg
import cv2
import torch
import torchvision.transforms.functional as TF
import numpy as np
import argparse
import open3d as o3d
from PIL import Image


def visualize_pcd(rgb, depth, scale):
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)
    # o3d.visualization.draw_geometries([rgb])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth,
        depth_scale=scale,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])


def normalize(x):
    x = x.astype(np.float32)
    min_v = x.min()
    max_v = x.max()
    x = (x - min_v) / ((max_v - min_v) + 1e-6)
    x = 1. - x
    x = np.clip(x * 0xffff, 0, 0xffff).astype(np.uint16)
    return x


def show_depth(args):
    # See https://github.com/isl-org/ZoeDepth/blob/main/hubconf.py
    if args.update:
        # Triggers fresh download of MiDaS repo
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    model_type = "ZoeD_N"
    model = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    rgb = Image.open(args.input).convert("RGB")
    rgb.load()
    src_w, src_h = rgb.size
    with torch.inference_mode():
        prediction = TF.to_tensor(model.infer_pil(rgb, output_type="pil"))
        print("output", prediction.shape)
        print("min", prediction.min().item(),
              "max", prediction.max().item())
        if args.clip is not None:
            prediction = torch.clamp(prediction, max=args.clip)

    # 16bit grayscale output
    depth = prediction.permute(1, 2, 0).cpu().numpy()
    rgb = (TF.to_tensor(rgb).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb = np.asarray(rgb, order="C")
    if args.output:
        cv2.imwrite(args.output, normalize(depth))

    visualize_pcd(rgb, depth, args.scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input image file")
    parser.add_argument("--output", "-o", type=str, help="output 16bit depth image.png")
    parser.add_argument("--clip", type=float, help="clip(pred, max=clip)")
    parser.add_argument("--scale", type=float, default=500, help="scale")
    parser.add_argument("--update", action="store_true", help="force update midas model")
    args = parser.parse_args()
    show_depth(args)
