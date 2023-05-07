# Show MiDaS monocular depth estimation result as 3d point cloud data
# from https://github.com/pytorch/hub/blob/master/intelisl_midas_v2.md
# pip3 install opencv-python timm open3d
# python -m playground.midas.show -i playground/jpeg_qtable/images/donut_q100.jpg
import cv2
import torch
import numpy as np
import argparse
import open3d as o3d


def visualize_pcd(rgb, depth):
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)
    # o3d.visualization.draw_geometries([rgb])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth,
        depth_scale=1000.0 * 256,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])


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

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(device)
    print("input", input_batch.shape)
    with torch.no_grad():
        prediction = midas(input_batch)
        print("output", prediction.shape)
        print("min", prediction.min().item(),
              "max", prediction.max().item(),
              "mean", prediction.mean().item())
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        if args.clip is not None:
            prediction = torch.clamp(prediction, min=args.clip)

    # 16bit grayscale output
    output = normalize(prediction).permute(1, 2, 0).cpu().numpy()
    depth = np.clip(output * 0xffff, 0, 0xffff).astype(np.uint16) // 2
    if args.output:
        cv2.imwrite(args.output, depth)

    visualize_pcd(rgb, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input image file")
    parser.add_argument("--output", "-o", type=str, help="output 16bit depth image.png")
    parser.add_argument("--clip", type=float, help="clip(pred, min=clip)")
    args = parser.parse_args()
    show_depth(args)
