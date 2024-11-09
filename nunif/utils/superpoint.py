"""PyTorch implementation of the SuperPoint model,
   derived from the TensorFlow re-implementation (2018).
   Authors: Rémi Pautrat, Paul-Edouard Sarlin

   MIT License
   https://github.com/rpautrat/SuperPoint/
"""
import torch.nn as nn
import torch
from collections import OrderedDict
from types import SimpleNamespace


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )


class SuperPoint(nn.Module):
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

        self.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, image):
        if image.shape[1] == 3:  # RGB to gray
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        # Decode the detection scores
        scores = self.detector(features)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        descriptors = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.conf.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }

    def load(self, map_location="cpu"):
        self.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://github.com/nagadomi/nunif/releases/download/0.0.0/superpoint_v6_from_tf.pth",
            weights_only=True, map_location=map_location))
        return self


def _visualize():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F
    from PIL import ImageDraw

    x1 = (IO.read_image("cc0/dog2.jpg") / 255.0)
    x1 = x1[:, :250, :250].unsqueeze(0)
    x2 = F.pad(TF.resize(TF.rotate(x1, 30), (200, 200)), (25,) * 4)

    x1 = x1.cuda()
    x2 = x2.cuda()
    model = SuperPoint().load().cuda()
    with torch.autocast(device_type=x1.device.type):
        ret1 = model(x1)
        ret2 = model(x2)
    d1 = ret1["descriptors"][0]
    d2 = ret2["descriptors"][0]
    distance = (d1.pow(2).sum(1, keepdim=True) - 2 * d1 @ d2.t() +
                d2.pow(2).sum(1, keepdim=True).t())
    match_index = torch.argmin(distance, dim=-1)
    min_distance = torch.gather(distance, dim=1, index=match_index.view(-1, 1)).view(-1)

    print("match score", "min", min_distance.min().item(), "max", min_distance.max().item(),
          "mean", min_distance.mean().item(), "median", min_distance.median().item())

    threshold = 10000.0  # min_distance.median()
    filter_index = min_distance < threshold

    k1 = ret1["keypoints"][0][filter_index]
    k2 = ret2["keypoints"][0][match_index][filter_index]

    img = TF.to_pil_image(torch.cat([x1, x2], dim=3).squeeze(0))

    # visualize
    gc = ImageDraw.Draw(img)
    k2_offset = x1.shape[3]

    if False:
        # line
        for xy1, xy2 in zip(k1, k2):
            x1, y1 = int(xy1[0].item()), int(xy1[1].item())
            x2, y2 = int(xy2[0].item()) + k2_offset, int(xy2[1].item())
            gc.line(((x1, y1), (x2, y2)), fill="green")
    # points
    for xy1, xy2 in zip(k1, k2):
        x1, y1 = int(xy1[0].item()), int(xy1[1].item())
        x2, y2 = int(xy2[0].item()) + k2_offset, int(xy2[1].item())
        gc.circle((x1, y1), radius=2, fill="red")
        gc.circle((x2, y2), radius=2, fill="blue")

    img.show()


def _benchmark():
    import torchvision.io as IO
    import time

    B = 8
    N = 100

    x = (IO.read_image("cc0/dog2.jpg") / 255.0)
    x = x[:, :256, :256].unsqueeze(0).repeat(B, 1, 1, 1).cuda()
    model = SuperPoint().load().cuda()

    with torch.autocast(device_type=x.device.type):
        model(x)
    torch.cuda.synchronize()

    t = time.time()
    N = 100
    with torch.autocast(device_type=x.device.type):
        for _ in range(N):
            model(x)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")

    # 800FPS on RTX3070ti


if __name__ == "__main__":
    _visualize()
    _benchmark()
