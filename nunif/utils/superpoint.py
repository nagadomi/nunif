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
import math
import torch.nn.functional as F


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

# The code below here was written by nagadomi

    def load(self, map_location="cpu"):
        self.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://github.com/nagadomi/nunif/releases/download/0.0.0/superpoint_v6_from_tf.pth",
            weights_only=True, map_location=map_location))
        return self

    @torch.inference_mode()
    def infer(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
            batch = False
        else:
            batch = True

        ret = self.forward(x)

        # convert to batch-first structure
        new_ret = []
        for i in range(x.shape[0]):
            new_ret.append({
                "keypoints": ret["keypoints"][i],
                "descriptors": ret["descriptors"][i],
                "keypoint_scores": ret["keypoint_scores"][i]
            })
        if not batch:
            new_ret = new_ret[0]

        return new_ret


def find_match_index(kp1, kp2, threshold=0.5, return_score=False):
    d1 = kp1["descriptors"]
    d2 = kp2["descriptors"]

    cosine_similarity = d1 @ d2.t()
    match_index = torch.argmax(cosine_similarity, dim=-1)
    max_similarity = torch.gather(cosine_similarity, dim=1, index=match_index.view(-1, 1)).view(-1)
    filter_index = max_similarity > threshold
    kp1_index = torch.arange(d1.shape[0], device=d1.device)[filter_index]
    kp2_index = match_index[filter_index]
    if return_score:
        return kp1_index, kp2_index, max_similarity[filter_index]
    else:
        return kp1_index, kp2_index


def find_rigid_transform(xy1, xy2, center=None, n=50, lr_translation=0.1, lr_scale_rotation=0.1, sigma=None,
                         disable_shift=False, disable_scale=False, disable_rotate=False):
    # TODO: batch
    assert torch.is_grad_enabled()
    xy1 = xy1.cpu()
    xy2 = xy2.cpu()
    translation = torch.zeros((1, 2), dtype=torch.float32, device=xy1.device, requires_grad=True)
    scale = torch.ones((1, 1), dtype=torch.float32, device=xy1.device, requires_grad=True)
    rotation = torch.zeros((1, 1), dtype=torch.float32, device=xy1.device, requires_grad=True)
    if disable_shift:
        translation.requires_grad_(False)
    if disable_scale:
        scale.requires_grad_(False)
    if disable_rotate:
        rotation.requires_grad_(False)

    param_groups = [
        {"params": [translation], "lr": lr_translation},
        {"params": [scale, rotation], "lr": lr_scale_rotation},
    ]
    optimizer = torch.optim.Adam(param_groups, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n, eta_min=lr_scale_rotation * 1e-3)

    if not torch.is_tensor(center):
        center = torch.tensor(center, dtype=torch.float32, device=xy1.device)

    xy1 = xy1 - center
    xy2 = xy2 - center
    norm_scale = xy1.abs().max()
    xy1 = xy1 / norm_scale
    xy2 = xy2 / norm_scale

    for i in range(n):
        optimizer.zero_grad()
        xy = xy1

        # rotate
        rcos = rotation.cos()
        rsin = rotation.sin()
        xy = torch.cat([xy[:, :1] * rcos - xy[:, 1:] * rsin,
                        xy[:, :1] * rsin + xy[:, 1:] * rcos], dim=1)

        # scale
        xy = xy * scale

        # translate
        xy = xy
        xy = xy + translation

        if sigma is not None and i > 0:
            loss = F.l1_loss(xy, xy2, reduction="none").mean(dim=1)
            mean, stdv = torch.std_mean(loss)
            mask = ((loss - mean) / stdv) < sigma
            loss = loss[mask].mean()
            # print(loss.item(), mask.sum() / xy.shape[0])
        else:
            loss = F.l1_loss(xy, xy2)
        loss.backward()
        optimizer.step()
        scheduler.step()

    shift = (translation.detach() * norm_scale).flatten().tolist()
    scale = scale.detach().item()
    angle = math.degrees(rotation.detach().item())
    center = center.flatten().tolist()

    return shift, scale, angle, center


@torch.inference_mode()
def apply_rigid_transform(x, shift, scale, angle, center, mode="bilinear", padding_mode="border"):
    # TODO: batch
    assert x.ndim == 3  # CHW
    height, width = x.shape[1:]
    center = torch.tensor(center, device=x.device, dtype=x.dtype)
    shift = torch.tensor(shift, device=x.device, dtype=x.dtype)
    angle = math.radians(angle)

    # inverse params
    shift = -shift
    scale = 1.0 / scale
    angle = -angle

    # backward warping
    py, px = torch.meshgrid(torch.linspace(0, height - 1, height, device=x.device, dtype=x.dtype),
                            torch.linspace(0, width - 1, width, device=x.device, dtype=x.dtype), indexing="ij")

    px = px - center[0]
    py = py - center[1]
    mesh_x = (px * math.cos(angle) - py * math.sin(angle))
    mesh_y = (px * math.sin(angle) + py * math.cos(angle))

    grid = torch.stack((mesh_x, mesh_y), 2).contiguous()
    grid = grid * scale
    grid = grid + (shift.view(1, 1, -1) + center.view(1, 1, -1))

    grid = (grid / torch.tensor([width - 1, height - 1], device=x.device, dtype=x.dtype).view(1, 1, -1)) * 2.0 - 1.0

    x = F.grid_sample(x.unsqueeze(0), grid.unsqueeze(0), mode="bilinear", padding_mode=padding_mode, align_corners=False)
    x = x[0]

    return x


def _visualize():
    import time
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    from PIL import ImageDraw

    x1 = (IO.read_image("cc0/dog2.jpg") / 255.0)
    x1 = x1[:, :250, :250]
    x2 = TF.pad(TF.resize(TF.rotate(x1, 30), (200, 200)), (25,) * 4)
    # x2 = TF.rotate(x1, 180)  # This will not result in {scale = 1, rotate = 180}, but {scale = -1, rotate=0}
    # x2 = F.pad(x1, [-25, 25, -5, 5])

    x1 = x1.unsqueeze(0).cuda()
    x2 = x2.unsqueeze(0).cuda()
    model = SuperPoint().load().cuda()
    with torch.autocast(device_type=x1.device.type):
        ret = model.infer(torch.cat([x1, x2], dim=0))

    # matching
    kp1_index, kp2_index = find_match_index(ret[0], ret[1])
    k1 = ret[0]["keypoints"][kp1_index]
    k2 = ret[1]["keypoints"][kp2_index]

    img = TF.to_pil_image(torch.cat([x1, x2], dim=3).squeeze(0))

    # visualize
    gc = ImageDraw.Draw(img)
    k2_offset = x1.shape[3]

    if False:
        # line
        for xy1, xy2 in zip(k1, k2):
            xx1, yy1 = int(xy1[0].item()), int(xy1[1].item())
            xx2, yy2 = int(xy2[0].item()) + k2_offset, int(xy2[1].item())
            gc.line(((xx1, yy1), (xx2, yy2)), fill="green")

    # points
    for xy1, xy2 in zip(k1, k2):
        xx1, yy1 = int(xy1[0].item()), int(xy1[1].item())
        xx2, yy2 = int(xy2[0].item()) + k2_offset, int(xy2[1].item())
        gc.circle((xx1, yy1), radius=2, fill="red")
        gc.circle((xx2, yy2), radius=2, fill="blue")

    # show matching
    img.show()
    time.sleep(1)

    # estimate transform
    shift, scale, angle, center = find_rigid_transform(k1, k2, center=[x1.shape[3] // 2, x1.shape[2] // 2])
    print(shift, scale, angle, center)
    # apply transform
    x3 = apply_rigid_transform(x1.squeeze(0), shift=shift, scale=scale, angle=angle, center=center)
    # show
    TF.to_pil_image(x2.squeeze(0)).show()
    time.sleep(1)
    TF.to_pil_image(x3).show()
    time.sleep(1)


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
            model.infer(x)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")

    # 850FPS on RTX3070ti


if __name__ == "__main__":
    _visualize()
    # _benchmark()
