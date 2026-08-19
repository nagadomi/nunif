# Pretrained RepVGG, Inference-only, flattened version.
# Original code and weights: https://github.com/DingXiaoH/RepVGG
#   MIT License Copyright (c) 2020 DingXiaoH

import torch
import torch.nn as nn
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(RepVGGBlock, self).__init__()
        self.rbr_reparam = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True
        )
        self.nonlinearity = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.nonlinearity(self.rbr_reparam(x))


class RepVGG(nn.Module):
    def __init__(self, num_blocks, width_multiplier, num_classes=1000):
        super(RepVGG, self).__init__()
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(3, self.in_planes, stride=2)
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(RepVGGBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def create_feature_extractor(self, return_nodes: list[str] | dict[str, str]) -> torch.fx.GraphModule:
        return create_feature_extractor(self, return_nodes=return_nodes)

    def get_graph_node_names(self) -> list[str]:
        train_nodes, eval_nodes = get_graph_node_names(self)
        return eval_nodes

    def preprocess(self, x):
        if x.ndim == 4:
            return (x - self.mean) / self.std
        else:
            return (x - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)


class RepVGG_B1(RepVGG):
    def __init__(self):
        super().__init__(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], num_classes=1000)

    @classmethod
    def from_pretrained(cls, map_location="cpu"):
        url = cls.CHECKPOINT_URL
        model = cls()
        model.eval()
        state_dict = torch.hub.load_state_dict_from_url(url, weights_only=True, map_location=map_location)
        model.load_state_dict(state_dict)
        return model

    CHECKPOINT_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/RepVGG-B1-deploy.pth"
    FEATURE_NODES = {
        # 1/2
        "stage0.nonlinearity": "l2",
        # 1/4
        "stage1.3.nonlinearity": "l4",
        # 1/8
        "stage2.5.nonlinearity": "l8",
        # 1/16
        "stage3.7.nonlinearity": "l16_h",
        # too close to "stage4.0.nonlinearity"
        "stage3.15.nonlinearity": "l16",
        # 1/32
        "stage4.0.nonlinearity": "l32",
    }
    FEATURE_CHANNELS = dict(
        l2=64,
        l4=128,
        l8=256,
        l16_h=512,
        l16=512,
        l32=2048,
    )


def _test_model():
    import argparse

    import torchvision.io as IO
    import torchvision.transforms as transforms

    from imagenet.class_names import CLASS_LABELS

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image")
    args = parser.parse_args()

    model = RepVGG_B1.from_pretrained()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    x = IO.read_image(args.input) / 255
    x = preprocess(x)
    x = x.unsqueeze(0)

    model = model.cuda()
    x = x.cuda()

    with torch.no_grad():
        x = model.preprocess(x)
        output = model(x).squeeze(0)

    prob = torch.nn.functional.softmax(output, dim=0)
    top_probs, top_ids = torch.topk(prob, 5)

    for i in range(5):
        cid = top_ids[i].item()
        prob = top_probs[i].item()
        print(f"{i + 1}: {CLASS_LABELS[cid]}({cid}):  {prob}")


def _test_features():
    from pprint import pprint

    model = RepVGG_B1.from_pretrained().cuda()
    print(model)
    feature_extractor = model.create_feature_extractor(RepVGG_B1.FEATURE_NODES)
    x = torch.rand((4, 3, 224, 224)).cuda()
    features = feature_extractor(x)
    print("** features, len=", len(features))
    for name, vec in features.items():
        print(name, vec.shape)

    print("** get_graph_node_names")
    pprint(model.get_graph_node_names())


if __name__ == "__main__":
    # _test_features()
    _test_model()
