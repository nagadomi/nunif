# reflection version of replication_pad2d_naive
import torch
import torch.nn as nn


def _detach_fn(x, flag: bool):
    if flag:
        return x.detach()
    else:
        return x


def reflection_pad2d_naive(x, padding: tuple[int, int, int, int], detach: bool = False):
    assert x.ndim == 4 and len(padding) == 4
    assert padding[0] <= x.shape[3] and padding[1] <= x.shape[3]
    assert padding[2] <= x.shape[2] and padding[3] <= x.shape[2]
    left, right, top, bottom = padding

    if left > 0 and right > 0:
        pad_l = torch.flip(_detach_fn(x[:, :, :, 1:left + 1], detach), dims=[3])
        pad_r = torch.flip(_detach_fn(x[:, :, :, -right - 1:-1], detach), dims=[3])
        x = torch.cat((pad_l, x, pad_r), dim=3)
    else:
        if left > 0:
            x = torch.cat((torch.flip(_detach_fn(x[:, :, :, 1:left + 1], detach), dims=[3]), x), dim=3)
        elif left < 0:
            x = x[:, :, :, -left:]
        if right > 0:
            x = torch.cat((x, torch.flip(_detach_fn(x[:, :, :, -right - 1:-1], detach), dims=[3])), dim=3)
        elif right < 0:
            x = x[:, :, :, :right]
    if top > 0 and bottom > 0:
        pad_t = torch.flip(_detach_fn(x[:, :, 1:top + 1, :], detach), dims=[2])
        pad_b = torch.flip(_detach_fn(x[:, :, -bottom - 1:-1, :], detach), dims=[2])
        x = torch.cat((pad_t, x, pad_b), dim=2)
    else:
        if top > 0:
            x = torch.cat((torch.flip(_detach_fn(x[:, :, 1:top + 1, :], detach), dims=[2]), x), dim=2)
        elif top < 0:
            x = x[:, :, -top:, :]
        if bottom > 0:
            x = torch.cat((x, torch.flip(_detach_fn(x[:, :, -bottom - 1:-1, :], detach), dims=[2])), dim=2)
        elif bottom < 0:
            x = x[:, :, :bottom, :]

    return x.contiguous()


def _loop_step(pad: int, base: int) -> tuple[int, int]:
    remain = 0
    if pad > (base - 1):
        remain = pad - (base - 1)
        pad = (base - 1)
    return pad, remain


def reflection_pad2d_loop(x, padding: tuple[int, int, int, int], detach: bool = False):
    # Limit one-step padding size to image size
    # For onnxruntime
    height, width = x.shape[2:]
    left, right, top, bottom = padding
    while left != 0 or right != 0 or top != 0 or bottom != 0:
        left_step, left = _loop_step(left, width)
        right_step, right = _loop_step(right, width)
        top_step, top = _loop_step(top, height)
        bottom_step, bottom = _loop_step(bottom, height)
        x = reflection_pad2d_naive(x, (left_step, right_step, top_step, bottom_step), detach=detach)
    return x


class ReflectionPad2dNaive(nn.Module):
    def __init__(self, padding, detach=False):
        super().__init__()
        assert isinstance(padding, (list, tuple)) and len(padding) == 4
        self.padding = padding
        self.detach = detach

    def forward(self, x):
        return reflection_pad2d_naive(x, self.padding, detach=self.detach)


def _test():
    padding = (1, 2, 3, 4)
    my_pad = ReflectionPad2dNaive(padding)
    nn_pad = nn.ReflectionPad2d(padding)

    x = torch.rand((4, 3, 8, 8))
    y1 = my_pad(x)
    y2 = nn_pad(x)
    assert (y1 - y2).abs().sum() == 0

    for _ in range(10):
        padding = torch.randint(-3, 3, (4,)).tolist()
        my_pad = ReflectionPad2dNaive(padding).cuda()
        nn_pad = nn.ReflectionPad2d(padding).cuda()
        x = torch.rand((4, 3, 8, 8)).cuda()
        y1 = my_pad(x)
        y2 = nn_pad(x)
        assert (y1 - y2).abs().sum() == 0


def _test_vis():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import time

    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)

    padding = [32] * 4
    my_pad = ReflectionPad2dNaive(padding)
    nn_pad = nn.ReflectionPad2d(padding)

    TF.to_pil_image(my_pad(x)[0]).show()
    time.sleep(1)
    TF.to_pil_image(nn_pad(x)[0]).show()


def _test_grad():
    # https://github.com/pytorch/pytorch/issues/68879
    conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
    x = torch.ones((1, 1, 6, 6), requires_grad=True)

    nn.init.constant_(conv.weight, 1)
    nn.init.constant_(conv.bias, 0)

    # all 1
    y = conv(reflection_pad2d_naive(x, (1,) * 4, detach=True)).sum()
    y.backward()
    print(x.grad)
    x.grad.zero_()

    # grad of the padded values is multiply computed
    y = conv(reflection_pad2d_naive(x, (1,) * 4, detach=False)).sum()
    y.backward()
    print(x.grad)


def _vis_loop():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF

    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)

    x = reflection_pad2d_loop(x, (640, -10, 320, -10))
    TF.to_pil_image(x[0]).show()


def _vis_loop2():
    import torchvision.transforms.functional as TF
    import time

    x = torch.zeros((1, 1, 4, 4))
    x[:, :, :, 0] = 1.0
    x[:, :, :, 2] = 1.0

    x = reflection_pad2d_loop(x, (10,) * 4)
    TF.to_pil_image(x[0]).show()
    time.sleep(2)

    x = torch.zeros((1, 1, 5, 5))
    x[:, :, :, 0] = 1.0
    x[:, :, :, 2] = 1.0
    x[:, :, :, 4] = 1.0

    x = reflection_pad2d_loop(x, (10,) * 4)
    TF.to_pil_image(x[0]).show()


def _test_loop():
    import torch.nn.functional as F
    for i in range(20):
        padding = (i, i, i, i)
        x = torch.rand((4, 3, 8, 8)).cuda()
        y1 = reflection_pad2d_loop(x, padding)
        y2 = F.pad(x, padding, mode="replicate")
        assert y1.shape == y2.shape


if __name__ == "__main__":
    _test()
    # _test_grad()
    # _test_vis()
    # _vis_loop()
    # _vis_loop2()
    # _test_loop()
