# reflection version of replication_pad2d_naive
import torch
import torch.nn as nn


def reflection_pad2d_naive(x, padding, detach=False):
    assert x.ndim == 4 and len(padding) == 4
    # TODO: over 2x size support
    assert padding[0] < x.shape[3] and padding[1] < x.shape[3]
    assert padding[2] < x.shape[2] and padding[3] < x.shape[2]
    left, right, top, bottom = padding

    detach_fn = lambda t: t.detach() if detach else t
    if left > 0:
        x = torch.cat((torch.flip(detach_fn(x[:, :, :, 1:left + 1]), dims=[3]), x), dim=3)
    elif left < 0:
        x = x[:, :, :, -left:]
    if right > 0:
        x = torch.cat((x, torch.flip(detach_fn(x[:, :, :, -right - 1:-1]), dims=[3])), dim=3)
    elif right < 0:
        x = x[:, :, :, :right]
    if top > 0:
        x = torch.cat((torch.flip(detach_fn(x[:, :, 1:top + 1, :]), dims=[2]), x), dim=2)
    elif top < 0:
        x = x[:, :, -top:, :]
    if bottom > 0:
        x = torch.cat((x, torch.flip(detach_fn(x[:, :, -bottom - 1:-1, :]), dims=[2])), dim=2)
    elif bottom < 0:
        x = x[:, :, :bottom, :]

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


if __name__ == "__main__":
    _test()
    _test_grad()
    # _test_vis()
