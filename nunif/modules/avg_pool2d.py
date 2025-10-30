import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAvgPool2d(nn.Module):
    # avg_pool2d using grouped conv2d.
    # faster than avg_pool2d when using torch.compile.
    def __init__(self, in_channels, kernel_size, stride=None, padding=0, count_include_pad=True, padding_mode="constant"):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.count_include_pad = count_include_pad
        self.count_pad_size = 1
        self.padding_mode = padding_mode
        in_channels = in_channels + self.count_pad_size if not count_include_pad else in_channels
        self.register_buffer("avg_kernel", self.gen_avg_kernel(in_channels, kernel_size), persistent=False)

    @staticmethod
    def gen_avg_kernel(in_channels, kernel_size):
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.reshape(1, 1, *kernel.shape)
        kernel = kernel.expand(in_channels, 1, *kernel.shape[2:]).contiguous()
        return kernel

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.count_include_pad:
            x = torch.cat([x, torch.ones((B, self.count_pad_size, H, W), dtype=x.dtype, device=x.device)], dim=1)
            padded_x = F.pad(x, (self.padding,) * 4, mode="constant", value=0)
            x = F.conv2d(padded_x, weight=self.avg_kernel, bias=None, stride=self.stride, groups=self.avg_kernel.shape[0])
            x, area = x[:, :-self.count_pad_size], x[:, -1:]
            return x / area
        else:
            padded_x = F.pad(x, (self.padding,) * 4, mode=self.padding_mode, value=0)
            x = F.conv2d(padded_x, weight=self.avg_kernel, bias=None, stride=self.stride, groups=self.avg_kernel.shape[0])
            return x


def _test():
    x = torch.rand((4, 3, 32, 32))
    conv_avg = ConvAvgPool2d(3, kernel_size=15, padding=7, stride=1, count_include_pad=False)
    z1 = F.avg_pool2d(x, kernel_size=15, padding=7, stride=1, count_include_pad=False)
    z2 = conv_avg(x)
    print((z1 - z2).abs().mean())

    conv_avg = ConvAvgPool2d(3, kernel_size=15, padding=7, stride=1, count_include_pad=True, padding_mode="constant")
    z1 = F.avg_pool2d(x, kernel_size=15, padding=7, stride=1, count_include_pad=True)
    z2 = conv_avg(x)
    print((z1 - z2).abs().mean())

    conv_avg = ConvAvgPool2d(3, kernel_size=4, padding=1, stride=2, count_include_pad=True, padding_mode="constant")
    z1 = F.avg_pool2d(x, kernel_size=4, padding=1, stride=2, count_include_pad=True)
    z2 = conv_avg(x)
    print((z1 - z2).abs().mean())


def _bench(in_channels, compile, kernel_size=7, count_include_pad=True):
    import time
    print(f"** in_channels={in_channels}, compile={compile}")

    K = kernel_size
    N = 100
    B = 4
    C = in_channels
    S = (B, C, 184, 184)
    device = "cuda:0"

    model = ConvAvgPool2d(
        in_channels=S[1],
        kernel_size=K, stride=1,
        padding=(K - 1) // 2,
        count_include_pad=count_include_pad,
    ).eval().to(device)
    avg_pool2d = nn.AvgPool2d(
        kernel_size=K,
        stride=1,
        padding=(K - 1) // 2,
        count_include_pad=count_include_pad,
    ).eval().to(device)
    if compile:
        model = torch.compile(model)
        avg_pool2d = torch.compile(avg_pool2d)
    x = torch.rand(S).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        model(x)
        avg_pool2d(x)

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            model(x)
    torch.cuda.synchronize()
    et = time.time() - t
    print("ConvAvgPool2d", et, 1 / (et / (B * N)), "FPS")

    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            avg_pool2d(x)
    torch.cuda.synchronize()
    et = time.time() - t
    print("nn.AvgPool2d", et, 1 / (et / (B * N)), "FPS")


def _bench_main():
    # nn.AvgPool2d is faster
    _bench(128, compile=False)
    # ConvAvgPool2d is faster
    _bench(128, compile=True)
    _bench(64, compile=True)
    _bench(256, compile=True)
    _bench(512, compile=True)
    _bench(1024, compile=True)


if __name__ == "__main__":
    _test()
    _bench_main()
