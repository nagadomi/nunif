from typing import Any

import torch


class OffloadFrame:
    def __init__(self, x: torch.Tensor, dtype: torch.dtype) -> None:
        assert dtype in {torch.uint8, torch.uint16}
        assert torch.is_floating_point(x)
        self.buffer = torch.empty(x.shape, dtype=dtype, device="cpu", pin_memory=torch.cuda.is_available())
        self.scale = torch.iinfo(dtype).max
        self.event = self._get_event(x.device)

        with torch.no_grad():
            offload_x = (x * self.scale).round().clamp(0, self.scale).to(dtype)

        if self.event is not None:
            self.buffer.copy_(offload_x, non_blocking=True)
            self.event.record()
        else:
            self.buffer.copy_(offload_x)

    def load(self, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        if self.event is not None:
            self.event.synchronize()

        with torch.no_grad():
            if device.type == "cpu":
                x = self.buffer.to(device) / float(self.scale)
            else:
                # .float() is required to workaround a bug causing uint16 division in MPS
                x = self.buffer.to(device, non_blocking=True).float() / float(self.scale)

        return x

    def cpu_buffer(self) -> torch.Tensor:
        if self.event is not None:
            self.event.synchronize()
        return self.buffer

    @staticmethod
    def _get_event(device: torch.device) -> Any:
        if device.type == "cuda":
            return torch.cuda.Event()
        elif device.type == "xpu":
            return torch.xpu.Event()

        return None


def _test():
    import time

    x = torch.ones((32, 1080, 1920, 3)).cuda()
    torch.cuda.synchronize()
    t = time.perf_counter()
    results = []
    for _ in range(10):
        for i in range(x.shape[0]):
            of = OffloadFrame(x[i], torch.uint8)
            x[i] = x[i] * x[i] * x[i]
            results.append(of)
    for res in results:
        assert torch.all(res.load(torch.device("cuda")) == 1.0)
    torch.cuda.synchronize()
    print((time.perf_counter() - t))


if __name__ == "__main__":
    _test()
