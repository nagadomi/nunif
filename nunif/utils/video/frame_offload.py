import contextlib
import time
from abc import ABC, abstractmethod
from typing import Any

import torch


class TensorPacker(ABC):
    @abstractmethod
    def pack(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def unpack(self, x: torch.Tensor, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
        pass


class Float32ToUInt8(TensorPacker):
    def pack(self, x):
        offload_x = (x * 255).round().clamp(0, 255).to(torch.uint8)
        return offload_x

    @classmethod
    def unpack(self, x, device, non_blocking=True):
        return x.to(device, non_blocking=non_blocking) / 255.0


class Float32ToUInt16(TensorPacker):
    def pack(self, x):
        offload_x = (x * 65535.0).round().clamp(0, 65535).to(torch.uint16)
        return offload_x

    def unpack(self, x, device, non_blocking=True):
        return x.to(device, non_blocking=non_blocking) / 65535.0


class Float32ToFloat16(TensorPacker):
    def pack(self, x):
        offload_x = x.to(torch.float16)
        return offload_x

    def unpack(self, x, device, non_blocking=True):
        return x.to(device=device, non_blocking=non_blocking).to(torch.float32)


class Float32ToFloat32(TensorPacker):
    def pack(self, x):
        return x

    def unpack(self, x, device, non_blocking=True):
        return x.to(device=device, non_blocking=non_blocking)


class OffloadResourceManager:
    """
    NOTE: Since `torch.empty(..., pin_memory=True)` itself is very slow, use this instead.
    """

    def __init__(self, size: torch.Size, device: torch.device, dtype: torch.dtype, preallocate=0):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.offload_device = torch.device("cpu")
        self.pin_memory = device.type in {"cuda", "xpu"}
        self.free_buffers = []
        self.busy_buffers: dict[int, torch.Tensor] = {}
        self.free_events = []
        self.busy_events: dict[int, Any] = {}

        for _ in range(preallocate):
            buffer = torch.empty(self.size, dtype=self.dtype, device=self.offload_device, pin_memory=self.pin_memory)
            event = self._create_event()
            self.free_buffers.append(buffer)
            if event is not None:
                self.free_events.append(event)

    def alloc_buffer(self):
        if not self.free_buffers:
            buffer = torch.empty(self.size, dtype=self.dtype, device=self.offload_device, pin_memory=self.pin_memory)
        else:
            buffer = self.free_buffers.pop()
        self.busy_buffers[id(buffer)] = buffer
        return buffer

    def free_buffer(self, buffer):
        buffer_id = id(buffer)
        if buffer_id in self.busy_buffers:
            del self.busy_buffers[buffer_id]
            self.free_buffers.append(buffer)
        else:
            raise ValueError("buffer is not a manager's resource")

    def alloc_event(self):
        if not self.free_events:
            event = self._create_event()
        else:
            event = self.free_events.pop()
        if event is not None:
            self.busy_events[id(event)] = event
        return event

    def free_event(self, event):
        if event is None:
            return

        event_id = id(event)
        if event_id in self.busy_events:
            del self.busy_events[event_id]
            self.free_events.append(event)
        else:
            raise ValueError("event is not a manager's resource")

    def _create_event(self) -> Any:
        if self.device.type == "cuda":
            return torch.cuda.Event()
        elif self.device.type == "xpu":
            return torch.xpu.Event()

        return None

    def __del__(self):
        self.free_buffers.clear()
        self.busy_buffers.clear()
        self.free_events.clear()
        self.busy_events.clear()


class OffloadedFrame:
    def __init__(self, x: torch.Tensor, dtype: torch.dtype, manager: OffloadResourceManager, stream=None) -> None:
        assert dtype in {torch.uint8, torch.uint16, torch.float16, torch.float32}
        assert torch.is_floating_point(x)
        self.manager = manager
        self.offload_event = self.manager.alloc_event()
        self.load_event = self.manager.alloc_event()
        self.stream = stream
        self.packer = self._get_packer(dtype)
        self.loadded_frame: torch.Tensor | None = None
        self.disposed: bool = False

        with self.stream_context(), torch.no_grad():
            assert manager.size == x.shape
            assert manager.dtype == dtype
            self.buffer = manager.alloc_buffer()
            offload_x = self.packer.pack(x)
            if self.offload_event is not None:
                self.buffer.copy_(offload_x, non_blocking=True)
                self.offload_event.record()
            else:
                self.buffer.copy_(offload_x)

    def prefetch(self, device: torch.device | str) -> None:
        if self.loadded_frame is not None:
            return

        device = torch.device(device)
        if self.offload_event is not None:
            self.offload_event.synchronize()
        with self.stream_context(), torch.no_grad():
            if self.load_event is None:
                self.loadded_frame = self.packer.unpack(self.buffer, device, non_blocking=False)
            else:
                self.loadded_frame = self.packer.unpack(self.buffer, device, non_blocking=True)
                self.load_event.record()

    def load(self, device: torch.device | str) -> torch.Tensor:
        assert not self.disposed

        self.prefetch(device)
        if self.load_event is not None:
            self.load_event.synchronize()

        assert self.loadded_frame is not None
        x = self.loadded_frame
        self.dispose()

        return x

    def dispose(self):
        assert not self.disposed
        self.manager.free_buffer(self.buffer)
        self.manager.free_event(self.load_event)
        self.manager.free_event(self.offload_event)
        self.buffer = None
        self.load_event = None
        self.offload_event_ = None
        self.loadded_frame = None
        self.disposed = True

    def cpu_buffer(self) -> torch.Tensor:
        assert not self.disposed
        if self.offload_event is not None:
            self.offload_event.synchronize()
        return self.buffer

    def stream_context(self):
        return self.stream if self.stream is not None else contextlib.nullcontext()

    @staticmethod
    def _get_event(device: torch.device) -> Any:
        if device.type == "cuda":
            return torch.cuda.Event()
        elif device.type == "xpu":
            return torch.xpu.Event()

        return None

    @staticmethod
    def _get_packer(dtype: torch.dtype) -> TensorPacker:
        if dtype == torch.uint8:
            return Float32ToUInt8()
        elif dtype == torch.uint16:
            return Float32ToUInt16()
        elif dtype == torch.float16:
            return Float32ToFloat16()
        elif dtype == torch.float32:
            return Float32ToFloat32()
        else:
            raise ValueError("dtype")


def _test():
    offload_stream = torch.cuda.Stream()
    dtype = torch.uint8
    device = torch.device("cpu")
    B = 4
    N = 10
    x = torch.ones((B, 2160, 3840, 3)).cuda()
    # x = torch.ones((B, 1080, 1920, 3)).cuda()
    torch.cuda.synchronize()
    t = time.perf_counter()
    manager = OffloadResourceManager(x[0].shape, dtype=dtype, device=device)
    for _ in range(N):
        results = []
        for i in range(x.shape[0]):
            t2 = time.perf_counter()
            of = OffloadedFrame(x[i], dtype, manager=manager, stream=offload_stream)
            print("W", time.perf_counter() - t2)
            x[i] = x[i] * x[i] * x[i]
            results.append(of)
        for i in range(len(results)):
            t2 = time.perf_counter()
            res = results.pop(0)
            if results:
                results[0].prefetch(device)
            xx = res.load(device)
            assert torch.all(xx == 1.0)
            xx = xx * xx * xx
            torch.cuda.synchronize()
            print("R", time.perf_counter() - t2)

    torch.cuda.synchronize()
    print(1.0 / ((time.perf_counter() - t) / (N * B)))


if __name__ == "__main__":
    _test()
