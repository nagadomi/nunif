import contextlib
import torch


def mps_is_available():
    return torch.backends.mps.is_available()


def xpu_is_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def create_device_name(device_id):
    if isinstance(device_id, (list, tuple)):
        assert len(device_id) > 0
        device_id = device_id[0]
    if device_id < 0:
        device_name = "cpu"
    else:
        if torch.cuda.is_available():
            device_name = 'cuda:%d' % device_id
        elif mps_is_available():
            device_name = 'mps:%d' % device_id
        elif xpu_is_available():
            device_name = 'xpu:%d' % device_id
        else:
            raise ValueError("No cuda/mps/xpu available. Use `--gpu -1` for CPU.")

    return device_name


def create_device(device_id):
    return torch.device(create_device_name(device_id))


def device_is(device, name):
    if isinstance(device, torch.device):
        return device.type == name
    else:
        return name in str(device)


def device_is_mps(device):
    return device_is(device, "mps")


def device_is_xpu(device):
    return device_is(device, "xpu")


def device_is_cpu(device):
    return device_is(device, "cpu")


def device_is_cuda(device):
    return device_is(device, "cuda")


def autocast(device, dtype=None, enabled=True):
    if device_is_cpu(device):
        # autocast on cpu is extremely slow for unknown reasons
        # disabled
        amp_device_type = "cpu"
        amp_dtype = torch.bfloat16
        enabled = False
    elif device_is_mps(device):
        # TODO: Test on macOS
        amp_device_type = "mps"
        amp_dtype = torch.float16
        enabled = False
    else:
        # For CUDA, MPS, XPU, and unknown devices, this passes through to the standard PyTorch implementation.
        amp_device_type = device.split(":")[0] if isinstance(device, str) else device.type
        amp_dtype = dtype

    return torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=enabled)


class DummyStream():
    def __init__(self, device=None, priority=0, **kwargs):
        pass

    def synchronize(self):
        pass

    def wait_stream(self, stream):
        pass

    def wait_event(self, event):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def create_stream(device):
    if device_is_cuda(device):
        return torch.cuda.Stream(device)
    elif device_is_xpu(device):
        return torch.xpu.Stream(device)
    else:
        return DummyStream()


def get_current_stream(device):
    if torch.cuda.is_available():
        return torch.cuda.current_stream()
    elif device_is_xpu(device):
        return torch.xpu.current_stream()
    else:
        return DummyStream()


def device_context(device):
    if device_is_cuda(device):
        return torch.cuda.device(device)
    elif device_is_xpu(device):
        return torch.xpu.device(device)
    else:
        return contextlib.nullcontext()
