import torch
from .. device import mps_is_available, xpu_is_available


if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}", torch.cuda.get_device_name(i))

if mps_is_available():
    if hasattr(torch.mps, "device_count"):
        for i in range(torch.mps.device_count()):
            print(f"mps:{i}")
    else:
        print("mps")

if xpu_is_available():
    for i in range(torch.xpu.device_count()):
        print(f"xpu:{i}", torch.xpu.get_device_name(i))
