import torch


if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}", torch.cuda.get_device_name(i))
if torch.mps.is_available():
    for i in range(torch.mps.device_count()):
        print(f"mps:{i}", torch.mps.get_device_name(i))
if hasattr(torch, "xpu") and torch.xpu.is_available():
    for i in range(torch.xpu.device_count()):
        print(f"xpu:{i}", torch.xpu.get_device_name(i))
