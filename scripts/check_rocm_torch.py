#!/usr/bin/env python3
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[FAIL] Could not import torch: {exc}")
        return 1

    print(f"torch: {torch.__version__}")
    hip_ver = getattr(torch.version, "hip", None)
    print(f"torch.version.hip: {hip_ver}")

    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() returned False")
        return 1

    try:
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(device)
        print(f"device: {name}")
        x = torch.randn((2, 2), device=device)
        y = x @ x
        print(f"sample matmul OK, mean={y.mean().item():.6f}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[FAIL] ROCm execution failed: {exc}")
        return 1

    print("[OK] ROCm/PyTorch looks usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
