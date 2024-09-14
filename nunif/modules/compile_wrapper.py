import torch
import sys
import os


_DISABLE_COMPILE = sys.platform != "linux" or os.getenv("NUNIF_DISABLE_COMPILE", False)


def compile(*args, **kwargs):
    if _DISABLE_COMPILE:
        return args[0]
    else:
        return torch.compile(*args, **kwargs)


def conditional_compile(env_name):
    def decorator(*args, **kwargs):
        env_names = env_name if isinstance(env_name, (list, tuple)) else [env_name]
        cond = any([int(os.getenv(name, "0")) for name in env_names])
        if not cond or _DISABLE_COMPILE:
            return args[0]
        else:
            return torch.compile(*args, **kwargs)
    return decorator
