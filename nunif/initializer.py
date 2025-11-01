import os
import torch
import random
import numpy as np
import secrets
import gc


def disable_image_lib_threads():
    # Disable OpenMP
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['OMP_THREAD_LIMIT'] = '1'

    # Disable ImageMagick's Threading
    os.environ['MAGICK_THREAD_LIMIT'] = '1'
    try:
        from wand.resource import limits
        limits["thread"] = 1
    except ImportError:
        pass

    # Disable OpenCV's Threading/OpenCL
    try:
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except ImportError:
        pass


def set_seed(seed):
    if seed is None or seed < 0:
        seed = secrets.randbelow(1_000_000_000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def gc_collect():
    gc.collect()

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
        torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
