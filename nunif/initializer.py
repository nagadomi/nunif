import os
import torch
import random
import numpy as np


def disable_image_lib_threads():
    # Disable OpenMP
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['OMP_THREAD_LIMIT'] = '1'

    # Disable ImageMagick's Threading
    os.environ['MAGICK_THREAD_LIMIT'] = '1'
    try:
        from wand.resource import limits
        limits["thread"] = 1
    except ModuleNotFoundError:
        pass

    # Disable OpenCV's Threading/OpenCL
    try:
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except ModuleNotFoundError:
        pass


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
