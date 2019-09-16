import os

def global_initialize():
    # Disable OpenMP
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OMP_THREAD_LIMIT'] = '1'

    # Disable ImageMagick's Threading
    os.environ['MAGICK_THREAD_LIMIT'] = '1'

    # Disable OpenCV's Threading/OpenCL
    try:
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except:
        pass

global_initialize()
