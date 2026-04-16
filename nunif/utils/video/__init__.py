import io
import mimetypes
import os

import av

# NOTE: This appears to be required to avoid a deadlock with HWAccel and thread_type="AUTO".
av.logging.set_level(None)


# Add video mimetypes that does not exist in mimetypes
mimetypes.add_type("video/x-ms-asf", ".asf")
mimetypes.add_type("video/x-ms-vob", ".vob")
mimetypes.add_type("video/divx", ".divx")
mimetypes.add_type("video/3gpp", ".3gp")
mimetypes.add_type("video/ogg", ".ogv")
mimetypes.add_type("video/3gpp2", ".3g2")
mimetypes.add_type("video/m2ts", ".m2ts")
mimetypes.add_type("video/m2ts", ".m2t")
mimetypes.add_type("video/m2ts", ".mts")
mimetypes.add_type("video/m2ts", ".ts")
mimetypes.add_type("video/vnd.rn-realmedia", ".rm")  # fake
mimetypes.add_type("video/x-flv", ".flv")  # Not defined on Windows
mimetypes.add_type("video/x-matroska", ".mkv")  # May not be defined for some reason

# Hide libva message
os.environ["LIBVA_MESSAGING_LEVEL"] = os.environ.get("LIBVA_MESSAGING_LEVEL", "1")

from .processor import *  # noqa
from .color_transform import TensorFrame  # noqa


def pyav_init_cuda_primary_context(max_devices=16):
    """
    Initialize the CUDA context in hwcontext_cuda before PyTorch initializes CUDA.
    This is required to use the primary context.
    Otherwise, stream synchronization with NVDEC is not possible.
    """
    import numpy as np
    import torch

    from .hwaccel import HW_DEVICES, create_hwaccel

    if "cuda" not in HW_DEVICES:
        return

    try:
        output_buffer = io.BytesIO()
        with av.open(output_buffer, mode="w", format="mp4") as container:
            stream = container.add_stream("libopenh264", rate=1)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"

            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        for device_id in range(max_devices):
            output_buffer.seek(0)
            hwaccel = create_hwaccel(device="cuda", device_id=device_id, disable_software_fallback=True)
            with av.open(output_buffer, mode="r", hwaccel=hwaccel) as container:
                for packet in container.demux(container.streams.video[0]):
                    for frame in packet.decode():
                        pass
    except:  # noqa
        # import sys
        # print(sys.exc_info(), file=sys.stderr)
        pass

    if torch.cuda.is_available():
        torch.cuda.init()
