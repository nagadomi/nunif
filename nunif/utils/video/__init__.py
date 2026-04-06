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
