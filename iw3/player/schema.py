from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ItemType(str, Enum):
    DIRECTORY = "directory"
    IMAGE = "image"
    VIDEO = "video"
    ARCHIVE = "archive"


class MediaItem(BaseModel):
    path: str
    name: str
    type: ItemType
    size: Optional[int] = None
    mtime: int
    stereo_format: Optional[str] = None


class StreamSettingsUpdate(BaseModel):
    divergence: Optional[float] = None
    convergence: Optional[float] = None
    ipd_offset: Optional[float] = None
    foreground_scale: Optional[float] = None
    edge_dilation: Optional[int] = None
    ema_normalize: Optional[bool] = None
    ema_decay: Optional[float] = None
    depth_aa: Optional[bool] = None
    preserve_screen_border: Optional[bool] = None
    synthetic_view: Optional[str] = None
    stream_fps: Optional[int] = None
    stream_quality: Optional[int] = None
