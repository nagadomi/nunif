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
