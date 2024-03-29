from os import path
import yaml
from datetime import datetime


FILENAME = "iw3_export.yml"
RGB_DIR = "rgb"
DEPTH_DIR = "depth"
AUDIO_FILE = "audio.m4a"
IMAGE_TYPE = "images"
VIDEO_TYPE = "video"


class ExportConfig:
    def __init__(self, type, basename=None, fps=None,
                 mapper=None, skip_mapper=None, skip_edge_dilation=None,
                 rgb_dir=None, depth_dir=None, audio_file=None,
                 user_data={}, updated_at=None):
        assert type in {IMAGE_TYPE, VIDEO_TYPE}
        self.type = type
        self.basename = basename
        self.fps = fps
        self.mapper = mapper
        self.skip_mapper = skip_mapper
        self.skip_edge_dilation = skip_edge_dilation
        self.rgb_dir = rgb_dir or RGB_DIR
        self.depth_dir = depth_dir or DEPTH_DIR
        self.audio_file = audio_file or AUDIO_FILE
        self.user_data = user_data
        self.updated_at = updated_at

    def save(self, file_path):
        config = {
            "type": self.type,
        }
        if self.basename:
            config.update({"basename": self.basename})
        config.update({"fps": float(self.fps)})
        config.update({
            "rgb_dir": self.rgb_dir,
            "depth_dir": self.depth_dir
        })
        if self.audio_file:
            config.update({"audio_file": self.audio_file})
        if self.mapper is not None:
            config.update({"mapper": self.mapper})
        if self.skip_mapper is not None:
            config.update({"skip_mapper": self.skip_mapper})
        if self.skip_edge_dilation is not None:
            config.update({"skip_edge_dilation": self.skip_edge_dilation})
        config.update({"updated_at": datetime.now().isoformat()})  # local time
        config.update({"user_data": self.user_data})

        with open(file_path, mode="w", encoding="utf-8") as f:
            yaml.dump(config, f, encoding="utf-8", default_flow_style=False, sort_keys=False)

    @staticmethod
    def load(file_path):
        with open(file_path, mode="r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        type = config.get("type")
        basename = config.get("basename")
        fps = config.get("fps")
        mapper = config.get("mapper", "none")
        user_data = config.get("user_data", {})
        if type not in {IMAGE_TYPE, VIDEO_TYPE}:
            raise ValueError(f"Unsupported type={type} in {file_path}")
        if type == "video":
            try:
                fps = float(fps)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid fps={fps} in in {file_path}")
        else:
            fps = None
        # TODO: Should support formula, but need to use a security-safe eval
        mapper = str(mapper)
        skip_mapper = config.get("skip_mapper", False)
        skip_edge_dilation = config.get("skip_edge_dilation", False)
        rgb_dir = config.get("rgb_dir")
        depth_dir = config.get("depth_dir")
        if type == VIDEO_TYPE:
            audio_file = config.get("audio_file")
        else:
            audio_file = None
        depth_dir = config.get("depth_dir")
        updated_at = config.get("updated_at", None)
        if updated_at is not None:
            try:
                updated_at = datetime.fromisoformat(updated_at)
            except (TypeError, ValueError):
                updated_at = None

        return ExportConfig(type, basename=basename, fps=fps,
                            mapper=mapper, skip_mapper=skip_mapper, skip_edge_dilation=skip_edge_dilation,
                            rgb_dir=rgb_dir, depth_dir=depth_dir, audio_file=audio_file,
                            user_data=user_data, updated_at=updated_at)

    def to_dict(self):
        return dict(
            type=self.type,
            basename=self.basename,
            fps=self.fps,
            mapper=self.mapper,
            skip_mapper=self.skip_mapper,
            skip_edge_dilation=self.skip_edge_dilation,
            rgb_dir=self.rgb_dir,
            depth_dir=self.depth_dir,
            audio_file=self.audio_file,
            user_data=self.user_data,
            updated_at=self.updated_at,
        )

    def __repr__(self):
        return repr(self.to_dict())

    def resolve_paths(self, base_dir):
        rgb_dir = resolve_path(base_dir, self.rgb_dir or "rgb_dir")
        depth_dir = resolve_path(base_dir, self.depth_dir or "depth_dir")
        audio_file = resolve_path(base_dir, self.audio_file)
        if not path.exists(rgb_dir):
            raise ValueError(f"ExportConfig: rgb_dir={rgb_dir} not found")
        if not path.exists(depth_dir):
            raise ValueError(f"ExportConfig: depth_dir={depth_dir} not found")
        if self.type == VIDEO_TYPE:
            if audio_file is not None and not path.exists(audio_file):
                audio_file = None
        else:
            audio_file = None
        return rgb_dir, depth_dir, audio_file


def resolve_path(base_dir, entry):
    if entry is None:
        return None
    if path.isabs(entry):
        return entry
    else:
        return path.join(base_dir, entry)
