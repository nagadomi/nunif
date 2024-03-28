import yaml


FILENAME = "iw3_export.yml"
RGB_DIR = "rgb"
DEPTH_DIR = "depth"
AUDIO_FILE = "audio.m4a"
IMAGE_TYPE = "images"
VIDEO_TYPE = "video"


class ExportConfig:
    def __init__(self, type, fps=None,
                 invert=False, mapper=None,
                 skip_mapper=None, skip_edge_dilation=None,
                 rgb_dir=None, depth_dir=None, audio_file=None,
                 user_data={}):
        assert type in {IMAGE_TYPE, VIDEO_TYPE}
        self.type = type
        self.fps = fps
        self.invert = invert
        self.mapper = mapper
        self.skip_mapper = skip_mapper
        self.skip_edge_dilation = skip_edge_dilation
        self.rgb_dir = rgb_dir or RGB_DIR
        self.depth_dir = depth_dir or DEPTH_DIR
        self.audio_file = audio_file or AUDIO_FILE
        self.user_data = user_data

    def save(self, file_path):
        config = {
            "type": self.type,
            "fps": float(self.fps),
        }
        if self.invert is not None:
            config.update({"invert": self.invert})
        if self.mapper is not None:
            config.update({"mapper": self.mapper})
        if self.skip_mapper is not None:
            config.update({"skip_mapper": self.skip_mapper})
        if self.skip_edge_dilation is not None:
            config.update({"skip_edge_dilation": self.skip_edge_dilation})
        if self.audio_file:
            config.update({"audio_file": self.audio_file})
        config.update({
            "rgb_dir": self.rgb_dir,
            "depth_dir": self.depth_dir,
            "user_data": self.user_data
        })
        with open(file_path, mode="w", encoding="utf-8") as f:
            yaml.dump(config, f, encoding="utf-8", default_flow_style=False, sort_keys=False)

    @staticmethod
    def load(file_path):
        with open(file_path, mode="r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        type = config.get("type")
        fps = config.get("fps")
        invert = config.get("invert", False)
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
        invert = bool(invert)
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

        return ExportConfig(type, fps=fps, invert=invert,
                            mapper=mapper, skip_mapper=skip_mapper, skip_edge_dilation=skip_edge_dilation,
                            rgb_dir=rgb_dir, depth_dir=depth_dir, audio_file=audio_file,
                            user_data=user_data)

    def to_dict(self):
        return dict(
            type=self.type,
            fps=self.fps,
            invert=self.invert,
            mapper=self.mapper,
            skip_mapper=self.skip_mapper,
            skip_edge_dilation=self.skip_edge_dilation,
            rgb_dir=self.rgb_dir,
            depth_dir=self.depth_dir,
            audio_file=self.audio_file,
            user_data=self.user_data,
        )

    def __repr__(self):
        return repr(self.to_dict())
