from .base_depth_model import BaseDepthModel
from .depth_anything_model import DepthAnythingModel  # noqa: F401
from .depth_anything_v3_model import DepthAnythingV3MonoModel  # noqa: F401
from .depth_pro_model import DepthProModel  # noqa: F401
from .null_depth_model import NullDepthModel  # noqa: F401
from .video_depth_anything_model import VideoDepthAnythingModel  # noqa: F401
from .video_depth_anything_streaming_model import VideoDepthAnythingStreamingModel  # noqa: F401
from .zoedepth_model import ZoeDepthModel  # noqa: F401


def create_depth_model(model_type: str) -> BaseDepthModel:
    cls = BaseDepthModel.get_depth_model_class(model_type)
    if cls is not None:
        return cls(model_type)
    raise ValueError(f"{model_type} is not supported")


def get_all_depth_models() -> list[str]:
    return BaseDepthModel.get_all_model_types()


def is_depth_aa_supported(model_type: str) -> bool:
    return BaseDepthModel.is_depth_aa_supported(model_type)


def is_resolution_change_supported(model_type: str) -> bool:
    return BaseDepthModel.is_resolution_change_supported(model_type)


def is_limit_resolution_supported(model_type: str) -> bool:
    return BaseDepthModel.is_limit_resolution_supported(model_type)
