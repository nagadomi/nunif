from . import depth_models  # noqa: F401
from .base_depth_model import BaseDepthModel


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
