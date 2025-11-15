from .zoedepth_model import ZoeDepthModel
from .depth_anything_model import DepthAnythingModel
from .depth_pro_model import DepthProModel
from .video_depth_anything_model import VideoDepthAnythingModel
from .video_depth_anything_streaming_model import VideoDepthAnythingStreamingModel
from .depth_anything_v3_model import DepthAnythingV3MonoModel
from .null_depth_model import NullDepthModel


def create_depth_model(model_type):
    if ZoeDepthModel.supported(model_type):
        model = ZoeDepthModel(model_type)
        return model
    elif DepthAnythingModel.supported(model_type):
        model = DepthAnythingModel(model_type)
        return model
    elif DepthAnythingV3MonoModel.supported(model_type):
        model = DepthAnythingV3MonoModel(model_type)
        return model
    elif DepthProModel.supported(model_type):
        model = DepthProModel(model_type)
        return model
    elif VideoDepthAnythingModel.supported(model_type):
        model = VideoDepthAnythingModel(model_type)
        return model
    elif VideoDepthAnythingStreamingModel.supported(model_type):
        model = VideoDepthAnythingStreamingModel(model_type)
        return model
    elif NullDepthModel.supported(model_type):
        model = NullDepthModel(model_type)
        return model
    else:
        raise ValueError(f"{model_type} is not supported")
