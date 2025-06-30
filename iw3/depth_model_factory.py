from . zoedepth_model import ZoeDepthModel
from . depth_anything_model import DepthAnythingModel
from . depth_pro_model import DepthProModel
from . video_depth_anything_model import VideoDepthAnythingModel
from . null_depth_model import NullDepthModel


def create_depth_model(model_type):
    if ZoeDepthModel.supported(model_type):
        model = ZoeDepthModel(model_type)
        return model
    elif DepthAnythingModel.supported(model_type):
        model = DepthAnythingModel(model_type)
        return model
    elif DepthProModel.supported(model_type):
        model = DepthProModel(model_type)
        return model
    elif VideoDepthAnythingModel.supported(model_type):
        model = VideoDepthAnythingModel(model_type)
        return model
    elif NullDepthModel.supported(model_type):
        model = NullDepthModel(model_type)
        return model
    else:
        raise ValueError(f"{model_type} is not supported")
