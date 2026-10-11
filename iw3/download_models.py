from .base_depth_model import BaseDepthModel
from .depth_model_factory import create_depth_model


def main():
    for cls in BaseDepthModel.get_registered_classes():
        cls.force_update()
    if not BaseDepthModel.has_model_checkpoint_file("ZoeD_Any_N") and not BaseDepthModel.has_model_checkpoint_file(
        "ZoeD_N"
    ):
        create_depth_model("ZoeD_Any_N").load(gpu=-1)


if __name__ == "__main__":
    main()
