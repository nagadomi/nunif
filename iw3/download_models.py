from .zoedepth_model import ZoeDepthModel
from .depth_anything_model import DepthAnythingModel
from .depth_pro_model import DepthProModel


def main():
    ZoeDepthModel.force_update()
    DepthAnythingModel.force_update()
    DepthProModel.force_update()
    if not ZoeDepthModel.has_checkpoint_file("ZoeD_N"):
        ZoeDepthModel("ZoeD_N").load(gpu=-1)


if __name__ == "__main__":
    main()
