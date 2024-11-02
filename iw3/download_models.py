from . import zoedepth_model as ZU
from . import depth_anything_model as DU
from . import depth_pro_model as PU


def main():
    ZU.force_update()
    DU.force_update()
    PU.force_update()
    if not ZU.has_model():
        ZU.load_model(gpu=-1)


if __name__ == "__main__":
    main()
