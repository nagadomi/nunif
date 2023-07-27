from .utils import load_depth_model, force_update_midas_model, force_update_zoedepth_model


def main():
    force_update_midas_model()
    force_update_zoedepth_model()
    load_depth_model(gpu=-1)


if __name__ == "__main__":
    main()
