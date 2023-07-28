from . import zoedepth_model as ZU


def main():
    ZU.force_update_midas()
    ZU.force_update_zoedepth()
    if not ZU.has_model():
        ZU.load_model(gpu=-1)


if __name__ == "__main__":
    main()
