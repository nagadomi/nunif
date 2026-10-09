from tests.shell_utils import run_module


def main() -> None:
    run_module("tests.smoke_data")
    run_module("tests.smoke_waifu2x")
    run_module("tests.smoke_stlizer")
    run_module("tests.smoke_iw3")


if __name__ == "__main__":
    main()
