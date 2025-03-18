from .utils import (
    init_win32,
    create_parser, set_state_args,
    iw3_desktop_main,
)


def cli_main():
    init_win32()

    parser = create_parser()
    args = parser.parse_args()
    set_state_args(args)
    iw3_desktop_main(args, init_wxapp=True)


if __name__ == "__main__":
    cli_main()
