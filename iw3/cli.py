from .utils import create_parser, set_state_args, iw3_main
from . import models # noqa


def main():
    parser = create_parser()
    args = parser.parse_args()
    set_state_args(args)
    iw3_main(args)


if __name__ == "__main__":
    main()
