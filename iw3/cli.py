from .utils import create_parser, iw3_main
from . import models # noqa


def main():
    parser = create_parser()
    args = parser.parse_args()
    iw3_main(args)


if __name__ == "__main__":
    main()
