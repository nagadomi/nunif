import argparse
from pprint import pprint
from nunif.addon import load_addons
from nunif.initializer import set_seed, disable_image_lib_threads


def create_default_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", "-i", type=str, required=True, help="input dataset directory")
    parser.add_argument("--data-dir", "-o", type=str, required=True, help="output training data directory")
    parser.add_argument("--seed", type=int, default=71, help="random seed")

    return parser


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="task", required=True)
    for addon in load_addons():
        subparser = addon.register_create_training_data(subparsers, create_default_parser())

    args = parser.parse_args()
    assert (args.handler is not None)

    disable_image_lib_threads()
    set_seed(args.seed)

    pprint(vars(args))
    args.handler(args)


if __name__ == "__main__":
    main()
