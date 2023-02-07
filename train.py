import argparse
from pprint import pprint
from nunif.addon import load_addons
from nunif.initializer import set_seed, disable_image_lib_threads
from nunif.training.trainer import create_trainer_default_parser


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="task", required=True)
    for addon in load_addons():
        subparser = addon.register_train(subparsers, create_trainer_default_parser())

    args = parser.parse_args()
    assert (args.handler is not None)

    pprint(vars(args))
    args.handler(args)


if __name__ == "__main__":
    main()
