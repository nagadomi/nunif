import argparse
import os
from pprint import pprint

from nunif.addon import load_addons
from nunif.initializer import init_torch_env
from nunif.training.trainer import create_trainer_default_parser


def main():
    init_torch_env()
    os.environ["NUNIF_TRAIN"] = "1"
    import torch

    torch._dynamo.config.recompile_limit = 32

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="task", required=True)
    for addon in load_addons():
        addon.register_train(subparsers, create_trainer_default_parser())

    args = parser.parse_args()
    assert args.handler is not None

    pprint(vars(args))
    try:
        args.handler(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
