import argparse
from nunif.addon import load_addons


def add_default_options(parser):
    subparser.add_argument("--dataset-dir", "-i", type=str, required=True, help="input dataset dir")
    subparser.add_argument("--data-dir", "-o", type=str, required=True, help="output data dir")
    

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="task", required=True)
    for addon in load_addons():
        subparser = addon.register_build_training_data(subparsers)
        if subparser is not None:
            add_default_options(subparser)
    args = parser.parse_args()
    assert(args.handler is not None)

    args.handler(args)


if __name__ == "__main__":
    main()
