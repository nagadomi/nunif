import argparse
from pprint import pprint
from nunif.addon import load_addons
from multiprocessing import cpu_count
from nunif.initializer import set_seed, disable_image_lib_threads


def create_default_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    num_workers = cpu_count() - 2
    if not num_workers > 0:
        num_workers = cpu_count

    parser.add_argument("--data-dir", "-i", type=str, required=True,
                        help="input training data directory that created by `create_training_data` command")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="output directory for trained model/checkpoint")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="minibatch size")
    parser.add_argument("--num-workers", type=int, default=num_workers,
                        help="number of worker processes for data loader")
    parser.add_argument("--max-epoch", type=int, default=200,
                        help="max epoch")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0],
                        help="device ids; if -1 is specified, use CPU")
    parser.add_argument("--learning-rate", type=float, default=0.00025,
                        help="learning rate")
    parser.add_argument("--learning-rate-decay", type=float, default=0.995,
                        help="learning rate decay")
    parser.add_argument("--learning-rate-decay-step", type=int, nargs="+", default=[1],
                        help="learning rate decay step; if multiple values are specified, use MultiStepLR")
    parser.add_argument("--disable-amp", action="store_true", help="disable AMP for some special reason")
    parser.add_argument("--resume", action="store_true", help="resume training from the latest checkpoint file")
    parser.add_argument("--reset-state", action="store_true", help="do not load best_score, optimizer and scheduler state when --resume")
    parser.add_argument("--seed", type=int, default=71, help="random seed")

    return parser


def main():
    default_parser = create_default_parser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="task", required=True)
    for addon in load_addons():
        subparser = addon.register_train(subparsers, default_parser)

    args = parser.parse_args()
    assert (args.handler is not None)

    disable_image_lib_threads()
    set_seed(args.seed)

    pprint(vars(args))
    args.handler(args)


if __name__ == "__main__":
    main()
