import argparse
from nunif.addon import load_addons
from multiprocessing import cpu_count
from nunif.initializer import set_seed, disable_image_lib_threads


def add_default_options(parser):
    num_workers = cpu_count() - 1
    if not num_workers > 0:
        num_workers = cpu_count
    parser.add_argument("--arch", type=str, required=True,
                        help="model arch")
    parser.add_argument("--data-dir", "-i", type=str, required=True,
                        help="training_data dir")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="model/checkpoint data dir")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="minibatch_size")
    parser.add_argument("--num-workers", type=int, default=num_workers,
                        help="number of worker process for data loading")
    parser.add_argument("--max-epoch", type=int, default=200,
                        help="max epoch")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0],
                        help="device ids. -1 for CPU")
    parser.add_argument("--learning-rate", type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument("--learning-rate-decay", type=float, default=0.995,
                        help="learning rate decay")
    parser.add_argument("--learning-rate-decay-step", type=int, nargs="+", default=[1],
                        help="learning rate decay step")
    parser.add_argument("--amp", action="store_true", help="with AMP")
    parser.add_argument("--resume", action="store_true", help="resume training from the latest checkpoint")
    parser.add_argument("--reset-state", action="store_true", help="reset optimizer,scheduler states for --resume")
    parser.add_argument("--seed", type=int, default=71, help="random seed")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="task", required=True)
    for addon in load_addons():
        subparser = addon.register_train(subparsers)
        if subparser is not None:
            add_default_options(subparser)
    args = parser.parse_args()

    disable_image_lib_threads()
    set_seed(args.seed)

    assert(args.handler is not None)
    print(vars(args))
    args.handler(args)


if __name__ == "__main__":
    main()
