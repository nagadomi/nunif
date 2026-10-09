# python -m nunif.cli.show_train_args -i models/photo_gan/*checkpoint.pth
import argparse
from pprint import pprint

import torch

from ..addon import load_addons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="input checkpoint files")
    args = parser.parse_args()

    load_addons()
    for checkpoint_file in sorted(args.input):
        print("----")
        print(checkpoint_file)
        meta = torch.load(checkpoint_file)
        meta = {k: meta.get(k, None) for k in ("name", "updated_at", "last_epoch", "train_kwargs")}
        pprint(meta, width=80, sort_dicts=False)


if __name__ == "__main__":
    main()
