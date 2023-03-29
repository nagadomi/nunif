import json
import os
from os import path
import torch
import argparse


DEFAULT_INPUT_DIR = path.relpath(path.join(path.dirname(__file__), "..", "..",
                                           "tmp", "search_qtable"))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", "-i", type=str, default=DEFAULT_INPUT_DIR, 
                        help="input dir")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output qtables.pth")
    args = parser.parse_args()

    qtables = []
    for fn in os.listdir(args.input_dir):
        if not fn.endswith(".json"):
            continue
        with open(path.join(args.input_dir, fn), mode="r", encoding="utf-8") as f:
            qtable = json.load(f)
            qtable = {int(k): v for k, v in qtable.items()}
            if qtable not in qtables:
                qtables.append(qtable)

    torch.save(qtables, args.output)
    print("qtables", len(qtables), "save", args.output)


if __name__ == "__main__":
    main()

