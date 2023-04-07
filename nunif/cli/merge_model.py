# merge 2 models
import os
from os import path
import argparse
from .. models import load_model, save_model
from .. models.utils import merge_state_dict
from .. addon import load_addons


def main():
    load_addons()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="input model files")
    parser.add_argument("--output", "-o", type=str, required=True, help="output model file")
    parser.add_argument("--weight", "-w", type=float, help="blend weight for input[0]")

    args = parser.parse_args()
    assert len(args.input) == 2
    if args.weight is not None:
        assert 0.0 < args.weight and args.weight < 1.0

    weight = args.weight if args.weight is not None else 0.5

    os.makedirs(path.dirname(args.output), exist_ok=True)

    a, _ = load_model(args.input[0])
    b, _ = load_model(args.input[1])
    state_dict = merge_state_dict(a.state_dict(), b.state_dict(), weight)
    a.load_state_dict(state_dict)

    save_model(a, args.output)


if __name__ == "__main__":
    main()
