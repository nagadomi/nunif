from nunif.models import load_model, save_model
from nunif.addon import load_addons
import os
import argparse


def main():
    load_addons()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="input model file or dir")
    parser.add_argument("--output", "-o", type=str, required=True, help="output model file or dir")

    args = parser.parse_args()
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for fn in os.listdir(args.input):
            if fn.endswith(".pth"):
                model, _ = load_model(os.path.join(args.input, fn))
                save_model(model, os.path.join(args.output, fn))
    else:
        model, _ = load_model(args.input)
        save_model(model, args.output)


if __name__ == "__main__":
    main()
