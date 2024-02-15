import argparse
from .. models import load_model
from .. addon import load_addons


def main():
    load_addons()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="input model file")

    args = parser.parse_args()
    model, _ = load_model(args.input)
    trainable_count = 0
    count = 0
    model.train()
    for p in model.parameters():
        if p.requires_grad:
            trainable_count += p.numel()
        count += p.numel()
    print(f"parameter count: {count:,}")
    print(f"trainable count: {trainable_count:,}")


if __name__ == "__main__":
    main()
