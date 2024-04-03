import torch
import matplotlib.pyplot as plt
import math


def softplus01_old(x, c):
    min_v = math.log(1 + math.exp(0 * 12.0 - c)) / (12 - c)
    max_v = math.log(1 + math.exp(1 * 12.0 - c)) / (12 - c)
    v = torch.log(1. + torch.exp(x * 12.0 - c)) / (12 - c)
    return (v - min_v) / (max_v - min_v)


def softplus01(x, bias, scale):
    min_v = math.log(1 + math.exp((0 - bias) * scale))
    max_v = math.log(1 + math.exp((1 - bias) * scale))
    v = torch.log(1. + torch.exp((x - bias) * scale))
    return (v - min_v) / (max_v - min_v)


def hardplus(x, scale):
    threshold = (1.0 - (1.0 / scale))
    index = threshold <= x
    y = torch.zeros_like(x)
    y[index] = (x[index] - threshold) * scale
    return y


def distance_to_dispary(x, c):
    c1 = 1.0 + c
    min_v = c / c1
    return ((c / (c1 - x)) - min_v) / (1.0 - min_v)


def find_softplus_v2_main():
    def find_softplus_v1_to_v2(c):
        # c=4, bias=0.333, scale=12
        # c=6, bias=0.5, scale=12
        # c=8.4, bias=0687, scale=12

        x = torch.linspace(0, 1, 1000)
        y = softplus01_old(x, c)
        best_score = 10000000
        hist = []
        for bias in torch.linspace(0, 1, 100).tolist():
            for scale in torch.linspace(0, 20, 100).tolist():
                y2 = softplus01(x, bias=bias, scale=scale)
                score = (y - y2).abs().mean().item()
                if best_score > score:
                    best_score = score
                    hist.append((score, dict(c=c, bias=bias, scale=scale)))

        print(f"** c={c} top 10:")
        for score, param in hist[-10:]:
            print("MAE", round(score, 5), "bias", round(param["bias"], 3), "scale", round(param["scale"], 3))

    # c=4, bias=0.333, scale=12
    # c=6, bias=0.5, scale=12
    # c=8.4, bias=0687, scale=12
    find_softplus_v1_to_v2(4)
    find_softplus_v1_to_v2(6)
    find_softplus_v1_to_v2(8.4)


def find_softplus_mul_main():
    def find_softplus(mul_scale, margin=0.2):
        threshold = (1.0 - (1.0 / mul_scale))
        threshold += threshold * margin
        x = torch.linspace(threshold, 1, 1000)
        y = hardplus(x, mul_scale)

        best_score = 10000000
        hist = []
        for bias in torch.linspace(0, 1, 100).tolist():
            # for scale in torch.linspace(0, 20, 100).tolist():
            scale = 12
            y2 = softplus01(x, bias=bias, scale=scale)
            score = (y - y2).abs().mean().item()
            if best_score > score:
                best_score = score
                hist.append((score, dict(bias=bias, scale=scale)))

        print(f"** mul_scale={mul_scale} top 10:")
        for score, param in hist[-10:]:
            print("MAE", round(score, 5), "bias", round(param["bias"], 3), "scale", round(param["scale"], 3))

    find_softplus(1.5)
    find_softplus(2)
    find_softplus(3)


def check_find_softplus_mul_main():
    x = torch.linspace(0, 1, 1000)

    plt.plot(x, x, label="none")

    y = hardplus(x, 1.5)
    y2 = softplus01(x, bias=0.343, scale=12)
    plt.plot(x, y, label="hard_mul_1.5")
    plt.plot(x, y2, label="soft_mul_1.5")

    y = hardplus(x, 2)
    y2 = softplus01(x, bias=0.515, scale=12)
    plt.plot(x, y, label="hard_mul_2")
    plt.plot(x, y2, label="soft_mul_2")

    y = hardplus(x, 3)
    y2 = softplus01(x, bias=0.687, scale=12)
    plt.plot(x, y, label="hard_mul_3")
    plt.plot(x, y2, label="soft_mul_3")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # find_softplus_v2_main()
    # find_softplus_mul_main()
    check_find_softplus_mul_main()
