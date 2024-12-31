from os import path
import torch
from fractions import Fraction
import hashlib


CACHE_VERSION = 1.0


def make_cache_path(output_path):
    cache_path = path.join(path.dirname(output_path), path.splitext("." + path.basename(output_path))[0] + ".stlizer")
    return cache_path


def sha256(s):
    return hashlib.sha256(s.encode()).digest()


def save_cache(cache_path, transforms, mean_match_scores, fps, args):
    if isinstance(fps, Fraction):
        fps = f"{fps.numerator}/{fps.denominator}"
    else:
        fps = float(fps)

    torch.save({"transforms": transforms,
                "mean_match_scores": mean_match_scores,
                "max_fps": args.max_fps,
                "fps": fps,
                "resolution": args.resolution,
                "input_file": sha256(path.basename(args.input)),
                "version": args.cache_version
                }, cache_path)


def try_load_cache(cache_path, args):
    if path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if "version" not in data:
            return None
        if args.max_fps != data["max_fps"]:
            return None
        if args.resolution != data["resolution"]:
            return None
        if sha256(path.basename(args.input)) != data["input_file"]:
            return None
        if isinstance(data["fps"], str):
            numerator, denominator = data["fps"].split("/")
            numerator = int(numerator)
            denominator = int(denominator)
            data["fps"] = Fraction(numerator, denominator)

        return data
    else:
        return None
