import os
from os import path
import torch
from fractions import Fraction
import hashlib
from platformdirs import user_cache_dir
from nunif.utils.home_dir import ensure_home_dir, is_nunif_home_set


CACHE_VERSION = 1.0
MD5_SALT = "stlizer"


def get_cache_dir():
    if is_nunif_home_set():
        cache_dir = path.join(ensure_home_dir("stlizer"), "cache")
    else:
        cache_dir = user_cache_dir(appname="stlizer", appauthor="nunif")

    if not path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cache_path(input_video_path):
    cache_dir = get_cache_dir()
    cache_filename = filepath_md5(input_video_path) + ".stlizer"
    cache_path = path.join(cache_dir, cache_filename)
    return cache_path


def md5(s):
    if s:
        return hashlib.md5((s + MD5_SALT).encode()).hexdigest()
    else:
        return ""


def filepath_md5(filepath):
    filepath = path.abspath(filepath)
    size = path.getsize(filepath)
    return md5(f"{filepath}_{str(size)}")


def save_cache(input_video_path, transforms, mean_match_scores, fps, args):
    cache_path = get_cache_path(input_video_path)
    if isinstance(fps, Fraction):
        fps = f"{fps.numerator}/{fps.denominator}"
    else:
        fps = float(fps)

    torch.save({"transforms": transforms,
                "mean_match_scores": mean_match_scores,
                "max_fps": args.max_fps,
                "fps": fps,
                "vf": md5(args.vf),
                "resolution": args.resolution,
                "version": CACHE_VERSION,
                }, cache_path)


def try_load_cache(input_video_path, args):
    cache_path = get_cache_path(input_video_path)
    if path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if "version" not in data:
            return None
        if args.max_fps != data["max_fps"]:
            return None
        if args.resolution != data["resolution"]:
            return None
        vf = data.get("vf", None)
        if vf is not None and md5(args.vf) != vf:
            return None
        if isinstance(data["fps"], str):
            numerator, denominator = data["fps"].split("/")
            numerator = int(numerator)
            denominator = int(denominator)
            data["fps"] = Fraction(numerator, denominator)

        return data
    else:
        return None


def purge_cache(input_video_path):
    cache_path = get_cache_path(input_video_path)
    if path.exists(cache_path):
        os.unlink(cache_path)


def list_cache_files():
    cache_dir = get_cache_dir()
    return (path.join(cache_dir, fn)
            for fn in os.listdir(cache_dir)
            if fn.endswith(".stlizer"))


def purge_cache_all():
    for cache_path in list_cache_files():
        os.unlink(cache_path)
