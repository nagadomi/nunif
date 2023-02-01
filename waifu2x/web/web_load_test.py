# test client for high server load
# Do not use this for remote web sites
# python3 -m waifu2x.web.web_load_test --image-dir /images --ntest 100

import requests
import argparse
import sys
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import random
import time
import os
from os import path


def send_random_request(i, host, port, file_path):
    style = random.choice(["art", "photo"])
    scale = str(random.choice([-1, 1, 2]))
    noise = str(random.choice([-1, 0, 1, 2, 3]))
    image_format = str(random.choice([0, 1]))
    data = {"style": style, "scale": scale, "noise": noise, "format": image_format}

    with open(file_path, "rb") as f:
        print("%05d" % i, data)
        res = requests.post(f"http://{host}:{port}/api",
                            files={'file': f}, data=data)
        if res.status_code != 200:
            print(f"Error {res.status_code} {res.reason}", file=sys.stderr)


def large_file(file_path, max_file_size, max_image_size):
    if path.getsize(file_path) > max_file_size:
        return True
    try:
        with open(file_path, "rb") as f, Image.open(f) as im:
            if max(im.size) > max_image_size:
                return True
    except UnidentifiedImageError:
        print("UnidentifiedImageError", file_path)
        return True
    return False


def dos(n, host, port, threads, files,
        max_file_size, max_image_size,
        sleep_range=[0.2, 2]):
    sleep_step = threads * 4
    futures = []
    with PoolExecutor(max_workers=threads) as pool:
        for i in range(n):
            while True:
                file_path = random.choice(files)
                if large_file(file_path, max_file_size, max_image_size):
                    #  print("large", file_path)
                    continue
                break
            time.sleep(random.uniform(sleep_range[0], sleep_range[1]))
            futures.append(pool.submit(send_random_request, i, host, port, file_path))
            if len(futures) > sleep_step:
                for f in futures:
                    f.result()
                futures = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8812, help="remote port")
    parser.add_argument("--image-dir", type=str, required=True, help="input image dir")
    parser.add_argument("--ntest", "-n", type=int, default=100, help="number of tries")
    parser.add_argument("--max-file-size", type=int, default=1024 * 1024 * 5, help="max file size (bytes)")
    parser.add_argument("--max-image-size", type=int, default=1500, help="max image size (width/height px)")
    parser.add_argument("--threads", type=int, default=4, help="number of tries")
    args = parser.parse_args()

    random.seed(71)

    files = [path.join(args.image_dir, fn)
             for fn in os.listdir(args.image_dir)
             if fn.endswith(".png") or fn.endswith(".jpg")]
    dos(args.ntest,
        host="localhost", port=args.port,
        threads=args.threads, files=files,
        max_file_size=args.max_file_size,
        max_image_size=args.max_image_size,
        sleep_range=[0.0, 0.5])
