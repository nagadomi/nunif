import sys
import os
from os import path
import posixpath
import torch
import argparse
import bottle
from bottle import request, HTTPResponse
import threading
import requests
import io
import time
import hashlib
from diskcache import Cache
from nunif.logger import logger, set_log_level
from nunif.utils import decode_image, save_image
from .utils import Waifu2x


DEFAULT_ART_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "cunet", "art"))
DEFAULT_PHOTO_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "upconv_7", "photo"))


class CacheGC():
    def __init__(self, cache, interval=60):
        self.cache = cache
        self.interval = interval
        self.last_expired_at = 0

    def gc(self):
        now = time.time()
        if self.last_expired_at + self.interval < now:
            self.cache.expire(now)
            self.cache.cull()
            self.last_expired_at = time.time()


def setup():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bind-addr", type=str, default="0.0.0.0",
                        help="0.0.0.0 for global, 127.0.0.1 for local")
    parser.add_argument("--port", type=int, default=8812, help="HTTP port number")
    parser.add_argument("--root", type=str, default="waifu2x/public_html",
                        help="web root directory")
    parser.add_argument("--backend", type=str, default="waitress",
                        help="server backend. It may not work except `waitress`.")
    parser.add_argument("--workers", type=int, default=1, help="The number of worker processes for gunicorn")
    parser.add_argument("--threads", type=int, default=32, help="The number of threads")
    parser.add_argument("--debug", action="store_true", help="debug=True for bottle")
    parser.add_argument("--max-body-size", type=int, default=5, help="maximum allowed size(MB) for uploaded files")

    parser.add_argument("--art-model-dir", type=str, default=DEFAULT_ART_MODEL_DIR, help="art model dir")
    parser.add_argument("--photo-model-dir", type=str, default=DEFAULT_PHOTO_MODEL_DIR, help="photo model dir")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[0], help="GPU device ids. -1 for CPU")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch size for tiled render")
    parser.add_argument("--tta", action="store_true", help="use TTA mode")
    parser.add_argument("--amp", action="store_true", help="with half float")

    parser.add_argument("--cache-ttl", type=int, default=120, help="cache TTL(min)")
    parser.add_argument("--cache-size-limit", type=int, default=10, help="cache size limit (GB)")
    parser.add_argument("--cache-dir", type=str, default=path.join("tmp", "waifu2x_cache"), help="cache dir")

    args = parser.parse_args()
    art_ctx = Waifu2x(model_dir=args.art_model_dir, gpus=args.gpu)
    photo_ctx = Waifu2x(model_dir=args.photo_model_dir, gpus=args.gpu)

    art_ctx.load_model_all()
    photo_ctx.load_model_all()

    cache = Cache(args.cache_dir, size_limit=args.cache_size_limit * 1073741824)
    cache_gc = CacheGC(cache, args.cache_ttl * 60)

    return args, art_ctx, photo_ctx, cache, cache_gc


global_lock = threading.RLock()
command_args, art_ctx, photo_ctx, cache, cache_gc = setup()
# HACK: Avoid unintended argparse in the backend(gunicorn).
sys.argv = [sys.argv[0]]


def get_image(request):
    # TODO: stream read and reject large file
    file_data = request.files.get("file", "")
    im, meta = None, None

    attached_filename = file_data.filename
    image_data = file_data.file.read()
    if image_data:
        im, meta = decode_image(image_data, attached_filename)
    else:
        url = request.forms.get("url", "")
        if url.startswith("http://") or url.startswith("https://"):
            key = "url_" + url
            image_data = cache.get(key, None)
            if image_data is not None:
                logger.debug(f"load url cache: {url}")
                im, meta = decode_image(image_data, posixpath.basename(url))
            else:
                res = requests.get(url)
                logger.debug(f"get url: {res.status_code} {len(res.content)} {url}")
                if res.status_code == 200:
                    image_data = res.content
                    im, meta = decode_image(image_data, posixpath.basename(url))
                    cache.set(key, image_data, expire=command_args.cache_ttl * 60)
    if image_data is not None and meta is not None:
        # sha1 for cache key
        meta["sha1"] = hashlib.sha1(image_data).hexdigest()
    return im, meta


def parse_request(request):
    style = request.forms.get("style", "photo")
    scale = int(request.forms.get("scale", "-1"))
    noise_level = int(request.forms.get("noise", "-1"))

    if style != "art":
        style = "photo"
    if scale != 2:
        scale = 1
    if not (-1 <= noise_level <= 3):
        noise_level = 2
    if scale == 1:
        method = "noise"
    else:
        if noise_level == -1:
            method = "scale"
        else:
            method = "noise_scale"

    if noise_level == -1:
        noise_level = 0

    return style, method, noise_level


def make_output_filename(style, method, noise_level, meta):
    base = meta["filename"] if meta["filename"] else meta["sha1"] + ".png"
    base = path.splitext(base)[0]
    if method == "noise":
        mode = f"{style}_noise{noise_level}"
    elif method == "noise_scale":
        mode = f"{style}_noise{noise_level}_scale"
    elif method == "scale":
        mode = f"{style}_scale"
    return f"{base}_waifu2x_{mode}.png"


@bottle.route("/api", method=["POST"])
def api():
    # {'url': 'https://ja.wikipedia.org/static/images/icons/wikipedia.png',
    #  'style': 'photo', 'noise': '1', 'scale': '2'}

    cache_gc.gc()
    im, meta = get_image(request)
    style, method, noise_level = parse_request(request)

    if im is None:
        bottle.abort(400, "ERROR: An error occurred. (unsupported image format/connection timeout/file is too large)")

    ctx_kwargs = {
        "im": im, "meta": meta,
        "method": method, "noise_level": noise_level,
        "tile_size": command_args.tile_size, "batch_size": command_args.batch_size,
        "tta": command_args.tta, "enable_amp": command_args.amp
    }
    output_filename = make_output_filename(style, method, noise_level, meta)
    key = meta["sha1"] + output_filename
    image_data = cache.get(key, None)
    if image_data is None:
        logger.debug(f"process: pid={os.getpid()}, tid={threading.get_ident()}, {output_filename}")
        with torch.no_grad():
            with global_lock:
                if style == "art":
                    rgb, alpha = art_ctx.convert_raw(**ctx_kwargs)
                else:
                    rgb, alpha = photo_ctx.convert_raw(**ctx_kwargs)
            z = Waifu2x.to_pil(rgb, alpha)

        with io.BytesIO() as buff:
            save_image(z, meta, buff)
            image_data = buff.getvalue()
            cache.set(key, image_data, expire=command_args.cache_ttl * 60)
    else:
        logger.debug(f"load cache: {output_filename}")

    res = HTTPResponse(status=200, body=image_data)
    res.set_header("Content-Type", "image/png")
    res.set_header("Content-Disposition", f'inline; filename="{output_filename}"')

    return res


def get_lang(accept_language):
    if accept_language:
        langs = accept_language.split(";")[0]
        if langs:
            return langs.split(",")[0]
    return None


def resolve_index_file(root_dir, accept_language):
    lang = get_lang(accept_language)
    if lang:
        lang = lang.translate(str.maketrans("", "", "./\\"))  # sanitize

    if lang == "pt-BR":
        index_file = "index.pt.html"
    elif lang == "es-ES":
        index_file = "index.es.html"
    elif lang in {"ca-ES", "ca-FR", "ca-IT", "ca-AD"}:
        index_file = "index.ca.html"
    else:
        index_file = f"index.{lang}.html"

    if path.exists(path.join(root_dir, index_file)):
        return index_file
    else:
        return "index.html"


@bottle.route("/<url:re:.*>", method=["GET"])
def static_file(url):
    """
    This assumes that `root` directory is flat.
    In production environment, this method is not used, instead it is directly sent from nginx.
    """
    print("url", url)
    url = url.replace("\\", "/")
    dirname = posixpath.dirname(url)
    basename = posixpath.basename(url)
    if dirname:
        bottle.abort(403)
    if not basename:
        basename = resolve_index_file(command_args.root, request.headers.get("Accept-Language", ""))

    return bottle.static_file(basename, root=command_args.root)


if __name__ == "__main__":
    # NOTE: This code expect the server to run with single-process multi-threading.
    import logging

    if command_args.debug:
        set_log_level(logging.DEBUG)

    bottle.run(host=command_args.bind_addr, port=command_args.port, debug=command_args.debug,
               server=command_args.backend,
               threads=command_args.threads, workers=command_args.workers, preload_app=True,
               max_request_body_size=command_args.max_body_size * 1024 * 1024)
