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
import json
from time import time
import hashlib
from configparser import ConfigParser
from diskcache import Cache
from enum import Enum
from nunif.logger import logger, set_log_level
# from nunif.utils.pil_io
from nunif.utils import wand_io as IL
from .utils import Waifu2x


DEFAULT_ART_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "cunet", "art"))
DEFAULT_PHOTO_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "upconv_7", "photo"))
BUFF_SIZE = 8192  # buffer block size for io access
SIZE_MB = 1024 * 1024
RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"


class ScaleOption(Enum):
    NONE = -1
    X16 = 1
    X20 = 2


class NoiseOption(Enum):
    NONE = -1
    JPEG_0 = 0
    JPEG_1 = 1
    JPEG_2 = 2
    JPEG_3 = 3


class StyleOption(Enum):
    ART = "art"
    PHOTO = "photo"


class CacheGC():
    def __init__(self, cache, interval=60):
        self.cache = cache
        self.interval = interval
        self.last_expired_at = 0

    def gc(self):
        now = time()
        if self.last_expired_at + self.interval < now:
            self.cache.expire(now)
            self.cache.cull()
            self.last_expired_at = time()


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
    parser.add_argument("--debug", action="store_true", help="Debug print")
    parser.add_argument("--max-body-size", type=int, default=5, help="maximum allowed size(MB) for uploaded files")
    parser.add_argument("--url-timeout", type=int, default=10, help="request_timeout for url")

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
    parser.add_argument("--enable-recaptcha", action="store_true", help="enable reCAPTCHA. it requires web-config.yml")
    parser.add_argument("--config", type=str, help="config file for API tokens")

    args = parser.parse_args()
    art_ctx = Waifu2x(model_dir=args.art_model_dir, gpus=args.gpu)
    photo_ctx = Waifu2x(model_dir=args.photo_model_dir, gpus=args.gpu)

    art_ctx.load_model_all()
    photo_ctx.load_model_all()

    cache = Cache(args.cache_dir, size_limit=args.cache_size_limit * 1073741824)
    cache_gc = CacheGC(cache, args.cache_ttl * 60)
    config = ConfigParser()
    if args.config:
        config.read(args.config)
    if "recaptcha" not in config:
        if args.enable_recaptcha:
            raise RuntimeError("--enable-recaptcha: No recaptcha setting in config file")
        else:
            config["recaptcha"] = {"site_key": "", "secret_key": ""}

    return args, config, art_ctx, photo_ctx, cache, cache_gc


global_lock = threading.RLock()
command_args, config, art_ctx, photo_ctx, cache, cache_gc = setup()
# HACK: Avoid unintended argparse in the backend(gunicorn).
sys.argv = [sys.argv[0]]


def fetch_uploaded_file(upload_file):
    max_body_size = command_args.max_body_size * SIZE_MB
    with io.BytesIO() as upload_buff:
        file_size = 0
        buff = upload_file.file.read(BUFF_SIZE)
        while buff:
            file_size += len(buff)
            upload_buff.write(buff)
            if file_size > max_body_size:
                logger.debug("fetch_uploaded_file: error: too lage")
                bottle.abort(413, f"Request entity too large (max: {command_args.max_body_size}MB)")
            buff = upload_file.file.read(BUFF_SIZE)

        image_data = upload_buff.getvalue()
        if image_data:
            logger.debug(f"fetch_uploaded_file: {round(len(image_data)/(SIZE_MB), 3)}MB")
        return image_data


def fetch_url_file(url):
    max_body_size = command_args.max_body_size * SIZE_MB
    timeout = command_args.url_timeout
    try:
        headers = {"Referer": url, "User-Agent": "waifu2x/web.py"}
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as res, io.BytesIO() as buff:
            if res.status_code != 200:
                logger.debug(f"fetch_url_file: error: status={res.status_code}, {url}")
                bottle.abort(400, "URL error")
            file_size = 0
            for chunk in res.iter_content(chunk_size=BUFF_SIZE):
                buff.write(chunk)
                file_size += len(chunk)
                if file_size > max_body_size:
                    logger.debug(f"fetch_url_file: error: too large, {url}")
                    bottle.abort(413, f"Request entity too large (max: {command_args.max_body_size}MB)")
            logger.debug(f"fetch_url_file: {round(file_size/(SIZE_MB), 3)}MB, {url}")
            return buff.getvalue()
    except requests.exceptions.Timeout:
        logger.debug(f"fetch_url_file: error: timeout, {url}")
        bottle.abort(400, "URL timeout")


def fetch_image(request):
    upload_file = request.files.get("file", "")
    im, meta = None, None

    image_data = None
    if upload_file:
        image_data = fetch_uploaded_file(upload_file)
    if image_data:
        im, meta = IL.decode_image(image_data, upload_file.filename,
                                   color="rgb", keep_alpha=True)
    else:
        url = request.forms.get("url", "")
        if url.startswith("http://") or url.startswith("https://"):
            key = "url_" + url
            image_data = cache.get(key, None)
            if image_data is not None:
                logger.debug(f"fetch_image: load cache: {url}")
                im, meta = IL.decode_image(image_data, posixpath.basename(url),
                                           color="rgb", keep_alpha=True)
            else:
                image_data = fetch_url_file(url)
                im, meta = IL.decode_image(image_data, posixpath.basename(url),
                                           color="rgb", keep_alpha=True)
                cache.set(key, image_data, expire=command_args.cache_ttl * 60)

    if image_data is not None and meta is not None:
        # sha1 for cache key
        meta["sha1"] = hashlib.sha1(image_data).hexdigest()
    return im, meta


def parse_request(request):
    try:
        style = StyleOption(request.forms.get("style", "photo"))
        scale = ScaleOption(int(request.forms.get("scale", "-1")))
        noise = NoiseOption(int(request.forms.get("noise", "-1")))
    except ValueError:
        bottle.abort(400, "Bad Request")

    if scale == ScaleOption.NONE:
        if noise == NoiseOption.NONE:
            method = "none"
        else:
            method = "noise"
    else:
        if noise == NoiseOption.NONE:
            method = "scale"
        else:
            method = "noise_scale"

    return style, method, scale, noise


def make_output_filename(style, method, noise, meta):
    base = meta["filename"] if meta["filename"] else meta["sha1"] + ".png"
    base = path.splitext(base)[0]
    if method == "noise":
        mode = f"{style.value}_noise{noise.value}"
    elif method == "noise_scale":
        mode = f"{style.value}_noise{noise.value}_scale"
    elif method == "scale":
        mode = f"{style.value}_scale"
    elif method == "none":
        mode = "none"
    return f"{base}_waifu2x_{mode}.png"


@bottle.get("/recaptcha_state.json")
def recaptcha_state():
    state = {
        "enabled": command_args.enable_recaptcha,
        "site_key": config["recaptcha"]["site_key"]
    }
    res = HTTPResponse(status=200, body=json.dumps(state))
    res.set_header("Content-Type", "application/json")
    return res


def verify_recaptcha(request):
    if not command_args.enable_recaptcha:
        return True

    timeout = command_args.url_timeout
    data = {
        "response": request.forms.get("recap", ""),
        "secret": config["recaptcha"]["secret_key"],
        "remoteip": request.remote_addr
    }
    try:
        res = requests.post(RECAPTCHA_VERIFY_URL, data=data, timeout=timeout)
        if res.status_code == 200:
            result = json.loads(res.text)
            if result["success"]:
                logger.debug("verify_recaptcha: success")
            else:
                logger.debug("verify_recaptcha: fail")
            return result["success"]
        else:
            logger.error(f"verify_recaptcha: HTTP Error {res.status_code} {res.text}")
            return False
    except requests.exceptions.Timeout:
        logger.error("verify_recaptcha: error: timeout")
        return False


def dump_meta(meta):
    return {k: v if isinstance(v, (str, int, float, type(None)))
            else str(type(v)) for k, v in meta.items()}


@bottle.post("/api")
def api():
    # {'url': 'https://ja.wikipedia.org/static/images/icons/wikipedia.png',
    #  'style': 'photo', 'noise': '1', 'scale': '2', 'recap': 'xxxxx'}
    if command_args.enable_recaptcha and not verify_recaptcha(request):
        bottle.abort(401, "reCAPTCHA Error")
    cache_gc.gc()
    try:
        im, meta = fetch_image(request)
    except:
        logger.error(f"api: fetch_image error: {sys.exc_info()[:2]}")
        im, meta = None, None
    if im is None:
        bottle.abort(400, "Image Load Error")
    logger.debug(f"api: image: {dump_meta(meta)}")

    style, method, scale, noise = parse_request(request)
    output_filename = make_output_filename(style, method, noise, meta)
    key = meta["sha1"] + output_filename

    image_data = cache.get(key, None)
    if image_data is None:
        t = time()
        if method == "none":
            logger.debug(f"api: forward: {style} {scale} {noise} "
                         f"pid={os.getpid()}-{threading.get_ident()}")
            z = im
        else:
            with torch.no_grad():
                rgb, alpha = IL.to_tensor(im, return_alpha=True)
                ctx_kwargs = {
                    "x": rgb, "alpha": alpha,
                    "method": method, "noise_level": noise.value,
                    "tile_size": command_args.tile_size, "batch_size": command_args.batch_size,
                    "tta": command_args.tta, "enable_amp": command_args.amp
                }
                with global_lock:
                    if style == StyleOption.ART:
                        rgb, alpha = art_ctx.convert(**ctx_kwargs)
                    else:
                        rgb, alpha = photo_ctx.convert(**ctx_kwargs)
                z = IL.to_image(rgb, alpha)
            logger.debug(f"api: forward: {style} {scale} {noise}, {round(time()-t, 2)}s, "
                         f"pid={os.getpid()}-{threading.get_ident()}")
        t = time()
        image_data = IL.encode_image(z, format="png", meta=meta)
        cache.set(key, image_data, expire=command_args.cache_ttl * 60)
        logger.debug(f"api: encode: {round(time()-t, 2)}s")
    else:
        logger.debug(f"api: load cache: {style} {scale} {noise}")

    if scale == ScaleOption.X16:
        # 1.6x
        im, meta = IL.decode_image(image_data, color="rgb", keep_alpha=True)
        w, h = int(im.size[0] * (1.6 / 2.0)), int(im.size[1] * (1.6 / 2.0))
        im.resize(w, h, "lanczos")
        image_data = IL.encode_image(im, format="png", meta=meta)

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


@bottle.get("/<url:re:.*>")
def static_file(url):
    """
    This assumes that `root` directory is flat.
    In production environment, this method is not used, instead it is directly sent from nginx.
    """
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
               max_request_body_size=command_args.max_body_size * SIZE_MB)
