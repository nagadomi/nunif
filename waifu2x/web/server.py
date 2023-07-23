import os
import sys
import math
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
import psutil
import gc
from enum import Enum
from urllib.parse import (
    quote as uri_encode,
    unquote_plus as uri_decode,
    urlparse,
)
import uuid
from nunif.logger import logger, set_log_level
from nunif.utils.filename import set_image_ext
from ..utils import Waifu2x


DEFAULT_ART_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "..", "pretrained_models"),
    "swin_unet", "art"))
DEFAULT_ART_SCAN_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "..", "pretrained_models"),
    "swin_unet", "art_scan"))
DEFAULT_PHOTO_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "..", "pretrained_models"),
    "swin_unet", "photo"))
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


class FormatOption(Enum):
    PNG = 0
    WEBP = 1


class StyleOption(Enum):
    ART = "art"
    PHOTO = "photo"
    ART_SCAN = "art_scan"


class CacheGC():
    def __init__(self, cache, interval=60):
        self.cache = cache
        self.interval = interval
        self.last_expired_at = 0
        self.proc = psutil.Process()

    def disk_size_mb(self):
        return round(self.cache.volume() / SIZE_MB, 2)

    def ram_size_mb(self):
        return round(self.proc.memory_info().rss / SIZE_MB, 2)

    def gc(self):
        now = time()
        if self.last_expired_at + self.interval < now:
            self.cache.expire(now)
            self.cache.cull()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.last_expired_at = time()
            logger.info(f"diskcache: cache={self.disk_size_mb()}MB, RAM={self.ram_size_mb()}")


def setup():
    default_gpu = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bind-addr", type=str, default="127.0.0.1",
                        help="0.0.0.0 for global, 127.0.0.1 for local")
    parser.add_argument("--port", type=int, default=8812, help="HTTP port number")
    parser.add_argument("--root", type=str, default=path.join(path.dirname(__file__), "public_html"),
                        help="web root directory")
    parser.add_argument("--backend", type=str, default="waitress",
                        help="server backend. It may not work except `waitress`.")
    parser.add_argument("--workers", type=int, default=1, help="The number of worker processes for gunicorn")
    parser.add_argument("--threads", type=int, default=32, help="The number of threads")
    parser.add_argument("--debug", action="store_true", help="Debug print")
    parser.add_argument("--max-body-size", type=int, default=5, help="maximum allowed size(MB) for uploaded files")
    parser.add_argument("--max-pixels", type=int, default=3000 * 3000, help="maximum number of output image pixels ")
    parser.add_argument("--url-timeout", type=int, default=10, help="request_timeout for url")

    parser.add_argument("--art-model-dir", type=str, default=DEFAULT_ART_MODEL_DIR, help="art model dir")
    parser.add_argument("--art-scan-model-dir", type=str, default=DEFAULT_ART_SCAN_MODEL_DIR, help="art scan model dir")
    parser.add_argument("--photo-model-dir", type=str, default=DEFAULT_PHOTO_MODEL_DIR, help="photo model dir")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu],
                        help="GPU device ids. -1 for CPU")
    parser.add_argument("--tile-size", type=int, default=256, help="tile size for tiled render")
    parser.add_argument("--batch-size", type=int, default=4, help="minibatch size for tiled render")
    parser.add_argument("--tta", action="store_true", help="use TTA mode")
    parser.add_argument("--disable-amp", action="store_true", help="disable AMP for some special reason")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--warmup", action="store_true", help="warmup at startup")
    parser.add_argument("--image-lib", type=str, choices=["pil", "wand"], default="pil",
                        help="image library to encode/decode images")
    parser.add_argument("--cache-ttl", type=int, default=30, help="cache TTL(min)")
    parser.add_argument("--cache-size-limit", type=int, default=10, help="cache size limit (GB)")
    parser.add_argument("--cache-dir", type=str, default=path.join("tmp", "waifu2x_cache"), help="cache dir")
    parser.add_argument("--enable-recaptcha", action="store_true", help="enable reCAPTCHA. it requires --config option")
    parser.add_argument("--config", type=str, help="config file for API tokens")
    parser.add_argument("--no-size-limit", action="store_true", help="No file/image size limits for private server")
    parser.add_argument("--torch-threads", type=int, help="The number of threads used for intraop parallelism on CPU")
    parser.add_argument("--exit", action="store_true", help="run setup() and exit")

    args = parser.parse_args()
    art_ctx = Waifu2x(model_dir=args.art_model_dir, gpus=args.gpu)
    art_scan_ctx = Waifu2x(model_dir=args.art_scan_model_dir, gpus=args.gpu)
    photo_ctx = Waifu2x(model_dir=args.photo_model_dir, gpus=args.gpu)

    art_ctx.load_model_all(load_4x=False)
    art_scan_ctx.load_model_all(load_4x=False)
    photo_ctx.load_model_all(load_4x=False)

    if args.compile:
        logger.info("Compiling models...")
        art_ctx.compile()
        art_scan_ctx.compile()
        photo_ctx.compile()
        if args.warmup:
            if args.batch_size != 1:
                logger.warning(("`--batch-size 1` is recommended."
                                "large batch size makes startup very slow."))
            art_ctx.warmup(tile_size=args.tile_size, batch_size=args.batch_size,
                           enable_amp=not args.disable_amp)
            art_scan_ctx.warmup(tile_size=args.tile_size, batch_size=args.batch_size,
                                enable_amp=not args.disable_amp)
            photo_ctx.warmup(tile_size=args.tile_size, batch_size=args.batch_size,
                             enable_amp=not args.disable_amp)
        logger.info("Done")

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

    return args, config, art_ctx, art_scan_ctx, photo_ctx, cache, cache_gc


def fetch_uploaded_file(upload_file):
    max_body_size = MAX_BODY_SIZE * SIZE_MB
    with io.BytesIO() as upload_buff:
        file_size = 0
        buff = upload_file.file.read(BUFF_SIZE)
        while buff:
            file_size += len(buff)
            upload_buff.write(buff)
            if file_size > max_body_size:
                logger.debug("fetch_uploaded_file: error: too large")
                bottle.abort(413, f"Request entity too large (max: {MAX_BODY_SIZE}MB)")
            buff = upload_file.file.read(BUFF_SIZE)

        image_data = upload_buff.getvalue()
        if image_data:
            logger.debug(f"fetch_uploaded_file: {round(len(image_data)/(SIZE_MB), 3)}MB")
        return image_data


def fetch_url_file(url):
    max_body_size = MAX_BODY_SIZE * SIZE_MB
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
                    bottle.abort(413, f"Request entity too large (max: {MAX_BODY_SIZE}MB)")
            logger.debug(f"fetch_url_file: {round(file_size/(SIZE_MB), 3)}MB, {url}")
            return buff.getvalue()
    except requests.exceptions.RequestException as e:
        logger.debug(f"fetch_url_file: error: {e}, {url}")
        bottle.abort(400, "URL Error")
    except UnicodeEncodeError as e:
        logger.debug(f"fetch_url_file: error: {e}, {url}")
        bottle.abort(400, "URL Error")


def fetch_image(request):
    upload_file = request.files.get("file", "")
    im, meta = None, None

    image_data = None
    if upload_file:
        image_data = fetch_uploaded_file(upload_file)
    if image_data:
        filename = upload_file.raw_filename
        if not filename:
            filename = str(uuid.uuid4())
        im, meta = IL.decode_image(image_data, filename,
                                   color="rgb", keep_alpha=True)
    else:
        url = request.forms.get("url", "")
        if url.startswith("http://") or url.startswith("https://"):
            key = "url_" + url
            image_data = cache.get(key, None)
            filename = posixpath.basename(urlparse(url).path)
            if not filename:
                filename = str(uuid.uuid4())
            else:
                filename = uri_decode(filename)
            if image_data is not None:
                logger.debug(f"fetch_image: load cache: {url}")
                im, meta = IL.decode_image(image_data, filename,
                                           color="rgb", keep_alpha=True)
            else:
                image_data = fetch_url_file(url)
                im, meta = IL.decode_image(image_data, filename,
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
        image_format = FormatOption(int(request.forms.get("format", "0")))
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
    if image_format == FormatOption.PNG:
        image_format = "png"
    else:
        image_format = "webp"

    return style, method, scale, noise, image_format


def make_output_filename(style, method, noise, image_format, meta):
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
    return set_image_ext(f"{base}_waifu2x_{mode}.png", image_format)


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
            logger.debug("verify_recaptcha: " + ("success" if result['success'] else "failure"))
            return result["success"]
        else:
            logger.error(f"verify_recaptcha: HTTP Error {res.status_code} {res.text}")
            bottle.abort(500, "reCAPTCHA Error")
    except requests.exceptions.RequestException as e:
        logger.error(f"verify_recaptcha: error: {e}")
        bottle.abort(500, "reCAPTCHA Error")


def dump_meta(meta):
    return {k: v if isinstance(v, (str, int, float, type(None)))
            else str(type(v)) for k, v in meta.items()}


def scale_16x(im, meta):
    w, h = int(im.size[0] * (1.6 / 2.0)), int(im.size[1] * (1.6 / 2.0))
    if meta["engine"] == "wand":
        im.resize(w, h, "lanczos")
    elif meta["engine"] == "pil":
        im = im.resize((w, h), 1)
    return im


@bottle.get("/api")
def api_get_error():
    bottle.abort(405, "Method Not Allowed")


@bottle.post("/api")
def api():
    # {'url': 'https://ja.wikipedia.org/static/images/icons/wikipedia.png',
    #  'style': 'photo', 'noise': '1', 'scale': '2', 'recap': 'xxxxx'}
    style, method, scale, noise, image_format = parse_request(request)

    if command_args.enable_recaptcha and not verify_recaptcha(request):
        bottle.abort(401, "reCAPTCHA Error")
    with global_lock:
        cache_gc.gc()

    im, meta = fetch_image(request)
    if im is None:
        bottle.abort(400, "Image Load Error")
    logger.debug(f"api: image: {dump_meta(meta)}")
    if scale != ScaleOption.NONE and im.size[0] * im.size[1] > MAX_SCALE_PIXELS:
        im.close()
        bottle.abort(413, "Request Image Too Large")
    if scale == ScaleOption.NONE and im.size[0] * im.size[1] > MAX_NOISE_PIXELS:
        im.close()
        bottle.abort(413, "Request Image Too Large")

    output_filename = make_output_filename(style, method, noise, image_format, meta)
    key = meta["sha1"] + output_filename
    image_data = cache.get(key, None)
    if image_data is None:
        t = time()
        if method == "none":
            logger.debug(f"api: forward: {style} {scale} {noise} {image_format}"
                         f"pid={os.getpid()}-{threading.get_ident()}")
            z = im
        else:
            with torch.inference_mode():
                rgb, alpha = IL.to_tensor(im, return_alpha=True)
                ctx_kwargs = {
                    "x": rgb, "alpha": alpha,
                    "method": method, "noise_level": noise.value,
                    "tile_size": command_args.tile_size, "batch_size": command_args.batch_size,
                    "tta": command_args.tta, "enable_amp": not command_args.disable_amp,
                }
                with global_lock:
                    if style == StyleOption.ART:
                        rgb, alpha = art_ctx.convert(**ctx_kwargs)
                    elif style == StyleOption.ART_SCAN:
                        rgb, alpha = art_scan_ctx.convert(**ctx_kwargs)
                    else:
                        rgb, alpha = photo_ctx.convert(**ctx_kwargs)
                z = IL.to_image(rgb, alpha)
            logger.debug(f"api: forward: {round(time()-t, 2)}s, {style} {scale} {noise} {image_format}, "
                         f"pid={os.getpid()}-{threading.get_ident()}")
        t = time()
        image_data = IL.encode_image(z, format=image_format, meta=meta)
        cache.set(key, image_data, expire=command_args.cache_ttl * 60)
        if im != z:
            im.close()
        z.close()
        logger.debug(f"api: encode: {round(time()-t, 2)}s")
    else:
        im.close()
        logger.debug(f"api: load cache: {style} {scale} {noise}")

    if scale == ScaleOption.X16:
        im, meta = IL.decode_image(image_data, keep_alpha=True)
        im = scale_16x(im, meta)
        image_data = IL.encode_image(im, format=image_format, meta=meta)
        im.close()

    res = HTTPResponse(status=200, body=image_data)
    res.set_header("Content-Type", f"image/{image_format}")
    res.set_header("Content-Disposition", f"inline; filename*=utf-8''{uri_encode(output_filename, safe='')}")

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


def main():
    # NOTE: This code expect the server to run with single-process multi-threading.
    import logging
    import faulthandler

    faulthandler.enable()

    global global_lock, command_args, config, art_ctx, art_scan_ctx, photo_ctx, cache, cache_gc
    global_lock = threading.RLock()
    command_args, config, art_ctx, art_scan_ctx, photo_ctx, cache, cache_gc = setup()
    # HACK: Avoid unintended argparse in the backend(gunicorn).
    sys.argv = [sys.argv[0]]

    global IL
    if command_args.image_lib == "wand":
        # 2x slow than pil_io but it supports 16bit output and various formats
        from nunif.utils import wand_io as IL
    else:
        from nunif.utils import pil_io as IL

    global MAX_NOISE_PIXELS, MAX_SCALE_PIXELS, MAX_BODY_SIZE
    if not command_args.no_size_limit:
        MAX_NOISE_PIXELS = command_args.max_pixels
        MAX_SCALE_PIXELS = (math.sqrt(command_args.max_pixels) / 2) ** 2
        MAX_BODY_SIZE = command_args.max_body_size
    else:
        MAX_NOISE_PIXELS = float("inf")
        MAX_SCALE_PIXELS = float("inf")
        MAX_BODY_SIZE = float("inf")

    if command_args.debug:
        set_log_level(logging.DEBUG)
    if command_args.torch_threads is not None:
        torch.set_num_threads(command_args.torch_threads)
        torch.set_num_interop_threads(command_args.torch_threads)

    if command_args.exit:
        return

    backend_kwargs = {}
    if command_args.backend == "waitress":
        if not command_args.no_size_limit:
            max_request_body_size = command_args.max_body_size * SIZE_MB
        else:
            max_request_body_size = 1073741824  # 1GB
        backend_kwargs = {
            "preload_app": True,
            "threads": command_args.threads,
            "outbuf_overflow": 20 * SIZE_MB,
            "inbuf_overflow": 20 * SIZE_MB,
            "max_request_body_size": max_request_body_size,
            "connection_limit": 256,
            "channel_timeout": 120,
        }
    elif command_args.backend == "gnunicon":
        # NOTE: gunicorn does not work due to `Cannot re-initialize CUDA in forked subprocess`.
        # Maybe gnunicon uses C-level fork() internally.
        backend_kwargs = {
            "preload_app": True,
            "workers": command_args.workers,
        }

    bottle.run(host=command_args.bind_addr, port=command_args.port, debug=command_args.debug,
               server=command_args.backend, **backend_kwargs)
