import argparse
import asyncio
import hashlib
import io
import ipaddress
import json
import math
import os
import posixpath
import socket
import threading
import uuid
from configparser import ConfigParser
from contextlib import asynccontextmanager
from enum import Enum
from os import path
from time import time
from typing import Any
from urllib.parse import (
    quote as uri_encode,
)
from urllib.parse import (
    unquote_plus as uri_decode,
)
from urllib.parse import (
    urljoin,
    urlparse,
)

import anyio
import filelock
import psutil
import requests
import torch
import urllib3
from diskcache import Cache
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from starlette.datastructures import UploadFile

from nunif.device import mps_is_available, xpu_is_available
from nunif.initializer import gc_collect
from nunif.logger import logger, set_log_level
from nunif.utils.filename import set_image_ext
from nunif.utils.home_dir import ensure_home_dir

from ..model_dir import MODEL_DIR
from ..utils import Waifu2x
from .public_dir import PUBLIC_DIR

DEFAULT_ART_MODEL_DIR = path.abspath(path.join(MODEL_DIR, "swin_unet_v3", "art"))
DEFAULT_ART_SCAN_MODEL_DIR = path.abspath(path.join(MODEL_DIR, "swin_unet", "art_scan"))
DEFAULT_PHOTO_MODEL_DIR = path.abspath(path.join(MODEL_DIR, "swin_unet", "photo"))
BUFF_SIZE = 8192  # buffer block size for io access
SIZE_MB = 1024 * 1024
MAX_URL_REDIRECTS = 5
RECAPTCHA_VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
TMP_DIR = ensure_home_dir("waifu2x", path.join(path.dirname(__file__), "..", "..", "tmp"))
COMPILE_LOCK = path.join(TMP_DIR, "waifu2x-web-compile.lock")
os.makedirs(TMP_DIR, exist_ok=True)

global_lock: threading.RLock = threading.RLock()
command_args: Any = None
config: ConfigParser = ConfigParser()
art_ctx: Any = None
art_scan_ctx: Any = None
photo_ctx: Any = None
cache: Any = None
cache_gc: Any = None
IL: Any = None
MAX_NOISE_PIXELS: float = 3000 * 3000
MAX_SCALE_PIXELS: float = 1500 * 1500
MAX_BODY_SIZE: float = 5
MAX_REQUEST_BODY_SIZE: int | None = 6 * SIZE_MB


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


class CacheGC:
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
            gc_collect()
            self.last_expired_at = time()
            logger.info(f"diskcache: cache={self.disk_size_mb()}MB, RAM={self.ram_size_mb()}")


class BodyTooLargeError(Exception):
    pass


class MaxBodySizeMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_size = MAX_REQUEST_BODY_SIZE
        if max_size is None:
            await self.app(scope, receive, send)
            return

        for header_name, header_val in scope.get("headers", []):
            if header_name == b"content-length":
                try:
                    if int(header_val) > max_size:
                        res = PlainTextResponse(
                            f"Request entity too large (max: {MAX_BODY_SIZE}MB)",
                            status_code=413,
                        )
                        await res(scope, receive, send)
                        return
                except ValueError:
                    res = PlainTextResponse("Invalid Content-Length", status_code=400)
                    await res(scope, receive, send)
                    return

        received = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_size:
                    raise BodyTooLargeError()
            return message

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLargeError:
            if not response_started:
                res = PlainTextResponse(
                    f"Request entity too large (max: {MAX_BODY_SIZE}MB)",
                    status_code=413,
                )
                await res(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if command_args is not None:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(40, int(command_args.threads))
    yield


app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)
app.add_middleware(MaxBodySizeMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> PlainTextResponse:
    return PlainTextResponse(content=str(exc.detail), status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> PlainTextResponse:
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return PlainTextResponse(content="Internal Server Error", status_code=500)


def setup():
    if torch.cuda.is_available() or mps_is_available() or xpu_is_available():
        default_gpu = 0
    else:
        default_gpu = -1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bind-addr", type=str, default="127.0.0.1", help="0.0.0.0 for global, 127.0.0.1 for local")
    parser.add_argument("--port", type=int, default=8812, help="HTTP port number")
    parser.add_argument("--root", type=str, default=PUBLIC_DIR, help="web root directory")
    parser.add_argument(
        "--backend",
        type=str,
        default="hypercorn",
        help="server backend (deprecated, ignored; always uses hypercorn)",
    )
    parser.add_argument("--workers", type=int, default=1, help="(deprecated, ignored)")
    parser.add_argument("--threads", type=int, default=32, help="The number of threads")
    parser.add_argument("--debug", action="store_true", help="Debug print")
    parser.add_argument("--max-body-size", type=int, default=5, help="maximum allowed size(MB) for uploaded files")
    parser.add_argument("--max-pixels", type=int, default=3000 * 3000, help="maximum number of output image pixels ")
    parser.add_argument("--url-timeout", type=int, default=10, help="request_timeout for url")

    parser.add_argument("--art-model-dir", type=str, default=DEFAULT_ART_MODEL_DIR, help="art model dir")
    parser.add_argument("--art-scan-model-dir", type=str, default=DEFAULT_ART_SCAN_MODEL_DIR, help="art scan model dir")
    parser.add_argument("--photo-model-dir", type=str, default=DEFAULT_PHOTO_MODEL_DIR, help="photo model dir")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu], help="GPU device ids. -1 for CPU")
    parser.add_argument("--tile-size", type=int, default=None, help="tile size for tiled render")
    parser.add_argument("--batch-size", type=int, default=None, help="minibatch size for tiled render")
    parser.add_argument("--tta", action="store_true", help="use TTA mode")
    parser.add_argument("--disable-amp", action="store_true", help="disable AMP for some special reason")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument("--warmup", action="store_true", help="warmup at startup")
    parser.add_argument(
        "--image-lib", type=str, choices=["pil", "wand"], default="pil", help="image library to encode/decode images"
    )
    parser.add_argument("--cache-ttl", type=int, default=30, help="cache TTL(min)")
    parser.add_argument("--cache-size-limit", type=int, default=10, help="cache size limit (GB)")
    parser.add_argument("--cache-dir", type=str, default=path.join(TMP_DIR, "waifu2x_cache"), help="cache dir")
    parser.add_argument("--enable-recaptcha", action="store_true", help="enable reCAPTCHA. it requires --config option")
    parser.add_argument(
        "--enable-turnstile", action="store_true", help="enable CloudFlare Turnstile. it requires --config option"
    )
    parser.add_argument(
        "--prefer-webp", action="store_true", help="If browser side PNG conversion is compatible, WebP is returned."
    )
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
        # Force `batch_size=1` to avoid recompilation.
        args.batch_size = 1
        compile_kwargs = {}
        if args.compile_mode != "default":
            compile_kwargs["mode"] = args.compile_mode
        compile_lock = filelock.FileLock(COMPILE_LOCK)
        with compile_lock:
            logger.info(f"Compiling models (mode={args.compile_mode})...")
            art_ctx.compile(**compile_kwargs)
            art_scan_ctx.compile(**compile_kwargs)
            photo_ctx.compile(**compile_kwargs)
            if args.warmup:
                art_ctx.warmup(tile_size=args.tile_size, batch_size=args.batch_size, enable_amp=not args.disable_amp)
                art_scan_ctx.warmup(
                    tile_size=args.tile_size, batch_size=args.batch_size, enable_amp=not args.disable_amp
                )
                photo_ctx.warmup(tile_size=args.tile_size, batch_size=args.batch_size, enable_amp=not args.disable_amp)
            logger.info("Done")

    gc_collect()

    cache = Cache(args.cache_dir, size_limit=args.cache_size_limit * 1073741824)
    cache_gc = CacheGC(cache, 60)
    config = ConfigParser()
    if args.config:
        config.read(args.config)
    if "recaptcha" not in config:
        if args.enable_recaptcha:
            raise RuntimeError("--enable-recaptcha: No recaptcha setting in config file")
        else:
            config["recaptcha"] = {"site_key": "", "secret_key": ""}
    if "turnstile" not in config:
        if args.enable_turnstile:
            raise RuntimeError("--enable-turnstile: No turnstile setting in config file")
        else:
            config["turnstile"] = {"site_key": "", "secret_key": ""}

    return args, config, art_ctx, art_scan_ctx, photo_ctx, cache, cache_gc


class UnsafeURLError(requests.exceptions.RequestException):
    pass


def is_public_ip(ip_str: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_str.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped
    return ip_obj.is_global and not ip_obj.is_multicast


def create_public_connection(
    host: str,
    port: int,
    timeout: Any = None,
    source_address: tuple[str, int] | None = None,
    socket_options: Any = None,
) -> socket.socket:
    host_clean = host.strip("[]")
    infos = socket.getaddrinfo(host_clean, port, 0, socket.SOCK_STREAM)
    if not infos:
        raise UnsafeURLError("No addresses resolved")
    for _, _, _, _, sockaddr in infos:
        if not isinstance(sockaddr[0], str) or not is_public_ip(sockaddr[0]):
            raise UnsafeURLError(f"Blocked non-public IP: {sockaddr[0]}")

    global_default_timeout = getattr(socket, "_GLOBAL_DEFAULT_TIMEOUT", None)
    last_err: OSError | None = None
    for family, socktype, proto, _, sockaddr in infos:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(family, socktype, proto)
            for opt in socket_options or []:
                sock.setsockopt(*opt)
            if timeout is not None and timeout is not global_default_timeout:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            last_err = e
            if sock is not None:
                sock.close()
    assert last_err is not None
    raise last_err


class SafeHTTPConnection(urllib3.connection.HTTPConnection):
    def _new_conn(self) -> socket.socket:
        return create_public_connection(
            self.host,
            self.port,
            timeout=self.timeout,
            source_address=self.source_address,
            socket_options=self.socket_options,
        )


class SafeHTTPSConnection(urllib3.connection.HTTPSConnection):
    def _new_conn(self) -> socket.socket:
        return create_public_connection(
            self.host,
            self.port,
            timeout=self.timeout,
            source_address=self.source_address,
            socket_options=self.socket_options,
        )


class SafeHTTPConnectionPool(urllib3.HTTPConnectionPool):
    ConnectionCls = SafeHTTPConnection  # type: ignore[assignment]


class SafeHTTPSConnectionPool(urllib3.HTTPSConnectionPool):
    ConnectionCls = SafeHTTPSConnection  # type: ignore[assignment]


class SafeHTTPAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": SafeHTTPConnectionPool,
            "https": SafeHTTPSConnectionPool,
        }


def validate_public_url(url: str) -> None:
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        raise HTTPException(status_code=400, detail="URL Error")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="URL Error")
    if port is not None and not (1 <= port <= 65535):
        raise HTTPException(status_code=400, detail="URL Error")


def new_safe_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    adapter = SafeHTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_uploaded_file(upload_file: UploadFile):
    max_body_size = MAX_BODY_SIZE * SIZE_MB
    with io.BytesIO() as upload_buff:
        file_size = 0
        buff = upload_file.file.read(BUFF_SIZE)
        while buff:
            file_size += len(buff)
            upload_buff.write(buff)
            if file_size > max_body_size:
                logger.debug("fetch_uploaded_file: error: too large")
                raise HTTPException(status_code=413, detail=f"Request entity too large (max: {MAX_BODY_SIZE}MB)")
            buff = upload_file.file.read(BUFF_SIZE)

        image_data = upload_buff.getvalue()
        if image_data:
            logger.debug(f"fetch_uploaded_file: {round(len(image_data) / SIZE_MB, 3)}MB")
        return image_data


def fetch_url_file(url: str):
    max_body_size = MAX_BODY_SIZE * SIZE_MB
    timeout = command_args.url_timeout
    current_url = url
    try:
        with new_safe_session() as session:
            for _ in range(MAX_URL_REDIRECTS + 1):
                validate_public_url(current_url)
                headers = {"Referer": current_url, "User-Agent": "waifu2x/web.py"}
                with session.get(
                    current_url, headers=headers, stream=True, timeout=timeout, allow_redirects=False
                ) as res:
                    if res.status_code in {301, 302, 303, 307, 308}:
                        location = res.headers.get("Location")
                        if not location:
                            raise HTTPException(status_code=400, detail="URL error")
                        current_url = urljoin(current_url, location)
                        continue
                    if res.status_code != 200:
                        logger.debug(f"fetch_url_file: error: status={res.status_code}, {current_url}")
                        raise HTTPException(status_code=400, detail="URL error")
                    with io.BytesIO() as buff:
                        file_size = 0
                        for chunk in res.iter_content(chunk_size=BUFF_SIZE):
                            buff.write(chunk)
                            file_size += len(chunk)
                            if file_size > max_body_size:
                                logger.debug(f"fetch_url_file: error: too large, {current_url}")
                                raise HTTPException(
                                    status_code=413, detail=f"Request entity too large (max: {MAX_BODY_SIZE}MB)"
                                )
                        logger.debug(f"fetch_url_file: {round(file_size / SIZE_MB, 3)}MB, {current_url}")
                        return buff.getvalue()
            logger.debug(f"fetch_url_file: error: too many redirects, {url}")
            raise HTTPException(status_code=400, detail="URL Error")
    except requests.exceptions.RequestException as e:
        logger.debug(f"fetch_url_file: error: {e}, {current_url}")
        raise HTTPException(status_code=400, detail="URL Error")
    except UnicodeEncodeError as e:
        logger.debug(f"fetch_url_file: error: {e}, {current_url}")
        raise HTTPException(status_code=400, detail="URL Error")


def check_image_pixels(image_data: bytes, max_pixels: float) -> None:
    if math.isinf(max_pixels):
        return
    if command_args.image_lib == "wand":
        from wand.image import Image as WandImage

        try:
            with WandImage(blob=image_data, ping=True) as probe:
                if probe.width * probe.height > max_pixels:
                    raise HTTPException(status_code=413, detail="Request Image Too Large")
        except HTTPException:
            raise
        except Exception:
            return
    else:
        from PIL import Image as PILImage
        from PIL import UnidentifiedImageError

        try:
            with PILImage.open(io.BytesIO(image_data)) as probe:
                if probe.size[0] * probe.size[1] > max_pixels:
                    raise HTTPException(status_code=413, detail="Request Image Too Large")
        except PILImage.DecompressionBombError:
            raise HTTPException(status_code=413, detail="Request Image Too Large")
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            return


def fetch_image(form: Any, max_pixels: float = float("inf")):
    upload_file = form.get("file")
    im, meta = None, None

    image_data = None
    if isinstance(upload_file, UploadFile):
        image_data = fetch_uploaded_file(upload_file)
    if image_data:
        filename = upload_file.filename if isinstance(upload_file, UploadFile) else None
        if not filename:
            filename = str(uuid.uuid4())
        check_image_pixels(image_data, max_pixels)
        im, meta = IL.decode_image(image_data, filename, color="rgb", keep_alpha=True, exif_transpose=True)
    else:
        url = form.get("url", "")
        if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
            key = "url_" + url
            image_data = cache.get(key, None)
            filename = posixpath.basename(urlparse(url).path)
            if not filename:
                filename = str(uuid.uuid4())
            else:
                filename = uri_decode(filename)
            if image_data is not None:
                logger.debug(f"fetch_image: load cache: {url}")
                check_image_pixels(image_data, max_pixels)
                im, meta = IL.decode_image(image_data, filename, color="rgb", keep_alpha=True, exif_transpose=True)
            else:
                image_data = fetch_url_file(url)
                check_image_pixels(image_data, max_pixels)
                im, meta = IL.decode_image(image_data, filename, color="rgb", keep_alpha=True, exif_transpose=True)
                cache.set(key, image_data, expire=command_args.cache_ttl * 60)

    if image_data is not None and meta is not None:
        # sha1 for cache key
        meta["sha1"] = hashlib.sha1(image_data).hexdigest()
    return im, meta


def parse_request(form: Any):
    try:
        style = StyleOption(str(form.get("style", "photo")))
        scale = ScaleOption(int(str(form.get("scale", "-1"))))
        noise = NoiseOption(int(str(form.get("noise", "-1"))))
        image_format_opt = FormatOption(int(str(form.get("format", "0"))))
    except ValueError:
        raise HTTPException(status_code=400, detail="Bad Request")

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
    if image_format_opt == FormatOption.PNG:
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


@app.get("/recaptcha_state.json")
async def recaptcha_state() -> Response:
    state = {
        "recaptcha_enabled": command_args.enable_recaptcha,
        "recaptcha_site_key": config["recaptcha"]["site_key"],
        "turnstile_enabled": command_args.enable_turnstile,
        "turnstile_site_key": config["turnstile"]["site_key"],
    }
    return Response(content=json.dumps(state), media_type="application/json")


def get_client_ip(request: Request) -> str:
    peer_ip = request.client.host if request.client else "127.0.0.1"
    try:
        peer_obj = ipaddress.ip_address(peer_ip)
        is_trusted_proxy = peer_obj.is_loopback or peer_obj.is_private
    except ValueError:
        is_trusted_proxy = False

    if is_trusted_proxy:
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",", 1)[0].strip()

    return peer_ip


def verify_recaptcha(request: Request, form: Any) -> bool:
    if not command_args.enable_recaptcha:
        return True

    timeout = command_args.url_timeout
    data = {
        "response": str(form.get("recap", "")),
        "secret": config["recaptcha"]["secret_key"],
        "remoteip": get_client_ip(request),
    }
    try:
        res = requests.post(RECAPTCHA_VERIFY_URL, data=data, timeout=timeout)
        if res.status_code == 200:
            result = json.loads(res.text)
            logger.debug("verify_recaptcha: " + ("success" if result["success"] else "failure"))
            return result["success"]
        else:
            logger.error(f"verify_recaptcha: HTTP Error {res.status_code} {res.text}")
            raise HTTPException(status_code=500, detail="reCAPTCHA Error")
    except requests.exceptions.RequestException as e:
        logger.error(f"verify_recaptcha: error: {e}")
        raise HTTPException(status_code=500, detail="reCAPTCHA Error")


def verify_turnstile(request: Request, form: Any) -> bool:
    if not command_args.enable_turnstile:
        return True

    timeout = command_args.url_timeout
    data = {
        "response": str(form.get("turnstile", "")),
        "secret": config["turnstile"]["secret_key"],
        "remoteip": get_client_ip(request),
    }
    try:
        res = requests.post(TURNSTILE_VERIFY_URL, data=data, timeout=timeout)
        if res.status_code == 200:
            result = json.loads(res.text)
            logger.debug("verify_turnstile: " + ("success" if result["success"] else "failure"))
            return result["success"]
        else:
            logger.error(f"verify_turnstile: HTTP Error {res.status_code} {res.text}")
            raise HTTPException(status_code=500, detail="Turnstile Error")
    except requests.exceptions.RequestException as e:
        logger.error(f"verify_turnstile: error: {e}")
        raise HTTPException(status_code=500, detail="Turnstile Error")


def dump_meta(meta):
    return {k: v if isinstance(v, (str, int, float, type(None))) else str(type(v)) for k, v in meta.items()}


def scale_16x(im, meta):
    w, h = int(im.size[0] * (1.6 / 2.0)), int(im.size[1] * (1.6 / 2.0))
    if meta["engine"] == "wand":
        im.resize(w, h, "lanczos")
    elif meta["engine"] == "pil":
        im = im.resize((w, h), 1)
    return im


def requires_png(meta):
    if "grayscale" in meta and meta["grayscale"]:
        return True

    if "depth" in meta and meta["depth"] > 8:
        return True

    if "icc_profile" in meta and meta["icc_profile"] is not None:
        return True

    return False


@app.get("/api")
def api_get() -> Response:
    return RedirectResponse(url="/", status_code=302)


def is_disconnected_sync(request: Request) -> bool:
    if getattr(request, "_is_disconnected", False):
        return True
    try:
        return bool(anyio.from_thread.run(request.is_disconnected))
    except RuntimeError:
        return False


def run_api_sync(request: Request, form: Any) -> tuple[bytes, str, str]:
    style, method, scale, noise, image_format = parse_request(form)

    if command_args.enable_recaptcha and not verify_recaptcha(request, form):
        raise HTTPException(status_code=401, detail="reCAPTCHA Error")
    if command_args.enable_turnstile and not verify_turnstile(request, form):
        raise HTTPException(status_code=401, detail="Turnstile Error")

    with global_lock:
        cache_gc.gc()
    if is_disconnected_sync(request):
        logger.debug("api: client disconnected before fetch_image")
        raise HTTPException(status_code=499, detail="Client Closed Request")

    max_pixels = MAX_SCALE_PIXELS if scale != ScaleOption.NONE else MAX_NOISE_PIXELS
    im, meta = fetch_image(form, max_pixels=max_pixels)
    if im is None:
        raise HTTPException(status_code=400, detail="Image Load Error")

    if command_args.prefer_webp and not requires_png(meta):
        image_format = "webp"

    logger.debug(f"api: image: {dump_meta(meta)}")
    if scale != ScaleOption.NONE and im.size[0] * im.size[1] > MAX_SCALE_PIXELS:
        im.close()
        raise HTTPException(status_code=413, detail="Request Image Too Large")
    if scale == ScaleOption.NONE and im.size[0] * im.size[1] > MAX_NOISE_PIXELS:
        im.close()
        raise HTTPException(status_code=413, detail="Request Image Too Large")

    output_filename = make_output_filename(style, method, noise, image_format, meta)
    key = meta["sha1"] + output_filename
    image_data = cache.get(key, None)
    if image_data is None:
        t = time()
        if method == "none":
            logger.debug(
                f"api: forward: {style} {scale} {noise} {image_format}pid={os.getpid()}-{threading.get_ident()}"
            )
            z = im
        else:
            if is_disconnected_sync(request):
                im.close()
                logger.debug("api: client disconnected before to_tensor")
                raise HTTPException(status_code=499, detail="Client Closed Request")
            with torch.inference_mode():
                rgb, alpha = IL.to_tensor(im, return_alpha=True, device=art_ctx.device)
                if rgb.shape[0] == 1:
                    rgb = rgb.repeat(3, 1, 1)
                ctx_kwargs = {
                    "x": rgb,
                    "alpha": alpha,
                    "method": method,
                    "noise_level": noise.value,
                    "tile_size": command_args.tile_size,
                    "batch_size": command_args.batch_size,
                    "tta": command_args.tta,
                    "enable_amp": not command_args.disable_amp,
                    "output_device": art_ctx.device,
                }
                if is_disconnected_sync(request):
                    del rgb, alpha
                    im.close()
                    logger.debug("api: client disconnected before lock acquisition")
                    raise HTTPException(status_code=499, detail="Client Closed Request")
                with global_lock:
                    if is_disconnected_sync(request):
                        del rgb, alpha
                        im.close()
                        logger.debug("api: client disconnected before model execution")
                        raise HTTPException(status_code=499, detail="Client Closed Request")
                    if style == StyleOption.ART:
                        rgb, alpha = art_ctx.convert(**ctx_kwargs)
                    elif style == StyleOption.ART_SCAN:
                        rgb, alpha = art_scan_ctx.convert(**ctx_kwargs)
                    else:
                        rgb, alpha = photo_ctx.convert(**ctx_kwargs)
                depth = meta["depth"] if "depth" in meta and meta["depth"] is not None else 8
                z = IL.to_image(rgb, alpha, depth=depth)
                del rgb, alpha
            logger.debug(
                f"api: forward: {round(time() - t, 2)}s, {style} {scale} {noise} {image_format}, "
                f"pid={os.getpid()}-{threading.get_ident()}"
            )
        t = time()
        image_data = IL.encode_image(z, format=image_format, meta=meta)
        cache.set(key, image_data, expire=command_args.cache_ttl * 60)
        if im != z:
            im.close()
        z.close()
        logger.debug(f"api: encode: {round(time() - t, 2)}s")
    else:
        im.close()
        logger.debug(f"api: load cache: {style} {scale} {noise}")

    if scale == ScaleOption.X16:
        im, meta = IL.decode_image(image_data, keep_alpha=True)
        im = scale_16x(im, meta)
        image_data = IL.encode_image(im, format=image_format, meta=meta)
        im.close()

    return image_data, image_format, output_filename


@app.post("/api")
async def api(request: Request) -> Response:
    # {'url': 'https://ja.wikipedia.org/static/images/icons/wikipedia.png',
    #  'style': 'photo', 'noise': '1', 'scale': '2', 'recap': 'xxxxx'}
    form = await request.form(max_files=1)
    try:
        image_data, image_format, output_filename = await anyio.to_thread.run_sync(run_api_sync, request, form)
    finally:
        await form.close()

    return Response(
        content=image_data,
        media_type=f"image/{image_format}",
        headers={"Content-Disposition": f"inline; filename*=utf-8''{uri_encode(output_filename, safe='')}"},
    )


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
    elif lang is not None and (lang.startswith("id-") or lang in {"in", "in-ID"}):
        index_file = "index.id.html"
    else:
        index_file = f"index.{lang}.html"

    if path.exists(path.join(root_dir, index_file)):
        return index_file
    else:
        return "index.html"


@app.get("/{url:path}")
async def static_file(url: str, request: Request) -> FileResponse:
    """
    This assumes that `root` directory is flat.
    In production environment, this method is not used, instead it is directly sent from nginx.
    """
    url = url.replace("\\", "/")
    dirname = posixpath.dirname(url)
    basename = posixpath.basename(url)
    if dirname:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not basename:
        basename = resolve_index_file(command_args.root, request.headers.get("accept-language", ""))

    abs_root = path.realpath(command_args.root)
    abs_path = path.realpath(path.join(abs_root, basename))
    if path.commonpath([abs_root, abs_path]) != abs_root or not path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(abs_path)


def create_hypercorn_config(bind_addr: str, port: int, debug: bool = False) -> Config:
    config = Config()
    config.bind = [f"{bind_addr}:{port}"]
    config.backlog = 256
    config.keep_alive_timeout = 120
    config.graceful_timeout = 1.0
    config.include_server_header = False
    config.errorlog = "-"
    if debug:
        config.accesslog = "-"
    else:
        config.accesslog = None
    return config


def main():
    # NOTE: This code expect the server to run with single-process multi-threading.
    import faulthandler
    import logging

    faulthandler.enable()

    global global_lock, command_args, config, art_ctx, art_scan_ctx, photo_ctx, cache, cache_gc
    global_lock = threading.RLock()
    command_args, config, art_ctx, art_scan_ctx, photo_ctx, cache, cache_gc = setup()

    global IL
    if command_args.image_lib == "wand":
        # 2x slow than pil_io but it supports 16bit output and various formats
        from nunif.utils import wand_io as IL
    else:
        from nunif.utils import pil_io as IL

    global MAX_NOISE_PIXELS, MAX_SCALE_PIXELS, MAX_BODY_SIZE, MAX_REQUEST_BODY_SIZE
    if not command_args.no_size_limit:
        MAX_NOISE_PIXELS = command_args.max_pixels
        MAX_SCALE_PIXELS = (math.sqrt(command_args.max_pixels) / 2) ** 2
        MAX_BODY_SIZE = command_args.max_body_size
        MAX_REQUEST_BODY_SIZE = (command_args.max_body_size + 1) * SIZE_MB
        if command_args.image_lib == "pil":
            from PIL import Image as PILImage

            PILImage.MAX_IMAGE_PIXELS = max(1, int(command_args.max_pixels))
    else:
        MAX_NOISE_PIXELS = float("inf")
        MAX_SCALE_PIXELS = float("inf")
        MAX_BODY_SIZE = float("inf")
        MAX_REQUEST_BODY_SIZE = 1024 * SIZE_MB

    if command_args.debug:
        set_log_level(logging.DEBUG)
    if command_args.torch_threads is not None:
        torch.set_num_threads(command_args.torch_threads)
        torch.set_num_interop_threads(command_args.torch_threads)

    if command_args.exit:
        return

    hypercorn_config = create_hypercorn_config(
        bind_addr=command_args.bind_addr,
        port=command_args.port,
        debug=command_args.debug,
    )
    try:
        asyncio.run(serve(app, hypercorn_config))  # type: ignore[arg-type]
    except KeyboardInterrupt:
        pass
