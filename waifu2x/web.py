import sys
import os
from os import path
import posixpath
import torch
import argparse
import bottle
from bottle import request, response, HTTPResponse
import threading
import requests
import io
from nunif.logger import logger
from nunif.utils import load_image, decode_image, save_image, ImageLoader
from .waifu2x import Waifu2x
from .models import CUNet, VGG7, UpConv7


DEFAULT_ART_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "cunet", "art"))
DEFAULT_PHOTO_MODEL_DIR = path.abspath(path.join(
    path.join(path.dirname(path.abspath(__file__)), "pretrained_models"),
    "upconv_7", "photo"))


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

    args = parser.parse_args()
    art_ctx = Waifu2x(model_dir=args.art_model_dir, gpus=args.gpu)
    photo_ctx = Waifu2x(model_dir=args.photo_model_dir, gpus=args.gpu)

    art_ctx.load_model_all()
    photo_ctx.load_model_all()

    return args, art_ctx, photo_ctx


global_lock = threading.RLock()
command_args, art_ctx, photo_ctx = setup()
# HACK: Avoid unintended argparse in the backend(gunicorn).
sys.argv = [sys.argv[0]]


def get_image(request):
    # TODO: stream read and reject large file
    file_data = request.files.get("file", "")
    im, meta = None, None

    attached_filename = file_data.filename
    attached_data = file_data.file.read()
    if attached_data:
        im, meta = decode_image(attached_data, attached_filename)
    else:
        url = request.forms.get("url", "")
        if url.startswith("http://") or url.startswith("https://"):
            logger.debug(f"request: {url}")
            res = requests.get(url)
            logger.debug(f"request: {res.status_code} {len(res.content)} {url}")
            if res.status_code == 200:
                im, meta = decode_image(res.content, posixpath.basename(url))
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
    base = meta["filename"] if meta["filename"] else uuid.uuid4() + ".png"
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
    print(dict(request.forms))
    # {'url': 'https://ja.wikipedia.org/static/images/icons/wikipedia.png',
    #  'style': 'photo', 'noise': '1', 'scale': '2'}
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
    with torch.no_grad():
        with global_lock:
            logger.debug(f"process: pid={os.getpid()}, tid={threading.get_ident()}")
            if style == "art":
                rgb, alpha = art_ctx.convert_raw(**ctx_kwargs)
            else:
                rgb, alpha = photo_ctx.convert_raw(**ctx_kwargs)
        z = Waifu2x.to_pil(rgb, alpha)

    output_filename = make_output_filename(style, method, noise_level, meta)
    with io.BytesIO() as image_data:
        save_image(z, meta, image_data)

        res = HTTPResponse(status=200, body=image_data.getvalue())
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
        lang = lang.translate(str.maketrans("", "", "./\\")) # sanitize

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
    bottle.run(host=command_args.bind_addr, port=command_args.port, debug=command_args.debug,
               server=command_args.backend, 
               threads=command_args.threads, workers=command_args.workers, preload_app=True,
               max_request_body_size=command_args.max_body_size * 1024 * 1024)
