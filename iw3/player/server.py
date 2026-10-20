import argparse
import asyncio
import ipaddress
import os
import secrets
import socket
import sys
from hypercorn.asyncio import serve
from hypercorn.config import Config
from .app import app
from .dir_config import CACHE_DIR, CERT_DIR, EXAMPLE_DIR
from .media_library import MediaLibrary
from .server_state import DEFAULT_SECRET_KEY_FILE as SECRET_KEY_FILE, ServerState
from .thumbnail_service import init_video_hwaccel


CERT_FILE = os.path.join(CERT_DIR, "server.crt")
KEY_FILE = os.path.join(CERT_DIR, "server.key")


def get_local_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def is_private_address(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return not ip_obj.is_unspecified and ip_obj.is_private
    except ValueError:
        return False


def is_loopback_address(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_loopback
    except ValueError:
        return False


def generate_self_signed_cert(cert_dir, cert_file, key_file, bind_addr="127.0.0.1"):
    from OpenSSL import crypto

    os.makedirs(cert_dir, exist_ok=True)
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)

        cert = crypto.X509()
        cert.set_version(2)  # v3
        cert.get_subject().ST = "Tokyo"
        cert.get_subject().L = "Local"
        cert.get_subject().O = "My App"  # noqa: E741
        cert.get_subject().CN = bind_addr
        cert.set_serial_number(int(os.getpid() + os.getppid() + os.urandom(1)[0]))
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)
        cert.set_issuer(cert.get_subject())

        try:
            ipaddress.ip_address(bind_addr)
            bind_san = f"IP:{bind_addr}".encode()
        except ValueError:
            bind_san = f"DNS:{bind_addr}".encode()

        alt_names = list({b"DNS:localhost", b"IP:127.0.0.1", bind_san})
        san_extension = crypto.X509Extension(b"subjectAltName", False, b", ".join(alt_names))
        cert.add_extensions([san_extension])

        cert.set_pubkey(k)
        cert.sign(k, "sha256")
        with open(cert_file, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(key_file, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
        print(f"Generated new self-signed certificate with SAN: {bind_addr}")


def create_parser():
    local_address = get_local_address()
    parser = argparse.ArgumentParser(description="WebXR View Server")
    parser.add_argument("--root", default=EXAMPLE_DIR, help="Root directory for image files")
    parser.add_argument("--port", type=int, default=1304, help="HTTP listen port")
    parser.add_argument("--bind-addr", type=str, default=local_address, help="HTTP listen address")
    parser.add_argument("--user", type=str, help="HTTP Basic Authentication username")
    parser.add_argument("--password", type=str, help="HTTP Basic Authentication password")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Directory for thumbnail cache")
    parser.add_argument("--cache-size-limit", type=int, default=1000, help="Cache size limit in MB")
    parser.add_argument("--clear-cache", action="store_true", help="Clear thumbnail cache on startup")
    parser.add_argument("--debug", action="store_true", help="Enable debug log window in VR")
    parser.add_argument("--debug-console", action="store_true", help="Enable debug log in browser console only")
    return parser


def set_state_args(args, stop_event=None):
    args.state = {"stop_event": stop_event}


async def serve_with_watchdog(args, config, shutdown_event):
    stop_event = getattr(args, "state", {}).get("stop_event")
    server_task = asyncio.create_task(serve(app, config, shutdown_trigger=shutdown_event.wait))
    try:
        if stop_event:
            while not stop_event.is_set():
                if server_task.done():
                    break
                await asyncio.sleep(0.1)
        else:
            await server_task
    finally:
        if not server_task.done():
            shutdown_event.set()
            timeout = (config.graceful_timeout or 0) + 1.0
            try:
                await asyncio.wait_for(server_task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if not server_task.done():
                    print("Graceful shutdown timed out, cancelling server task...")
                    server_task.cancel()
                    try:
                        await server_task
                    except asyncio.CancelledError:
                        pass

        if server_task.done():
            try:
                await server_task
            except asyncio.CancelledError:
                pass


def server_main(args):
    if args.bind_addr is None:
        args.bind_addr = get_local_address()
        if not is_private_address(args.bind_addr):
            raise RuntimeError(f"Detected IP address({args.bind_addr}) is not LAN Address. Specify --bind-addr")

    if is_loopback_address(args.bind_addr):
        print(f"Warning: {args.bind_addr} is loopback only.", file=sys.stderr)

    auth = (args.user, args.password) if (args.user or args.password) else None
    if not is_private_address(args.bind_addr) and auth is None:
        raise RuntimeError(f"({args.bind_addr}) is not LAN Address. Specify --user/--password")

    init_video_hwaccel()
    generate_self_signed_cert(CERT_DIR, CERT_FILE, KEY_FILE, args.bind_addr)
    if not os.path.exists(SECRET_KEY_FILE):
        print(f"Generating new secret key: {SECRET_KEY_FILE}")
        with open(SECRET_KEY_FILE, "wb") as f:
            f.write(secrets.token_bytes(32))

    args.secret_key_file = SECRET_KEY_FILE
    if hasattr(app.state, "server_state") and app.state.server_state is not None:
        app.state.server_state.close()
        app.state.server_state = None
        app.state.media_library = None

    state = ServerState(args)
    app.state.server_state = state
    app.state.media_library = MediaLibrary(state)

    print(f"Starting iw3-player server on https://{args.bind_addr}:{args.port}")

    config = Config()
    config.bind = [f"{args.bind_addr}:{args.port}"]
    config.certfile = CERT_FILE
    config.keyfile = KEY_FILE
    config.accesslog = None
    config.errorlog = None
    config.graceful_timeout = 1.0  # Allow 1 second for connections to close during shutdown

    shutdown_event = asyncio.Event()

    try:
        asyncio.run(serve_with_watchdog(args, config, shutdown_event))
    except KeyboardInterrupt:
        shutdown_event.set()
        print("\nShutdown requested by user (Ctrl+C)")
    finally:
        state.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    server_main(args)
