import argparse
import asyncio
import datetime
import ipaddress
import os
import secrets
import socket
import sys
import threading
from typing import Any, cast

from hypercorn.asyncio.lifespan import Lifespan
from hypercorn.asyncio.tcp_server import TCPServer
from hypercorn.asyncio.worker_context import WorkerContext
from hypercorn.config import Config
from hypercorn.events import Closed
from hypercorn.typing import Event as HypercornEvent
from hypercorn.typing import Framework, LifespanState
from hypercorn.utils import repr_socket_addr, wrap_app

from .app import app
from .dir_config import CACHE_DIR, CERT_DIR, EXAMPLE_DIR
from .media_library import MediaLibrary
from .server_state import DEFAULT_SECRET_KEY_FILE as SECRET_KEY_FILE
from .server_state import ServerState
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


def _write_private_file(file_path: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(file_path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            fd = -1
            f.write(data)
    finally:
        if fd != -1:
            os.close(fd)


def generate_self_signed_cert(
    cert_dir: str,
    cert_file: str,
    key_file: str,
    bind_addr: str = "127.0.0.1",
) -> None:
    os.makedirs(cert_dir, mode=0o700, exist_ok=True)
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Tokyo"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "My App"),
                x509.NameAttribute(NameOID.COMMON_NAME, bind_addr),
            ]
        )

        san_entries: list[x509.GeneralName] = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]
        try:
            bind_san: x509.GeneralName = x509.IPAddress(ipaddress.ip_address(bind_addr))
        except ValueError:
            bind_san = x509.DNSName(bind_addr)
        if bind_san not in san_entries:
            san_entries.append(bind_san)

        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=10 * 365))
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .sign(private_key=key, algorithm=hashes.SHA256())
        )

        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        _write_private_file(
            key_file,
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ),
        )
        print(f"Generated new self-signed certificate with SAN: {bind_addr}")


def ensure_server_credentials(
    bind_addr: str,
    cert_dir: str = CERT_DIR,
    cert_file: str = CERT_FILE,
    key_file: str = KEY_FILE,
    secret_key_file: str = SECRET_KEY_FILE,
) -> tuple[str, str, str]:
    generate_self_signed_cert(cert_dir, cert_file, key_file, bind_addr)
    if not os.path.exists(secret_key_file):
        print(f"Generating new secret key: {secret_key_file}")
        _write_private_file(secret_key_file, secrets.token_bytes(32))
    return cert_file, key_file, secret_key_file


def create_hypercorn_config(
    bind_addr: str,
    port: int,
    cert_file: str | None = CERT_FILE,
    key_file: str | None = KEY_FILE,
    graceful_timeout: float = 1.0,
) -> Config:
    config = Config()
    config.bind = [f"{bind_addr}:{port}"]
    if cert_file is not None and key_file is not None:
        config.certfile = cert_file
        config.keyfile = key_file
    config.accesslog = None
    config.errorlog = None
    config.graceful_timeout = graceful_timeout
    return config


def init_server_runtime(
    args: Any,
    bind_addr: str,
    port: int,
    enable_media_library: bool = True,
    graceful_timeout: float = 1.0,
) -> tuple[ServerState, Config]:
    cert_file, key_file, secret_key_file = ensure_server_credentials(bind_addr)
    args.secret_key_file = secret_key_file

    if hasattr(app.state, "server_state") and app.state.server_state is not None:
        app.state.server_state.close()
        app.state.server_state = None
        app.state.media_library = None

    state = ServerState(args)
    app.state.server_state = state
    app.state.media_library = MediaLibrary(state) if enable_media_library else None
    config = create_hypercorn_config(
        bind_addr=bind_addr,
        port=port,
        cert_file=cert_file,
        key_file=key_file,
        graceful_timeout=graceful_timeout,
    )
    return state, config


def validate_bind_address_and_auth(args: Any) -> tuple[str, str] | None:
    if args.bind_addr is None:
        args.bind_addr = get_local_address()
        if not is_private_address(args.bind_addr):
            raise RuntimeError(f"Detected IP address({args.bind_addr}) is not LAN Address. Specify --bind-addr")

    if is_loopback_address(args.bind_addr):
        print(
            (
                f"Warning: {args.bind_addr} is only accessible from this PC. "
                "If this option is not explicitly enabled, the network connection might be unavailable."
            ),
            file=sys.stderr,
        )

    if args.user or args.password:
        auth: tuple[str, str] | None = (args.user or "", args.password or "")
    else:
        auth = None

    if not is_private_address(args.bind_addr) and auth is None:
        raise RuntimeError(f"({args.bind_addr}) is not LAN Address. Specify --user/--password")

    return auth


def _add_player_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--root", default=EXAMPLE_DIR, help="Root directory for image files")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Directory for thumbnail cache")
    parser.add_argument("--cache-size-limit", type=int, default=1000, help="Cache size limit in MB")
    parser.add_argument("--clear-cache", action="store_true", help="Clear thumbnail cache on startup")
    parser.add_argument("--debug", action="store_true", help="Enable debug log window in VR")
    parser.add_argument("--debug-console", action="store_true", help="Enable debug log in browser console only")
    parser.add_argument("--enable-stream", action="store_true", help="Enable desktop screenshot streaming")
    return parser


class _PlayerArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        effective_args = sys.argv[1:] if args is None else list(args)
        if (
            "--enable-stream" in effective_args
            or getattr(namespace, "enable_stream", False)
            or bool(self.get_default("enable_stream"))
        ):
            full_parser = create_parser(enable_stream=True)
            full_parser.set_defaults(**self._defaults)
            return full_parser.parse_args(args=args, namespace=namespace)
        return super().parse_args(args=args, namespace=namespace)


def create_parser(enable_stream: bool = False) -> argparse.ArgumentParser:
    if enable_stream:
        from .. import utils as IW3U
        from ..desktop.utils import add_desktop_stream_args

        parser = IW3U.create_parser(required_true=False)
        add_desktop_stream_args(parser, default_port=1304)
        parser.description = "WebXR View Server"
        return _add_player_args(parser)

    parser = _PlayerArgumentParser(description="WebXR View Server")
    parser.add_argument("--port", type=int, default=1304, help="HTTP listen port")
    parser.add_argument("--bind-addr", type=str, default=get_local_address(), help="HTTP listen address")
    parser.add_argument("--user", type=str, help="HTTP Basic Authentication username")
    parser.add_argument("--password", type=str, help="HTTP Basic Authentication password")
    return _add_player_args(parser)


def set_state_args(args, args_lock=None, stop_event=None, fps_event=None, tqdm_fn=None, depth_model=None):
    if getattr(args, "enable_stream", False):
        from ..desktop.utils import set_state_args as set_desktop_state_args

        set_desktop_state_args(
            args,
            args_lock=args_lock,
            stop_event=stop_event,
            fps_event=fps_event,
            tqdm_fn=tqdm_fn,
            depth_model=depth_model,
        )
    else:
        args.state = {
            "stop_event": stop_event,
            "fps_event": fps_event,
            "tqdm_fn": tqdm_fn,
            "args_lock": args_lock,
        }


def _notify_stream_shutdown(args: argparse.Namespace) -> None:
    stream_hub = getattr(args, "stream_hub", None)
    if stream_hub is None and hasattr(app.state, "server_state") and app.state.server_state is not None:
        stream_hub = getattr(app.state.server_state, "stream_hub", None)
    if stream_hub is not None:
        stream_hub.shutdown_event.set()
        stream_hub.clear_frames(stop_clients=True)


async def _wait_for_shutdown_signal(
    stop_event: threading.Event | None,
    shutdown_event: asyncio.Event,
    terminate_event: HypercornEvent,
) -> None:
    if stop_event is not None:
        while not stop_event.is_set() and not shutdown_event.is_set() and not terminate_event.is_set():
            await asyncio.sleep(0.05)
        return

    shutdown_task = asyncio.create_task(shutdown_event.wait())
    terminate_task = asyncio.create_task(terminate_event.wait())
    try:
        await asyncio.wait({shutdown_task, terminate_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (shutdown_task, terminate_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(shutdown_task, terminate_task, return_exceptions=True)


async def _close_pending_connections(
    active_connections: dict[asyncio.Task[None], TCPServer],
    graceful_timeout: float | None,
) -> None:
    async def _wait_active(timeout: float | None) -> set[asyncio.Task[None]]:
        if not active_connections:
            return set()
        _, pending_tasks = await asyncio.wait(list(active_connections.keys()), timeout=timeout)
        return pending_tasks

    pending = await _wait_active(graceful_timeout)
    if not pending:
        return

    # Close protocols before cancelling tasks so Hypercorn's h2/TaskGroup does not
    # schedule _idle_timeout onto a shutting-down TaskGroup.
    close_tasks: list[asyncio.Task[None]] = []
    for task in list(pending):
        tcp_server = active_connections.get(task)
        if tcp_server is None:
            continue
        if hasattr(tcp_server, "protocol"):
            close_tasks.append(asyncio.create_task(tcp_server.protocol.handle(Closed())))
        try:
            tcp_server.writer.close()
        except Exception:
            pass

    pending = await _wait_active(0.5)

    if pending:
        for task in list(pending):
            tcp_server = active_connections.get(task)
            if tcp_server is not None:
                transport = getattr(tcp_server.writer, "transport", None)
                if transport is not None:
                    transport.abort()
        pending = await _wait_active(0.5)

    for close_task in close_tasks:
        if not close_task.done():
            close_task.cancel()
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)

    for _ in range(2):
        if not pending:
            break
        for task in list(pending):
            task.cancel()
        _, pending = await asyncio.wait(pending, timeout=0.5)


async def serve_with_watchdog(
    args: argparse.Namespace,
    config: Config,
    shutdown_event: asyncio.Event,
) -> None:
    stop_event = getattr(args, "state", {}).get("stop_event")
    loop = asyncio.get_running_loop()
    wrapped_app = wrap_app(cast(Framework, app), config.wsgi_max_body_size, "asgi")
    lifespan_state: LifespanState = {}
    lifespan = Lifespan(wrapped_app, config, loop, lifespan_state)
    lifespan_task = loop.create_task(lifespan.handle_lifespan())

    context = WorkerContext(None)
    servers: list[asyncio.Server] = []
    active_connections: dict[asyncio.Task[None], TCPServer] = {}
    interrupted = False

    async def _server_callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task(loop)
        tcp_server = TCPServer(wrapped_app, loop, config, context, lifespan_state, reader, writer)
        if task is not None:
            active_connections[task] = tcp_server
            task.add_done_callback(lambda t: active_connections.pop(t, None))
        await tcp_server

    try:
        await lifespan.wait_for_startup()
        if lifespan_task.done():
            exception = lifespan_task.exception()
            if exception is not None:
                raise exception

        sockets = config.create_sockets()
        ssl_context = config.create_ssl_context() if config.ssl_enabled else None
        ssl_handshake_timeout = config.ssl_handshake_timeout if config.ssl_enabled else None

        for sock in sockets.secure_sockets:
            servers.append(
                await asyncio.start_server(
                    _server_callback,
                    backlog=config.backlog,
                    ssl=ssl_context,
                    sock=sock,
                    ssl_handshake_timeout=ssl_handshake_timeout,
                )
            )
            bind = repr_socket_addr(sock.family, sock.getsockname())
            await config.log.info(f"Running on https://{bind} (CTRL + C to quit)")

        for sock in sockets.insecure_sockets:
            servers.append(await asyncio.start_server(_server_callback, backlog=config.backlog, sock=sock))
            bind = repr_socket_addr(sock.family, sock.getsockname())
            await config.log.info(f"Running on http://{bind} (CTRL + C to quit)")

        try:
            await _wait_for_shutdown_signal(stop_event, shutdown_event, context.terminate)
        except asyncio.CancelledError:
            interrupted = True
            current_task = asyncio.current_task(loop)
            if current_task is not None:
                current_task.uncancel()
    finally:
        _notify_stream_shutdown(args)
        await context.terminated.set()

        # Close listening sockets first without calling wait_closed(), which in
        # Python 3.12+ blocks until all active client connections disconnect.
        for server in servers:
            server.close()

        await _close_pending_connections(active_connections, config.graceful_timeout)

        for server in servers:
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

        try:
            await asyncio.wait_for(lifespan.wait_for_shutdown(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        if not lifespan_task.done():
            lifespan_task.cancel()
        try:
            await lifespan_task
        except (asyncio.CancelledError, Exception):
            pass

    if interrupted:
        print("\nShutdown requested by user (Ctrl+C)")


def server_main(args, init_wxapp=True):
    enable_stream = bool(getattr(args, "enable_stream", False))
    if not hasattr(args, "state") or not isinstance(args.state, dict):
        set_state_args(args)
    elif enable_stream and "depth_model" not in args.state:
        existing_state = args.state
        set_state_args(
            args,
            args_lock=existing_state.get("args_lock"),
            stop_event=existing_state.get("stop_event"),
            fps_event=existing_state.get("fps_event"),
            tqdm_fn=existing_state.get("tqdm_fn"),
            depth_model=existing_state.get("depth_model"),
        )

    validate_bind_address_and_auth(args)

    if enable_stream:
        depth_model = args.state.get("depth_model")
        if depth_model is not None and not depth_model.loaded():
            ok = depth_model.download(
                args.depth_model,
                stop_event=args.state.get("stop_event"),
                tqdm_fn=args.state.get("tqdm_fn"),
            )
            if not ok:
                return args

    init_video_hwaccel()
    if enable_stream and getattr(args, "stream_hub", None) is None:
        from nunif.utils.win32 import disable_win32_power_throttling

        from ..desktop.service import DesktopStreamService
        from ..desktop.utils import get_stream_codec_config, resolve_player_stereo_format
        from ..desktop.video_stream_encoder import validate_stream_video_codec
        from .stream_hub import StreamFrameHub

        disable_win32_power_throttling()
        video_codec, available_codecs, _ = get_stream_codec_config(args)
        gpu = getattr(args, "gpu", None)
        gpu_id = gpu[0] if isinstance(gpu, (list, tuple)) and len(gpu) > 0 else None
        validate_stream_video_codec(video_codec, gpu_id=gpu_id)
        stream_hub = StreamFrameHub(
            lock=threading.Lock(),
            frame_width=1920,
            frame_height=1080,
            fps=int(getattr(args, "stream_fps", 30)),
            stereo_format=resolve_player_stereo_format(args),
            video_codec=video_codec,
            available_stream_codecs=available_codecs,
        )
        desktop_service = DesktopStreamService(
            args=args,
            stream_hub=stream_hub,
            idle_timeout=10.0,
            init_wxapp=init_wxapp,
        )
        desktop_service.authorize_capture()
        args.stream_hub = stream_hub
        args.desktop_service = desktop_service

    state, config = init_server_runtime(
        args,
        bind_addr=args.bind_addr,
        port=args.port,
        enable_media_library=True,
        graceful_timeout=1.0,
    )

    stream_url = f"https://{args.bind_addr}:{args.port}"
    print(f"Starting iw3-player server on {stream_url}")
    fps_event = args.state.get("fps_event")
    if fps_event is not None:
        fps_event.set_url(stream_url)

    shutdown_event = asyncio.Event()

    try:
        asyncio.run(serve_with_watchdog(args, config, shutdown_event))
    except KeyboardInterrupt:
        shutdown_event.set()
        print("\nShutdown requested by user (Ctrl+C)")
    finally:
        state.close()
        if args.state.get("stop_event") and args.state["stop_event"].is_set():
            args.state["stop_event"].clear()

    return args
