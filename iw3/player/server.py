import os
import sys
import argparse
import asyncio
import socket
import ipaddress
from hypercorn.config import Config
from hypercorn.asyncio import serve
from .app import app
from .server_state import ServerState
from .media_library import MediaLibrary
from .dir_config import CERT_DIR, CACHE_DIR, EXAMPLE_DIR


CERT_FILE = os.path.join(CERT_DIR, "server.crt")
KEY_FILE = os.path.join(CERT_DIR, "server.key")
SECRET_KEY_FILE = os.path.join(CERT_DIR, "secret.key")


def get_local_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
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
    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    if not os.path.exists(cert_dir):
        os.makedirs(cert_dir)
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Tokyo"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "My App"),
            x509.NameAttribute(NameOID.COMMON_NAME, bind_addr),
        ])
        now = datetime.now(timezone.utc)
        # Use typed SAN entries; pyOpenSSL removed its X509Extension API.
        addresses = dict.fromkeys([ipaddress.ip_address("127.0.0.1"),
                                   ipaddress.ip_address(bind_addr)])
        alt_names = [x509.DNSName("localhost")]
        alt_names.extend(x509.IPAddress(address) for address in addresses)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(k.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=10 * 365))
            .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
            .sign(k, hashes.SHA256())
        )
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(key_file, "wb") as f:
            f.write(k.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))
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
            # Wait for graceful shutdown, but enforce a hard timeout
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
            # Capture startup errors (like port in use)
            try:
                await server_task
            except asyncio.CancelledError:
                pass


def server_main(args):
    # Initialize global state attached to app
    args.secret_key_file = SECRET_KEY_FILE
    if hasattr(app.state, "server_state") and app.state.server_state is not None:
        app.state.server_state.close()
        app.state.server_state = None
        app.state.media_library = None

    state = ServerState(args)
    app.state.server_state = state
    app.state.media_library = MediaLibrary(state)

    generate_self_signed_cert(CERT_DIR, CERT_FILE, KEY_FILE, args.bind_addr)
    if not os.path.exists(SECRET_KEY_FILE):
        import secrets

        print(f"Generating new secret key: {SECRET_KEY_FILE}")
        with open(SECRET_KEY_FILE, "wb") as f:
            f.write(secrets.token_bytes(32))

    if args.bind_addr is None:
        args.bind_addr = get_local_address()
        if not is_private_address(args.bind_addr):
            raise RuntimeError(f"Detected IP address({args.bind_addr}) is not LAN Address. Specify --bind-addr")

    if is_loopback_address(args.bind_addr):
        print(f"Warning: {args.bind_addr} is loopback only.", file=sys.stderr)

    auth = (args.user, args.password) if (args.user or args.password) else None
    if not is_private_address(args.bind_addr) and auth is None:
        raise RuntimeError(f"({args.bind_addr}) is not LAN Address. Specify --user/--password")

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
    except Exception as e:
        # Re-raise to be caught by GUI
        raise e


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    server_main(args)
