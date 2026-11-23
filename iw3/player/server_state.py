import os
from typing import Any

from diskcache import Cache

from .dir_config import CACHE_DIR, CERT_DIR, EXAMPLE_DIR

DEFAULT_SECRET_KEY_FILE = os.path.join(CERT_DIR, "secret.key")


class ServerState:
    """Holds the server configuration and shared state."""

    def __init__(self, args: Any = None) -> None:
        root = getattr(args, "root", None) or EXAMPLE_DIR
        cache_dir = getattr(args, "cache_dir", None) or CACHE_DIR
        cache_size_limit = getattr(args, "cache_size_limit", 1000)

        self.mode: str = getattr(args, "mode", "player")
        self.enable_file_library: bool = bool(getattr(args, "enable_file_library", self.mode == "player"))
        self.stream_hub = getattr(args, "stream_hub", None)
        self.desktop_service = getattr(args, "desktop_service", None)

        self.image_root = os.path.abspath(root)
        self.auth_user = getattr(args, "user", None)
        self.auth_password = getattr(args, "password", None)
        self.debug_mode = bool(getattr(args, "debug", False))
        self.debug_console = bool(getattr(args, "debug_console", False))
        self.secret_key_file = getattr(args, "secret_key_file", None) or DEFAULT_SECRET_KEY_FILE

        self.cache_db: Any = None
        if self.enable_file_library:
            self.cache_db = Cache(cache_dir, size_limit=cache_size_limit * 1024 * 1024)
            if getattr(args, "clear_cache", False):
                self.cache_db.clear()

    def close(self) -> None:
        if self.desktop_service is not None:
            self.desktop_service.stop()
        if self.cache_db is not None:
            self.cache_db.close()
