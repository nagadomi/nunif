import os

from diskcache import Cache

from .dir_config import CACHE_DIR, CERT_DIR, EXAMPLE_DIR


class ServerState:
    """Holds the server configuration and shared state."""

    def __init__(self, args):
        if args is not None:
            self.image_root = os.path.abspath(args.root)
            self.auth_user = args.user
            self.auth_password = args.password
            self.debug_mode = args.debug
            self.debug_console = args.debug_console
            self.cache_db = Cache(args.cache_dir, size_limit=args.cache_size_limit * 1024 * 1024)
            if args.clear_cache:
                self.cache_db.clear()
            self.secret_key_file = args.secret_key_file
        else:
            self.image_root = EXAMPLE_DIR
            self.auth_user = None
            self.auth_password = None
            self.debug_mode = False
            self.debug_console = False
            self.cache_db = Cache(CACHE_DIR, size_limit=1000 * 1024 * 1024)
            self.secret_key_file = os.path.join(CERT_DIR, "secret.txt")

    def close(self):
        self.cache_db.close()
