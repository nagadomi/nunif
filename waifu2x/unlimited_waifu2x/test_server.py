# python3 -m waifu2x.unlimited_waifu2x.test_server
# View at http://localhost:8812/
# Do not use this server in product environments.
import argparse
import asyncio
from os import path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config

ROOT_DIR = path.abspath(path.join(path.dirname(__file__), "public_html"))
app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
cors_enabled = False


@app.get("/{url:path}")
async def static_file(url: str) -> FileResponse:
    """Serve static files from public_html with optional cross-origin isolation headers."""
    if not url:
        url = "index.html"
    abs_root = path.realpath(ROOT_DIR)
    abs_path = path.realpath(path.join(abs_root, url.lstrip("/")))
    if path.commonpath([abs_root, abs_path]) != abs_root or not path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Not found")

    response = FileResponse(abs_path)
    if cors_enabled:
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    return response


def main() -> None:
    """Parse CLI arguments and run the Hypercorn test server."""
    global cors_enabled
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--cors",
        action="store_true",
        help=(
            "Add CORS header for testing wasm multi-threading."
            " google-chrome does not work for security reasons (localhost CORS), use firefox"
        ),
    )
    parser.add_argument("--bind-addr", type=str, default="127.0.0.1", help="0.0.0.0 for global, 127.0.0.1 for local")
    parser.add_argument("--port", type=int, default=8812, help="HTTP port number")
    args = parser.parse_args()
    cors_enabled = args.cors

    config = Config()
    config.bind = [f"{args.bind_addr}:{args.port}"]
    try:
        asyncio.run(serve(app, config))  # type: ignore[arg-type]
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
