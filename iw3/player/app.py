import base64
import os
import secrets
import sys
from contextlib import asynccontextmanager
from typing import Callable, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .dir_config import PUBLIC_DIR
from .media_library import MediaLibrary
from .schema import MediaItem
from .server_state import ServerState

# auto_error=False prevents automatic 401 response from the dependency itself
security = HTTPBasic(auto_error=False)


async def authenticate(request: Request, credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    state: ServerState = request.app.state.server_state
    if state.auth_user is None and state.auth_password is None:
        return None

    if credentials:
        is_user_ok = secrets.compare_digest(credentials.username, state.auth_user or "")
        is_pass_ok = secrets.compare_digest(credentials.password, state.auth_password or "")
        if is_user_ok and is_pass_ok:
            return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if hasattr(app.state, "server_state") and app.state.server_state is not None:
        app.state.server_state.close()


app = FastAPI(lifespan=lifespan, dependencies=[Depends(authenticate)])


def get_library(request: Request) -> MediaLibrary:
    return request.app.state.media_library


def get_state(request: Request) -> ServerState:
    return request.app.state.server_state


def _list_public_files(
    subdir: str,
    extensions: tuple[str, ...],
    sort_key: Optional[Callable[[str], str]] = None,
) -> List[str]:
    target_dir = os.path.join(PUBLIC_DIR, subdir)
    if not os.path.exists(target_dir):
        return []
    files = []
    try:
        for entry in os.scandir(target_dir):
            if entry.is_file() and entry.name.lower().endswith(extensions):
                files.append(entry.name)
    except Exception as e:
        print(f"Asset scan error ({subdir}): {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal server error")
    return sorted(files, key=sort_key)


@app.get("/api/files", response_model=List[MediaItem])
async def list_files(path: str = "/", library: MediaLibrary = Depends(get_library)):
    return library.list_files(path)


@app.get("/api/file/info", response_model=MediaItem)
async def get_file_info(path: str, library: MediaLibrary = Depends(get_library)):
    return library.get_file_info(path)


@app.get("/api/environments")
async def list_environments():
    return _list_public_files("environments", (".hdr", ".exr", ".glb"))


@app.get("/api/luts")
async def list_luts():
    return _list_public_files("lut", (".cube",), sort_key=lambda p: os.path.splitext(os.path.basename(p))[0])


@app.get("/api/key")
async def get_key(state: ServerState = Depends(get_state)):
    if not os.path.exists(state.secret_key_file):
        raise HTTPException(status_code=404, detail="Secret key not found")
    try:
        with open(state.secret_key_file, "rb") as f:
            key_data = f.read()
            return {"key": base64.b64encode(key_data).decode("utf-8")}
    except Exception as e:
        print(f"Key error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/media")
async def serve_media(path: str, library: MediaLibrary = Depends(get_library)):
    return library.get_file_response(path)


@app.get("/api/thumbnail")
async def serve_thumbnail(
    request: Request,
    path: str,
    library: MediaLibrary = Depends(get_library),
):
    return library.get_thumbnail(path, if_none_match=request.headers.get("if-none-match"))


@app.get("/api/subtitles")
async def serve_subtitles(
    request: Request,
    path: str,
    library: MediaLibrary = Depends(get_library),
):
    return library.get_subtitles(path, if_none_match=request.headers.get("if-none-match"))


@app.post("/api/cache/clear")
async def clear_cache(state: ServerState = Depends(get_state)):
    try:
        state.cache_db.clear()
        return {"status": "ok", "message": "Cache cleared"}
    except Exception as e:
        print(f"Cache clear error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.get("/js/debug_log.js")
async def serve_debug_log_js(state: ServerState = Depends(get_state)):
    if state.debug_mode:
        filename = "debug_log.js"
    elif state.debug_console:
        filename = "debug_log_console.js"
    else:
        filename = "debug_log_empty.js"
    return FileResponse(os.path.join(PUBLIC_DIR, "js", filename))


@app.get("/{filename:path}")
async def serve_static(filename: str):
    if not filename or filename == "index.html":
        return FileResponse(os.path.join(PUBLIC_DIR, "index.html"), headers={"Cache-Control": "no-cache"})

    # Securely join path with PUBLIC_DIR
    abs_public = os.path.abspath(PUBLIC_DIR)
    target_path = os.path.join(abs_public, filename.lstrip("/"))
    abs_path = os.path.abspath(target_path)

    if not os.path.isfile(abs_path) or os.path.commonpath([abs_public, abs_path]) != abs_public:
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(abs_path, headers={"Cache-Control": "no-cache"})
