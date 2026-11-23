import asyncio
import base64
import json
import os
import secrets
import struct
import sys
from contextlib import asynccontextmanager
from typing import Callable, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.requests import HTTPConnection
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from iw3.desktop.video_stream_encoder import resolve_stream_video_codec

from .dir_config import PUBLIC_DIR
from .media_library import MediaLibrary
from .origin_check import is_same_origin_request
from .schema import MediaItem, StreamSettingsUpdate
from .server_state import ServerState

# auto_error=False prevents automatic 401 response from the dependency itself
security = HTTPBasic(auto_error=False)


async def authenticate(conn: HTTPConnection, credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    state: ServerState = conn.app.state.server_state
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


def _is_same_origin_connection(conn: HTTPConnection) -> bool:
    return is_same_origin_request(
        origin=conn.headers.get("origin"),
        sec_fetch_site=conn.headers.get("sec-fetch-site"),
        host=conn.headers.get("host"),
        scheme=conn.url.scheme,
    )


async def verify_same_origin(conn: HTTPConnection) -> None:
    if not _is_same_origin_connection(conn):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-origin request forbidden",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if hasattr(app.state, "server_state") and app.state.server_state is not None:
        app.state.server_state.close()


app = FastAPI(lifespan=lifespan, dependencies=[Depends(authenticate)])


def get_library(request: Request) -> MediaLibrary:
    library = getattr(request.app.state, "media_library", None)
    if library is None:
        raise HTTPException(status_code=404, detail="File library is disabled")
    return library


def get_state(request: Request) -> ServerState:
    return request.app.state.server_state


def get_stream_hub(state: ServerState = Depends(get_state)):
    stream_hub = getattr(state, "stream_hub", None)
    if stream_hub is None:
        raise HTTPException(status_code=404, detail="Desktop stream is not active")
    return stream_hub


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


@app.post("/api/cache/clear", dependencies=[Depends(verify_same_origin)])
async def clear_cache(state: ServerState = Depends(get_state)):
    try:
        if state.cache_db is not None:
            state.cache_db.clear()
        return {"status": "ok", "message": "Cache cleared"}
    except Exception as e:
        print(f"Cache clear error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.get("/api/config")
async def get_config(state: ServerState = Depends(get_state)):
    mode = getattr(state, "mode", "player")
    enable_file_library = getattr(state, "enable_file_library", mode == "player")
    stream_hub = getattr(state, "stream_hub", None)
    return {
        "mode": mode,
        "features": {
            "file_library": bool(enable_file_library),
            "desktop_stream": stream_hub is not None,
        },
    }


def _get_stream_service(state: ServerState):
    return getattr(state, "desktop_service", None) or getattr(getattr(state, "stream_hub", None), "service", None)


async def _ensure_desktop_service_running(state: ServerState) -> None:
    desktop_service = getattr(state, "desktop_service", None)
    if desktop_service is None or desktop_service.is_running():
        return
    try:
        await asyncio.to_thread(desktop_service.ensure_running)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream/info")
async def get_stream_info(stream_hub=Depends(get_stream_hub)):
    return stream_hub.get_stream_info()


@app.post("/api/stream/start", dependencies=[Depends(verify_same_origin)])
async def start_desktop_stream(
    state: ServerState = Depends(get_state),
    stream_hub=Depends(get_stream_hub),
):
    await _ensure_desktop_service_running(state)
    return stream_hub.get_stream_info()


@app.post("/api/stream/stop", dependencies=[Depends(verify_same_origin)])
async def stop_desktop_stream(
    state: ServerState = Depends(get_state),
    _stream_hub=Depends(get_stream_hub),
):
    desktop_service = getattr(state, "desktop_service", None)
    if desktop_service is not None:
        await asyncio.to_thread(desktop_service.stop)
        return {"active": bool(desktop_service.is_running())}
    return {"active": True}


@app.get("/api/stream/settings")
async def get_stream_settings(
    state: ServerState = Depends(get_state),
    _stream_hub=Depends(get_stream_hub),
):
    service = _get_stream_service(state)
    if service is None:
        return {}
    return service.get_settings()


@app.post("/api/stream/settings", dependencies=[Depends(verify_same_origin)])
async def update_stream_settings(
    payload: StreamSettingsUpdate,
    state: ServerState = Depends(get_state),
    _stream_hub=Depends(get_stream_hub),
):
    service = _get_stream_service(state)
    if service is None:
        raise HTTPException(status_code=404, detail="Desktop stream service is unavailable")
    updates = payload.model_dump(exclude_none=True)
    return service.update_settings(updates)


@app.get("/api/stream", dependencies=[Depends(verify_same_origin)])
async def serve_desktop_stream(
    state: ServerState = Depends(get_state),
    stream_hub=Depends(get_stream_hub),
):
    await _ensure_desktop_service_running(state)
    return StreamingResponse(
        stream_hub.async_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


def _parse_requested_codecs(raw_codecs: str | None) -> list[str]:
    if not raw_codecs:
        return ["hevc", "h264"]
    parsed = [c.strip().lower() for c in raw_codecs.split(",") if c.strip().lower() in {"h264", "hevc"}]
    return parsed


def _can_accept_ws_codecs(stream_hub, requested_codecs: list[str]) -> bool:
    if not requested_codecs:
        return False
    available_codecs = getattr(stream_hub, "available_stream_codecs", None)
    if available_codecs is not None:
        available_set = {c.lower() for c in available_codecs if c}
        effective_codecs = [c for c in requested_codecs if c.lower() in available_set]
        if not effective_codecs:
            return False
    else:
        effective_codecs = requested_codecs
    return (
        resolve_stream_video_codec(
            video_codec=getattr(stream_hub, "video_codec", "libx264"),
            client_codecs=effective_codecs,
        )
        is not None
    )


@app.websocket("/api/stream/ws")
async def serve_desktop_stream_ws(websocket: WebSocket):
    if not _is_same_origin_connection(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    state: ServerState = websocket.app.state.server_state
    stream_hub = getattr(state, "stream_hub", None)
    await websocket.accept()

    if stream_hub is None:
        await websocket.send_json({"type": "error", "message": "Desktop stream is not active"})
        await websocket.close(code=1008)
        return

    requested_codecs = _parse_requested_codecs(websocket.query_params.get("codecs"))

    try:
        await _ensure_desktop_service_running(state)
    except HTTPException as e:
        await websocket.send_json({"type": "error", "message": str(e.detail)})
        await websocket.close(code=1011)
        return

    if not _can_accept_ws_codecs(stream_hub, requested_codecs):
        await websocket.send_json({"type": "fallback", "reason": "codec_unavailable"})
        await websocket.close(code=1000)
        return

    subscriber, _ = stream_hub.subscribe_ws(requested_codecs=requested_codecs)

    def _signal_sender_stop() -> None:
        while subscriber.queue.full():
            try:
                subscriber.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            subscriber.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def _receive_control() -> None:
        try:
            while True:
                event = await websocket.receive()
                if event.get("type") == "websocket.disconnect":
                    break
                raw_text = event.get("text")
                if not raw_text:
                    continue
                try:
                    msg = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get("type")
                if msg_type == "request_keyframe":
                    stream_hub.request_keyframe()
                elif msg_type == "pause":
                    stream_hub.set_ws_paused(subscriber.client_id, True)
                elif msg_type == "resume":
                    stream_hub.set_ws_paused(subscriber.client_id, False)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            _signal_sender_stop()

    recv_task = asyncio.create_task(_receive_control())
    last_codec_string: str | None = None
    last_video_codec: str | None = None
    last_width = 0
    last_height = 0
    last_epoch = -1

    try:
        while not stream_hub.shutdown_event.is_set():
            item = await subscriber.queue.get()
            if item is None:
                break
            if isinstance(item, dict):
                await websocket.send_json(item)
                await websocket.close(code=1000)
                break

            packet = item
            if (
                packet.codec_string != last_codec_string
                or packet.video_codec != last_video_codec
                or packet.width != last_width
                or packet.height != last_height
                or packet.epoch != last_epoch
            ):
                await websocket.send_json(
                    {
                        "type": "config",
                        "codec": packet.codec_string,
                        "codecFamily": packet.codec_family,
                        "videoCodec": packet.video_codec,
                        "width": packet.width,
                        "height": packet.height,
                        "fps": stream_hub.fps,
                        "epoch": packet.epoch,
                    }
                )
                last_codec_string = packet.codec_string
                last_video_codec = packet.video_codec
                last_width = packet.width
                last_height = packet.height
                last_epoch = packet.epoch

            header = struct.pack(
                "<IIQ",
                packet.epoch & 0xFFFFFFFF,
                1 if packet.is_keyframe else 0,
                packet.timestamp_us,
            )
            await websocket.send_bytes(header + packet.data)
            stream_hub.record_ws_frame_sent(subscriber.client_id)
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    finally:
        if not recv_task.done():
            recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        stream_hub.unsubscribe_ws(subscriber.client_id)


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
