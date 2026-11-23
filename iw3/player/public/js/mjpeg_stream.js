import * as THREE from "three";
import { BaseDesktopStream } from "./desktop_stream_base.js";

const BOUNDARY_BYTES = new TextEncoder().encode("--frame");
const HEADER_END_BYTES = new Uint8Array([13, 10, 13, 10]); // \r\n\r\n
const JPEG_SOI_BYTES = new Uint8Array([0xff, 0xd8]);
const JPEG_EOI_BYTES = new Uint8Array([0xff, 0xd9]);

const INITIAL_RX_CAPACITY = 512 * 1024;
const MAX_HEADER_BYTES = 64 * 1024;
const MAX_FRAME_BYTES = 32 * 1024 * 1024;

const STATE_FIND_BOUNDARY = 0;
const STATE_READ_HEADER = 1;
const STATE_READ_BODY = 2;
const STATE_READ_FALLBACK_SOI = 3;
const STATE_READ_FALLBACK_EOI = 4;

function indexOfBytes(haystack, needle, start = 0, end = haystack.length) {
    const limit = end - needle.length;
    if (limit < start || needle.length === 0) return -1;
    const first = needle[0];
    for (let i = start; i <= limit; i++) {
        if (haystack[i] !== first) continue;
        let matched = true;
        for (let j = 1; j < needle.length; j++) {
            if (haystack[i + j] !== needle[j]) {
                matched = false;
                break;
            }
        }
        if (matched) return i;
    }
    return -1;
}

/**
 * Live MJPEG multipart stream loader backed by fetch ReadableStream + createImageBitmap.
 */
class MjpegStream extends BaseDesktopStream {
    constructor(streamUrl = "/api/stream", infoUrl = "/api/stream/info", debugLogInstance = null) {
        super(infoUrl, debugLogInstance, new THREE.Texture());
        this.streamUrl = streamUrl;

        this._pendingBytes = null;
        this._isDecoding = false;
        this._abortController = null;
        this._reconnectResolve = null;
        this._textDecoder = new TextDecoder("ascii");

        this._texWidth = 0;
        this._texHeight = 0;
        this._lastInfoPollTime = 0;
    }

    get codecName() {
        return "mjpeg";
    }

    start(initialInfo = null) {
        if (initialInfo) {
            this._applyStreamInfo(initialInfo, false);
        }
        this.isActive = true;
        this.isPaused = false;

        return new Promise((resolve) => {
            this._firstFrameResolve = resolve;
            this._connectLoop().catch((e) => {
                this.debugLogInstance?.log(`[ERROR] MJPEG loop error: ${e.message}`);
                this._resolveFirstFrame({ cancelled: true });
            });
        });
    }

    _onStreamInfo(info, _notifyChange, { widthChanged, heightChanged, tokenChanged }) {
        if (info.stream_uri) {
            this.streamUrl = info.stream_uri;
        }
        if (widthChanged || heightChanged || tokenChanged) {
            this._needsTextureDispose = true;
        }
    }

    _readWithTimeout(reader, timeoutMs = 3000) {
        let timer = null;
        const timeoutPromise = new Promise((_, reject) => {
            timer = setTimeout(() => {
                if (this._abortController) {
                    this._abortController.abort();
                }
                reject(new Error("Stream read timeout"));
            }, timeoutMs);
        });
        return Promise.race([reader.read(), timeoutPromise]).finally(() => {
            if (timer !== null) {
                clearTimeout(timer);
            }
        });
    }

    _sleepReconnect(ms) {
        return new Promise((resolve) => {
            this._reconnectResolve = resolve;
            this._reconnectTimer = setTimeout(() => {
                this._reconnectTimer = null;
                this._reconnectResolve = null;
                resolve();
            }, ms);
        });
    }

    async _connectLoop() {
        let isReconnect = false;
        while (this.isActive) {
            if (isReconnect || !this.streamToken) {
                const infoOk = await this._pollStreamInfo();
                if (!this.isActive) break;
                if (!infoOk) {
                    await this._sleepReconnect(1000);
                    continue;
                }
            }

            this._abortController = new AbortController();
            let connected = false;
            try {
                const response = await fetch(this.streamUrl, {
                    signal: this._abortController.signal,
                    cache: "no-store",
                    credentials: "same-origin",
                });
                if (response.status === 401) {
                    this.dispatchEvent(new CustomEvent("auth-required"));
                }
                if (!response.ok || !response.body) {
                    throw new Error(`HTTP ${response.status}`);
                }
                connected = true;
                if (isReconnect) {
                    this._needsTextureDispose = true;
                    await this._pollStreamInfo();
                }
                isReconnect = true;
                this._lastInfoPollTime = performance.now();
                const reader = response.body.getReader();
                await this._readMultipartStream(reader);
            } catch (e) {
                if (!this.isActive) {
                    break;
                }
                this.debugLogInstance?.log(`[WARN] Stream disconnected (${e.message}), reconnecting...`);
            } finally {
                this._abortController = null;
            }

            isReconnect = true;
            if (!this.isActive) break;
            if (!connected) {
                await this._sleepReconnect(1000);
            }
        }
        this._resolveFirstFrame({ cancelled: true });
    }

    _appendRxChunk(rx, chunk) {
        const chunkLen = chunk.length;
        if (rx.writePos + chunkLen > rx.buf.length) {
            const unread = rx.writePos - rx.readPos;
            if (unread + chunkLen <= rx.buf.length) {
                if (unread > 0 && rx.readPos > 0) {
                    rx.buf.copyWithin(0, rx.readPos, rx.writePos);
                }
            } else {
                const needed = unread + chunkLen;
                let newCap = rx.buf.length;
                while (newCap < needed) {
                    newCap *= 2;
                }
                const nextBuf = new Uint8Array(newCap);
                if (unread > 0) {
                    nextBuf.set(rx.buf.subarray(rx.readPos, rx.writePos), 0);
                }
                rx.buf = nextBuf;
            }
            rx.scanPos = Math.max(0, rx.scanPos - rx.readPos);
            rx.readPos = 0;
            rx.writePos = unread;
        }
        rx.buf.set(chunk, rx.writePos);
        rx.writePos += chunkLen;
    }

    async _readMultipartStream(reader) {
        const rx = {
            buf: new Uint8Array(INITIAL_RX_CAPACITY),
            readPos: 0,
            writePos: 0,
            scanPos: 0,
        };
        let state = STATE_FIND_BOUNDARY;
        let expectedLength = 0;

        while (this.isActive) {
            const now = performance.now();
            if (now - this._lastInfoPollTime > 2000) {
                this._lastInfoPollTime = now;
                const prevToken = this.streamToken;
                this._pollStreamInfo().then((ok) => {
                    if (
                        ok &&
                        prevToken &&
                        this.streamToken &&
                        this.streamToken !== prevToken &&
                        this._abortController
                    ) {
                        this._abortController.abort();
                    }
                });
            }

            const { value, done } = await this._readWithTimeout(reader, 3000);
            if (done) break;
            if (!value || value.length === 0) continue;

            this._appendRxChunk(rx, value);

            let latestFrameStart = -1;
            let latestFrameEnd = -1;

            while (this.isActive) {
                if (state === STATE_FIND_BOUNDARY) {
                    const searchStart = Math.max(rx.readPos, rx.scanPos);
                    const boundaryIdx = indexOfBytes(rx.buf, BOUNDARY_BYTES, searchStart, rx.writePos);
                    if (boundaryIdx === -1) {
                        // Keep tail bytes in case boundary straddles chunk boundary
                        const available = rx.writePos - rx.readPos;
                        const keep = Math.min(available, BOUNDARY_BYTES.length - 1);
                        rx.readPos = rx.writePos - keep;
                        rx.scanPos = rx.readPos;
                        break;
                    }
                    rx.readPos = boundaryIdx;
                    rx.scanPos = boundaryIdx + BOUNDARY_BYTES.length;
                    state = STATE_READ_HEADER;
                } else if (state === STATE_READ_HEADER) {
                    const searchStart = Math.max(rx.readPos, rx.scanPos - (HEADER_END_BYTES.length - 1));
                    const headerEnd = indexOfBytes(rx.buf, HEADER_END_BYTES, searchStart, rx.writePos);
                    if (headerEnd === -1) {
                        if (rx.writePos - rx.readPos > MAX_HEADER_BYTES) {
                            rx.readPos = rx.writePos;
                            rx.scanPos = rx.writePos;
                            state = STATE_FIND_BOUNDARY;
                        } else {
                            rx.scanPos = rx.writePos;
                        }
                        break;
                    }
                    const headerText = this._textDecoder.decode(rx.buf.subarray(rx.readPos, headerEnd));
                    const match = /content-length:\s*(\d+)/i.exec(headerText);
                    const frameStart = headerEnd + HEADER_END_BYTES.length;
                    const contentLength = match ? parseInt(match[1], 10) : -1;

                    rx.readPos = frameStart;
                    rx.scanPos = frameStart;
                    if (contentLength > 0 && contentLength <= MAX_FRAME_BYTES) {
                        expectedLength = contentLength;
                        state = STATE_READ_BODY;
                    } else {
                        state = STATE_READ_FALLBACK_SOI;
                    }
                } else if (state === STATE_READ_BODY) {
                    if (rx.writePos - rx.readPos < expectedLength) {
                        break;
                    }
                    latestFrameStart = rx.readPos;
                    latestFrameEnd = rx.readPos + expectedLength;
                    rx.readPos = latestFrameEnd;
                    rx.scanPos = latestFrameEnd;
                    expectedLength = 0;
                    state = STATE_FIND_BOUNDARY;
                } else if (state === STATE_READ_FALLBACK_SOI) {
                    const searchStart = Math.max(rx.readPos, rx.scanPos - 1);
                    const soi = indexOfBytes(rx.buf, JPEG_SOI_BYTES, searchStart, rx.writePos);
                    if (soi === -1) {
                        if (rx.writePos - rx.readPos > MAX_HEADER_BYTES) {
                            rx.readPos = rx.writePos;
                            rx.scanPos = rx.writePos;
                            state = STATE_FIND_BOUNDARY;
                        } else {
                            rx.scanPos = rx.writePos;
                        }
                        break;
                    }
                    rx.readPos = soi;
                    rx.scanPos = soi + 2;
                    state = STATE_READ_FALLBACK_EOI;
                } else if (state === STATE_READ_FALLBACK_EOI) {
                    const searchStart = Math.max(rx.readPos + 2, rx.scanPos - 1);
                    const eoi = indexOfBytes(rx.buf, JPEG_EOI_BYTES, searchStart, rx.writePos);
                    if (eoi === -1) {
                        if (rx.writePos - rx.readPos > MAX_FRAME_BYTES) {
                            rx.readPos = rx.writePos;
                            rx.scanPos = rx.writePos;
                            state = STATE_FIND_BOUNDARY;
                        } else {
                            rx.scanPos = rx.writePos;
                        }
                        break;
                    }
                    latestFrameStart = rx.readPos;
                    latestFrameEnd = eoi + JPEG_EOI_BYTES.length;
                    rx.readPos = latestFrameEnd;
                    rx.scanPos = latestFrameEnd;
                    state = STATE_FIND_BOUNDARY;
                }
            }

            if (latestFrameStart >= 0) {
                const jpegBytes = rx.buf.slice(latestFrameStart, latestFrameEnd);
                this._scheduleDecode(jpegBytes);
            }

            if (rx.readPos === rx.writePos) {
                rx.readPos = 0;
                rx.writePos = 0;
                rx.scanPos = 0;
            }
        }
    }

    _scheduleDecode(jpegBytes) {
        if (!this.isActive || this.isPaused) return;
        this._pendingBytes = jpegBytes;
        if (!this._isDecoding) {
            this._decodeLatestLoop();
        }
    }

    async _decodeLatestLoop() {
        this._isDecoding = true;
        try {
            while (this._pendingBytes !== null && this.isActive && !this.isPaused) {
                const bytes = this._pendingBytes;
                this._pendingBytes = null;

                try {
                    const blob = new Blob([bytes], { type: "image/jpeg" });
                    const bitmap = await createImageBitmap(blob);

                    if (!this.isActive) {
                        bitmap.close();
                        break;
                    }

                    const prevTexWidth = this._texWidth;
                    const prevTexHeight = this._texHeight;
                    const texSizeChanged =
                        (prevTexWidth > 0 && bitmap.width !== prevTexWidth) ||
                        (prevTexHeight > 0 && bitmap.height !== prevTexHeight);
                    this._texWidth = bitmap.width;
                    this._texHeight = bitmap.height;

                    const prevImage = this.texture.image;
                    this.texture.image = bitmap;
                    if (prevImage && typeof prevImage.close === "function") {
                        prevImage.close();
                    }

                    // Dispose texture before re-upload so Three.js reallocates immutable WebGL storage for the new dimensions.
                    if (texSizeChanged || this._needsTextureDispose) {
                        this._needsTextureDispose = false;
                        this.texture.dispose();
                    }

                    const dimChanged = texSizeChanged || this.width !== bitmap.width || this.height !== bitmap.height;
                    this.width = bitmap.width;
                    this.height = bitmap.height;
                    if (this.height > 0) {
                        this.aspectRatio = this.width / this.height;
                    }
                    this.hasNewFrame = true;
                    const wasWaitingFirstFrame = Boolean(this._firstFrameResolve);
                    this._resolveFirstFrame({ cancelled: false });

                    if (dimChanged && !wasWaitingFirstFrame) {
                        this.dispatchEvent(
                            new CustomEvent("stream-updated", {
                                detail: {
                                    width: this.width,
                                    height: this.height,
                                    stereo_format: this.stereoFormat,
                                    stream_token: this.streamToken,
                                    tokenChanged: false,
                                },
                            })
                        );
                        if (texSizeChanged) {
                            this._pollStreamInfo();
                        }
                    }
                } catch (e) {
                    this.debugLogInstance?.log(`[WARN] Frame decode skipped: ${e.message}`);
                }
            }
        } finally {
            this._isDecoding = false;
        }
    }

    consumeNewFrame() {
        if (this.hasNewFrame) {
            this.hasNewFrame = false;
            this.texture.needsUpdate = true;
            return true;
        }
        return false;
    }

    _onPaused() {
        this._pendingBytes = null;
    }

    _onStop() {
        this._pendingBytes = null;
        if (this._reconnectResolve) {
            this._reconnectResolve();
            this._reconnectResolve = null;
        }
        if (this._abortController) {
            this._abortController.abort();
            this._abortController = null;
        }
    }
}

export { MjpegStream };
