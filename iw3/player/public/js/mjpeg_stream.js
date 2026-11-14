import * as THREE from "three";

const BOUNDARY_BYTES = new TextEncoder().encode("--frame");
const HEADER_END_BYTES = new Uint8Array([13, 10, 13, 10]); // \r\n\r\n
const JPEG_SOI_BYTES = new Uint8Array([0xff, 0xd8]);
const JPEG_EOI_BYTES = new Uint8Array([0xff, 0xd9]);

function indexOfBytes(haystack, needle, start = 0) {
    const limit = haystack.length - needle.length;
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

function concatBytes(a, b) {
    if (a.length === 0) return b;
    if (b.length === 0) return a;
    const out = new Uint8Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
}

/**
 * Live MJPEG multipart stream loader backed by fetch ReadableStream + createImageBitmap.
 */
class MjpegStream extends EventTarget {
    constructor(streamUrl = "/stream.jpg", infoUrl = "/api/stream/info", debugLogInstance = null) {
        super();
        this.streamUrl = streamUrl;
        this.infoUrl = infoUrl;
        this.debugLogInstance = debugLogInstance;

        this.texture = new THREE.Texture();
        this.texture.colorSpace = THREE.SRGBColorSpace;
        this.texture.wrapS = THREE.ClampToEdgeWrapping;
        this.texture.wrapT = THREE.ClampToEdgeWrapping;
        this.texture.generateMipmaps = false;
        this.texture.minFilter = THREE.LinearFilter;
        this.texture.magFilter = THREE.LinearFilter;
        this.texture.flipY = false;

        this.isActive = false;
        this.isPaused = false;
        this.hasNewFrame = false;
        this.width = 0;
        this.height = 0;
        this.aspectRatio = 16 / 9;
        this.stereoFormat = null;
        this.processToken = null;

        this._pendingBytes = null;
        this._isDecoding = false;
        this._abortController = null;
        this._reconnectTimer = null;
        this._reconnectResolve = null;
        this._firstFrameResolve = null;
        this._textDecoder = new TextDecoder("ascii");

        this._texWidth = 0;
        this._texHeight = 0;
        this._infoWidth = 0;
        this._infoHeight = 0;
        this._needsTextureDispose = false;
        this._lastInfoPollTime = 0;
        this._pollPromise = null;
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

    _resolveFirstFrame(result) {
        if (this._firstFrameResolve) {
            const resolve = this._firstFrameResolve;
            this._firstFrameResolve = null;
            resolve(result);
        }
    }

    _applyStreamInfo(info, notifyChange = true) {
        if (!info) return;
        const width = info.width || 0;
        const height = info.height || 0;
        const format = info.stereo_format || null;
        const token = info.process_token || null;
        if (info.stream_uri) {
            this.streamUrl = info.stream_uri;
        }

        const widthChanged = Boolean(width && (this._infoWidth ? width !== this._infoWidth : width !== this.width));
        const heightChanged = Boolean(height && (this._infoHeight ? height !== this._infoHeight : height !== this.height));
        const formatChanged = Boolean(format && this.stereoFormat && format !== this.stereoFormat);
        const tokenChanged = Boolean(token && this.processToken && token !== this.processToken);
        const initialMetaLoaded = Boolean((!this.stereoFormat && format) || (!this.processToken && token));

        if (widthChanged || heightChanged || tokenChanged) {
            this._needsTextureDispose = true;
        }

        if (width) {
            this._infoWidth = width;
            if (widthChanged || !this.width) {
                this.width = width;
            }
        }
        if (height) {
            this._infoHeight = height;
            if (heightChanged || !this.height) {
                this.height = height;
            }
        }
        if (this.width && this.height) {
            this.aspectRatio = this.width / this.height;
        }
        if (format) this.stereoFormat = format;
        if (token) this.processToken = token;

        if (notifyChange && (widthChanged || heightChanged || formatChanged || tokenChanged || initialMetaLoaded)) {
            this.dispatchEvent(
                new CustomEvent("stream-updated", {
                    detail: {
                        ...info,
                        width: this.width,
                        height: this.height,
                        stereo_format: this.stereoFormat,
                        process_token: this.processToken,
                        tokenChanged,
                    },
                })
            );
        }
    }

    async _pollStreamInfo() {
        if (this._pollPromise) {
            return await this._pollPromise;
        }
        this._pollPromise = (async () => {
            try {
                const res = await fetch(this.infoUrl, {
                    cache: "no-store",
                    credentials: "same-origin",
                });
                if (res.ok) {
                    const info = await res.json();
                    this._applyStreamInfo(info, true);
                    return true;
                }
                if (res.status === 401) {
                    this.dispatchEvent(new CustomEvent("auth-required"));
                }
            } catch {
                // Server might still be restarting; retry on next tick
            } finally {
                this._pollPromise = null;
            }
            return false;
        })();
        return await this._pollPromise;
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
            if (isReconnect || !this.processToken) {
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

    async _readMultipartStream(reader) {
        let buffer = new Uint8Array(0);

        while (this.isActive) {
            const now = performance.now();
            if (now - this._lastInfoPollTime > 2000) {
                this._lastInfoPollTime = now;
                const prevToken = this.processToken;
                this._pollStreamInfo().then((ok) => {
                    if (
                        ok &&
                        prevToken &&
                        this.processToken &&
                        this.processToken !== prevToken &&
                        this._abortController
                    ) {
                        this._abortController.abort();
                    }
                });
            }

            const { value, done } = await this._readWithTimeout(reader, 3000);
            if (done) break;
            if (!value || value.length === 0) continue;

            buffer = concatBytes(buffer, value);

            while (this.isActive) {
                const boundaryIdx = indexOfBytes(buffer, BOUNDARY_BYTES, 0);
                if (boundaryIdx === -1) {
                    // Keep tail bytes in case boundary straddles chunk boundary
                    const keep = Math.min(buffer.length, BOUNDARY_BYTES.length - 1);
                    buffer = buffer.slice(buffer.length - keep);
                    break;
                }

                const headerEnd = indexOfBytes(buffer, HEADER_END_BYTES, boundaryIdx);
                if (headerEnd === -1) {
                    if (buffer.length - boundaryIdx > 64 * 1024) {
                        buffer = new Uint8Array(0);
                    } else if (boundaryIdx > 0) {
                        buffer = buffer.slice(boundaryIdx);
                    }
                    break;
                }

                const headerText = this._textDecoder.decode(buffer.subarray(boundaryIdx, headerEnd));
                const match = /content-length:\s*(\d+)/i.exec(headerText);
                const frameStart = headerEnd + HEADER_END_BYTES.length;
                const contentLength = match ? parseInt(match[1], 10) : -1;

                if (contentLength > 0 && contentLength <= 32 * 1024 * 1024) {
                    const frameEnd = frameStart + contentLength;
                    if (buffer.length < frameEnd) {
                        if (boundaryIdx > 0) {
                            buffer = buffer.slice(boundaryIdx);
                        }
                        break;
                    }
                    const jpegBytes = buffer.slice(frameStart, frameEnd);
                    buffer = buffer.slice(frameEnd);
                    this._scheduleDecode(jpegBytes);
                } else {
                    // Fallback when Content-Length header is absent
                    const soi = indexOfBytes(buffer, JPEG_SOI_BYTES, frameStart);
                    if (soi === -1) {
                        if (buffer.length - frameStart > 64 * 1024) {
                            buffer = new Uint8Array(0);
                        }
                        break;
                    }
                    const eoi = indexOfBytes(buffer, JPEG_EOI_BYTES, soi + 2);
                    if (eoi === -1) {
                        if (boundaryIdx > 0) {
                            buffer = buffer.slice(boundaryIdx);
                        }
                        break;
                    }
                    const frameEnd = eoi + JPEG_EOI_BYTES.length;
                    const jpegBytes = buffer.slice(soi, frameEnd);
                    buffer = buffer.slice(frameEnd);
                    this._scheduleDecode(jpegBytes);
                }
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
                    const bitmap = await createImageBitmap(blob, { imageOrientation: "flipY" });

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
                                    process_token: this.processToken,
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

    pause() {
        if (!this.isPaused) {
            this.isPaused = true;
            this._pendingBytes = null;
            this.dispatchEvent(new CustomEvent("play-state-change", { detail: { isPlaying: false } }));
        }
    }

    play() {
        if (this.isPaused) {
            this.isPaused = false;
            this.dispatchEvent(new CustomEvent("play-state-change", { detail: { isPlaying: true } }));
        }
    }

    togglePause() {
        if (this.isPaused) {
            this.play();
        } else {
            this.pause();
        }
    }

    stop() {
        this.isActive = false;
        this._pendingBytes = null;
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
        if (this._reconnectResolve) {
            this._reconnectResolve();
            this._reconnectResolve = null;
        }
        if (this._abortController) {
            this._abortController.abort();
            this._abortController = null;
        }
        if (this.texture) {
            if (this.texture.image && typeof this.texture.image.close === "function") {
                this.texture.image.close();
                this.texture.image = null;
            }
            this.texture.dispose();
        }
        this._resolveFirstFrame({ cancelled: true });
    }
}

export { MjpegStream };
