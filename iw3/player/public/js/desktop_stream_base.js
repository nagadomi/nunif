import * as THREE from "three";

/**
 * Shared base class for live desktop stream transports (MJPEG and WebCodecs).
 */
export class BaseDesktopStream extends EventTarget {
    constructor(infoUrl = "/api/stream/info", debugLogInstance = null, texture = new THREE.Texture()) {
        super();
        this.infoUrl = infoUrl;
        this.debugLogInstance = debugLogInstance;

        this.texture = texture;
        this.texture.colorSpace = THREE.SRGBColorSpace;
        this.texture.wrapS = THREE.ClampToEdgeWrapping;
        this.texture.wrapT = THREE.ClampToEdgeWrapping;
        this.texture.generateMipmaps = false;
        this.texture.minFilter = THREE.LinearFilter;
        this.texture.magFilter = THREE.LinearFilter;
        this.texture.flipY = false;
        this.texture.userData.shaderFlipY = true;

        this.isActive = false;
        this.isPaused = false;
        this.hasNewFrame = false;
        this.width = 0;
        this.height = 0;
        this.aspectRatio = 16 / 9;
        this.stereoFormat = null;
        this.streamToken = null;
        this.bufferMs = null;

        this._infoWidth = 0;
        this._infoHeight = 0;
        this._needsTextureDispose = false;
        this._firstFrameResolve = null;
        this._reconnectTimer = null;
        this._pollTimer = null;
        this._pollPromise = null;
    }

    _resolveFirstFrame(result) {
        if (this._firstFrameResolve) {
            const resolve = this._firstFrameResolve;
            this._firstFrameResolve = null;
            resolve(result);
        }
    }

    _triggerError(message) {
        if (!this.isActive) return;
        const detailMessage = message || "Desktop stream error";
        this.debugLogInstance?.log(`[ERROR] Stream error: ${detailMessage}`);
        const hadPendingFirstFrame = Boolean(this._firstFrameResolve);
        if (hadPendingFirstFrame) {
            this._resolveFirstFrame({ cancelled: false, fallback: false, error: detailMessage });
            this.stop();
        } else {
            this.stop();
            this.dispatchEvent(new CustomEvent("stream-error", { detail: { message: detailMessage } }));
        }
    }

    _onStreamInfo(info, _notifyChange, _changes) {
        return info;
    }

    _applyStreamInfo(info, notifyChange = true) {
        if (!info) return;
        const width = Number(info.width) || 0;
        const height = Number(info.height) || 0;
        const format = info.stereo_format || null;
        const token = info.stream_token || null;

        const widthChanged = Boolean(width && (this._infoWidth ? width !== this._infoWidth : width !== this.width));
        const heightChanged = Boolean(height && (this._infoHeight ? height !== this._infoHeight : height !== this.height));
        const formatChanged = Boolean(format && this.stereoFormat && format !== this.stereoFormat);
        const tokenChanged = Boolean(token && this.streamToken && token !== this.streamToken);
        const initialMetaLoaded = Boolean((!this.stereoFormat && format) || (!this.streamToken && token));
        const hasBufferMs = typeof info.buffer_ms === "number" && Number.isFinite(info.buffer_ms);
        const nextBufferMs = hasBufferMs ? Math.max(0, Math.min(500, Math.round(info.buffer_ms))) : null;
        const bufferMsChanged = hasBufferMs && this.bufferMs !== null && nextBufferMs !== this.bufferMs;

        this._onStreamInfo(info, notifyChange, { widthChanged, heightChanged, tokenChanged, bufferMsChanged });

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
        if (token) this.streamToken = token;
        if (hasBufferMs) this.bufferMs = nextBufferMs;

        if (notifyChange && (widthChanged || heightChanged || formatChanged || tokenChanged || initialMetaLoaded || bufferMsChanged)) {
            this.dispatchEvent(
                new CustomEvent("stream-updated", {
                    detail: {
                        ...info,
                        width: this.width,
                        height: this.height,
                        stereo_format: this.stereoFormat,
                        stream_token: this.streamToken,
                        buffer_ms: this.bufferMs,
                        tokenChanged,
                        bufferMsChanged,
                    },
                })
            );
        }
    }

    async _pollStreamInfo() {
        if (!this.isActive) {
            return false;
        }
        if (this._pollPromise) {
            return await this._pollPromise;
        }
        this._pollPromise = (async () => {
            try {
                const res = await fetch(this.infoUrl, {
                    cache: "no-store",
                    credentials: "same-origin",
                });
                if (!this.isActive) {
                    return false;
                }
                if (res.ok) {
                    const info = await res.json();
                    if (!this.isActive) {
                        return false;
                    }
                    this._applyStreamInfo(info, true);
                    return true;
                }
                if (res.status === 401) {
                    this.dispatchEvent(new CustomEvent("auth-required"));
                }
            } catch {
                // Server might still be restarting; retry on next poll tick
            } finally {
                this._pollPromise = null;
            }
            return false;
        })();
        return await this._pollPromise;
    }

    _startInfoPolling(intervalMs = 3000) {
        this._stopInfoPolling();
        this._pollTimer = setInterval(() => {
            if (!this.isActive) return;
            this._pollStreamInfo();
        }, intervalMs);
    }

    _stopInfoPolling() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
    }

    _clearReconnectTimer() {
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
    }

    _onPaused() {}

    _onPlayed() {}

    _onStop() {}

    setBufferMs(_ms) {}

    getBufferStatus() {
        return {
            supported: false,
            configuredMs: 0,
            effectiveMs: 0,
            isClamped: false,
            queueLength: 0,
            frameIntervalMs: null,
        };
    }

    pause() {
        if (!this.isPaused) {
            this.isPaused = true;
            this.hasNewFrame = false;
            this._onPaused();
            this.dispatchEvent(new CustomEvent("play-state-change", { detail: { isPlaying: false } }));
        }
    }

    play() {
        if (this.isPaused) {
            this.isPaused = false;
            this._onPlayed();
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
        this.hasNewFrame = false;
        this._clearReconnectTimer();
        this._stopInfoPolling();
        this._onStop();
        if (this.texture) {
            const currentFrame = this.texture.image;
            if (currentFrame && typeof currentFrame.close === "function") {
                currentFrame.close();
            }
            this.texture.image = null;
            this.texture.dispose();
        }
        this._resolveFirstFrame({ cancelled: true, fallback: false });
    }
}
