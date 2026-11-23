import * as THREE from "three";
import { BaseDesktopStream } from "./desktop_stream_base.js";

const HEADER_BYTE_LENGTH = 16;
const FLAG_KEYFRAME = 1;

const CODEC_PROBES = {
    hevc: ["hev1.1.6.L120.90", "hvc1.1.6.L120.90"],
    h264: ["avc1.640028", "avc1.42001f"],
};

export function getRequestedVideoCodecOverride() {
    if (typeof window === "undefined") {
        return null;
    }
    const raw = new URLSearchParams(window.location.search).get("video_codec");
    if (!raw) {
        return null;
    }
    const value = raw.trim().toLowerCase();
    if (value === "jpeg" || value === "mjpeg" || value === "jpeg-gpu" || value === "jpeg_gpu") {
        return "jpeg";
    }
    if (value === "h264" || value === "hevc") {
        return value;
    }
    return null;
}

async function isCodecFamilySupported(family, width, height) {
    if (typeof VideoDecoder === "undefined" || typeof VideoDecoder.isConfigSupported !== "function") {
        return false;
    }
    const codedWidth = width > 0 ? width : 1920;
    const codedHeight = height > 0 ? height : 1080;
    const candidates = CODEC_PROBES[family] || [];

    for (const hardwareAcceleration of ["prefer-hardware", "no-preference"]) {
        for (const codec of candidates) {
            try {
                const res = await VideoDecoder.isConfigSupported({
                    codec,
                    codedWidth,
                    codedHeight,
                    hardwareAcceleration,
                    optimizeForLatency: true,
                });
                if (res?.supported) {
                    return true;
                }
            } catch {
                // Ignore unsupported probe variant and try the next candidate
            }
        }
    }
    return false;
}

export async function probeSupportedStreamCodecs(families = ["h264"], width = 1920, height = 1080) {
    if (typeof VideoDecoder === "undefined" || !Array.isArray(families) || families.length === 0) {
        return [];
    }
    const supported = [];
    for (const family of families) {
        if (await isCodecFamilySupported(family, width, height)) {
            supported.push(family);
        }
    }
    return supported;
}

/**
 * Live WebSocket + WebCodecs video stream loader backed by THREE.VideoFrameTexture.
 */
class WebCodecsStream extends BaseDesktopStream {
    constructor(
        wsPath = "/api/stream/ws",
        infoUrl = "/api/stream/info",
        requestedCodecs = ["hevc", "h264"],
        debugLogInstance = null
    ) {
        const texture = new THREE.VideoFrameTexture();
        texture.image = null;
        super(infoUrl, debugLogInstance, texture);
        this.wsPath = wsPath;
        this.requestedCodecs = requestedCodecs.length > 0 ? [...requestedCodecs] : ["hevc", "h264"];

        this.activeCodec = null;
        this.activeVideoCodec = null;
        this.activeCodecFamily = null;

        this._ws = null;
        this._activeWs = null;
        this._decoder = null;
        this._decoderCodec = null;
        this._decoderWidth = 0;
        this._decoderHeight = 0;
        this._waitingForKeyframe = true;
        this._pendingFrame = null;
        this._hasDecodedFrame = false;
        this._lastKeyframeRequestMs = 0;
        this._messageQueue = Promise.resolve();
    }

    get codecName() {
        return this.activeVideoCodec || this.activeCodecFamily || this.activeCodec || "webcodecs";
    }

    start(initialInfo = null) {
        if (initialInfo) {
            this._applyStreamInfo(initialInfo, false);
        }
        this.isActive = true;
        this.isPaused = false;
        this._hasDecodedFrame = false;
        this._messageQueue = Promise.resolve();

        return new Promise((resolve) => {
            this._firstFrameResolve = resolve;
            this._connect();
            this._startInfoPolling();
        });
    }

    _buildWebSocketUrl() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const url = new URL(this.wsPath, `${protocol}//${window.location.host}`);
        if (this.requestedCodecs.length > 0) {
            url.searchParams.set("codecs", this.requestedCodecs.join(","));
        }
        return url.toString();
    }

    _onStreamInfo(info) {
        if (info.ws_uri) {
            this.wsPath = info.ws_uri;
        }
    }

    _sendControl(payload) {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            try {
                this._ws.send(JSON.stringify(payload));
            } catch {
                // Ignore transient send errors while closing
            }
        }
    }

    _requestKeyframeThrottled() {
        const now = performance.now();
        if (now - this._lastKeyframeRequestMs < 500) {
            return;
        }
        this._lastKeyframeRequestMs = now;
        this._sendControl({ type: "request_keyframe" });
    }

    _closeDecoder() {
        if (this._decoder) {
            try {
                if (this._decoder.state !== "closed") {
                    this._decoder.close();
                }
            } catch {
                // Ignore close error on already errored decoder
            }
            this._decoder = null;
        }
        this._decoderCodec = null;
        this._decoderWidth = 0;
        this._decoderHeight = 0;
        this._waitingForKeyframe = true;
    }

    async _configureDecoder(msg) {
        const codec = msg.codec;
        const codedWidth = Number(msg.width) || this.width || 1920;
        const codedHeight = Number(msg.height) || this.height || 1080;

        if (
            this._decoder &&
            this._decoder.state === "configured" &&
            this._decoderCodec === codec &&
            this._decoderWidth === codedWidth &&
            this._decoderHeight === codedHeight
        ) {
            this.activeVideoCodec = msg.videoCodec || this.activeVideoCodec;
            this.activeCodecFamily = msg.codecFamily || this.activeCodecFamily;
            return true;
        }

        let config = null;
        for (const hardwareAcceleration of ["prefer-hardware", "no-preference"]) {
            const candidate = {
                codec,
                codedWidth,
                codedHeight,
                hardwareAcceleration,
                optimizeForLatency: true,
            };
            try {
                const res = await VideoDecoder.isConfigSupported(candidate);
                if (res?.supported) {
                    config = res.config || candidate;
                    break;
                }
            } catch {
                // Try next acceleration mode
            }
        }

        if (!this.isActive) {
            return false;
        }
        if (!config) {
            this._triggerFallback(`unsupported_decoder_config:${codec}`);
            return false;
        }

        this._closeDecoder();
        const decoder = new VideoDecoder({
            output: (frame) => {
                this._onDecodedFrame(frame);
            },
            error: (err) => {
                this.debugLogInstance?.log(`[WARN] VideoDecoder error: ${err?.message || err}`);
                this._closeDecoder();
                if (!this.isActive) return;
                if (this._firstFrameResolve) {
                    this._triggerFallback("decoder_error");
                } else {
                    this._requestKeyframeThrottled();
                }
            },
        });

        try {
            decoder.configure(config);
        } catch (err) {
            this.debugLogInstance?.log(`[WARN] VideoDecoder.configure failed: ${err?.message || err}`);
            try {
                decoder.close();
            } catch {
                // ignore
            }
            this._triggerFallback("decoder_configure_failed");
            return false;
        }

        this._decoder = decoder;
        this._decoderCodec = codec;
        this._decoderWidth = codedWidth;
        this._decoderHeight = codedHeight;
        this.activeCodec = codec;
        this.activeVideoCodec = msg.videoCodec || null;
        this.activeCodecFamily = msg.codecFamily || null;
        this._waitingForKeyframe = true;
        this.debugLogInstance?.log(`WebCodecs decoder configured: ${codec} (${msg.videoCodec || "unknown"}, ${codedWidth}x${codedHeight})`);
        if (!this._firstFrameResolve) {
            this.dispatchEvent(
                new CustomEvent("stream-updated", {
                    detail: {
                        width: this.width || codedWidth,
                        height: this.height || codedHeight,
                        stereo_format: this.stereoFormat,
                        stream_token: this.streamToken,
                        active_video_codec: this.codecName,
                        tokenChanged: false,
                    },
                })
            );
        }
        return true;
    }

    _onDecodedFrame(frame) {
        if (!this.isActive || this.isPaused) {
            frame.close();
            return;
        }

        const frameW = frame.displayWidth || frame.codedWidth || 0;
        const frameH = frame.displayHeight || frame.codedHeight || 0;
        const dimChanged = Boolean(frameW && frameH && (frameW !== this.width || frameH !== this.height));
        if (frameW > 0 && frameH > 0) {
            if (dimChanged && this.texture.image) {
                this._needsTextureDispose = true;
            }
            this.width = frameW;
            this.height = frameH;
            this.aspectRatio = frameW / frameH;
        }

        // Before the first render pass, attach frame directly so consumers see valid texture dimensions.
        if (!(typeof VideoFrame !== "undefined" && this.texture.image instanceof VideoFrame)) {
            if (this._pendingFrame) {
                this._pendingFrame.close();
                this._pendingFrame = null;
            }
            this.texture.setFrame(frame);
            this.hasNewFrame = true;
        } else {
            if (this._pendingFrame) {
                this._pendingFrame.close();
            }
            this._pendingFrame = frame;
            this.hasNewFrame = true;
        }

        this._hasDecodedFrame = true;
        const wasWaitingFirstFrame = Boolean(this._firstFrameResolve);
        this._resolveFirstFrame({ cancelled: false, fallback: false });

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
        }
    }

    _triggerFallback(reason) {
        if (!this.isActive) return;
        this.debugLogInstance?.log(`[INFO] WebCodecs fallback to MJPEG (${reason})`);
        const hadPendingFirstFrame = Boolean(this._firstFrameResolve);
        if (hadPendingFirstFrame) {
            this._resolveFirstFrame({ cancelled: false, fallback: true, reason });
            this.stop();
        } else {
            this.stop();
            this.dispatchEvent(new CustomEvent("fallback-mjpeg", { detail: { reason } }));
        }
    }

    async _handleWsMessage(ws, event) {
        if (!this.isActive || this._activeWs !== ws) return;

        if (typeof event.data === "string") {
            let msg;
            try {
                msg = JSON.parse(event.data);
            } catch {
                return;
            }
            if (!msg || typeof msg !== "object") return;
            if (msg.type === "fallback") {
                this._triggerFallback(msg.reason || "server_fallback");
                return;
            }
            if (this._ws !== ws) return;
            if (msg.type === "config") {
                await this._configureDecoder(msg);
            } else if (msg.type === "error") {
                this.debugLogInstance?.log(`[WARN] Stream WS error: ${msg.message || "unknown"}`);
            }
            return;
        }

        if (this._ws !== ws) return;

        if (!(event.data instanceof ArrayBuffer) || event.data.byteLength <= HEADER_BYTE_LENGTH) {
            return;
        }
        if (this.isPaused || !this._decoder || this._decoder.state !== "configured") {
            return;
        }

        const view = new DataView(event.data);
        const flags = view.getUint32(4, true);
        const isKeyframe = (flags & FLAG_KEYFRAME) !== 0;
        const lowTs = view.getUint32(8, true);
        const highTs = view.getUint32(12, true);
        const timestampUs = highTs * 0x100000000 + lowTs;

        if (this._waitingForKeyframe) {
            if (!isKeyframe) {
                return;
            }
            this._waitingForKeyframe = false;
        }

        // Reset queued P-frames if decode falls behind real-time.
        if (this._decoder.decodeQueueSize > 3 && !isKeyframe) {
            try {
                this._decoder.reset();
                if (this._decoderCodec) {
                    await this._configureDecoder({
                        codec: this._decoderCodec,
                        width: this.width,
                        height: this.height,
                        codecFamily: this.activeCodecFamily,
                        videoCodec: this.activeVideoCodec,
                    });
                }
            } catch {
                this._waitingForKeyframe = true;
            }
            this._requestKeyframeThrottled();
            return;
        }

        const payload = new Uint8Array(event.data, HEADER_BYTE_LENGTH);
        try {
            this._decoder.decode(
                new EncodedVideoChunk({
                    type: isKeyframe ? "key" : "delta",
                    timestamp: timestampUs,
                    data: payload,
                })
            );
        } catch (err) {
            this.debugLogInstance?.log(`[WARN] WebCodecs decode error: ${err?.message || err}`);
            this._waitingForKeyframe = true;
            this._requestKeyframeThrottled();
        }
    }

    _connect() {
        if (!this.isActive) return;

        const wsUrl = this._buildWebSocketUrl();
        let ws;
        try {
            ws = new WebSocket(wsUrl);
        } catch (err) {
            this.debugLogInstance?.log(`[WARN] WebSocket open failed: ${err?.message || err}`);
            if (!this._hasDecodedFrame || this._firstFrameResolve) {
                this._triggerFallback("ws_open_failed");
            } else {
                this._scheduleReconnect();
            }
            return;
        }
        ws.binaryType = "arraybuffer";
        this._ws = ws;
        this._activeWs = ws;

        ws.onopen = () => {
            if (this._ws !== ws || !this.isActive) return;
            this._waitingForKeyframe = true;
            if (this.isPaused) {
                this._sendControl({ type: "pause" });
            }
        };

        ws.onmessage = (event) => {
            // Serialize message handling so async VideoDecoder.isConfigSupported completes
            // before the immediately following initial IDR binary frame is processed.
            this._messageQueue = this._messageQueue
                .then(() => this._handleWsMessage(ws, event))
                .catch((err) => {
                    this.debugLogInstance?.log(`[WARN] WS message handling error: ${err?.message || err}`);
                });
        };

        ws.onerror = () => {
            if (this._ws !== ws || !this.isActive) return;
            this.debugLogInstance?.log("[WARN] Stream WebSocket connection error");
        };

        ws.onclose = () => {
            if (this._ws !== ws) return;
            this._ws = null;
            this._waitingForKeyframe = true;
            if (!this.isActive) return;
            if (!this._hasDecodedFrame || this._firstFrameResolve) {
                this._triggerFallback("ws_closed_before_first_frame");
                return;
            }
            this._scheduleReconnect();
        };
    }

    _scheduleReconnect() {
        if (!this.isActive || this._reconnectTimer) return;
        this._reconnectTimer = setTimeout(() => {
            this._reconnectTimer = null;
            if (this.isActive) {
                this._connect();
            }
        }, 1000);
    }

    consumeNewFrame() {
        if (!this.hasNewFrame) {
            return false;
        }
        this.hasNewFrame = false;

        if (this._pendingFrame) {
            if (this._needsTextureDispose) {
                this._needsTextureDispose = false;
                this.texture.dispose();
            }
            const prevFrame = this.texture.image;
            if (prevFrame && typeof prevFrame.close === "function") {
                prevFrame.close();
            }
            const nextFrame = this._pendingFrame;
            this._pendingFrame = null;
            this.texture.setFrame(nextFrame);
        }
        return true;
    }

    _onPaused() {
        if (this._pendingFrame) {
            this._pendingFrame.close();
            this._pendingFrame = null;
        }
        this._sendControl({ type: "pause" });
    }

    _onPlayed() {
        this._waitingForKeyframe = true;
        this._sendControl({ type: "resume" });
        this._requestKeyframeThrottled();
    }

    _onStop() {
        this._hasDecodedFrame = false;
        this._activeWs = null;
        if (this._ws) {
            const ws = this._ws;
            this._ws = null;
            try {
                ws.close(1000);
            } catch {
                // ignore
            }
        }
        this._closeDecoder();
        if (this._pendingFrame) {
            this._pendingFrame.close();
            this._pendingFrame = null;
        }
    }
}

export { WebCodecsStream };
