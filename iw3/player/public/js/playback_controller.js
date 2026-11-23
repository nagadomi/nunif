import * as THREE from "three";
import { STEREO_FORMATS, PREFETCH_CONFIG } from "./constants.js";
import { MjpegStream } from "./mjpeg_stream.js";
import {
    WebCodecsStream,
    getRequestedVideoCodecOverride,
    probeSupportedStreamCodecs,
} from "./webcodecs_stream.js";

/**
 * Controller responsible for media resource management (images & videos),
 * texture caching, playback lifecycle, and WebXR Layers integration.
 */
class PlaybackController extends EventTarget {
    constructor(renderer, scene, debugLogInstance) {
        super();
        this.renderer = renderer;
        this.scene = scene;
        this.debugLogInstance = debugLogInstance;

        this.textureLoader = new THREE.TextureLoader();
        this.textureCache = new Map(); // path -> { texture, aspectRatio }
        
        this.videoElement = null;
        this.videoTexture = null;
        this.desktopStream = null;
        this.currentLoadId = 0;
        this.isXRLayersActive = false;
        this.xrBinding = null;
    }

    get streamTexture() {
        return this.desktopStream ? this.desktopStream.texture : null;
    }

    get streamCodecName() {
        return this.desktopStream ? (this.desktopStream.codecName || null) : null;
    }

    get isStreamActive() {
        return !!(this.desktopStream && this.desktopStream.isActive);
    }

    get isPlaying() {
        if (this.isStreamActive) {
            return !this.desktopStream.isPaused;
        }
        return !!(this.videoElement && !this.videoElement.paused);
    }

    get currentTime() {
        return this.videoElement ? this.videoElement.currentTime : 0;
    }

    get duration() {
        return this.videoElement ? this.videoElement.duration : 0;
    }

    play() {
        if (this.isStreamActive) {
            this.desktopStream.play();
            return;
        }
        if (this.videoElement) {
            const playPromise = this.videoElement.play();
            if (playPromise !== undefined) {
                playPromise.catch((e) => {
                    this.debugLogInstance?.log(`[WARN] Autoplay prevented: ${e.message}`);
                });
            }
        }
    }

    pause() {
        if (this.isStreamActive) {
            this.desktopStream.pause();
            return;
        }
        if (this.videoElement) {
            this.videoElement.pause();
        }
    }

    togglePlayPause() {
        if (this.isStreamActive) {
            this.desktopStream.togglePause();
            return;
        }
        if (!this.videoElement) return;
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    consumeNewStreamFrame() {
        return this.desktopStream ? this.desktopStream.consumeNewFrame() : false;
    }

    destroyStream(notifyServer = false) {
        if (this.desktopStream) {
            this.desktopStream.stop();
            this.desktopStream = null;
            this.dispatchEvent(new CustomEvent('video-destroyed'));
            if (notifyServer) {
                fetch('/api/stream/stop', { method: 'POST', credentials: 'same-origin' }).catch(() => {});
            }
        }
    }

    _bindStreamEvents(stream, item) {
        stream.addEventListener('play-state-change', (e) => {
            if (this.desktopStream === stream) {
                this.dispatchEvent(new CustomEvent('play-state-change', { detail: e.detail }));
            }
        });
        stream.addEventListener('stream-updated', (e) => {
            if (this.desktopStream === stream) {
                this.dispatchEvent(new CustomEvent('stream-updated', { detail: e.detail }));
            }
        });
        stream.addEventListener('auth-required', () => {
            if (this.desktopStream === stream) {
                this.dispatchEvent(new CustomEvent('stream-auth-required'));
            }
        });
        stream.addEventListener('fallback-mjpeg', async () => {
            if (this.desktopStream !== stream) return;
            const loadId = this.currentLoadId;
            const streamUrl = item.stream_uri || '/api/stream';
            const fallbackStream = new MjpegStream(streamUrl, '/api/stream/info', this.debugLogInstance);
            this._bindStreamEvents(fallbackStream, item);
            this.desktopStream = fallbackStream;
            const res = await fallbackStream.start(item.stream_info || null);
            if (this.currentLoadId !== loadId || this.desktopStream !== fallbackStream || res?.cancelled) {
                if (this.desktopStream === fallbackStream && this.currentLoadId !== loadId) {
                    this.destroyStream();
                }
                return;
            }
            fallbackStream.texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
            this.dispatchEvent(
                new CustomEvent('stream-updated', {
                    detail: {
                        ...(item.stream_info || {}),
                        width: fallbackStream.width,
                        height: fallbackStream.height,
                        stereo_format: fallbackStream.stereoFormat,
                        stream_token: fallbackStream.streamToken,
                        active_video_codec: fallbackStream.codecName,
                        tokenChanged: false,
                    },
                })
            );
        });
    }

    async _startDesktopStream(item, loadId, finalFormat) {
        const info = item.stream_info || {};
        const urlOverride = getRequestedVideoCodecOverride();
        const serverVideoCodec = (info.video_codec || '').toLowerCase();
        const isServerJpeg = serverVideoCodec === 'jpeg' || serverVideoCodec === 'jpeg-gpu' || serverVideoCodec === 'mjpeg';
        const serverFamilies = Array.isArray(info.available_stream_codecs)
            ? info.available_stream_codecs.map((c) => c.toLowerCase())
            : ['h264'];

        let candidateFamilies = serverFamilies;
        if (urlOverride === 'h264' || urlOverride === 'hevc') {
            candidateFamilies = serverFamilies.filter((c) => c === urlOverride);
        }

        let requestedCodecs = [];
        if (urlOverride !== 'jpeg' && !isServerJpeg && candidateFamilies.length > 0 && info.ws_uri !== null) {
            requestedCodecs = await probeSupportedStreamCodecs(
                candidateFamilies,
                info.width || 1920,
                info.height || 1080
            );
        }

        if (this.currentLoadId !== loadId) {
            return { success: false, cancelled: true };
        }

        if (requestedCodecs.length > 0) {
            const wsStream = new WebCodecsStream(
                info.ws_uri || '/api/stream/ws',
                '/api/stream/info',
                requestedCodecs,
                this.debugLogInstance
            );
            this.desktopStream = wsStream;
            this._bindStreamEvents(wsStream, item);
            const startResult = await wsStream.start(info);
            if (this.currentLoadId !== loadId || this.desktopStream !== wsStream) {
                if (this.desktopStream === wsStream && this.currentLoadId !== loadId) {
                    this.destroyStream();
                }
                return { success: false, cancelled: true };
            }
            if (!startResult?.cancelled && !startResult?.fallback) {
                wsStream.texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
                const detail = {
                    item,
                    texture: wsStream.texture,
                    isVideo: false,
                    isStream: true,
                    aspectRatio: wsStream.aspectRatio,
                    format: finalFormat,
                    isXRActive: false
                };
                this.dispatchEvent(new CustomEvent('media-loaded', { detail }));
                return { success: true, detail };
            }
            if (startResult?.cancelled && !startResult?.fallback) {
                return { success: false, cancelled: true };
            }
        }

        const streamUrl = item.stream_uri || '/api/stream';
        const mjpegStream = new MjpegStream(streamUrl, '/api/stream/info', this.debugLogInstance);
        this.desktopStream = mjpegStream;
        this._bindStreamEvents(mjpegStream, item);

        const startResult = await mjpegStream.start(info);
        if (this.currentLoadId !== loadId || this.desktopStream !== mjpegStream || startResult?.cancelled) {
            if (this.desktopStream === mjpegStream && this.currentLoadId !== loadId) {
                this.destroyStream();
            }
            return { success: false, cancelled: true };
        }

        mjpegStream.texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
        const detail = {
            item,
            texture: mjpegStream.texture,
            isVideo: false,
            isStream: true,
            aspectRatio: mjpegStream.aspectRatio,
            format: finalFormat,
            isXRActive: false
        };
        this.dispatchEvent(new CustomEvent('media-loaded', { detail }));
        return { success: true, detail };
    }

    seek(time) {
        if (!this.videoElement || isNaN(this.videoElement.duration)) return;
        this.videoElement.currentTime = THREE.MathUtils.clamp(time, 0, this.videoElement.duration);
    }

    setVolume(volume) {
        if (this.videoElement) {
            this.videoElement.volume = volume;
        }
    }

    setPlaybackRate(playbackRate) {
        if (this.videoElement) {
            this.videoElement.playbackRate = playbackRate;
        }
    }

    setMuted(muted) {
        if (this.videoElement) {
            this.videoElement.muted = muted;
        }
    }

    setLoop(loop) {
        if (this.videoElement) {
            this.videoElement.loop = loop;
        }
    }

    destroyVideo() {
        this.dispatchEvent(new CustomEvent('video-destroyed'));
        if (this.videoTexture) {
            this.videoTexture.dispose();
            this.videoTexture = null;
        }
        if (this.videoElement) {
            this.videoElement.onloadedmetadata = null;
            this.videoElement.onerror = null;
            this.videoElement.onended = null;
            this.videoElement.onstalled = null;
            this.videoElement.ontimeupdate = null;
            this.videoElement.onplay = null;
            this.videoElement.onpause = null;
            this.videoElement.pause();
            this.videoElement.src = "";
            this.videoElement.load();
            this.videoElement = null;
        }
    }

    _configureImageTexture(texture, path) {
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.magFilter = THREE.LinearFilter;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.generateMipmaps = true;
        texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();

        const tw = texture.image.width || 1;
        const th = texture.image.height || 1;
        const aspectRatio = tw / th;
        this.textureCache.set(path, { texture, aspectRatio });
        return aspectRatio;
    }

    /**
     * Loads a media item (image or video).
     * @param {Object} item Gallery media item
     * @param {Object} options Configuration options
     * @returns {Promise<void>}
     */
    load(item, options = {}) {
        return new Promise((resolve) => {
            const {
                forcedFormat = null,
                targetFormat = null,
                loop = false,
                volume = 1.0,
                muted = false,
                playbackRate = 1.0,
                useXRLayers = false,
                domeXRScreen = null,
                environmentManager = null
            } = options;

            const loadId = ++this.currentLoadId;
            const url = `/api/media?path=${encodeURIComponent(item.path)}`;
            const finalFormat = targetFormat || item.stereo_format || STEREO_FORMATS.SBS_FULL;

            if (item.type === 'stream') {
                if (forcedFormat && this.desktopStream && this.desktopStream.isActive) {
                    const detail = {
                        item,
                        texture: this.desktopStream.texture,
                        isVideo: false,
                        isStream: true,
                        aspectRatio: this.desktopStream.aspectRatio,
                        format: finalFormat,
                        isXRActive: false
                    };
                    this.dispatchEvent(new CustomEvent('media-loaded', { detail }));
                    resolve({ success: true, detail });
                    return;
                }

                this.destroyVideo();
                this.destroyStream();

                this._startDesktopStream(item, loadId, finalFormat).then(resolve);
            } else if (item.type === 'video') {
                if (!(forcedFormat && this.videoElement)) {
                    this.destroyStream(true);
                    this.destroyVideo();
                    this.videoElement = document.createElement('video');
                    this.videoElement.crossOrigin = "anonymous";
                    this.videoElement.preload = "auto";
                    this.videoElement.autoplay = true;
                    this.videoElement.playsInline = true;
                    this.videoElement.setAttribute('playsinline', '');
                    this.videoElement.setAttribute('webkit-playsinline', '');
                    this.videoElement.loop = loop;
                    this.videoElement.volume = volume;
                    this.videoElement.defaultPlaybackRate = playbackRate;
                    this.videoElement.playbackRate = playbackRate;
                    this.videoElement.defaultMuted = muted;
                    this.videoElement.muted = muted;
                    if (muted) {
                        this.videoElement.setAttribute('muted', '');
                    }
                    this.videoElement.src = url;

                    this.videoElement.onended = () => {
                        this.dispatchEvent(new CustomEvent('ended', { detail: { item } }));
                    };

                    this.videoElement.ontimeupdate = () => {
                        this.dispatchEvent(new CustomEvent('timeupdate', { 
                            detail: { currentTime: this.videoElement.currentTime, duration: this.videoElement.duration } 
                        }));
                    };

                    this.videoElement.onplay = () => {
                        this.dispatchEvent(new CustomEvent('play-state-change', { detail: { isPlaying: true } }));
                    };

                    this.videoElement.onpause = () => {
                        this.dispatchEvent(new CustomEvent('play-state-change', { detail: { isPlaying: false } }));
                    };

                    this.videoElement.onstalled = () => {
                        this.debugLogInstance?.log(`[WARN] Video stalled: ${item.name}`);
                    };
                }

                const handleReady = async () => {
                    if (this.currentLoadId !== loadId) {
                        if (!forcedFormat) this.destroyVideo();
                        resolve({ success: false, cancelled: true });
                        return;
                    }

                    const vw = this.videoElement.videoWidth;
                    const vh = this.videoElement.videoHeight;

                    if (vw === 0 || vh === 0) {
                        const errDetail = { message: "Error: No video track or unsupported codec" };
                        this.dispatchEvent(new CustomEvent('error', { detail: errDetail }));
                        resolve({ success: false, error: errDetail.message });
                        return;
                    }

                    if (useXRLayers && domeXRScreen && environmentManager) {
                        await this.setupXRLayers(domeXRScreen, environmentManager);
                        if (this.isXRLayersActive) {
                            if (this.videoElement?.paused) {
                                this.play();
                            }
                            const detail = {
                                item,
                                texture: null,
                                isVideo: true,
                                aspectRatio: vw / vh,
                                format: finalFormat,
                                isXRActive: true
                            };
                            this.dispatchEvent(new CustomEvent('media-loaded', { detail }));
                            resolve({ success: true, detail });
                            return;
                        }
                    }

                    let texture = this.videoTexture;
                    if (!texture) {
                        texture = new THREE.VideoTexture(this.videoElement);
                        texture.colorSpace = THREE.SRGBColorSpace;
                        texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
                        
                        // VideoTexture does not support WebGL glGenerateMipmap (causes GL_INVALID_OPERATION)
                        // and mipmap generation for streaming video frames is extremely performance-heavy.
                        texture.generateMipmaps = false;
                        texture.minFilter = THREE.LinearFilter;
                        texture.magFilter = THREE.LinearFilter;
                        texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
                        this.videoTexture = texture;
                    }
                    // At this point readyState >= HAVE_CURRENT_DATA (2), so texImage2D will succeed without "no video" error.
                    texture.needsUpdate = true;

                    if (this.videoElement?.paused) {
                        this.play();
                    }

                    const detail = {
                        item,
                        texture,
                        isVideo: true,
                        aspectRatio: vw / vh,
                        format: finalFormat,
                        isXRActive: false
                    };
                    this.dispatchEvent(new CustomEvent('media-loaded', { detail }));
                    resolve({ success: true, detail });
                };

                const waitForFrame = () => {
                    if (this.currentLoadId !== loadId) {
                        if (!forcedFormat) this.destroyVideo();
                        resolve({ success: false, cancelled: true });
                        return;
                    }
                    // Ensure the first frame pixel data is decoded (readyState >= 2 = HAVE_CURRENT_DATA)
                    // before binding the texture to avoid WebGL "INVALID_VALUE: texImage2D: no video"
                    if (this.videoElement.readyState >= 2) {
                        handleReady();
                    } else {
                        this.videoElement.addEventListener('loadeddata', () => {
                            if (this.currentLoadId === loadId) {
                                handleReady();
                            }
                        }, { once: true });
                    }
                };

                this.videoElement.onloadedmetadata = waitForFrame;
                this.videoElement.onerror = () => {
                    if (this.currentLoadId === loadId) {
                        const err = this.videoElement.error;
                        let msg = "Video Load Error";
                        if (err) {
                            switch (err.code) {
                                case 1: msg = "Aborted"; break;
                                case 2: msg = "Network Error"; break;
                                case 3: msg = "Decode Error (Unsupported codec?)"; break;
                                case 4: msg = "Format Not Supported"; break;
                            }
                        }
                        const errDetail = { message: `Error: ${msg}`, path: item.path };
                        this.dispatchEvent(new CustomEvent('error', { detail: errDetail }));
                        this.destroyVideo();
                        resolve({ success: false, error: errDetail.message });
                    }
                };

                if (forcedFormat) {
                    waitForFrame();
                } else {
                    this.videoElement.load();
                }
            } else {
                // Image Loading
                this.destroyStream(true);
                this.destroyVideo();

                // Check texture cache
                const cached = this.textureCache.get(item.path);
                if (cached) {
                    const detail = {
                        item,
                        texture: cached.texture,
                        isVideo: false,
                        aspectRatio: cached.aspectRatio,
                        format: finalFormat,
                        isXRActive: false
                    };
                    this.dispatchEvent(new CustomEvent('media-loaded', { detail }));
                    resolve({ success: true, detail });
                    return;
                }

                this.textureLoader.load(
                    url,
                    (texture) => {
                        if (this.currentLoadId !== loadId) {
                            texture.dispose();
                            resolve({ success: false, cancelled: true });
                            return;
                        }
                        const aspectRatio = this._configureImageTexture(texture, item.path);
                        const detail = {
                            item,
                            texture,
                            isVideo: false,
                            aspectRatio,
                            format: finalFormat,
                            isXRActive: false
                        };
                        this.dispatchEvent(new CustomEvent('media-loaded', { detail }));
                        resolve({ success: true, detail });
                    },
                    undefined,
                    (_err) => {
                        if (this.currentLoadId === loadId) {
                            const errDetail = { message: "Error: Image Load Failed", path: item.path };
                            this.dispatchEvent(new CustomEvent('error', { detail: errDetail }));
                            resolve({ success: false, error: errDetail.message });
                        }
                    }
                );
            }
        });
    }

    prefetchImages(gallery, currentIndex) {
        if (!gallery || currentIndex === -1) return;

        const { count } = PREFETCH_CONFIG;
        const total = gallery.length;
        const targetIndices = [];

        for (let i = 1; i <= count; i++) {
            const next = currentIndex + i;
            const prev = currentIndex - i;
            if (next < total) targetIndices.push(next);
            if (prev >= 0) targetIndices.push(prev);
        }

        targetIndices.forEach(idx => {
            const item = gallery[idx];
            if (!item || item.type !== 'image' || this.textureCache.has(item.path)) return;

            const url = `/api/media?path=${encodeURIComponent(item.path)}`;
            this.textureLoader.load(url, (texture) => {
                this._configureImageTexture(texture, item.path);
                this.pruneCache(gallery, currentIndex);
            });
        });
    }

    pruneCache(gallery, currentIndex) {
        if (this.textureCache.size <= PREFETCH_CONFIG.maxSize) return;

        const pathsInCache = Array.from(this.textureCache.keys());
        let furthestPath = null;
        let maxDistance = -1;

        pathsInCache.forEach(path => {
            const idx = gallery ? gallery.findIndex(i => i.path === path) : -1;
            if (idx === -1) {
                maxDistance = Infinity;
                furthestPath = path;
                return;
            }
            const distance = Math.abs(idx - currentIndex);
            if (distance > maxDistance) {
                maxDistance = distance;
                furthestPath = path;
            }
        });

        if (furthestPath) {
            const entry = this.textureCache.get(furthestPath);
            if (entry?.texture) {
                entry.texture.dispose();
            }
            this.textureCache.delete(furthestPath);
        }
    }

    clearCache() {
        this.textureCache.forEach(entry => {
            if (entry?.texture) entry.texture.dispose();
        });
        this.textureCache.clear();
    }

    async setupXRLayers(domeXRScreen, environmentManager) {
        this.debugLogInstance?.log("[XR-Layers] setupXRLayers() called");
        const session = this.renderer.xr.getSession();
        const sessionHasLayers = session?.enabledFeatures ? session.enabledFeatures.includes('layers') : false;
        if (!session || !this.videoElement || !window.XRMediaBinding || !sessionHasLayers) {
            this.debugLogInstance?.log(`[XR-Layers] Skip: session=${!!session}, video=${!!this.videoElement}, binding=${!!window.XRMediaBinding}, layers=${sessionHasLayers}`);
            return;
        }

        if (!this.isXRLayersActive) {
            this.debugLogInstance?.log("[XR-Layers] Activating layers...");
            const refSpace = this.renderer.xr.getReferenceSpace();
            await domeXRScreen.initXRLayer(session, this.videoElement, refSpace);

            if (domeXRScreen.xrLayer) {
                try {
                    const baseLayer = this.renderer.xr.getBaseLayer();
                    this.debugLogInstance?.log("[XR-Layers] Updating Session Render State...");
                    session.updateRenderState({
                        layers: [domeXRScreen.xrLayer, baseLayer]
                    });

                    if (environmentManager) {
                        environmentManager.setBackgroundLocked(true);
                    } else {
                        this.scene.background = null;
                    }
                    this.renderer.setClearAlpha(0);

                    this.isXRLayersActive = true;
                    this.debugLogInstance?.log("[XR-Layers] Composition Active with Three.js baseLayer");
                } catch (e) {
                    this.debugLogInstance?.log(`[XR-Layers] Setup ERROR: ${e.message}`);
                    this.teardownXRLayers(domeXRScreen, environmentManager);
                }
            } else {
                this.debugLogInstance?.log("[XR-Layers] Failed to create video layer");
            }
        }
    }

    teardownXRLayers(domeXRScreen, environmentManager) {
        if (!this.isXRLayersActive) return;

        this.debugLogInstance?.log("[XR-Layers] Tearing down...");
        try {
            const session = this.renderer.xr.getSession();
            const baseLayer = this.renderer.xr.getBaseLayer();
            if (session && baseLayer) {
                session.updateRenderState({ layers: [baseLayer] });
            }
        } catch (e) {
            this.debugLogInstance?.log(`[XR-Layers] Teardown Session Error: ${e.message}`);
        }

        if (environmentManager) {
            environmentManager.setBackgroundLocked(false);
        } else {
            this.scene.background = new THREE.Color(0x000000);
        }
        this.renderer.setClearAlpha(1);

        if (domeXRScreen) {
            domeXRScreen.destroyXRLayer();
        }

        this.isXRLayersActive = false;
        this.xrBinding = null;
        this.debugLogInstance?.log("[XR-Layers] Teardown complete");
    }
}

export { PlaybackController };
