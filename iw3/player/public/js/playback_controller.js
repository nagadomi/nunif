import * as THREE from "three";
import { STEREO_FORMATS, PREFETCH_CONFIG } from "./constants.js";

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
        this.currentLoadId = 0;
        this.isXRLayersActive = false;
        this.xrBinding = null;
    }

    get isPlaying() {
        return !!(this.videoElement && !this.videoElement.paused);
    }

    get currentTime() {
        return this.videoElement ? this.videoElement.currentTime : 0;
    }

    get duration() {
        return this.videoElement ? this.videoElement.duration : 0;
    }

    play() {
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
        if (this.videoElement) {
            this.videoElement.pause();
        }
    }

    togglePlayPause() {
        if (!this.videoElement) return;
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
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

            if (item.type === 'video') {
                if (!(forcedFormat && this.videoElement)) {
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
        if (!session || !this.videoElement || !window.XRMediaBinding) {
            this.debugLogInstance?.log(`[XR-Layers] Skip: session=${!!session}, video=${!!this.videoElement}, binding=${!!window.XRMediaBinding}`);
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

                    this.scene.background = null;
                    this.renderer.setClearAlpha(0);
                    if (environmentManager) environmentManager.isBackgroundLocked = true;

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

        if (environmentManager) environmentManager.isBackgroundLocked = false;
        this.scene.background = new THREE.Color(0x000000);
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
