import * as THREE from "three";
import { StereoScreen } from './stereo_screen.js';
import { StereoDomeScreen } from './stereo_dome_screen.js';
import { StereoDomeXRScreen } from './stereo_dome_xr_screen.js';
import { UIManager } from './ui_manager.js';
import { LIMITS, DEFAULTS, STEREO_FORMATS } from "./constants.js";
import { EnvironmentManager } from './environment_manager.js';
import { SubtitleWindow } from './window_subtitle.js';
import { GalleryManager } from './gallery_manager.js';
import { InputManager } from './input_manager.js';
import { PlaybackController } from './playback_controller.js';
import { ScreenTransformController } from './screen_transform_controller.js';
import { UIUtils } from './ui_common.js';
import { storage } from './storage.js';

class StereoPlayer extends THREE.Group {
    constructor(renderer, camera, scene, debugLogInstance) {
        super();
        this.renderer = renderer;
        this.camera = camera;
        this.scene = scene;
        this.debugLogInstance = debugLogInstance;

        storage.setDebugLog(debugLogInstance);

        this.isARMode = false;
        this.scene.background = new THREE.Color(0x000000);
        this.appConfig = { mode: 'player', features: { file_library: true, desktop_stream: false } };
        this.streamItem = null;
        this.streamFormatOverride = null;
        this.configReady = this._fetchAppConfig();

        // Managers
        this.galleryManager = new GalleryManager(debugLogInstance);
        this.environmentManager = new EnvironmentManager(scene, debugLogInstance);
        this.scene.add(this.environmentManager);

        // Playback Controller
        this.playbackController = new PlaybackController(renderer, scene, debugLogInstance);
        this.playbackController.addEventListener('ended', (e) => {
            this.uiManager.updateFileConfig(e.detail.item, { playback_time: 0 });
            this.uiManager.syncUI(true);
        });
        this.playbackController.addEventListener('error', (e) => {
            this.uiManager.showNotification(e.detail.message);
            this.debugLogInstance.log(`[ERROR] ${e.detail.message}`);
            this.activeScreen.clearScreen();
        });
        this.playbackController.addEventListener('play-state-change', () => {
            this.uiManager.syncUI(true);
        });
        this.playbackController.addEventListener('video-destroyed', () => {
            if (this.planeScreen) this.planeScreen.resetTexture();
            if (this.domeScreen) this.domeScreen.resetTexture();
        });
        this.playbackController.addEventListener('stream-updated', (e) => {
            this.onStreamUpdated(e.detail);
        });
        this.playbackController.addEventListener('stream-auth-required', () => {
            if (!this.renderer.xr.isPresenting) {
                window.location.reload();
            }
        });

        // Screen & Subtitle
        this.uiManager = new UIManager(scene, camera, renderer, debugLogInstance, this);
        
        const screenOptions = {
            onTransformChange: (key, val) => this.uiManager.onSliderChange(key, val, false),
            onSizeChange: () => this.uiManager.syncUI(true),
            debugLogInstance: this.debugLogInstance
        };
        this.planeScreen = new StereoScreen(new THREE.Texture(), 1.0, screenOptions);
        this.domeScreen = new StereoDomeScreen(new THREE.Texture(), 1.0, screenOptions);
        this.domeXRScreen = new StereoDomeXRScreen(screenOptions);
        
        this.add(this.planeScreen);
        this.add(this.domeScreen);
        this.add(this.domeXRScreen);
        
        this.activeScreen = this.planeScreen;
        this.planeScreen.visible = false;
        this.domeScreen.visible = false;
        this.domeXRScreen.visible = false;

        this.subtitleWindow = new SubtitleWindow();
        this.activeScreen.add(this.subtitleWindow);

        this.setupAxesGuide();

        // Gesture & Transform Controller
        this.transformController = new ScreenTransformController({
            onTransformChange: (key, val) => this.uiManager.onSliderChange(key, val, false),
            onSaveSettings: () => this.uiManager.saveSettings(),
            onVibrate: (controllerIndex, intensity, duration) => {
                const pointer = this.uiManager.xrPointers[controllerIndex];
                if (pointer) UIUtils.vibratePointer(pointer.id, intensity, duration, this.uiManager);
            },
            onDragStateChange: (isDragging) => this.setAxesGuideVisible(isDragging),
            debugLogInstance: this.debugLogInstance
        });

        this.inputManager = new InputManager(this, this.uiManager);

        this.currentTrackUrl = null;
        this.lastPlaybackTimeSave = 0;

        this.clock = new THREE.Clock();
        this.isXRActive = false;
        this.backgroundIntensity = DEFAULTS.screen_bg_color;

        // Experimental: Toggle between individual eye tracking (correct parallax) 
        // and head-center fixed (traditional viewing).
        this.useEyeTracking = true;

        // Caches for tracking
        this._tempPosL = new THREE.Vector3();
        this._tempPosR = new THREE.Vector3();
        this._headWorldPos = new THREE.Vector3();

        this.rotation.order = 'YXZ';
        this.trackDevice = DEFAULTS.screen_track_device;
        this._resetScreenFrames = 5;
        this._resetScreenYaw = false;

        this.setupXRSessionEvents();
        
        // Initial state for non-XR mode
        this.activeScreen.updateEyeVisibility(false);
        this.activeScreen.clearScreen();

        this.initModeAsync();
    }

    async _fetchAppConfig() {
        for (let attempt = 0; attempt < 3; attempt++) {
            try {
                const res = await fetch('/api/config', { cache: 'no-store', credentials: 'same-origin' });
                if (res.ok) {
                    this.appConfig = await res.json();
                    return this.appConfig;
                }
                if (res.status === 401 && !this.renderer.xr.isPresenting) {
                    window.location.reload();
                    return this.appConfig;
                }
            } catch {
                // Retry while server is starting up
            }
            if (attempt < 2) {
                await new Promise((r) => setTimeout(r, 500));
            }
        }
        return this.appConfig;
    }

    async initModeAsync() {
        const config = await this.configReady;
        if (this.uiManager?.ready) {
            await this.uiManager.ready;
        }
        this.uiManager.setAppConfig(config);
        if (config.mode === 'desktop') {
            await this.loadStream();
        } else {
            this.galleryManager.fetchDirectory('/').then(() => {
                this.debugLogInstance.log(`${this.galleryManager.playbackGallery.length} items found`);
            });
        }
    }

    setBackgroundLevel(val) {
        this.environmentManager.setBackgroundLevel(val);
        // If XR Layers are active, we must keep scene.background as null 
        // to ensure the video layer remains visible behind the UI.
        if (this.isXRLayersActive) {
            this.scene.background = null;
        }
    }

    setColorParameter(key, val) {
        this.planeScreen.setColorParameter(key, val);
        this.domeScreen.setColorParameter(key, val);
    }

    setLutTexture(tex) {
        this.planeScreen.setLutTexture(tex);
        this.domeScreen.setLutTexture(tex);
    }

    async switchScreen(format) {
        if (this._isSwitching) return;
        this._isSwitching = true;

        try {
            const isDome = (format === STEREO_FORMATS.SBS_180);
            const isPresenting = this.renderer.xr.isPresenting;
            const settingOn = this.uiManager.settingsManager.get('render_xr_layers');
            const bindingExists = !!window.XRMediaBinding;
            const hasVideo = !!this.videoElement;
            const session = this.renderer.xr.getSession();
            const sessionHasLayers = session?.enabledFeatures ? session.enabledFeatures.includes('layers') : false;

            const useXRLayers = isDome && isPresenting && settingOn && bindingExists && sessionHasLayers && hasVideo;
            
            this.debugLogInstance.log(`[XR-Layers] Evaluated: ${useXRLayers} (dome=${isDome}, vr=${isPresenting}, set=${settingOn}, bind=${bindingExists}, layers=${sessionHasLayers}, vid=${hasVideo})`);

            let nextScreen;
            if (useXRLayers) {
                nextScreen = this.domeXRScreen;
            } else if (isDome) {
                nextScreen = this.domeScreen;
            } else {
                nextScreen = this.planeScreen;
            }
            
            const prevScreen = this.activeScreen;

            if (nextScreen !== prevScreen) {
                if (!useXRLayers && this.isXRLayersActive) {
                    this.teardownXRLayers();
                }
                prevScreen.visible = false;
                this.activeScreen = nextScreen;
                this.setAxesGuideVisible(false);
                
                if (this.activeScreen.supportsSubtitles) {
                    this.activeScreen.add(this.subtitleWindow);
                    this.subtitleWindow.visible = true;
                } else {
                    if (this.subtitleWindow.parent) this.subtitleWindow.parent.remove(this.subtitleWindow);
                    this.subtitleWindow.visible = false;
                    this.uiManager.updateSubtitleText(""); 
                }
                
                // Apply effective settings for the newly activated screen type
                if (this.uiManager.settingsManager) {
                    this.uiManager.settingsManager.applyAll(this.activeScreen.screenType);
                }
            }
            
            this.activeScreen.visible = true;
            this.activeScreen.updateEyeVisibility(this.renderer.xr.isPresenting);
            this.environmentManager.visible = !isDome;

            // Sync XR Layers if active
            if (useXRLayers && this.videoElement) {
                this.debugLogInstance.log("[XR-Layers] Branch -> setupXRLayers");
                await this.setupXRLayers();
            } else {
                this.debugLogInstance.log(`[XR-Layers] Branch -> teardown (use=${useXRLayers}, vid=${!!this.videoElement})`);
                this.teardownXRLayers();
            }

            this.updateMatrixWorld(true);
            this.activeScreen.updatePositions();
        } catch (e) {
            this.debugLogInstance.log(`[XR-Layers] FATAL ERROR: ${e.message}`);
            console.error(e);
        } finally {
            this._isSwitching = false;
        }
    }

    get videoElement() {
        return this.playbackController.videoElement;
    }

    get videoTexture() {
        return this.playbackController.videoTexture;
    }

    get isPlaying() {
        return this.playbackController.isPlaying;
    }

    get isXRLayersActive() {
        return this.playbackController.isXRLayersActive;
    }

    get controllers() {
        return this.inputManager.controllers;
    }

    get gamepads() {
        return this.inputManager.gamepads;
    }

    async loadMedia(index, forcedFormat = null) {
        const file = this.galleryManager.playbackGallery[index];
        if (!file) return;

        this.streamItem = null;
        this.streamFormatOverride = null;

        // IMMEDIATELY stop periodic saves to avoid race conditions during async operations
        const wasPlaying = this.isPlaying;
        const oldVideoTime = (this.videoElement && wasPlaying) ? this.videoElement.currentTime : null;
        const oldFile = wasPlaying ? this.galleryManager.getCurrentItem() : null;

        // Clear previous state
        this.uiManager.updateSubtitleText(""); 
        
        // Save old video position before switching
        if (oldFile && oldVideoTime !== null && !forcedFormat) {
            await this.uiManager.settingsManager.updateFileState({ playback_time: oldVideoTime }, oldFile);
        }
        
        // Update playing item and gallery state
        this.galleryManager.setPlayingItem(index);

        // Load settings and state for the new file
        await this.uiManager.settingsManager.loadForFile(file);

        // If no user override or saved state, query file info if stereo format is undefined
        if (!forcedFormat && !this.uiManager.settingsManager.getFileState('stereo_format') && !file.stereo_format) {
            try {
                const res = await fetch(`/api/file/info?path=${encodeURIComponent(file.path)}`);
                if (res.ok) {
                    const info = await res.json();
                    if (info && info.stereo_format) {
                        file.stereo_format = info.stereo_format;
                    }
                }
            } catch {
                // Ignore network errors and keep current format
            }
        }

        if (forcedFormat) {
            const isDefault = (forcedFormat === file.stereo_format);
            await this.uiManager.settingsManager.updateFileState({ stereo_format: isDefault ? undefined : forcedFormat });
        }

        const finalFormat = forcedFormat || this.uiManager.settingsManager.getFileState('stereo_format') || file.stereo_format || STEREO_FORMATS.SBS_FULL;
        await this.switchScreen(finalFormat);
        await this.uiManager.settingsManager.applyForLoadedFile(this.activeScreen.screenType);

        // Teardown XR layers if active before starting the next media load
        this.teardownXRLayers();

        this.debugLogInstance.log(`Loading: ${file.name}`);

        const isDome = (finalFormat === STEREO_FORMATS.SBS_180);
        const isPresenting = this.renderer.xr.isPresenting;
        const settingOn = this.uiManager.settingsManager.get('render_xr_layers');
        const bindingExists = !!window.XRMediaBinding;
        const session = this.renderer.xr.getSession();
        const sessionHasLayers = session?.enabledFeatures ? session.enabledFeatures.includes('layers') : false;
        const useXRLayers = isDome && isPresenting && settingOn && bindingExists && sessionHasLayers;

        const options = {
            forcedFormat,
            targetFormat: finalFormat,
            loop: this.uiManager.videoRepeat,
            volume: this.uiManager.currentVolume,
            muted: this.uiManager.isMuted,
            playbackRate: this.uiManager.playbackRate,
            useXRLayers,
            domeXRScreen: this.domeXRScreen,
            environmentManager: this.environmentManager
        };

        const result = await this.playbackController.load(file, options);
        if (!result.success) {
            if (!result.cancelled) {
                this.activeScreen.clearScreen();
            }
            return;
        }

        const { detail } = result;
        if (detail.isVideo) {
            if (detail.isXRActive) {
                await this.switchScreen(finalFormat);
                this.uiManager.settingsManager.applyAll(this.activeScreen.screenType);
            } else {
                this.activeScreen.updateTexture(detail.texture, detail.aspectRatio, detail.format);
            }
            this.lastPlaybackTimeSave = performance.now();
            this.loadSubtitles(file.path);

            const savedPlaybackTime = this.uiManager.settingsManager.getFileState('playback_time');
            if (savedPlaybackTime !== null && savedPlaybackTime !== undefined) {
                this.seek(savedPlaybackTime);
            }

            if (this.videoElement?.paused) {
                this.playbackController.play();
            }
        } else {
            this.uiManager.availableSubtitles = [];
            this.uiManager.currentSubtitleIndex = -1;
            this.activeScreen.updateTexture(detail.texture, detail.aspectRatio, detail.format);
            this.playbackController.prefetchImages(this.galleryManager.playingGallery, this.galleryManager.playingIndex);
        }

        // Save recent file info (encrypted)
        storage.set('recent_path', this.galleryManager.currentPath);
        storage.set('recent_file_name', file.name);
        this.uiManager.hideRecentButton();
        this.uiManager.syncUI(true);
    }

    async loadStream(forcedFormat = null) {
        if (this.uiManager?.ready) {
            await this.uiManager.ready;
        }

        if (!forcedFormat) {
            const oldFile = this.galleryManager.getCurrentItem();
            const oldVideoTime = (this.videoElement && !isNaN(this.videoElement.currentTime))
                ? this.videoElement.currentTime
                : null;
            this.uiManager.updateSubtitleText("");
            this.uiManager.availableSubtitles = [];
            this.uiManager.currentSubtitleIndex = -1;
            if (oldFile && oldVideoTime !== null) {
                await this.uiManager.settingsManager.updateFileState({ playback_time: oldVideoTime }, oldFile);
            }
            this.playbackController.destroyVideo();
            this.galleryManager.setPlayingItem(-1);
            if (this.appConfig?.mode !== 'desktop') {
                this.uiManager.showNotification('Starting Desktop Stream...', 10000);
            }
        }

        const infoEndpoint = forcedFormat ? '/api/stream/info' : '/api/stream/start';
        const infoMethod = forcedFormat ? 'GET' : 'POST';
        let info = null;
        let errorDetail = null;
        while (!info) {
            try {
                const res = await fetch(infoEndpoint, {
                    method: infoMethod,
                    cache: 'no-store',
                    credentials: 'same-origin',
                });
                if (res.ok) {
                    info = await res.json();
                    break;
                }
                if (res.status === 401 && !this.renderer.xr.isPresenting) {
                    window.location.reload();
                    return false;
                }
                try {
                    const errData = await res.json();
                    if (errData?.detail) errorDetail = errData.detail;
                } catch {
                    // ignore non-json error body
                }
                this.debugLogInstance.log(`[WARN] Stream info returned HTTP ${res.status}`);
            } catch (e) {
                this.debugLogInstance.log(`[WARN] Failed to fetch stream info: ${e.message}`);
            }

            if (forcedFormat && this.streamItem?.stream_info) {
                info = this.streamItem.stream_info;
                break;
            }
            if (this.appConfig?.mode !== 'desktop') {
                this.uiManager.showNotification(
                    errorDetail ? `Stream unavailable: ${errorDetail}` : 'Stream unavailable',
                    3000
                );
                this.uiManager.syncUI(true);
                return false;
            }
            await new Promise((r) => setTimeout(r, 1000));
        }

        if (this.appConfig?.mode !== 'desktop' && !forcedFormat) {
            this.uiManager.hideNotification();
        }

        const streamFormat = info.stereo_format || STEREO_FORMATS.SBS_HALF;
        const streamName = this._formatStreamName(info.width, info.height, null);

        const streamItem = {
            path: '__desktop_stream__',
            name: streamName,
            type: 'stream',
            mtime: 0,
            stereo_format: streamFormat,
            stream_uri: info.stream_uri || '/api/stream',
            stream_info: info,
        };
        this.streamItem = streamItem;

        this.uiManager.updateSubtitleText("");
        this.uiManager.availableSubtitles = [];
        this.uiManager.currentSubtitleIndex = -1;

        await this.uiManager.settingsManager.loadForFile(streamItem);
        if (this.uiManager.settingsManager.getFileState('stereo_format') !== null) {
            await this.uiManager.settingsManager.updateFileState({ stereo_format: undefined });
        }
        this.streamFormatOverride = forcedFormat || null;

        const finalFormat = this.streamFormatOverride || streamItem.stereo_format || STEREO_FORMATS.SBS_HALF;
        await this.switchScreen(finalFormat);
        await this.uiManager.settingsManager.applyForLoadedFile(this.activeScreen.screenType);
        this.teardownXRLayers();

        this.debugLogInstance.log(`Loading stream: ${streamName} (${finalFormat})`);
        const result = await this.playbackController.load(streamItem, {
            forcedFormat,
            targetFormat: finalFormat,
        });
        if (!result.success) {
            if (!result.cancelled) {
                this.streamItem = null;
                this.streamFormatOverride = null;
                this.activeScreen.clearScreen();
            }
            this.uiManager.syncUI(true);
            return false;
        }

        const { detail } = result;
        const activeWidth = this.playbackController.desktopStream?.width || info.width || 0;
        const activeHeight = this.playbackController.desktopStream?.height || info.height || 0;
        const activeCodec = this.playbackController.streamCodecName || null;
        streamItem.name = this._formatStreamName(activeWidth, activeHeight, activeCodec);
        this.activeScreen.updateTexture(detail.texture, detail.aspectRatio, detail.format);
        this.uiManager?.streamSettings?.fetchSettings?.();
        this.uiManager.syncUI(true);
        return true;
    }

    _formatStreamName(width, height, codecName) {
        const hasDims = Boolean(width && height);
        if (hasDims && codecName) {
            return `Desktop (${width}x${height}, ${codecName})`;
        }
        if (hasDims) {
            return `Desktop (${width}x${height})`;
        }
        if (codecName) {
            return `Desktop (${codecName})`;
        }
        return 'Desktop Stream';
    }

    async stopStream() {
        this.playbackController.destroyStream(true);
        this.streamItem = null;
        this.streamFormatOverride = null;
        this.activeScreen.clearScreen();
        if (this.uiManager?.activeSubMenu === this.uiManager?.streamSettings) {
            this.uiManager.switchSubMenu(null);
        }
        this.uiManager.syncUI(true);
    }

    async onStreamUpdated(info) {
        if (!info || !this.streamItem) return;
        try {
            const prevBackendFormat = this.streamItem.stereo_format;
            if (info.tokenChanged || (info.stereo_format && info.stereo_format !== prevBackendFormat)) {
                this.streamFormatOverride = null;
            }

            this.streamItem.stream_info = { ...this.streamItem.stream_info, ...info };
            if (info.stereo_format) {
                this.streamItem.stereo_format = info.stereo_format;
            }
            const streamWidth = info.width || this.playbackController.desktopStream?.width || this.streamItem.stream_info?.width || 0;
            const streamHeight = info.height || this.playbackController.desktopStream?.height || this.streamItem.stream_info?.height || 0;
            const streamCodec = this.playbackController.streamCodecName || info.active_video_codec || null;
            this.streamItem.name = this._formatStreamName(streamWidth, streamHeight, streamCodec);

            const finalFormat = this.streamFormatOverride || this.streamItem.stereo_format || STEREO_FORMATS.SBS_HALF;
            if (finalFormat !== this.activeScreen.currentStereoFormat) {
                await this.switchScreen(finalFormat);
                await this.uiManager.settingsManager.applyForLoadedFile(this.activeScreen.screenType);
            }

            const aspectRatio = (info.width && info.height)
                ? (info.width / info.height)
                : (this.playbackController.desktopStream?.aspectRatio || this.activeScreen.aspectRatio);
            const tex = this.playbackController.streamTexture;
            if (tex) {
                this.activeScreen.updateTexture(tex, aspectRatio, finalFormat);
            }
            if (
                info.bufferMsChanged &&
                typeof info.buffer_ms === 'number' &&
                !this.playbackController.hasUrlBufferOverride &&
                !this.uiManager?.streamSettings?._isDragging
            ) {
                this.playbackController.setStreamBufferMs(info.buffer_ms);
            }
            this.uiManager?.streamSettings?.fetchSettings?.();
            this.uiManager.syncUI(true);
        } catch (e) {
            this.debugLogInstance?.log(`[ERROR] Stream update failed: ${e.message}`);
            if (!this.renderer.xr.isPresenting) {
                window.location.reload();
            }
        }
    }

    async loadSubtitles(path) {
        if (!this.videoElement) return;
        this.videoElement.querySelectorAll('track').forEach(t => t.remove());
        this.uiManager.availableSubtitles = [];

        try {
            const response = await fetch(`/api/subtitles?path=${encodeURIComponent(path)}`);
            if (response.ok) {
                const subs = await response.json();
                this.uiManager.availableSubtitles = subs;
                if (subs.length > 0) {
                    const index = THREE.MathUtils.clamp(this.uiManager.currentSubtitleIndex, 0, subs.length - 1);
                    this.uiManager.currentSubtitleIndex = index;
                    this.setSubtitleTrack(index);
                } else {
                    this.uiManager.updateSubtitleText("");
                }
            } else {
                this.uiManager.updateSubtitleText("");
            }
        } catch {
            this.uiManager.updateSubtitleText("");
        }
    }

    setSubtitleTrack(index) {
        if (!this.videoElement || !this.uiManager.availableSubtitles[index]) return;
        if (this.currentTrackUrl) URL.revokeObjectURL(this.currentTrackUrl);
        this.videoElement.querySelectorAll('track').forEach(t => t.remove());

        const sub = this.uiManager.availableSubtitles[index];
        const blob = new Blob([sub.vtt], { type: 'text/vtt' });
        this.currentTrackUrl = URL.createObjectURL(blob);

        const track = document.createElement('track');
        track.kind = 'subtitles'; track.label = sub.title; track.srclang = 'und';
        track.src = this.currentTrackUrl; track.default = true;
        this.videoElement.appendChild(track);

        const handleTrackReady = () => {
            const textTrack = this.videoElement.textTracks[0];
            if (textTrack) {
                textTrack.mode = this.uiManager.showSubtitles ? 'hidden' : 'disabled';
                
                const update = () => {
                    const cue = textTrack.activeCues ? textTrack.activeCues[0] : null;
                    this.uiManager.updateSubtitleText(cue ? cue.text : "");
                };
                textTrack.oncuechange = update;
                // Initial check for current cue
                setTimeout(update, 100);
            }
        };
        if (track.readyState === 2) handleTrackReady(); else track.addEventListener('load', handleTrackReady);
    }

    togglePlayPause() {
        this.playbackController.togglePlayPause();
    }

    updateSubtitlesMode(show) {
        if (!this.videoElement || !this.videoElement.textTracks[0]) return;
        const textTrack = this.videoElement.textTracks[0];
        textTrack.mode = show ? 'hidden' : 'disabled';
        if (show) {
            const cue = textTrack.activeCues[0];
            this.uiManager.updateSubtitleText(cue ? cue.text : "");
        } else {
            this.uiManager.updateSubtitleText("");
        }
    }

    seek(time) {
        this.playbackController.seek(time);
    }

    loadNextImage(direction) {
        if (this.playbackController.isStreamActive) return;
        const next = this.galleryManager.getNextIndex(direction);
        if (next !== -1) this.loadMedia(next);
    }

    setARMode(isAR) {
        if (this.isARMode === isAR) return;
        this.isARMode = isAR;
        if (this.debugLogInstance) {
            this.debugLogInstance.log(`[XR] setARMode: ${isAR}`);
        }
        this.environmentManager.setARMode(isAR);
        this.planeScreen?.setARMode?.(isAR);
        this.domeScreen?.setARMode?.(isAR);
        if (this.uiManager) {
            this.uiManager.isARMode = isAR;
            this.uiManager.syncUI(true);
        }
    }

    resetScreenPosition(options = false) {
        const resetYaw = typeof options === 'boolean' ? options : !!options?.resetYaw;
        if (this.camera) {
            this.position.copy(this.camera.position);
            if (resetYaw) {
                const euler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
                this.rotation.y = euler.y;
            }
        }
        this._resetScreenFrames = 5;
        this._resetScreenYaw = resetYaw;
    }

    resetScreenTransform() {
        this.resetScreenPosition(true);
    }

    setTrackDevice(track) {
        this.trackDevice = !!track;
        if (this.trackDevice) {
            this.resetScreenPosition();
        }
    }

    setupXRSessionEvents() {
        this.renderer.xr.addEventListener('sessionstart', () => {
            const blendMode = this.renderer.xr.getEnvironmentBlendMode?.() ?? this.renderer.xr.getSession()?.environmentBlendMode;
            if (blendMode) {
                this.setARMode(blendMode === 'alpha-blend' || blendMode === 'additive');
            }
            this.activeScreen.updateEyeVisibility(true);
            this.subtitleWindow.updateEyeVisibility(true);
            this.resetScreenTransform();

            const refSpace = this.renderer.xr.getReferenceSpace?.();
            if (refSpace) {
                refSpace.addEventListener('reset', () => {
                    this.rotation.y = 0;
                    this.resetScreenTransform();
                    if (this.debugLogInstance) {
                        this.debugLogInstance.log('[XR] Reference space reset detected, reset screen transform');
                    }
                });
            }
        });
        this.renderer.xr.addEventListener('sessionend', async () => {
            this.setARMode(false);
            this.activeScreen.updateEyeVisibility(false);
            this.subtitleWindow.updateEyeVisibility(false);
            await this.uiManager.saveSettings();
            window.location.reload();
        });
    }

    toggleControllerUIVisibility(isVisible) {
        this.inputManager.toggleControllerUIVisibility(isVisible);
    }

    setupAxesGuide() {
        this.axesGuideGroup = new THREE.Group();
        this.axesGuideGroup.visible = false;

        const guideLength = 1.0; // 1m total length
        const guideRadius = 0.0015; // 3mm diameter
        const guideDistance = 0.65; // Positioned between origin (0m) and UI panel (0.9m)

        const laserMaterial = new THREE.MeshBasicMaterial({
            color: 0xff3344, // Vivid laser red
            transparent: true,
            opacity: 0.85,
            depthTest: false
        });
        laserMaterial.renderOrder = 9999;

        const centerFixedMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff, // White indicator for center zero reference
            transparent: true,
            opacity: 0.95,
            depthTest: false
        });
        centerFixedMaterial.renderOrder = 9999;

        // Dynamic horizontal bar (X axis) that tracks Pos Y
        const xGeo = new THREE.CylinderGeometry(guideRadius, guideRadius, guideLength, 8);
        xGeo.rotateZ(Math.PI / 2);
        this.axesGuideHorizontalBar = new THREE.Mesh(xGeo, laserMaterial);
        this.axesGuideHorizontalBar.renderOrder = 9999;

        // Fixed vertical bar (Y axis)
        const yGeo = new THREE.CylinderGeometry(guideRadius, guideRadius, guideLength, 8);
        const yMesh = new THREE.Mesh(yGeo, laserMaterial);
        yMesh.renderOrder = 9999;

        // Short fixed horizontal bar at center (Y = 0 reference)
        const centerBarLength = 0.15; // 15cm reference notch
        const centerBarGeo = new THREE.CylinderGeometry(guideRadius * 1.3, guideRadius * 1.3, centerBarLength, 8);
        centerBarGeo.rotateZ(Math.PI / 2);
        const centerBarMesh = new THREE.Mesh(centerBarGeo, centerFixedMaterial);
        centerBarMesh.renderOrder = 9999;

        // Center zero dot
        const dotGeo = new THREE.SphereGeometry(guideRadius * 1.8, 8, 8);
        const dotMesh = new THREE.Mesh(dotGeo, centerFixedMaterial);
        dotMesh.renderOrder = 9999;

        const crosshair = new THREE.Group();
        crosshair.add(this.axesGuideHorizontalBar);
        crosshair.add(yMesh);
        crosshair.add(centerBarMesh);
        crosshair.add(dotMesh);
        crosshair.position.set(0, 0, -guideDistance);

        this.axesGuideGroup.add(crosshair);
        this.add(this.axesGuideGroup);
    }

    setAxesGuideVisible(visible) {
        if (!this.axesGuideGroup) return;
        const isDome = this.activeScreen && (this.activeScreen.screenType === 'dome' || this.activeScreen.screenType === 'dome_xr');
        this.axesGuideGroup.visible = visible && isDome;
        if (this.axesGuideGroup.visible) {
            this.updateAxesGuide();
        }
    }

    updateAxesGuide() {
        if (!this.axesGuideGroup || !this.axesGuideGroup.visible || !this.activeScreen) return;

        const panDeg = this.activeScreen._unitToAngle ? this.activeScreen._unitToAngle(this.activeScreen.currentScreenX) : 0;
        const rollDeg = this.activeScreen.currentRoll || 0;
        const yOffset = this.activeScreen.currentScreenY || 0;

        this.axesGuideGroup.rotation.set(
            0,
            THREE.MathUtils.degToRad(panDeg),
            THREE.MathUtils.degToRad(rollDeg),
            'YXZ'
        );

        if (this.axesGuideHorizontalBar) {
            this.axesGuideHorizontalBar.position.y = yOffset;
        }
    }

    handleAnimation(delta) {
        if (delta === undefined) delta = this.clock.getDelta() * 1000;
        if (this.renderer.xr.isPresenting && !this.isXRActive) {
            this.isXRActive = true;
            if (this.uiManager.isMainMenuVisible) {
                this.toggleControllerUIVisibility(true);
                setTimeout(() => { if (this.renderer.xr.isPresenting) this.uiManager.showMainMenu(); }, 500);
            }
        } else if (!this.renderer.xr.isPresenting && this.isXRActive) this.isXRActive = false;

        if (this.renderer.xr.isPresenting) {
            const blendMode = this.renderer.xr.getEnvironmentBlendMode?.() ?? this.renderer.xr.getSession()?.environmentBlendMode;
            if (blendMode) {
                const isAR = (blendMode === 'alpha-blend' || blendMode === 'additive');
                if (this.isARMode !== isAR) {
                    this.setARMode(isAR);
                }
            }
        }
        
        const isDome = this.activeScreen && (this.activeScreen.screenType === 'dome' || this.activeScreen.screenType === 'dome_xr');
        
        if (isDome || this.trackDevice || this._resetScreenFrames > 0) {
            this.position.copy(this.camera.position);
        }
        if (this._resetScreenFrames > 0) {
            if (this._resetScreenYaw && this.camera) {
                const euler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
                this.rotation.y = euler.y;
            }
            if (this._resetScreenFrames > 0) {
                this._resetScreenFrames--;
                if (this._resetScreenFrames === 0) {
                    this._resetScreenYaw = false;
                }
            }
        }
        this.updateMatrixWorld(true);
        this.environmentManager.update(delta);
        
        if (this.useEyeTracking) {
            // Experimental: Calculate exact eye offsets from world positions
            const xrCamera = this.renderer.xr.getCamera(this.camera);
            if (xrCamera.cameras.length >= 2) {
                this._headWorldPos.setFromMatrixPosition(this.camera.matrixWorld);
                this._tempPosL.setFromMatrixPosition(xrCamera.cameras[0].matrixWorld).sub(this._headWorldPos);
                this._tempPosR.setFromMatrixPosition(xrCamera.cameras[1].matrixWorld).sub(this._headWorldPos);

                this.activeScreen.updatePositions(this._tempPosL, this._tempPosR);
            } else {
                this.activeScreen.updatePositions();
            }
        } else {
            // Standard: Center fixed (less CPU usage, traditional feeling)
            this.activeScreen.updatePositions();
        }

        this.uiManager.update(delta);
        this.inputManager.update(delta);

        if (this.axesGuideGroup?.visible) {
            this.updateAxesGuide();
        }

        // Periodically save playback position
        if (this.isPlaying && this.videoElement) {
            const now = performance.now();
            if (now - this.lastPlaybackTimeSave > 5000) {
                this.savePlaybackPosition();
            }
        }

        // Pre-upload dirty dynamic video texture before renderer.render().
        // This avoids interleaved texture uploads during transparent/MSAA draws in WebXR multiview,
        // preventing screen and GUI element flickering on Meta Quest / Chromium (Three.js #33458 / #32151).
        const videoTex = this.videoTexture;
        const videoElem = this.videoElement;
        if (videoTex && videoElem && videoElem.readyState >= 2) {
            if (this.isPlaying) {
                videoTex.needsUpdate = true;
            }
            this.renderer.initTexture(videoTex);
        }

        const streamTex = this.playbackController.streamTexture;
        if (streamTex) {
            this.playbackController.consumeNewStreamFrame();
            const hasValidStreamImage = streamTex.isVideoFrameTexture
                ? (typeof VideoFrame !== "undefined" && streamTex.image instanceof VideoFrame)
                : Boolean(streamTex.image);
            if (hasValidStreamImage) {
                this.renderer.initTexture(streamTex);
            }
        }
    }

    async savePlaybackPosition() {
        if (!this.videoElement || !this.isPlaying) return;
        const currentItem = this.galleryManager.getCurrentItem();
        if (currentItem) {
            await this.uiManager.updateFileConfig(currentItem, { playback_time: this.videoElement.currentTime });
            this.lastPlaybackTimeSave = performance.now();
        }
    }

    setRotationX(radian) {
        this.rotation.x = THREE.MathUtils.clamp(radian, LIMITS.screen_tiltMin * Math.PI / 180, LIMITS.screen_tiltMax * Math.PI / 180);
    }

    async exitXR() { 
        const session = this.renderer.xr.getSession(); 
        if (session) {
            session.end(); 
        } else {
            await this.uiManager.saveSettings();
            window.location.reload();
        }
    }

    async setupXRLayers() {
        await this.playbackController.setupXRLayers(this.domeXRScreen, this.environmentManager);
    }

    teardownXRLayers() {
        this.playbackController.teardownXRLayers(this.domeXRScreen, this.environmentManager);
    }
}

export { StereoPlayer };
