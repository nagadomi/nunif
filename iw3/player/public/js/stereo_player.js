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

        this.scene.background = new THREE.Color(0x000000);

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

        this.setupXRSessionEvents();
        
        // Initial state for non-XR mode
        this.activeScreen.updateEyeVisibility(false);
        this.activeScreen.clearScreen();

        this.galleryManager.fetchDirectory('/').then(() => {
            this.debugLogInstance.log(`${this.galleryManager.playbackGallery.length} items found`);
        });
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

            const useXRLayers = isDome && isPresenting && settingOn && bindingExists && hasVideo;
            
            this.debugLogInstance.log(`[XR-Layers] Evaluated: ${useXRLayers} (dome=${isDome}, vr=${isPresenting}, set=${settingOn}, bind=${bindingExists}, vid=${hasVideo})`);

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
        const useXRLayers = isDome && isPresenting && settingOn && bindingExists;

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
        const next = this.galleryManager.getNextIndex(direction);
        if (next !== -1) this.loadMedia(next);
    }

    setupXRSessionEvents() {
        this.renderer.xr.addEventListener('sessionstart', () => {
            this.activeScreen.updateEyeVisibility(true);
            this.subtitleWindow.updateEyeVisibility(true);
        });
        this.renderer.xr.addEventListener('sessionend', async () => {
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
        
        this.position.copy(this.camera.position);
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
        if (videoTex && this.videoElement && this.videoElement.readyState >= 2) {
            this.renderer.initTexture(videoTex);
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
