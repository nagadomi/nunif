import * as THREE from "three";
import { StereoScreen } from './stereo_screen.js';
import { UIManager } from './ui_manager.js';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';
import { LIMITS, DEFAULTS, STEREO_FORMATS, PREFETCH_CONFIG } from "./constants.js";
import { EnvironmentManager } from './environment_manager.js';
import { SubtitleWindow } from './window_subtitle.js';
import { GalleryManager } from './gallery_manager.js';
import { InputManager } from './input_manager.js';
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

        // Screen & Subtitle
        this.stereoScreen = new StereoScreen(new THREE.Texture(), 1.0);
        this.add(this.stereoScreen);
        this.stereoScreen.clearScreen(); 
        this.subtitleWindow = new SubtitleWindow();
        this.stereoScreen.add(this.subtitleWindow);

        this.uiManager = new UIManager(scene, camera, renderer, debugLogInstance, this);
        this.inputManager = new InputManager(this, this.uiManager);

        // Controller tracking
        this.controllers = [null, null];
        this.gamepads = [null, null];
        this.controllerRays = [null, null];
        this.controllerGrips = [null, null];

        this.textureLoader = new THREE.TextureLoader();
        this.textureCache = new Map(); // path -> { texture, aspectRatio }

        // Video state
        this.videoElement = null;
        this.currentLoadId = 0;
        this.currentTrackUrl = null;
        this.lastPlaybackTimeSave = 0;

        // Interactive states
        this.isControllerRotating = false;
        this.activeRotationController = null;
        this.initialRotationX = DEFAULTS.screen_tilt * Math.PI / 180;
        this.initialControllerRotationX = 0;

        this.isControllerTranslating = false;
        this.activeTranslationController = null;
        this.initialControllerPosition = new THREE.Vector3();
        this.initialScreenX = 0;
        this.initialScreenY = 0;

        this.selectStartTime = [0, 0];

        this.raycaster = new THREE.Raycaster();
        this.tempMatrix = new THREE.Matrix4();
        this.clock = new THREE.Clock();
        this.isXRActive = false;
        this.backgroundIntensity = DEFAULTS.screen_bg_color;

        this.setupControllers();
        
        // Initial state for non-XR mode
        this.stereoScreen.updateEyeVisibility(false);
        this.subtitleWindow.updateEyeVisibility(false);

        this.galleryManager.fetchDirectory('/').then(() => {
            this.debugLogInstance.log(`${this.galleryManager.playbackGallery.length} items found`);
        });
    }

    setBackgroundLevel(val) {
        this.environmentManager.setBackgroundLevel(val);
    }

    async loadMedia(index, forcedFormat = null) {
        const file = this.galleryManager.playbackGallery[index];
        if (!file) return;

        // IMMEDIATELY stop periodic saves to avoid race conditions during async operations
        const wasPlaying = this.isPlaying;
        const oldVideoTime = (this.videoElement && wasPlaying) ? this.videoElement.currentTime : null;
        const oldFile = wasPlaying ? this.galleryManager.getCurrentItem() : null;

        const loadId = ++this.currentLoadId;
        const url = `/api/image?path=${encodeURIComponent(file.path)}`;

        // Clear previous state
        this.uiManager.updateSubtitleText(""); 
        
        // Save old video position before switching
        if (oldFile && oldVideoTime !== null && !forcedFormat) {
            await this.uiManager.updateFileConfig(oldFile, { playback_time: oldVideoTime });
        }
        
        // Update playing item and gallery state
        this.galleryManager.setPlayingItem(index);

        if (forcedFormat) {
            // If the forced format is the same as the default detected format, remove the override
            const isDefault = (forcedFormat === file.stereo_format);
            await this.uiManager.updateFileConfig(file, { stereo_format: isDefault ? undefined : forcedFormat });
        }

        let targetFormat = forcedFormat;
        let targetConfig = null;
        if (!targetFormat) {
            targetConfig = await this.uiManager.getFileConfig(file);
            if (targetConfig && targetConfig.stereo_format) targetFormat = targetConfig.stereo_format;
        } else {
            targetConfig = await this.uiManager.getFileConfig(file);
        }

        // Apply file-specific settings using the centralized manager
        if (targetConfig) {
            await this.uiManager.applyFileConfig(targetConfig);
        }

        if (this.videoElement && !forcedFormat) {
            // Save current playback position before switching
            if (this.isPlaying) {
                const currentItem = this.galleryManager.getCurrentItem();
                if (currentItem) {
                    await this.uiManager.updateFileConfig(currentItem, { playback_time: this.videoElement.currentTime });
                }
            }

            if (this.currentTrackUrl) URL.revokeObjectURL(this.currentTrackUrl);
            this.videoElement.onloadedmetadata = this.videoElement.onerror = null;
            this.videoElement.pause(); this.videoElement.src = ""; this.videoElement.load();
            this.videoElement = null; 
            
            // Dispose current textures
            if (this.stereoScreen.displayScreenLeft.material.map) {
                this.stereoScreen.displayScreenLeft.material.map.dispose();
            }
            if (this.stereoScreen.displayScreenRight.material.map) {
                this.stereoScreen.displayScreenRight.material.map.dispose();
            }
        }

        this.debugLogInstance.log(`Loading: ${file.name}`);

        if (file.type === 'video') {
            if (!(forcedFormat && this.videoElement)) {
                this.videoElement = document.createElement('video');
                this.videoElement.src = url;
                this.videoElement.crossOrigin = "anonymous";
                this.videoElement.preload = "auto";
                this.videoElement.autoplay = true;
                this.videoElement.loop = this.uiManager.videoRepeat;
                this.videoElement.volume = this.uiManager.currentVolume;
                this.uiManager.applyPlaybackSpeed();

                this.videoElement.onended = () => {
                    this.uiManager.updateFileConfig(file, { playback_time: 0 });
                    this.uiManager.syncUI(true);
                };

                this.videoElement.onstalled = () => {
                    this.debugLogInstance.log(`[WARN] Video stalled: ${file.name}`);
                };
            }

            const handleMetadata = () => {
                if (this.currentLoadId !== loadId) { if (!forcedFormat) { this.videoElement.pause(); this.videoElement.src = ""; } return; }
                // Source loads and format/mipmap changes may reset media properties.
                this.uiManager.applyPlaybackSpeed();
                const vw = this.videoElement.videoWidth;
                const vh = this.videoElement.videoHeight;
                
                if (vw === 0 || vh === 0) {
                    this.uiManager.showNotification("Error: No video track or unsupported codec");
                    this.debugLogInstance.log(`[ERROR] Video dimensions are 0x0. Audio may play but no image.`);
                    // We don't call updateTexture to avoid invalid geometry, or we call it with a black texture
                    this.stereoScreen.clearScreen();
                    return;
                }

                const texture = new THREE.VideoTexture(this.videoElement);
                texture.colorSpace = THREE.SRGBColorSpace;
                texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
                
                // Mipmap and Filtering based on settings
                const useMipmap = this.uiManager.renderSettings.values.render_video_mipmap;
                if (useMipmap) {
                    texture.generateMipmaps = true;
                    texture.minFilter = THREE.LinearMipmapLinearFilter;
                } else {
                    texture.generateMipmaps = false;
                    texture.minFilter = THREE.LinearFilter;
                }
                texture.magFilter = THREE.LinearFilter;
                texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();

                const format = targetFormat || file.stereo_format || STEREO_FORMATS.SBS_FULL;
                this.stereoScreen.updateTexture(texture, vw / vh, format);
                this.lastPlaybackTimeSave = performance.now(); // Reset periodic save timer
                this.loadSubtitles(file.path);

                // Save recent file info (encrypted)
                storage.set('recent_path', this.galleryManager.currentPath);
                storage.set('recent_file_name', file.name);
                this.uiManager.hideRecentButton();

                // Restore playback position
                if (targetConfig && targetConfig.playback_time !== undefined) {
                    this.seek(targetConfig.playback_time);
                }
            };

            this.videoElement.onloadedmetadata = handleMetadata;
            this.videoElement.onerror = () => {
                if (this.currentLoadId === loadId) {
                    const error = this.videoElement.error;
                    let msg = "Video Load Error";
                    if (error) {
                        switch (error.code) {
                            case 1: msg = "Aborted"; break;
                            case 2: msg = "Network Error"; break;
                            case 3: msg = "Decode Error (Unsupported codec?)"; break;
                            case 4: msg = "Format Not Supported"; break;
                        }
                    }
                    this.uiManager.showNotification(`Error: ${msg}`);
                    this.debugLogInstance.log(`[ERROR] ${msg}: ${file.path}`);
                    this.stereoScreen.clearScreen();
                    this.videoElement = null;
                }
            };
            if (forcedFormat) handleMetadata(); else this.videoElement.load();
        } else {
            // Check cache first
            const cached = this.textureCache.get(file.path);
            if (cached) {
                const format = targetFormat || file.stereo_format || STEREO_FORMATS.SBS_FULL;
                this.stereoScreen.updateTexture(cached.texture, cached.aspectRatio, format);
                this.uiManager.syncUI(true);
                
                // Save recent file info (encrypted)
                storage.set('recent_path', this.galleryManager.currentPath);
                storage.set('recent_file_name', file.name);
                this.uiManager.hideRecentButton();

                // Trigger prefetch for neighbors
                this.prefetchImages();
                return;
            }

            // Clear subtitle state for images
            this.uiManager.availableSubtitles = [];
            this.uiManager.currentSubtitleIndex = -1;

            this.textureLoader.load(url, (texture) => {
                texture.colorSpace = THREE.SRGBColorSpace;
                texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
                texture.magFilter = THREE.LinearFilter;
                texture.minFilter = THREE.LinearMipmapLinearFilter;
                texture.generateMipmaps = true;
                texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
                const tw = texture.image.width || 1, th = texture.image.height || 1;
                const aspectRatio = tw / th;
                const format = targetFormat || file.stereo_format || STEREO_FORMATS.SBS_FULL;

                // Add to cache
                this.textureCache.set(file.path, { texture, aspectRatio });
                this.pruneCache();

                this.stereoScreen.updateTexture(texture, aspectRatio, format);
                this.uiManager.syncUI(true);

                // Save recent file info (encrypted)
                storage.set('recent_path', this.galleryManager.currentPath);
                storage.set('recent_file_name', file.name);
                this.uiManager.hideRecentButton();

                // Trigger prefetch for neighbors
                this.prefetchImages();
            }, undefined, (_err) => {
                if (this.currentLoadId === loadId) {
                    this.uiManager.showNotification("Error: Image Load Failed");
                    this.debugLogInstance.log(`[ERROR] Image Load Failed: ${file.path}`);
                    this.stereoScreen.clearScreen();
                }
            });
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
        } catch (_e) {
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
        if (!this.videoElement) return;
        if (this.isPlaying) { this.pause(); }
        else { this.play(); }
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
        if (!this.videoElement) return;
        this.videoElement.currentTime = THREE.MathUtils.clamp(time, 0, this.videoElement.duration);
    }

    loadNextImage(direction) {
        const next = this.galleryManager.getNextIndex(direction);
        if (next !== -1) this.loadMedia(next);
    }

    setupControllers() {
        const controllerModelFactory = new XRControllerModelFactory();
        const rayWidth = 0.003, rayLength = 5.0;
        const rayGeometry = new THREE.CylinderGeometry(rayWidth, rayWidth, rayLength, 8);
        rayGeometry.rotateX(-Math.PI / 2); rayGeometry.translate(0, 0, -rayLength / 2);
        const rayMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });

        // Handle XR Session end to reload page
        this.renderer.xr.addEventListener('sessionstart', () => {
            this.stereoScreen.updateEyeVisibility(true);
            this.subtitleWindow.updateEyeVisibility(true);
        });
        this.renderer.xr.addEventListener('sessionend', async () => {
            this.stereoScreen.updateEyeVisibility(false);
            this.subtitleWindow.updateEyeVisibility(false);
            await this.uiManager.saveSettings();
            window.location.reload();
        });

        for (let i = 0; i < 2; i++) {
            this.controllers[i] = this.renderer.xr.getController(i);
            this.scene.add(this.controllers[i]);
            this.controllers[i].addEventListener('connected', (event) => {
                this.gamepads[i] = event.data.gamepad;
                const grip = this.renderer.xr.getControllerGrip(i);
                const model = controllerModelFactory.createControllerModel(grip);
                model.visible = this.uiManager.isMainMenuVisible;
                grip.add(model); this.scene.add(grip); this.controllerGrips[i] = model;
                const ray = new THREE.Mesh(rayGeometry, rayMaterial);
                ray.visible = this.uiManager.isMainMenuVisible;
                this.controllers[i].add(ray); this.controllerRays[i] = ray;
            });
            this.controllers[i].addEventListener('selectstart', (e) => this.onSelectStart(e));
            this.controllers[i].addEventListener('selectend', (e) => this.onSelectEnd(e));
            this.controllers[i].addEventListener('squeezestart', (e) => this.onSqueezeStart(e));
            this.controllers[i].addEventListener('squeezeend', (e) => this.onSqueezeEnd(e));
        }
    }

    checkUIHit(controller) {
        this.tempMatrix.identity().extractRotation(controller.matrixWorld);
        this.raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
        this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(this.tempMatrix);
        this.raycaster.near = 0;
        this.raycaster.far = Infinity;
        return this.raycaster.intersectObjects(this.uiManager.getInteractableMeshes(), true);
    }

    onSelectStart(event) {
        const controller = event.target;
        if (this.checkUIHit(controller).length > 0) return;
        const index = this.controllers.indexOf(controller);
        if (index !== -1) this.selectStartTime[index] = performance.now();
        this.activeRotationController = controller; this.isControllerRotating = true;
        this.initialRotationX = this.rotation.x;
        const quat = new THREE.Quaternion(); controller.getWorldQuaternion(quat);
        this.initialControllerRotationX = new THREE.Euler().setFromQuaternion(quat, 'YXZ').x;
    }

    onSelectEnd(event) {
        const controller = event.target;
        if (controller !== this.activeRotationController) return;
        const index = this.controllers.indexOf(controller);
        const duration = performance.now() - this.selectStartTime[index];
        const rotDelta = Math.abs(this.rotation.x - this.initialRotationX);

        if (duration < 500 && rotDelta < 0.05) {
            if (this.uiManager.isAnyMenuVisible() || this.uiManager.activeSubMenu === this.uiManager.explorer) {
                if (this.checkUIHit(controller).length === 0) {
                    this.uiManager.hideMainMenu(true); this.toggleControllerUIVisibility(false);
                }
            } else {
                this.uiManager.showMainMenu(index); this.toggleControllerUIVisibility(true);
            }
        } else if (this.isControllerRotating) {
            // Save when rotation (Tilt) ends
            this.uiManager.saveSettings();
        }
        this.isControllerRotating = false; this.activeRotationController = null;
    }

    onSqueezeStart(event) {
        const controller = event.target;
        this.activeTranslationController = controller; this.isControllerTranslating = true;
        controller.getWorldPosition(this.initialControllerPosition);
        this.initialScreenX = this.stereoScreen.currentScreenX;
        this.initialScreenY = this.stereoScreen.currentScreenY;
    }

    onSqueezeEnd(event) {
        if (event.target === this.activeTranslationController) {
            this.isControllerTranslating = false;
            this.activeTranslationController = null;
            this.uiManager.saveSettings();
        }
    }

    toggleControllerUIVisibility(isVisible) {
        for (let i = 0; i < 2; i++) {
            if (this.controllerRays[i]) this.controllerRays[i].visible = isVisible;
            if (this.controllerGrips[i]) this.controllerGrips[i].visible = isVisible;
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
        this.updateEnvironmentFade(delta);
        this.uiManager.update(delta);
        this.inputManager.update(delta);

        // Periodically save playback position
        if (this.isPlaying && this.videoElement) {
            const now = performance.now();
            if (now - this.lastPlaybackTimeSave > 5000) {
                this.savePlaybackPosition();
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

    get isPlaying() {
        return !!(this.videoElement && !this.videoElement.paused);
    }

    play() {
        if (this.videoElement) {
            this.videoElement.play();
        }
    }

    pause() {
        if (this.videoElement) {
            this.videoElement.pause();
        }
    }

    exitXR() { 
        const session = this.renderer.xr.getSession(); 
        if (session) session.end(); 
    }
    async setEnvironment(filename) { await this.environmentManager.loadEnvironment(filename); }

    updateEnvironmentFade(delta) {
        this.environmentManager.update(delta);
    }

    prefetchImages() {
        const gm = this.galleryManager;
        if (!gm.playingGallery || gm.playingIndex === -1) return;

        const { count } = PREFETCH_CONFIG;
        const total = gm.playingGallery.length;
        const targetIndices = [];

        for (let i = 1; i <= count; i++) {
            const next = gm.playingIndex + i;
            const prev = gm.playingIndex - i;
            if (next < total) targetIndices.push(next);
            if (prev >= 0) targetIndices.push(prev);
        }

        targetIndices.forEach(idx => {
            const item = gm.playingGallery[idx];
            if (!item || item.type !== 'image' || this.textureCache.has(item.path)) return;

            const url = `/api/image?path=${encodeURIComponent(item.path)}`;
            this.textureLoader.load(url, (texture) => {
                texture.colorSpace = THREE.SRGBColorSpace;
                texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
                texture.magFilter = THREE.LinearFilter;
                texture.minFilter = THREE.LinearMipmapLinearFilter;
                texture.generateMipmaps = true;
                texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
                
                const tw = texture.image.width || 1, th = texture.image.height || 1;
                this.textureCache.set(item.path, { texture, aspectRatio: tw / th });
                this.pruneCache();
            });
        });
    }

    pruneCache() {
        if (this.textureCache.size <= PREFETCH_CONFIG.maxSize) return;

        const gm = this.galleryManager;
        const pathsInCache = Array.from(this.textureCache.keys());
        
        // Find the furthest item from current playing index
        let furthestPath = null;
        let maxDistance = -1;

        pathsInCache.forEach(path => {
            const idx = gm.playingGallery.findIndex(i => i.path === path);
            if (idx === -1) {
                // Not in current gallery anymore, prune immediately
                maxDistance = Infinity;
                furthestPath = path;
                return;
            }
            const distance = Math.abs(idx - gm.playingIndex);
            if (distance > maxDistance) {
                maxDistance = distance;
                furthestPath = path;
            }
        });

        if (furthestPath) {
            const entry = this.textureCache.get(furthestPath);
            if (entry && entry.texture) {
                entry.texture.dispose();
            }
            this.textureCache.delete(furthestPath);
        }
    }
}

export { StereoPlayer };
