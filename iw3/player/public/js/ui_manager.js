import * as THREE from 'three';
import { LUTCubeLoader } from 'three';
import { Container, Text, reversePainterSortStable } from '@pmndrs/uikit';
import { forwardHtmlEvents, createRayPointer } from '@pmndrs/pointer-events';
import { DEFAULTS, UI_CONFIG, COLORS, FONT_CONFIG, getSettingLimits } from './constants.js';
import { storage } from './storage.js';
import { UIUtils, ThreeUtils } from './ui_common.js';

// Import split UI components
import { MainMenu } from './menu_main.js';
import { ScreenSettingsMenu } from './menu_screen_settings.js';
import { ColorSettingsMenu } from './menu_color_settings.js';
import { EnvironmentSettingsMenu } from './menu_environment_settings.js';
import { RenderSettingsMenu } from './menu_render_settings.js';
import { ExplorerWindow } from './window_explorer.js';
import { SubtitleSettingsMenu } from './menu_subtitle_settings.js';
import { SettingsManager } from './settings_manager.js';

class UIManager extends THREE.Group {
    constructor(scene, camera, renderer, debugLogInstance, stereoPlayer) {
        super();
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        this.debugLogInstance = debugLogInstance;
        this.stereoPlayer = stereoPlayer;
        this.settingsManager = new SettingsManager(this);

        this.pixelSize = UI_CONFIG.pixelSize;
        this.renderer.setTransparentSort(reversePainterSortStable);

        // --- UI Visibility State ---
        this.isMainMenuVisible = false;
        this.isNotificationVisible = false;
        this.activeSubMenu = null;

        // Subtitle Track State
        this.availableSubtitles = [];
        this.currentSubtitleIndex = -1;
        this._unmuteVolume = DEFAULTS.screen_volume;
        this._currentLutName = "None";

        // Explorer State
        this.currentPage = 0;
        this.itemsPerPage = 15;
        this.shouldJumpToCurrentItem = false;
        this.shouldJumpToPath = null;

        // Environment Lists
        this.skyboxes = ["None"];
        this.models = ["None"];
        this.luts = ["None"];
        this.availableFrameRates = [0]; // 0 = Auto
        this.currentSkyboxIndex = 0;
        this.currentModelIndex = 0;
        this.currentLutIndex = 0;

        // Feedback State
        this._atLimit = {};
        this.fonts = [
            { id: 'Auto', label: 'Auto', languages: 'Automatic detection' },
            { id: 'NotoSansJP', label: 'Japanese', languages: 'Japanese (JIS Level 1 & 2)' },
            { id: 'NotoSansSC', label: 'Chinese (Simplified)', languages: 'Simplified Chinese (Standard table)' },
            { id: 'NotoSansTC', label: 'Chinese (Traditional)', languages: 'Traditional Chinese (MOE Standard)' },
            { id: 'NotoSans', label: 'Latin / Cyrillic', languages: 'English, Russian, European, etc.' },
            { id: 'NotoSansKR', label: 'Korean', languages: 'Korean (All Hangul syllables)' }
        ];

        // Events
        this.abortController = new AbortController();
        this.pointerEvents = forwardHtmlEvents(renderer.domElement, () => this.camera, this.scene, {
            signal: this.abortController.signal,
            recursive: true,
            filter: (object) => ThreeUtils.isHierarchyVisible(object)
        });
        this.xrPointers = [null, null];
        this.ignorePointerUntilReleased = [false, false];

        this.visible = false;
        this.currentZ = DEFAULTS.uiZ;
        this.smoothedOcclusion = 0;
        this.occlusionRaycaster = new THREE.Raycaster();
        this.occlusionRaycaster.layers.enable(1);
        this.occlusionRaycaster.layers.enable(2);

        // FPS & Status metrics
        const FPS_WINDOW = 500;
        this.fpsHistory = new Float32Array(FPS_WINDOW);
        this.fpsIndex = 0;
        this.fpsCount = 0;
        this.currentFPS = 0;
        this.lastStatusUpdateTime = 0;
        this.lastOcclusionUpdateTime = 0;

        this.initAsync();
        this.scene.add(this);
    }

    async initAsync() {
        await storage.ready;
        await this.settingsManager.init();
        this.galleryManager.loadSettings();

        this.setupUI();
        this.updateMenuPositions();

        this.settingsManager.applyAll();

        this.fetchEnvironments();
        this.fetchLuts();

        // Check for recent file
        const recentPath = storage.get('recent_path');
        const recentFile = storage.get('recent_file_name');
        if (recentPath && recentFile) {
            this.mainMenu.recentButton.setProperties({ display: 'flex' });
        }

        this.showMainMenu();
    }

    async toggleMenuAlignment() {
        const next = this.menuAlignment === 'right' ? 'left' : 'right';
        await this.settingsManager.set('menu_alignment', next);
        this.syncUI(true);
    }

    updateMenuPositions() {
        const offset = this.menuAlignment === 'right' ? UI_CONFIG.menuMarginLeft : -UI_CONFIG.menuMarginLeft;
        const targetMenus = [this.screenSettings, this.colorSettings, this.environmentSettings, this.renderSettings, this.subtitleSettings];
        targetMenus.forEach(menu => {
            if (menu && menu.container) {
                menu.container.setProperties({ marginLeft: offset });
            }
        });
    }

    async openRecentFile() {
        const recentPath = storage.get('recent_path');
        const recentFile = storage.get('recent_file_name');
        if (!recentPath || !recentFile) return;

        try {
            // Load the gallery for the saved path
            await this.galleryManager.fetchDirectory(recentPath);
            // Use playbackGallery (which respects the current sort mode) to find the file
            const gallery = this.galleryManager.playbackGallery;
            const index = gallery.findIndex(item => item.name === recentFile);
            
            if (index !== -1) {
                await this.loadMedia(index);
                this.hideRecentButton();
            } else {
                this.showNotification("Recent file not found");
                this.hideRecentButton();
            }
        } catch (e) {
            this.debugLogInstance.log(`[ERROR] Failed to open recent file: ${e.message}`);
            this.hideRecentButton();
        }
    }

    hideRecentButton() {
        if (this.mainMenu?.recentButton) {
            this.mainMenu.recentButton.setProperties({ display: 'none' });
        }
    }

    setupUI() {
        let savedFont = this.settingsManager.get('render_font') || 'Auto';

        // Self-healing: Reset to Auto if the saved font is no longer available (e.g. NotoSansRU)
        if (savedFont !== 'Auto' && !this.fonts.find(f => f.id === savedFont)) {
            this.debugLogInstance.log(`[WARN] Font "${savedFont}" not found, resetting to Auto.`);
            savedFont = 'Auto';
            setTimeout(() => this.onSliderChange('render_font', 'Auto'), 0);
        }
        
        let defaultFont = 'NotoSans'; 
        if (savedFont !== 'Auto') {
            defaultFont = savedFont;
        } else {
            const lang = navigator.language;
            if (lang.startsWith('ja')) {
                defaultFont = 'NotoSansJP';
            } else if (lang.startsWith('zh')) {
                if (lang === 'zh-TW' || lang === 'zh-HK' || lang === 'zh-MO') {
                    defaultFont = 'NotoSansTC';
                } else {
                    defaultFont = 'NotoSansSC';
                }
            } else if (lang.startsWith('ko')) {
                defaultFont = 'NotoSansKR';
            }
        }

        this.mainContainer = new Container({
            flexDirection: 'column', alignItems: 'center', pixelSize: this.pixelSize,
            width: 1600, height: 1200, ...FONT_CONFIG, fontFamily: defaultFont
        });
        this.mainContainer.frustumCulled = false;
        this.add(this.mainContainer);

        this.mainMenu = new MainMenu(this, defaultFont);
        this.screenSettings = new ScreenSettingsMenu(this, defaultFont);
        this.colorSettings = new ColorSettingsMenu(this, defaultFont);
        this.environmentSettings = new EnvironmentSettingsMenu(this, defaultFont);
        this.renderSettings = new RenderSettingsMenu(this, defaultFont);
        this.subtitleSettings = new SubtitleSettingsMenu(this, defaultFont);
        this.explorer = new ExplorerWindow(this, defaultFont, this.galleryManager);

        this.allSettingsMenus = [this.screenSettings, this.colorSettings, this.environmentSettings, this.renderSettings, this.subtitleSettings];
        [...this.allSettingsMenus, this.explorer].forEach(menu => {
            if (menu?.container) {
                menu.container.visible = false;
            }
        });

        this.mainContainer.add(this.mainMenu.container);
        this.mainContainer.add(this.screenSettings.container);
        this.mainContainer.add(this.colorSettings.container);
        this.mainContainer.add(this.environmentSettings.container);
        this.mainContainer.add(this.renderSettings.container);
        this.mainContainer.add(this.subtitleSettings.container);
        this.mainContainer.add(this.explorer.container);

        this.notificationContainer = new Container({
            flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            paddingLeft: 24, paddingRight: 24, paddingTop: 12, paddingBottom: 12,
            backgroundColor: COLORS.button, borderRadius: 16,
            display: 'none', pixelSize: this.pixelSize,
            depthTest: false,
            renderOrder: 99999,
            ...FONT_CONFIG, fontFamily: defaultFont
        });
        this.scene.add(this.notificationContainer);
        this.notificationText = new Text({ 
            text: '', fontSize: 32, color: COLORS.text,
            depthTest: false,
            renderOrder: 99999 
        });
        this.notificationContainer.add(this.notificationText);

        this.mainContainer.position.set(0, -0.5, DEFAULTS.uiZ);
        this.renderOrder = 999;
    }

    get galleryManager() {
        return this.stereoPlayer ? this.stereoPlayer.galleryManager : null;
    }

    get environmentManager() {
        return this.stereoPlayer ? this.stereoPlayer.environmentManager : null;
    }

    get activeScreen() {
        return this.stereoPlayer ? this.stereoPlayer.activeScreen : null;
    }

    get videoElement() {
        return this.stereoPlayer ? this.stereoPlayer.videoElement : null;
    }

    get currentItem() {
        return this.galleryManager ? this.galleryManager.getCurrentItem() : null;
    }

    get screenType() {
        return this.activeScreen ? this.activeScreen.screenType : 'plane';
    }

    get currentStereoFormat() {
        return this.activeScreen ? this.activeScreen.currentStereoFormat : null;
    }

    get showSubtitles() {
        return this.settingsManager.get('subtitle_visible');
    }

    get currentVolume() {
        return this.settingsManager.get('screen_volume');
    }

    get playbackRate() {
        return this.settingsManager.get('video_speed');
    }

    get isMuted() {
        return this.settingsManager.get('video_muted');
    }

    get videoRepeat() {
        return this.settingsManager.get('video_repeat');
    }

    get menuAlignment() {
        return this.settingsManager.get('menu_alignment');
    }

    togglePlayPause() {
        if (this.stereoPlayer) this.stereoPlayer.togglePlayPause();
    }

    async exitXR() {
        if (this.stereoPlayer) await this.stereoPlayer.exitXR();
    }

    loadNextMedia(direction) {
        if (this.stereoPlayer) this.stereoPlayer.loadNextImage(direction);
    }

    async loadMedia(index, format = null) {
        if (this.stereoPlayer) {
            return await this.stereoPlayer.loadMedia(index, format);
        }
    }

    seek(seconds) {
        if (this.stereoPlayer) {
            this.stereoPlayer.seek(seconds);
        }
    }

    savePlaybackPosition() {
        if (this.stereoPlayer) this.stereoPlayer.savePlaybackPosition();
    }

    updateTargetFPS(val) {
        if (this.renderer?.xr?.isPresenting) {
            const session = this.renderer.xr.getSession();
            if (session && session.updateTargetFrameRate) {
                let targetFPS = 0;
                if (val === 0) {
                    targetFPS = this.availableFrameRates.find(r => r >= 30) || 0;
                } else if (this.availableFrameRates.includes(val)) {
                    targetFPS = val;
                }
                if (targetFPS > 0) {
                    session.updateTargetFrameRate(targetFPS).catch(e => this.debugLogInstance.log(`[ERROR] FPS Change Error: ${e.message}`));
                }
            }
        }
    }

    // --- Visibility Control ---

    /**
     * Shows a temporary notification.
     * @param {string} text 
     * @param {number} duration 
     * @param {THREE.Vector3} [worldPosition] - Optional specific world position
     */
    showNotification(text, duration = 1000, worldPosition = null) {
        if (!this.notificationContainer) return;
        
        if (this.notificationTimeout) {
            clearTimeout(this.notificationTimeout);
        }

        this.notificationText.setProperties({ text: text });
        this.notificationContainer.setProperties({ display: 'flex' });
        this.isNotificationVisible = true;
        
        if (worldPosition) {
            // Position at specific world coordinates
            this.notificationContainer.position.copy(worldPosition);
            // Face the camera lens directly (prevents tilting)
            this.notificationContainer.quaternion.copy(this.camera.quaternion);
            
            // Offset slightly towards the camera to avoid Z-fighting with the screen
            const camPos = new THREE.Vector3();
            this.camera.getWorldPosition(camPos);
            const dir = new THREE.Vector3().subVectors(camPos, worldPosition).normalize();
            this.notificationContainer.position.addScaledVector(dir, 0.1);
        } else {
            // Default position: in front of the camera
            const pos = new THREE.Vector3(), quat = new THREE.Quaternion();
            this.camera.updateMatrixWorld(true);
            this.camera.getWorldPosition(pos);
            this.camera.getWorldQuaternion(quat);
            
            this.notificationContainer.position.copy(pos);
            this.notificationContainer.quaternion.copy(quat);
            
            // Offset forward and slightly down relative to camera
            const offset = new THREE.Vector3(0, -0.4, DEFAULTS.uiZ - 0.4);
            offset.applyQuaternion(quat);
            this.notificationContainer.position.add(offset);
        }
        
        this.notificationContainer.visible = true;

        this.notificationTimeout = setTimeout(() => {
            this.notificationContainer.setProperties({ display: 'none' });
            this.notificationContainer.visible = false;
            this.isNotificationVisible = false;
            this.notificationTimeout = null;
        }, duration);
    }

    switchSubMenu(targetMenu) {
        if (this.activeSubMenu === targetMenu) {
            if (this.activeSubMenu) {
                this.activeSubMenu.container.setProperties({ display: 'none' });
                this.activeSubMenu.container.visible = false;
                this.activeSubMenu = null;
            }
            if (this.mainContainer) this.mainContainer.update(0);
            return;
        }
        if (this.activeSubMenu) {
            this.activeSubMenu.container.setProperties({ display: 'none' });
            this.activeSubMenu.container.visible = false;
        }
        this.activeSubMenu = targetMenu;
        if (this.activeSubMenu) {
            this.activeSubMenu.container.setProperties({ display: 'flex' });
            this.activeSubMenu.container.visible = true;
            if (this.activeSubMenu === this.explorer) {
                this.shouldJumpToCurrentItem = true;
                this.fetchDirectory(this.galleryManager.currentPath);
            } else { this.syncUI(true); }
        }
        if (this.mainContainer) this.mainContainer.update(0);
    }

    showMainMenu(triggeringControllerIndex = -1) {
        if (!this.mainContainer) return;

        // If opened via VR controller trigger, we must ensure any active 'down' state
        // is cleared BEFORE the menu becomes visible. Otherwise, the 'up' event
        // (selectend) might trigger a click on a menu button that just appeared.
        if (triggeringControllerIndex !== -1 && this.xrPointers[triggeringControllerIndex]) {
            const pointer = this.xrPointers[triggeringControllerIndex];
            if (pointer.getButtonsDown().has(0)) {
                const ev = { pointerId: pointer.id, button: 0, buttons: 0, timeStamp: performance.now() };
                pointer.up(ev);
            }
            this.ignorePointerUntilReleased[triggeringControllerIndex] = true;
        }

        const pos = new THREE.Vector3(), quat = new THREE.Quaternion();
        this.camera.updateMatrixWorld(true); this.camera.getWorldPosition(pos); this.camera.getWorldQuaternion(quat);
        this.position.copy(pos); this.quaternion.copy(quat);
        this.visible = this.isMainMenuVisible = true;
        this.mainContainer.visible = true;
        this.mainContainer.setProperties({ display: 'flex' });
        this.smoothedOcclusion = 0; this.currentZ = DEFAULTS.uiZ;

        this.syncUI(true);
        this.mainContainer.update(0);
    }

    hideMainMenu(force = false) { 
        if (this.isMainMenuVisible || force) { 
            this.visible = this.isMainMenuVisible = false;
            if (this.mainContainer) {
                this.mainContainer.visible = false;
                this.mainContainer.setProperties({ display: 'none' });
            }
            this.switchSubMenu(null); 
            this.ignorePointerUntilReleased = [false, false];
            if (this.mainContainer) this.mainContainer.update(0);

            // Clean up VR pointer hover/down states
            for (let i = 0; i < 2; i++) {
                const pointer = this.xrPointers[i];
                if (pointer) {
                    const ev = { pointerId: pointer.id, button: 0, buttons: 0, timeStamp: performance.now() };
                    pointer.exit(ev);
                }
            }
        } 
    }
    toggleMainMenuVisibility() { if (this.isMainMenuVisible) this.hideMainMenu(); else this.showMainMenu(); }
    isAnyMenuVisible() { return this.isMainMenuVisible || this.activeSubMenu !== null; }
    getInteractableMeshes() {
        if (!this.visible) return [];
        const m = [this.mainMenu.container];
        if (this.activeSubMenu) m.push(this.activeSubMenu.container);
        return m;
    }

    // --- Data Handlers ---

    async fetchEnvironments() {
        try {
            const response = await fetch('/api/environments');
            const list = await response.json();
            
            this.skyboxes = ["None", ...list.filter(f => !f.toLowerCase().endsWith('.glb'))];
            this.models = ["None", ...list.filter(f => f.toLowerCase().endsWith('.glb'))];

            const skyboxName = this.settingsManager.get('environment_name');
            if (skyboxName) {
                const idx = this.skyboxes.indexOf(skyboxName);
                if (idx !== -1) {
                    this.currentSkyboxIndex = idx;
                    if (skyboxName !== "None") this.environmentManager.loadSkybox(skyboxName);
                }
            }

            const modelName = this.settingsManager.get('environment_model_name');
            if (modelName) {
                const idx = this.models.indexOf(modelName);
                if (idx !== -1) {
                    this.currentModelIndex = idx;
                    if (modelName !== "None") this.environmentManager.loadModel(modelName);
                }
            }
        } catch (e) { this.debugLogInstance.log(`Env Error: ${e.message}`); }
    }

    async fetchLuts() {
        try {
            const response = await fetch('/api/luts');
            const list = await response.json();
            this.luts = ["None", ...list];

            const lutName = this.settingsManager.get('color_lut');
            if (lutName) {
                const idx = this.luts.indexOf(lutName);
                if (idx !== -1) {
                    this.currentLutIndex = idx;
                    if (lutName !== "None") this.applyLut(lutName);
                }
            }
        } catch (e) { this.debugLogInstance.log(`LUT Error: ${e.message}`); }
    }

    async applyLut(filename) {
        if (this._currentLutName === filename) return;
        this._currentLutName = filename;

        if (filename === "None") {
            this.stereoPlayer?.setLutTexture(null);
            return;
        }

        const loader = new LUTCubeLoader();
        try {
            const result = await loader.loadAsync(`/lut/${filename}`);
            const tex3d = result.texture3D;
            
            tex3d.minFilter = THREE.LinearFilter;
            tex3d.magFilter = THREE.LinearFilter;
            tex3d.needsUpdate = true;
            this.stereoPlayer?.setLutTexture(tex3d);
        } catch (e) {
            this.debugLogInstance.log(`LUT Load Error: ${e.message}`);
        }
    }

    _cycleIndex(currentIndex, len, delta) {
        return ((Math.max(0, currentIndex) + delta) % len + len) % len;
    }

    changeLut(delta) {
        this.currentLutIndex = this._cycleIndex(this.currentLutIndex, this.luts.length, delta);
        this.onSliderChange('color_lut', this.luts[this.currentLutIndex]);
    }

    changeSkybox(delta) {
        this.currentSkyboxIndex = this._cycleIndex(this.currentSkyboxIndex, this.skyboxes.length, delta);
        this.onSliderChange('environment_name', this.skyboxes[this.currentSkyboxIndex]);
    }

    changeModel(delta) {
        this.currentModelIndex = this._cycleIndex(this.currentModelIndex, this.models.length, delta);
        this.onSliderChange('environment_model_name', this.models[this.currentModelIndex]);
    }

    changeFont(delta) {
        const currentId = this.settingsManager.get('render_font') || 'Auto';
        const nextIndex = this._cycleIndex(this.fonts.findIndex(f => f.id === currentId), this.fonts.length, delta);
        this.onSliderChange('render_font', this.fonts[nextIndex].id);
    }

    changeFPS(delta) {
        const currentFPS = this.settingsManager.get('render_fps') || 0;
        const nextIndex = this._cycleIndex(this.availableFrameRates.indexOf(currentFPS), this.availableFrameRates.length, delta);
        this.onSliderChange('render_fps', this.availableFrameRates[nextIndex]);
    }

    async fetchDirectory(path) {
        try {
            // Clear current view before fetching to avoid showing stale data
            this.explorer.render([], path, 0, this.itemsPerPage);

            const items = await this.galleryManager.fetchDirectory(path);
            this.currentPage = 0;

            let targetPath = null;
            if (this.shouldJumpToCurrentItem) {
                const currentItem = this.currentItem;
                if (currentItem) targetPath = currentItem.path;
                this.shouldJumpToCurrentItem = false;
            } else if (this.shouldJumpToPath) {
                targetPath = this.shouldJumpToPath;
                this.shouldJumpToPath = null;
            }

            if (targetPath) {
                const itemIdx = items.findIndex(item => item.path === targetPath);
                if (itemIdx !== -1) this.currentPage = Math.floor(itemIdx / this.itemsPerPage);
            }

            this.explorer.render(items, path, this.currentPage, this.itemsPerPage, targetPath);
        } catch (e) {
            this.showNotification("Error: Failed to fetch file list");
            this.debugLogInstance.log(`[ERROR] Fetch Directory Failed: ${e.message}`);
            // Close explorer if it fails to load
            this.switchSubMenu(null);
        }
    }

    async setSortMode(mode) {
        await this.galleryManager.setSortMode(mode);
        this.currentPage = 0;
        if (this.activeSubMenu === this.explorer) {
            this.explorer.render(this.galleryManager.items, this.galleryManager.currentPath, this.currentPage, this.itemsPerPage);
        }
    }

    changePage(delta) {
        const max = Math.ceil(this.galleryManager.items.length / this.itemsPerPage) - 1;
        const next = THREE.MathUtils.clamp(this.currentPage + delta, 0, Math.max(0, max));
        if (next !== this.currentPage) {
            this.currentPage = next;
            if (this.activeSubMenu === this.explorer) {
                this.explorer.render(this.galleryManager.items, this.galleryManager.currentPath, this.currentPage, this.itemsPerPage);
            }
        }
    }

    async openImageFromExplorer(allItems, targetItem) {
        const idx = this.galleryManager.findPlaybackIndexByPath(targetItem.path);
        if (idx !== -1) {
            await this.loadMedia(idx);
            this.hideMainMenu(true);
            if (this.stereoPlayer) this.stereoPlayer.toggleControllerUIVisibility(false);
        }
    }

    navigateUp() {
        const currentPath = this.galleryManager.currentPath;
        const parent = this.galleryManager.getParentPath();
        if (parent) {
            // Set target path to jump to (strip leading slash to match item.path)
            this.shouldJumpToPath = currentPath.startsWith('/') ? currentPath.substring(1) : currentPath;
            this.fetchDirectory(parent);
        }
    }

    toggleVideoRepeat() {
        this.settingsManager.set('video_repeat', !this.videoRepeat);
    }

    toggleMute() {
        this.setMuted(!this.isMuted);
    }

    setMuted(muted) {
        if (!muted && this.currentVolume === 0) {
            const restoreVol = this._unmuteVolume || DEFAULTS.screen_volume;
            this.settingsManager.set('screen_volume', restoreVol);
        } else if (muted && this.currentVolume > 0) {
            this._unmuteVolume = this.currentVolume;
        }
        this.settingsManager.set('video_muted', muted);
        this.syncUI(true);
    }

    setPlaybackRate(speed, shouldSave = true) {
        this.settingsManager.set('video_speed', speed, { isUserAction: shouldSave });
        this.syncUI(true);
    }

    resetPlaybackRate() {
        this.setPlaybackRate(DEFAULTS.video_speed, true);
    }

    changePlaybackRate(delta) {
        const min = 0.25;
        const max = 4.0;
        const current = this.playbackRate ?? DEFAULTS.video_speed;
        const newSpeed = Math.round(THREE.MathUtils.clamp(current + delta, min, max) * 100) / 100;
        this.setPlaybackRate(newSpeed, true);
    }

    updateSubtitleText(text) { 
        if (this.stereoPlayer?.subtitleWindow) {
            const fontSize = this.settingsManager.get('subtitle_font_size');
            this.stereoPlayer.subtitleWindow.updateText(text, fontSize); 
        }
    }
    toggleSubtitles() { 
        const isVisible = this.settingsManager.get('subtitle_visible');
        this.onSliderChange('subtitle_visible', !isVisible); 
    }
    changeSubtitle(delta) {
        if (this.availableSubtitles.length === 0) return;
        this.currentSubtitleIndex = this._cycleIndex(this.currentSubtitleIndex, this.availableSubtitles.length, delta);
        this.stereoPlayer?.setSubtitleTrack(this.currentSubtitleIndex);

        const currentItem = this.currentItem;
        if (currentItem) {
            this.updateFileConfig(currentItem, { subtitle_index: this.currentSubtitleIndex });
        }
    }

    // --- Event Handlers ---

    async handleSlider(e, track, id, min, max, shouldSave = true) {
        if (!e.point) return;
        const local = track.worldToLocal(e.point.clone());
        const percent = THREE.MathUtils.clamp(0.5 + local.x, 0, 1);
        
        const rawPercent = 0.5 + local.x; // Keep for limit detection
        const atMin = rawPercent <= 0.01;
        const atMax = rawPercent >= 0.99;
        if (atMin || atMax) {
            const limitKey = `${id}_${atMin ? 'min' : 'max'}`;
            if (!this._atLimit[limitKey]) {
                UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticLimitIntensity, UI_CONFIG.hapticLimitDuration, this);
                this._atLimit[limitKey] = true;
            }
        } else {
            this._atLimit[`${id}_min`] = false;
            this._atLimit[`${id}_max`] = false;
        }

        const limits = getSettingLimits(id, this.screenType, this.activeScreen);
        const actualMin = limits.min ?? min;
        const actualMax = limits.max ?? max;
        
        await this.onSliderChange(id, actualMin + (actualMax - actualMin) * percent, shouldSave);
        this.syncUI(false);
    }

    handleVideoSeek(e, track, shouldSave = true) {
        if (!e.point || !this.videoElement) return;
        const local = track.worldToLocal(e.point.clone());
        const percent = THREE.MathUtils.clamp(0.5 + local.x, 0, 1);
        this.seek(percent * this.videoElement.duration);
        if (shouldSave) this.savePlaybackPosition();
    }

    handleVolumeChange(e, track, shouldSave = true) {
        if (!e.point) return;
        const local = track.worldToLocal(e.point.clone());
        let percent = THREE.MathUtils.clamp(0.5 + local.x, 0, 1);
        if (percent <= 0.05) {
            percent = 0;
        } else if (percent >= 0.95) {
            percent = 1;
        }
        if (this.isMuted && percent > 0) {
            this.settingsManager.set('video_muted', false, { isUserAction: shouldSave });
        }
        this.onSliderChange('screen_volume', percent, shouldSave);
    }

    async onSliderChange(id, val, shouldSave = true) {
        await this.settingsManager.set(id, val, { isUserAction: shouldSave });
    }

    async onFormatChange(fmt) {
        if (this.galleryManager.playingIndex !== -1) {
            await this.loadMedia(this.galleryManager.playingIndex, fmt);
        }
    }

    // --- Sync & Storage ---

    async saveSettings() { 
        await this.settingsManager.flush();
    }

    async updateFileConfig(file, updates) { 
        await this.settingsManager.updateFileState(updates, file);
    }

    syncUI(updateSlowMetrics = false) {
        if (!this.mainContainer || !this.visible) return;

        const now = performance.now();
        const slowMetricsUpdated = (now - this.lastStatusUpdateTime > 1000) || updateSlowMetrics;
        const isVideo = !!this.videoElement;

        const mainParams = {
            videoProgress: isVideo ? (this.videoElement.currentTime / this.videoElement.duration || 0) : 0,
            isPlaying: isVideo && !this.videoElement.paused,
            volume: this.currentVolume,
            isMuted: this.isMuted,
            speed: this.playbackRate,
            currentTime: isVideo ? this.videoElement.currentTime : 0,
            duration: isVideo ? this.videoElement.duration : 0
        };

        if (slowMetricsUpdated) {
            mainParams.currentFileName = this.currentItem ? this.currentItem.name : "";
            mainParams.isVideoActive = isVideo;
            mainParams.fps = this.currentFPS;
            mainParams.slowMetricsUpdated = true;
        }

        this.mainMenu.sync(mainParams);
        this.activeSubMenu?.sync?.();
    }

    update(delta) {
        if (!this.mainContainer) return;
        const now = performance.now();
        const FPS_WINDOW = 500;

        // Always update FPS buffer even if UI is hidden
        if (delta > 0) { 
            this.fpsHistory[this.fpsIndex] = delta; 
            this.fpsIndex = (this.fpsIndex + 1) % FPS_WINDOW; 
            this.fpsCount = Math.min(this.fpsCount + 1, FPS_WINDOW); 
        }

        let slowMetricsUpdated = false;
        if (now - this.lastStatusUpdateTime > 500) {
            if (this.fpsCount > 0) { 
                let sum = 0; 
                for (let i = 0; i < this.fpsCount; i++) sum += this.fpsHistory[i]; 
                this.currentFPS = 1000 / (sum / this.fpsCount); 
            }
            slowMetricsUpdated = true; 
            this.lastStatusUpdateTime = now;
        }

        if (this.visible) {
            this.syncUI(slowMetricsUpdated);
        }

        const isVideo = !!this.videoElement;
        const hasActiveSubtitle = this.availableSubtitles.length > 0 && this.currentSubtitleIndex >= 0;
        const sub = this.stereoPlayer?.subtitleWindow;
        const s = this.activeScreen;
        if (sub && s.supportsSubtitles) {
            const getSubVal = (k) => this.settingsManager.get(k);
            const settings = { 
                y: getSubVal('subtitle_y'), 
                z: getSubVal('subtitle_z'), 
                scale: getSubVal('subtitle_font_size'),
                disparity: getSubVal('subtitle_eye_sep')
            };
            const screenParams = {
                aspectRatio: s.aspectRatio,
                physicalScreenHeight: s.physicalScreenHeight,
                currentScreenX: s.currentScreenX,
                currentScreenY: s.currentScreenY,
                currentZoom: s.currentZoom
            };
            sub.updateLayout(screenParams, settings, isVideo && hasActiveSubtitle && getSubVal('subtitle_visible'));
        }

        if (this.visible) {
            this.mainContainer.update(delta);
            
            // Heavy raycasting for occlusion detection: throttle to ~20fps (50ms)
            if (now - this.lastOcclusionUpdateTime > 50) {
                this.updateOcclusionState();
                this.lastOcclusionUpdateTime = now;
            }
            // Smooth movement (lerp) needs to run every frame to stay fluid
            this.applyOcclusionSmoothing(delta);
        }

        if (this.isNotificationVisible) {
            this.notificationContainer.update(delta);
        }
        
        if (this.pointerEvents?.update) this.pointerEvents.update();
        if (this.renderer.xr.isPresenting && this.isMainMenuVisible) {
            for (let i = 0; i < 2; i++) {
                const c = this.stereoPlayer.controllers[i]; if (!c) continue;
                if (!this.xrPointers[i]) {
                    this.xrPointers[i] = createRayPointer(
                        () => this.camera,
                        { current: c },
                        {},
                        {
                            intersectionRecursive: true,
                            filter: (object) => ThreeUtils.isHierarchyVisible(object)
                        }
                    );
                }
                const pointer = this.xrPointers[i]; 
                const pressed = this.stereoPlayer.gamepads[i]?.buttons[0].pressed;
                
                const ev = { pointerId: pointer.id, button: 0, buttons: pressed ? 1 : 0, timeStamp: performance.now() };
                pointer.move(this.scene, ev);
                
                if (pressed) {
                    if (!this.ignorePointerUntilReleased[i] && !pointer.getButtonsDown().has(0)) {
                        pointer.down(ev);
                    }
                } else {
                    if (pointer.getButtonsDown().has(0)) {
                        if (!this.ignorePointerUntilReleased[i]) {
                            pointer.up(ev);
                        }
                    }
                    this.ignorePointerUntilReleased[i] = false;
                }
            }
        }
    }

    updateOcclusionState() {
        this.updateMatrixWorld(true);
        
        // --- Optimization: Skip raycasting if screen is too far or not occludable ---
        const s = this.activeScreen;
        if (!s.canOccludeUI) {
            this.targetOcclusion = 0;
            return;
        }

        const screenDist = Math.abs(s.currentZoom);
        
        // If screen is significantly behind the UI (UI is at ~0.9m), no need to check.
        // Using 1.5m as a safe threshold.
        if (screenDist > 1.5) {
            this.targetOcclusion = 0;
            return;
        }

        const cameraPos = new THREE.Vector3(); this.camera.getWorldPosition(cameraPos);
        let isOccluded = false; const targets = this.getInteractableMeshes();
        const rayDir = new THREE.Vector3(), tempPoint = new THREE.Vector3(), referencePoint = new THREE.Vector3();
        const offsets = [[0, 0], [300, 200], [-300, 200], [300, -200], [-300, -200]];
        const buffer = this.smoothedOcclusion > 0.5 ? UI_CONFIG.occlusionHysteresisNear : UI_CONFIG.occlusionHysteresisFar;
        for (const target of targets) {
            target.updateMatrixWorld(true);
            for (const [ox, oy] of offsets) {
                tempPoint.set(ox * this.pixelSize, -oy * this.pixelSize, 0); target.localToWorld(tempPoint); this.worldToLocal(tempPoint);
                referencePoint.set(tempPoint.x, tempPoint.y, DEFAULTS.uiZ); this.localToWorld(referencePoint);
                rayDir.copy(referencePoint).sub(cameraPos).normalize(); this.occlusionRaycaster.set(cameraPos, rayDir);
                const hits = this.occlusionRaycaster.intersectObject(this.activeScreen, true);
                if (hits.length > 0 && hits[0].distance < cameraPos.distanceTo(referencePoint) + buffer) { isOccluded = true; break; }
            }
            if (isOccluded) break;
        }
        this.targetOcclusion = isOccluded ? 1 : 0;
    }

    applyOcclusionSmoothing(delta) {
        const d = Math.min(delta, 100);
        const lerpFactor = 1 - Math.exp(-d / UI_CONFIG.occlusionTimeConstant);
        this.smoothedOcclusion += lerpFactor * ((this.targetOcclusion || 0) - this.smoothedOcclusion);
        this.mainContainer.position.z = DEFAULTS.uiZ + (DEFAULTS.uiForwardLimit * this.smoothedOcclusion);
    }
}

export { UIManager };
