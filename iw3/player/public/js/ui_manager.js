import * as THREE from 'three';
import { LUTCubeLoader } from 'three';
import { Container, Text, reversePainterSortStable } from '@pmndrs/uikit';
import { forwardHtmlEvents, createRayPointer } from '@pmndrs/pointer-events';
import { LIMITS, DEFAULTS, UI_CONFIG, COLORS, FONT_CONFIG, SETTINGS_METADATA } from './constants.js';
import { storage } from './storage.js';
import { PlaybackSpeed } from './playback_speed.js';
import { UIUtils } from './ui_common.js';

// Import split UI components
import { MainMenu } from './menu_main.js';
import { ScreenSettingsMenu } from './menu_screen_settings.js';
import { ColorSettingsMenu } from './menu_color_settings.js';
import { EnvironmentSettingsMenu } from './menu_environment_settings.js';
import { RenderSettingsMenu } from './menu_render_settings.js';
import { ExplorerWindow } from './window_explorer.js';
import { SubtitleSettingsMenu } from './menu_subtitle_settings.js';

class UIManager extends THREE.Group {
    constructor(scene, camera, renderer, debugLogInstance, stereoPlayer) {
        super();
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        this.debugLogInstance = debugLogInstance;
        this.stereoPlayer = stereoPlayer;

        this.pixelSize = UI_CONFIG.pixelSize;
        this.renderer.setTransparentSort(reversePainterSortStable);

        // --- UI Visibility State ---
        this.isMainMenuVisible = false;
        this.isNotificationVisible = false;
        this.activeSubMenu = null;

        // Subtitle State
        this.availableSubtitles = [];
        this.currentSubtitleIndex = -1;
        this.showSubtitles = DEFAULTS.subtitle_visible;
        
        // Helpers
        this.currentVolume = DEFAULTS.screen_volume;
        this.playbackSpeed = new PlaybackSpeed();
        this.subY = DEFAULTS.subtitle_y;
        this.subZ = DEFAULTS.subtitle_z;
        this.subFontSize = DEFAULTS.subtitle_font_size;
        this.subEyeSep = DEFAULTS.subtitle_eye_sep;

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
            recursive: true
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

        this.videoRepeat = DEFAULTS.video_repeat;
        this.menuAlignment = DEFAULTS.menu_alignment;
        this.isApplyingConfig = false; // Flag to block saving during loadMedia
        this.dirtyGroups = new Set(); // Track modified isFileSpecific groups

        this.initAsync();
        this.scene.add(this);
    }

    async initAsync() {
        await storage.ready;
        this.stereoPlayer.galleryManager.loadSettings();
        
        // Load general settings
        const general = storage.get('general', {});
        this.menuAlignment = general.menu_alignment || DEFAULTS.menu_alignment;

        this.setupUI();
        this.updateMenuPositions(); // Apply loaded alignment

        this.isApplyingConfig = true; // Block dirty flags during initial load
        await this.loadSettings();
        this.isApplyingConfig = false;
        this.dirtyGroups.clear();

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

    toggleMenuAlignment() {
        this.menuAlignment = this.menuAlignment === 'right' ? 'left' : 'right';
        this.updateMenuPositions();
        this.saveSettings();
        // Sync UI to update the button icon
        this.syncUI(true);
    }

    updateMenuPositions() {
        const offset = this.menuAlignment === 'right' ? UI_CONFIG.menuMarginLeft : -UI_CONFIG.menuMarginLeft;
        // Apply to all settings menus except explorer and main menu
        const targetMenus = [this.screenSettings, this.colorSettings, this.environmentSettings, this.renderSettings, this.subtitleSettings, this.speedMenu];
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
            await this.stereoPlayer.galleryManager.fetchDirectory(recentPath);
            // Use playbackGallery (which respects the current sort mode) to find the file
            const gallery = this.stereoPlayer.galleryManager.playbackGallery;
            const index = gallery.findIndex(item => item.name === recentFile);
            
            if (index !== -1) {
                await this.stereoPlayer.loadMedia(index);
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
        const renderSettings = storage.get('render', {});
        let savedFont = renderSettings.render_font || 'Auto';

        // Self-healing: Reset to Auto if the saved font is no longer available (e.g. NotoSansRU)
        if (savedFont !== 'Auto' && !this.fonts.find(f => f.id === savedFont)) {
            this.debugLogInstance.log(`[WARN] Font "${savedFont}" not found, resetting to Auto.`);
            savedFont = 'Auto';
            // Schedule save to update storage
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
        this.add(this.mainContainer);

        this.mainMenu = new MainMenu(this, defaultFont);
        this.speedMenu = this.mainMenu.speedMenu;
        this.screenSettings = new ScreenSettingsMenu(this, defaultFont);
        this.colorSettings = new ColorSettingsMenu(this, defaultFont);
        this.environmentSettings = new EnvironmentSettingsMenu(this, defaultFont);
        this.renderSettings = new RenderSettingsMenu(this, defaultFont);
        this.subtitleSettings = new SubtitleSettingsMenu(this, defaultFont);
        this.explorer = new ExplorerWindow(this, defaultFont);

        this.allSettingsMenus = [this.screenSettings, this.colorSettings, this.environmentSettings, this.renderSettings, this.subtitleSettings];

        this.mainContainer.add(this.mainMenu.container);
        this.mainContainer.add(this.speedMenu.container);
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
        this.activeSubMenu?.onClose?.();
        if (this.activeSubMenu === targetMenu) {
            if (this.activeSubMenu) { this.activeSubMenu.container.setProperties({ display: 'none' }); this.activeSubMenu = null; }
            return;
        }
        if (this.activeSubMenu) this.activeSubMenu.container.setProperties({ display: 'none' });
        this.activeSubMenu = targetMenu;
        if (this.activeSubMenu) {
            this.activeSubMenu.container.setProperties({ display: 'flex' });
            if (this.activeSubMenu === this.explorer) {
                this.shouldJumpToCurrentItem = true;
                this.fetchDirectory(this.stereoPlayer.galleryManager.currentPath);
            } else { this.syncUI(true); }
        }
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
        this.smoothedOcclusion = 0; this.currentZ = DEFAULTS.uiZ;

        this.syncUI(true);
    }

    hideMainMenu(force = false) { 
        if (this.isMainMenuVisible || force) { 
            this.visible = this.isMainMenuVisible = false; 
            this.switchSubMenu(null); 
            this.ignorePointerUntilReleased = [false, false];
        } 
    }
    toggleMainMenuVisibility() { if (this.isMainMenuVisible) this.hideMainMenu(); else this.showMainMenu(); }
    toggleScreenSettingsVisibility() { this.switchSubMenu(this.screenSettings); }
    toggleColorSettingsVisibility() { this.switchSubMenu(this.colorSettings); }
    toggleEnvironmentSettingsVisibility() { this.switchSubMenu(this.environmentSettings); }
    toggleRenderSettingsVisibility() { this.switchSubMenu(this.renderSettings); }
    toggleSubtitleSettingsVisibility() { this.switchSubMenu(this.subtitleSettings); }
    toggleExplorerVisibility() { this.switchSubMenu(this.explorer); }
    isAnyMenuVisible() { return this.isMainMenuVisible; }
    getInteractableMeshes() {
        if (!this.visible) return [];
        const m = [this.mainMenu.container];
        if (this.activeSubMenu) m.push(this.activeSubMenu.container);
        return m;
    }

    // --- Data Handlers ---

    /**
     * Applies a file-specific configuration object to the UI and player.
     * @param {Object} config - The configuration object from storage.
     */
    async applyFileConfig(config) {
        this.isApplyingConfig = true;
        this.dirtyGroups.clear();

        if (config) {
            this.debugLogInstance.log(`[Config] Applied saved config (keys: ${Object.keys(config).join(', ')})`);

            for (const [key, meta] of Object.entries(SETTINGS_METADATA)) {
                if (config[key] !== undefined && meta.isFileSpecific && !meta.isInternal) {
                    const uiKey = meta.uiKey || key;
                    const uiVal = meta.toUI ? meta.toUI(config[key]) : config[key];
                    if (key === 'screen_height') {
                        this.debugLogInstance.log(`[Config] Load Screen Size: ${config[key]}m (idx: ${uiVal})`);
                    }
                    await this.onSliderChange(uiKey, uiVal, false);
                }
            }

            if (config.subtitle_index !== undefined) {
                this.currentSubtitleIndex = config.subtitle_index;
                // Note: StereoPlayer.loadMedia handles the actual track switching
            }
        }
        
        this.isApplyingConfig = false;
        this.dirtyGroups.clear();
        this.syncUI();
    }

    /**
     * Shows a notification at the bottom-center of the screen with the current size index.
     */
    showScreenSizeNotification() {
        const screen = this.stereoPlayer.stereoScreen;
        const sizeIdx = Math.round(this.screenSettings.values.screen_height_index);
        
        // Calculate world position for bottom-center of the actual mesh
        // Using a vertical offset of -0.2 (20% of mesh height) to stay clear of the image
        const mesh = screen.displayScreenLeft;
        const bottomCenter = new THREE.Vector3(0, -0.7, 0); 
        mesh.updateMatrixWorld(true);
        mesh.localToWorld(bottomCenter);

        this.showNotification(`Size: ${sizeIdx}`, 1000, bottomCenter);
    }

    /**
     * Gets all current values for isFileSpecific settings in a group.
     * @param {string} groupName 
     * @returns {Object}
     */
    getGroupConfig(groupName) {
        const config = {};
        for (const [key, meta] of Object.entries(SETTINGS_METADATA)) {
            if (meta.group === groupName && meta.isFileSpecific) {
                if (meta.isInternal) {
                    if (key === 'subtitle_index') config[key] = this.currentSubtitleIndex;
                } else {
                    const uiKey = meta.uiKey || key;
                    for (const menu of this.allSettingsMenus) {
                        if (menu.handlesKey(uiKey)) {
                            const val = menu.values[uiKey];
                            config[key] = meta.fromUI ? meta.fromUI(val) : val;
                            break;
                        }
                    }
                }
            }
        }
        return config;
    }

    /**
     * Generates a file-specific configuration object from the current state.
     * @returns {Object} A configuration object suitable for updateFileConfig.
     */
    getCurrentFileConfig() {
        const config = {};
        for (const [key, meta] of Object.entries(SETTINGS_METADATA)) {
            if (meta.isFileSpecific) {
                if (meta.isInternal) {
                    if (key === 'subtitle_index') config[key] = this.currentSubtitleIndex;
                    // Note: playback_time and stereo_format are handled by StereoPlayer/GalleryManager
                } else {
                    const uiKey = meta.uiKey || key;
                    for (const menu of this.allSettingsMenus) {
                        if (menu.handlesKey(uiKey)) {
                            const val = menu.values[uiKey];
                            config[key] = meta.fromUI ? meta.fromUI(val) : val;
                            break;
                        }
                    }
                }
            }
        }
        return config;
    }

    async fetchEnvironments() {
        try {
            const response = await fetch('/api/environments');
            const list = await response.json();
            
            this.skyboxes = ["None", ...list.filter(f => !f.toLowerCase().endsWith('.glb'))];
            this.models = ["None", ...list.filter(f => f.toLowerCase().endsWith('.glb'))];

            const skyboxName = this.environmentSettings.values.environment_name;
            if (skyboxName) {
                const idx = this.skyboxes.indexOf(skyboxName);
                if (idx !== -1) {
                    this.currentSkyboxIndex = idx;
                    if (skyboxName !== "None") this.stereoPlayer.environmentManager.loadSkybox(skyboxName);
                }
            }

            const modelName = this.environmentSettings.values.environment_model_name;
            if (modelName) {
                const idx = this.models.indexOf(modelName);
                if (idx !== -1) {
                    this.currentModelIndex = idx;
                    if (modelName !== "None") this.stereoPlayer.environmentManager.loadModel(modelName);
                }
            }
        } catch (e) { this.debugLogInstance.log(`Env Error: ${e.message}`); }
    }

    async fetchLuts() {
        try {
            const response = await fetch('/api/luts');
            const list = await response.json();
            this.luts = ["None", ...list];

            const lutName = this.colorSettings.values.color_lut;
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
        if (filename === "None") {
            this.stereoPlayer.stereoScreen.setLutTexture(null);
            return;
        }

        const loader = new LUTCubeLoader();
        try {
            const result = await loader.loadAsync(`/lut/${filename}`);
            const tex3d = result.texture3D;
            
            tex3d.minFilter = THREE.LinearFilter;
            tex3d.magFilter = THREE.LinearFilter;
            tex3d.needsUpdate = true;
            this.stereoPlayer.stereoScreen.setLutTexture(tex3d);
        } catch (e) {
            this.debugLogInstance.log(`LUT Load Error: ${e.message}`);
        }
    }

    changeLut(delta) {
        const len = this.luts.length;
        this.currentLutIndex = (this.currentLutIndex + delta + len) % len;
        const lut = this.luts[this.currentLutIndex];
        this.applyLut(lut);
        this.onSliderChange('color_lut', lut);
    }

    resetLut() {
        this.currentLutIndex = 0;
        const lut = this.luts[0]; // "None"
        this.applyLut(lut);
        this.onSliderChange('color_lut', lut);
    }

    changeSkybox(delta) {
        const len = this.skyboxes.length;
        this.currentSkyboxIndex = (this.currentSkyboxIndex + delta + len) % len;
        const skybox = this.skyboxes[this.currentSkyboxIndex];
        this.stereoPlayer.environmentManager.loadSkybox(skybox);
        this.onSliderChange('environment_name', skybox);
    }

    changeModel(delta) {
        const len = this.models.length;
        this.currentModelIndex = (this.currentModelIndex + delta + len) % len;
        const model = this.models[this.currentModelIndex];
        this.stereoPlayer.environmentManager.loadModel(model);
        this.onSliderChange('environment_model_name', model);
    }

    resetSkybox() {
        this.currentSkyboxIndex = 0;
        const skybox = this.skyboxes[0]; // "None"
        this.stereoPlayer.environmentManager.loadSkybox(skybox);
        this.onSliderChange('environment_name', skybox);
    }

    resetModel() {
        this.currentModelIndex = 0;
        const model = this.models[0]; // "None"
        this.stereoPlayer.environmentManager.loadModel(model);
        this.onSliderChange('environment_model_name', model);
    }

    changeFont(delta) {
        const renderVals = this.renderSettings.values;
        const currentId = renderVals.render_font || 'Auto';
        const currentIndex = this.fonts.findIndex(f => f.id === currentId);
        const len = this.fonts.length;
        const nextIndex = (currentIndex + delta + len) % len;
        const nextFont = this.fonts[nextIndex];
        this.onSliderChange('render_font', nextFont.id);
    }

    changeFPS(delta) {
        const currentFPS = this.renderSettings.values.render_fps || 0;
        const len = this.availableFrameRates.length;
        let currentIndex = this.availableFrameRates.indexOf(currentFPS);
        if (currentIndex === -1) currentIndex = 0;
        
        const nextIndex = (currentIndex + delta + len) % len;
        const nextFPS = this.availableFrameRates[nextIndex];
        this.onSliderChange('render_fps', nextFPS);
    }

    async fetchDirectory(path) {
        try {
            // Clear current view before fetching to avoid showing stale data
            this.explorer.render([], path, 0, this.itemsPerPage);

            const items = await this.stereoPlayer.galleryManager.fetchDirectory(path);
            this.currentPage = 0;

            let targetPath = null;
            if (this.shouldJumpToCurrentItem) {
                const currentItem = this.stereoPlayer.galleryManager.getCurrentItem();
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
        await this.stereoPlayer.galleryManager.setSortMode(mode);
        this.currentPage = 0;
        if (this.activeSubMenu === this.explorer) {
            this.explorer.render(this.stereoPlayer.galleryManager.items, this.stereoPlayer.galleryManager.currentPath, this.currentPage, this.itemsPerPage);
        }
    }

    changePage(delta) {
        const max = Math.ceil(this.stereoPlayer.galleryManager.items.length / this.itemsPerPage) - 1;
        const next = THREE.MathUtils.clamp(this.currentPage + delta, 0, Math.max(0, max));
        if (next !== this.currentPage) {
            this.currentPage = next;
            if (this.activeSubMenu === this.explorer) {
                this.explorer.render(this.stereoPlayer.galleryManager.items, this.stereoPlayer.galleryManager.currentPath, this.currentPage, this.itemsPerPage);
            }
        }
    }

    async openImageFromExplorer(allItems, targetItem) {
        const idx = this.stereoPlayer.galleryManager.findPlaybackIndexByPath(targetItem.path);
        if (idx !== -1) {
            await this.stereoPlayer.loadMedia(idx);
            this.hideMainMenu(true);
            this.stereoPlayer.toggleControllerUIVisibility(false);
        }
    }

    navigateUp() {
        const currentPath = this.stereoPlayer.galleryManager.currentPath;
        const parent = this.stereoPlayer.galleryManager.getParentPath();
        if (parent) {
            // Set target path to jump to (strip leading slash to match item.path)
            this.shouldJumpToPath = currentPath.startsWith('/') ? currentPath.substring(1) : currentPath;
            this.fetchDirectory(parent);
        }
    }

    toggleVideoRepeat() {
        this.videoRepeat = !this.videoRepeat;
        if (this.stereoPlayer.videoElement) {
            this.stereoPlayer.videoElement.loop = this.videoRepeat;
        }
        this.saveSettings();
    }

    updateSubtitleText(text) { this.stereoPlayer.subtitleWindow.updateText(text, this.subtitleSettings.values.subtitle_font_size); }
    toggleSubtitles() { this.onSliderChange('subtitle_visible', !this.subtitleSettings.values.subtitle_visible); }
    changeSubtitle(delta) {
        if (this.availableSubtitles.length === 0) return;
        const len = this.availableSubtitles.length;
        this.currentSubtitleIndex = (this.currentSubtitleIndex + delta + len) % len;
        this.stereoPlayer.setSubtitleTrack(this.currentSubtitleIndex);

        // Save immediately to file config
        const currentItem = this.stereoPlayer.galleryManager.getCurrentItem();
        if (currentItem) {
            this.updateFileConfig(currentItem, { subtitle_index: this.currentSubtitleIndex });
        }
    }

    // --- Event Handlers ---

    handleSlider(e, track, id, min, max, shouldSave = true) {
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

        let actualMin = min, actualMax = max;
        if (id === 'screen_tx' || id === 'screen_ty') {
            const s = this.stereoPlayer.stereoScreen;
            const limit = LIMITS.screen_txMax * s.physicalScreenHeight * (id === 'screen_tx' ? s.aspectRatio : 1.0);
            actualMin = -limit; actualMax = limit;
        }
        this.onSliderChange(id, actualMin + (actualMax - actualMin) * percent, shouldSave);
    }

    handleVideoSeek(e, track, shouldSave = true) {
        if (!e.point || !this.stereoPlayer.videoElement) return;
        const local = track.worldToLocal(e.point.clone());
        const percent = THREE.MathUtils.clamp(0.5 + local.x, 0, 1);
        this.stereoPlayer.seek(percent * this.stereoPlayer.videoElement.duration);
        if (shouldSave) this.stereoPlayer.savePlaybackPosition();
    }

    handleVolumeChange(e, track, shouldSave = true) {
        if (!e.point) return;
        const local = track.worldToLocal(e.point.clone());
        const percent = THREE.MathUtils.clamp(0.5 + local.x, 0, 1);
        this.onSliderChange('screen_volume', percent, shouldSave);
    }

    applyPlaybackSpeed() {
        if (!this.playbackSpeed.applyTo(this.stereoPlayer.videoElement)) {
            this.showNotification(`Speed unsupported; kept ${this.playbackSpeed.rate}x`);
        }
    }

    async setPlaybackSpeed(rate, shouldSave = true) {
        this.playbackSpeed.setRate(rate);
        this.applyPlaybackSpeed();
        this.syncUI();
        if (shouldSave) await storage.set('playback_speed', this.playbackSpeed.toJSON());
    }

    async setPlaybackSpeedBounds(min, max) {
        if (!this.playbackSpeed.setBounds(min, max)) return false;
        await this.setPlaybackSpeed(this.playbackSpeed.rate);
        return true;
    }

    resetPlaybackSpeed() {
        const speed = this.playbackSpeed;
        speed.setBounds(Math.min(speed.min, 1), Math.max(speed.max, 1));
        return this.setPlaybackSpeed(1);
    }

    restorePlaybackSpeedDefaults() {
        this.playbackSpeed = new PlaybackSpeed();
        return this.setPlaybackSpeed(1);
    }

    handlePlaybackSpeed(e, track, shouldSave = true) {
        if (!this.visible || !e.point) return;
        const local = track.worldToLocal(e.point.clone());
        const percent = THREE.MathUtils.clamp(0.5 + local.x, 0, 1);
        const speed = this.playbackSpeed;
        const rate = percent === 0 ? speed.min : percent === 1 ? speed.max :
            Math.round((speed.min + (speed.max - speed.min) * percent) * 100) / 100;
        return this.setPlaybackSpeed(rate, shouldSave);
    }

    async onSliderChange(id, val, shouldSave = true) {
        // Capture the applying state synchronously to avoid race conditions after await
        const wasApplyingDuringCall = this.isApplyingConfig;

        for (const menu of this.allSettingsMenus) {
            if (menu.handlesKey(id)) { 
                await menu.onValueChange(id, val, shouldSave); 

                // Mark group as dirty if an isFileSpecific setting was touched by user
                // and we are NOT currently in the middle of a bulk application/load
                if (!wasApplyingDuringCall) {
                    const entry = Object.entries(SETTINGS_METADATA).find(([k, m]) => k === id || m.uiKey === id);
                    if (entry && entry[1].isFileSpecific) {
                        this.dirtyGroups.add(entry[1].group);
                    }
                }

                // Automatically update per-file config if this key is marked as file-specific
                if (shouldSave && !wasApplyingDuringCall) {
                    // Find metadata by either the key itself or its uiKey
                    const entry = Object.entries(SETTINGS_METADATA).find(([k, m]) => k === id || m.uiKey === id);
                    if (entry) {
                        const [fileKey, meta] = entry;
                        if (meta.isFileSpecific) {
                            const currentItem = this.stereoPlayer.galleryManager.getCurrentItem();
                            if (currentItem) {
                                // Save all isFileSpecific settings in this group to 'pin' the state
                                this.debugLogInstance.log(`[Config] User changed ${fileKey}. Pinning group '${meta.group}'.`);
                                const updates = this.getGroupConfig(meta.group);
                                await this.updateFileConfig(currentItem, updates);
                                this.dirtyGroups.delete(meta.group);
                            }
                        }
                    }
                }

                // Handle real-time FPS change during XR
                if (id === 'render_fps' && this.renderer.xr.isPresenting) {
                    const session = this.renderer.xr.getSession();
                    if (session && session.updateTargetFrameRate) {
                        let targetFPS = val;
                        if (val === 0) {
                            // Find lowest supported rate >= 30 from the available list
                            targetFPS = this.availableFrameRates.find(r => r >= 30) || 0;
                        }
                        if (targetFPS > 0) {
                            session.updateTargetFrameRate(targetFPS).catch(e => this.debugLogInstance.log(`[ERROR] FPS Change Error: ${e.message}`));
                        }
                    }
                }
                return; 
            }
        }
    }

    async onFormatChange(fmt) {
        if (this.stereoPlayer.galleryManager.playingIndex !== -1) {
            await this.stereoPlayer.loadMedia(this.stereoPlayer.galleryManager.playingIndex, fmt);
        }
    }

    // --- Sync & Storage ---

    async loadSettings() { 
        this.playbackSpeed = new PlaybackSpeed(storage.get('playback_speed'));
        this.applyPlaybackSpeed();
        await this.screenSettings.load();
        await this.colorSettings.load();
        await this.environmentSettings.load();
        await this.renderSettings.load();
        await this.subtitleSettings.load();

        const general = storage.get('general', {});
        this.videoRepeat = general.video_repeat !== undefined ? general.video_repeat : DEFAULTS.video_repeat;
        this.menuAlignment = general.menu_alignment || DEFAULTS.menu_alignment;
    }
    
    async saveSettings() { 
        await Promise.all([
            storage.set('playback_speed', this.playbackSpeed.toJSON()),
            this.screenSettings.save(),
            this.colorSettings.save(),
            this.environmentSettings.save(),
            this.renderSettings.save(),
            this.subtitleSettings.save(),
            storage.set('general', { 
                video_repeat: this.videoRepeat,
                menu_alignment: this.menuAlignment
            })
        ]);

        // Also save per-file settings if active and they have been modified by the user
        const currentItem = this.stereoPlayer.galleryManager.getCurrentItem();
        if (currentItem && !this.isApplyingConfig && this.dirtyGroups.size > 0) {
            for (const groupName of this.dirtyGroups) {
                this.debugLogInstance.log(`[Config] Periodic save for group '${groupName}'`);
                const updates = this.getGroupConfig(groupName);
                await this.updateFileConfig(currentItem, updates);
            }
            this.dirtyGroups.clear();
        }
    }
    async getFileConfig(file) { return await storage.getFileConfig(file); }
    async updateFileConfig(file, updates) { await storage.updateFileConfig(file, updates); }
    async clearFileConfig(file) { await storage.clearFileConfig(file); }

    syncUI(updateSlowMetrics = false) {
        if (!this.mainContainer || !this.visible) return;

        const now = performance.now();
        const slowMetricsUpdated = (now - this.lastStatusUpdateTime > 1000) || updateSlowMetrics;
        const isVideo = !!this.stereoPlayer.videoElement;
        const s = this.stereoPlayer.stereoScreen;

        // 1. High-frequency sync: Video progress and Volume (needed for smooth slider feedback)
        this.mainMenu.sync({
            videoProgress: isVideo ? (this.stereoPlayer.videoElement.currentTime / this.stereoPlayer.videoElement.duration || 0) : 0,
            isPlaying: isVideo && !this.stereoPlayer.videoElement.paused,
            volume: this.currentVolume,
            currentTime: isVideo ? this.stereoPlayer.videoElement.currentTime : 0,
            duration: isVideo ? this.stereoPlayer.videoElement.duration : 0
        });

        // 2. Low-frequency sync: FPS, Clock, Battery, Filename, etc.
        if (slowMetricsUpdated) {
            const mainParams = {
                currentFileName: (this.stereoPlayer.galleryManager.getCurrentItem()) ? this.stereoPlayer.galleryManager.getCurrentItem().name : "",
                isVideoActive: isVideo,
                fps: this.currentFPS,
                slowMetricsUpdated: true, 
                showSubtitles: this.subtitleSettings.values.subtitle_visible,
                skybox: this.skyboxes[this.currentSkyboxIndex],
                model: this.models[this.currentModelIndex],
                volume: this.currentVolume
            };
            this.mainMenu.sync(mainParams);
        }

        // 3. High-frequency sub-menu sync: Settings menus need real-time feedback for sliders
        if (this.activeSubMenu) {
            const currentFontId = this.renderSettings.values.render_font || 'Auto';
            const currentFont = this.fonts.find(f => f.id === currentFontId) || this.fonts[0];
            
            if (this.activeSubMenu === this.screenSettings) {
                this.screenSettings.sync({ effective_physical_screen_height: s.physicalScreenHeight, aspect_ratio: s.aspectRatio, currentFormat: s.currentStereoFormat });
            } else if (this.activeSubMenu === this.colorSettings) {
                this.colorSettings.sync();
            } else if (this.activeSubMenu === this.environmentSettings) {
                this.environmentSettings.sync();
            } else if (this.activeSubMenu === this.renderSettings) {
                this.renderSettings.sync({ currentFontLabel: currentFont.label, currentFontLanguages: currentFont.languages });
            } else if (this.activeSubMenu === this.subtitleSettings) {
                this.subtitleSettings.sync({ currentSubtitleTitle: this.availableSubtitles[this.currentSubtitleIndex]?.title || "None" });
            }
        }
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

        const isVideo = !!this.stereoPlayer.videoElement;
        const hasActiveSubtitle = this.availableSubtitles.length > 0 && this.currentSubtitleIndex >= 0;
        const sub = this.stereoPlayer?.subtitleWindow;
        if (sub) {
            const screen = this.stereoPlayer.stereoScreen;
            const settings = { 
                y: this.subtitleSettings.values.subtitle_y, 
                z: this.subtitleSettings.values.subtitle_z, 
                scale: this.subtitleSettings.values.subtitle_font_size,
                disparity: this.subtitleSettings.values.subtitle_eye_sep
            };
            const screenParams = {
                aspectRatio: screen.aspectRatio,
                physicalScreenHeight: screen.physicalScreenHeight,
                currentScreenX: screen.currentScreenX,
                currentScreenY: screen.currentScreenY,
                currentZoom: screen.currentZoom
            };
            sub.updateLayout(screenParams, settings, isVideo && hasActiveSubtitle && this.subtitleSettings.values.subtitle_visible);
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
        if (this.renderer.xr.isPresenting) {
            for (let i = 0; i < 2; i++) {
                const c = this.stereoPlayer.controllers[i]; if (!c) continue;
                if (!this.xrPointers[i]) this.xrPointers[i] = createRayPointer(() => this.camera, { current: c }, {}, { intersectionRecursive: true });
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
        
        // --- Optimization: Skip raycasting if screen is too far ---
        const s = this.stereoPlayer.stereoScreen;
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
                const hits = this.occlusionRaycaster.intersectObject(this.stereoPlayer.stereoScreen, true);
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
