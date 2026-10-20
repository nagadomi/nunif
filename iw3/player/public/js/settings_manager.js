import * as THREE from 'three';
import { SETTINGS_METADATA, getSettingDefault, getSettingLimits } from './constants.js';
import { storage } from './storage.js';

/**
 * SettingsManager manages active settings and per-file overrides.
 *
 * State Model:
 * - `this.groups`: Active settings for each group ('screen', 'screen_dome', 'subtitle',
 *   'color', 'environment', 'render', 'general'), persisted to IndexedDB so the last
 *   adjusted state is preserved across both media switches and browser reloads.
 * - `this.currentFileConfig`: Per-file overrides for the currently loaded media file.
 *   When a file is loaded, its overrides are merged into `this.groups` so untouched
 *   settings naturally inherit their previous active values.
 * - `this.dirtyGroups` / `this.dirtyFileKeys`: Tracks uncommitted changes during
 *   continuous slider or controller drags and flushes them to IndexedDB on release.
 */
export class SettingsManager {
    constructor(uiManager) {
        this.uiManager = uiManager;
        this.currentFile = null;
        this.currentFileConfig = {};

        this.groups = {
            screen: {},
            screen_dome: {},
            subtitle: {},
            color: {},
            environment: {},
            render: {},
            general: {}
        };

        this.dirtyGroups = new Set();
        this.dirtyFileKeys = new Set();
    }

    /**
     * Initializes active group settings from storage cache.
     */
    async init() {
        for (const group of Object.keys(this.groups)) {
            const saved = storage.get(group, {});
            this.groups[group] = { ...saved };
        }
    }

    /**
     * Checks if the given screen type is Dome or Dome XR.
     */
    isDome(screenType) {
        return screenType === 'dome' || screenType === 'dome_xr';
    }

    /**
     * Resolves the storage key for a settings group based on screen type.
     */
    getGroupKey(groupName, screenType = this.uiManager?.screenType) {
        if (groupName === 'screen') {
            return this.isDome(screenType) ? 'screen_dome' : 'screen';
        }
        return groupName;
    }

    /**
     * Resolves the effective value for a given key.
     */
    get(key, screenType = this.uiManager?.screenType) {
        const meta = SETTINGS_METADATA[key];
        const groupKey = this.getGroupKey(meta?.group || 'general', screenType);
        const val = this.groups[groupKey]?.[key];
        if (val !== undefined) {
            return val;
        }
        return getSettingDefault(key, screenType);
    }

    /**
     * Sets a setting value, applies it to the scene, and optionally commits it to storage.
     */
    async set(key, val, options = {}) {
        const commit = options.commit ?? options.isUserAction ?? true;
        const screenType = options.screenType || this.uiManager?.screenType;
        const meta = SETTINGS_METADATA[key];
        const group = meta?.group || 'general';
        const groupKey = this.getGroupKey(group, screenType);

        let finalVal = val;
        if (typeof finalVal === 'number' && meta && (meta.min !== undefined || meta.max !== undefined)) {
            const { min, max } = getSettingLimits(key, screenType, this.uiManager?.activeScreen);
            finalVal = THREE.MathUtils.clamp(finalVal, min, max);
            if (key === 'screen_height_index') {
                finalVal = Math.round(finalVal);
            }
        }

        if (!this.groups[groupKey]) {
            this.groups[groupKey] = {};
        }
        this.groups[groupKey][key] = finalVal;
        this.dirtyGroups.add(groupKey);

        if (meta?.isFileSpecific && !meta?.isInternal && this.currentFile) {
            for (const [k, m] of Object.entries(SETTINGS_METADATA)) {
                if (m.isFileSpecific && !m.isInternal) {
                    const valToSave = (k === key) ? finalVal : this.get(k, screenType);
                    this.currentFileConfig[k] = valToSave;
                    this.dirtyFileKeys.add(k);
                }
            }
        }

        this.applySettingToTarget(key, finalVal, screenType);

        if (commit) {
            await this.flush();
        }
    }

    /**
     * Persists all dirty groups and dirty file-specific keys to IndexedDB.
     */
    async flush() {
        if (this.dirtyGroups.size > 0) {
            const groupsToSave = Array.from(this.dirtyGroups);
            this.dirtyGroups.clear();
            for (const groupKey of groupsToSave) {
                await storage.set(groupKey, this.groups[groupKey]);
            }
        }

        if (this.currentFile && this.dirtyFileKeys.size > 0) {
            const updates = {};
            for (const key of this.dirtyFileKeys) {
                updates[key] = this.currentFileConfig[key];
            }
            this.dirtyFileKeys.clear();
            await storage.updateFileConfig(this.currentFile, updates);
            if (this.uiManager?.debugLogInstance) {
                this.uiManager.debugLogInstance.log(`[Settings] File config saved (${Object.keys(updates).length} keys)`);
            }
        } else {
            this.dirtyFileKeys.clear();
        }
    }

    /**
     * Applies a single setting directly to the active screen / subsystems.
     */
    applySettingToTarget(key, val, _screenType = this.uiManager?.screenType) {
        const u = this.uiManager;
        if (!u) return;
        const p = u.stereoPlayer;
        const s = u.activeScreen;

        switch (key) {
            case 'screen_tilt':
                p?.setRotationX(val * Math.PI / 180);
                break;
            case 'screen_zoom':
                s?.setZoom?.(val);
                break;
            case 'screen_eye_sep':
                s?.setEyeSeparation?.(val);
                break;
            case 'screen_curvature':
                s?.setCurvature?.(val);
                break;
            case 'screen_edge_fade':
                s?.setEdgeFade?.(val);
                break;
            case 'screen_tx':
                s?.setTranslation?.(val, s.currentScreenY);
                break;
            case 'screen_ty':
                s?.setTranslation?.(s.currentScreenX, val);
                break;
            case 'screen_roll':
                s?.setRoll?.(val);
                break;
            case 'screen_bg_color':
                p?.setBackgroundLevel(val);
                break;
            case 'screen_height_index':
                s?.setPhysicalScreenHeight?.(Math.pow(2, Math.round(val)));
                break;
            case 'screen_volume':
                p?.playbackController?.setVolume(val);
                break;
            case 'video_speed':
                if (u.videoElement) u.videoElement.defaultPlaybackRate = val;
                p?.playbackController?.setPlaybackRate(val);
                break;
            case 'video_repeat':
                p?.playbackController?.setLoop(val);
                break;
            case 'video_muted':
                p?.playbackController?.setMuted(val);
                break;
            case 'menu_alignment':
                u.updateMenuPositions();
                break;
            case 'subtitle_visible':
                p?.updateSubtitlesMode(val);
                break;
            case 'color_contrast':
            case 'color_brightness':
            case 'color_hue':
            case 'color_saturation':
            case 'color_gamma':
                p?.setColorParameter(key, val);
                break;
            case 'color_lut':
                if (u.luts?.length > 1) {
                    const idx = u.luts.indexOf(val);
                    u.currentLutIndex = idx !== -1 ? idx : 0;
                    u.applyLut(u.luts[u.currentLutIndex]);
                } else if (val === 'None') {
                    u.currentLutIndex = 0;
                    u.applyLut('None');
                }
                break;
            case 'environment_rotation':
                u.environmentManager?.setRotation(val);
                break;
            case 'environment_tilt':
                u.environmentManager?.setTilt(val);
                break;
            case 'environment_intensity':
                u.environmentManager?.setIntensity(val);
                break;
            case 'environment_model_scale':
                u.environmentManager?.setModelScale(val);
                break;
            case 'environment_model_y':
                u.environmentManager?.setModelY(val);
                break;
            case 'environment_name':
                if (u.skyboxes?.length > 1) {
                    const idx = u.skyboxes.indexOf(val);
                    u.currentSkyboxIndex = idx !== -1 ? idx : 0;
                    u.environmentManager?.loadSkybox(u.skyboxes[u.currentSkyboxIndex]);
                } else if (val === 'None') {
                    u.currentSkyboxIndex = 0;
                    u.environmentManager?.loadSkybox('None');
                }
                break;
            case 'environment_model_name':
                if (u.models?.length > 1) {
                    const idx = u.models.indexOf(val);
                    u.currentModelIndex = idx !== -1 ? idx : 0;
                    u.environmentManager?.loadModel(u.models[u.currentModelIndex]);
                } else if (val === 'None') {
                    u.currentModelIndex = 0;
                    u.environmentManager?.loadModel('None');
                }
                break;
            case 'render_fps':
                u.updateTargetFPS?.(val);
                break;
        }
    }

    /**
     * Resets both per-file overrides and active settings for a given group,
     * restoring default metadata values for the active screen type.
     */
    async resetGroup(groupName, screenType = this.uiManager?.screenType) {
        const groupKey = this.getGroupKey(groupName, screenType);

        // Clear per-file overrides for this group
        if (this.currentFile) {
            const updates = {};
            for (const [key, meta] of Object.entries(SETTINGS_METADATA)) {
                if (meta.group === groupName && meta.isFileSpecific) {
                    updates[key] = undefined;
                    delete this.currentFileConfig[key];
                    this.dirtyFileKeys.delete(key);
                }
            }
            if (Object.keys(updates).length > 0) {
                await storage.updateFileConfig(this.currentFile, updates);
            }
        }

        // Clear active settings for this group to revert to metadata defaults
        this.groups[groupKey] = {};
        this.dirtyGroups.delete(groupKey);
        await storage.set(groupKey, {});

        if (this.uiManager?.debugLogInstance) {
            this.uiManager.debugLogInstance.log(`[Settings] Reset group '${groupKey}'`);
        }

        // Re-apply defaults to target and sync UI
        this.applyGroup(groupName, screenType);
        if (this.uiManager) this.uiManager.syncUI(true);
    }

    /**
     * Applies all settings in a group to target using current effective values.
     */
    applyGroup(groupName, screenType = this.uiManager?.screenType) {
        for (const [key, meta] of Object.entries(SETTINGS_METADATA)) {
            if (meta.group === groupName && !meta.isInternal) {
                const effectiveVal = this.get(key, screenType);
                this.applySettingToTarget(key, effectiveVal, screenType);
            }
        }
    }

    /**
     * Applies settings when loading a media file.
     * - Merges any file-specific overrides from `currentFileConfig` into `this.groups` and applies them.
     * - Settings without explicit file overrides remain untouched (inheriting the last active state).
     */
    async applyForLoadedFile(screenType = this.uiManager?.screenType) {
        if (this.currentFileConfig) {
            for (const [key, meta] of Object.entries(SETTINGS_METADATA)) {
                if (meta.isFileSpecific && !meta.isInternal && this.currentFileConfig[key] !== undefined) {
                    const val = this.currentFileConfig[key];
                    const groupKey = this.getGroupKey(meta.group, screenType);
                    if (!this.groups[groupKey]) {
                        this.groups[groupKey] = {};
                    }
                    this.groups[groupKey][key] = val;
                    this.dirtyGroups.add(groupKey);
                    this.applySettingToTarget(key, val, screenType);
                }
            }
            await this.flush();
        }
        if (this.uiManager) this.uiManager.syncUI(true);
    }

    /**
     * Applies all effective settings to the active screen and UI.
     */
    applyAll(screenType = this.uiManager?.screenType) {
        for (const [key, meta] of Object.entries(SETTINGS_METADATA)) {
            if (!meta.isInternal) {
                const effectiveVal = this.get(key, screenType);
                this.applySettingToTarget(key, effectiveVal, screenType);
            }
        }
        if (this.uiManager) this.uiManager.syncUI(true);
    }

    /**
     * Gets a media-specific state (playback_time, stereo_format, subtitle_index).
     */
    getFileState(key, defaultVal = null) {
        if (this.currentFileConfig && this.currentFileConfig[key] !== undefined) {
            return this.currentFileConfig[key];
        }
        return defaultVal;
    }

    /**
     * Updates media-specific state.
     */
    async updateFileState(updates, targetFile = this.currentFile) {
        if (!targetFile) return;
        if (targetFile === this.currentFile) {
            for (const [k, v] of Object.entries(updates)) {
                if (v === undefined || v === null) {
                    delete this.currentFileConfig[k];
                } else {
                    this.currentFileConfig[k] = v;
                }
            }
        }
        await storage.updateFileConfig(targetFile, updates);
    }

    /**
     * Lifecycle: loads persistent configuration for a given media file.
     */
    async loadForFile(file) {
        await this.flush();
        this.currentFile = file;
        this.currentFileConfig = file ? (await storage.getFileConfig(file)) || {} : {};
        if (this.currentFileConfig.subtitle_index !== undefined && this.uiManager) {
            this.uiManager.currentSubtitleIndex = this.currentFileConfig.subtitle_index;
        }
        if (this.uiManager?.debugLogInstance && file) {
            const keys = Object.keys(this.currentFileConfig);
            if (keys.length > 0) {
                this.uiManager.debugLogInstance.log(`[Settings] Loaded file config (${keys.join(', ')})`);
            }
        }
    }
}

