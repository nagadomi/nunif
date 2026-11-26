import * as THREE from 'three';
import { COLORS, UI_CONFIG } from './constants.js';
import { UIUtils } from './ui_common.js';

const THROTTLE_MS = 100;

const STREAM_SETTINGS_DEFS = {
    divergence: {
        label: '3D Strength',
        min: 0.0,
        max: 10.0,
        step: 0.05,
        decimals: 2,
        default: 1.0,
    },
    convergence: {
        label: 'Convergence Plane',
        min: 0.0,
        max: 1.0,
        step: 0.05,
        decimals: 2,
        default: 0.5,
    },
    foreground_scale: {
        label: 'Foreground Scale',
        min: -3.0,
        max: 3.0,
        step: 0.05,
        decimals: 2,
        default: 0.0,
    },
    edge_dilation: {
        label: 'Edge Fix',
        min: 0,
        max: 10,
        step: 1,
        decimals: 0,
        default: 2,
    },
    buffer_ms: {
        label: 'Stream Buffer',
        min: 0,
        max: 200,
        step: 25,
        decimals: 0,
        default: 0,
    },
};

const STREAM_SETTING_KEYS = Object.keys(STREAM_SETTINGS_DEFS);

class StreamSettingsMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = UIUtils.createSubMenuContainer(defaultFont, {
            positionTop: UI_CONFIG.menuPositionTop.stream
        });
        this.headerBtns = UIUtils.setupWindowHeader(this.container, 'Stream Settings', () => this.reset(), this.uiManager);

        this.values = {};
        this._lastSentValues = {};
        const urlOverrideMs = this.uiManager?.stereoPlayer?.playbackController?.urlBufferOverrideMs ?? null;
        for (const key of STREAM_SETTING_KEYS) {
            const defVal = key === 'buffer_ms' && urlOverrideMs !== null ? urlOverrideMs : STREAM_SETTINGS_DEFS[key].default;
            this.values[key] = defVal;
            this._lastSentValues[key] = defVal;
        }

        this.pendingUpdates = {};
        this._lastSendTime = 0;
        this._throttleTimer = null;
        this._isSending = false;
        this._flushRequested = false;
        this._isDragging = false;

        this.sliders = {};
        for (const key of STREAM_SETTING_KEYS) {
            const def = STREAM_SETTINGS_DEFS[key];
            this.sliders[key] = UIUtils.createSlider(
                this.container,
                def.label,
                key,
                def.min,
                def.max,
                this.uiManager,
                () => this.resetSetting(key),
                null,
                {
                    showValue: true,
                    onSlider: (e, track, id, min, max, shouldCommit) =>
                        this.handleSlider(e, track, id, min, max, shouldCommit),
                    onCommit: () => this.commitChanges(),
                }
            );
        }
    }

    _snapValue(rawValue, def) {
        const clamped = THREE.MathUtils.clamp(rawValue, def.min, def.max);
        const stepped = Math.round(clamped / def.step) * def.step;
        return Number(THREE.MathUtils.clamp(stepped, def.min, def.max).toFixed(def.decimals));
    }

    _formatValue(value, def) {
        if (typeof value !== 'number' || Number.isNaN(value)) {
            return def.default.toFixed(def.decimals);
        }
        return value.toFixed(def.decimals);
    }

    _applyServerSettings(data) {
        if (!data || typeof data !== 'object') return;
        const pc = this.uiManager?.stereoPlayer?.playbackController;
        for (const key of STREAM_SETTING_KEYS) {
            if (key === 'buffer_ms' && pc?.hasUrlBufferOverride) {
                this.values.buffer_ms = pc.urlBufferOverrideMs;
                this._lastSentValues.buffer_ms = pc.urlBufferOverrideMs;
                continue;
            }
            if (typeof data[key] === 'number' && !(key in this.pendingUpdates)) {
                this.values[key] = data[key];
                this._lastSentValues[key] = data[key];
                if (key === 'buffer_ms') {
                    pc?.setStreamBufferMs(data[key]);
                }
            }
        }
    }

    _clearThrottleTimer() {
        if (this._throttleTimer !== null) {
            clearTimeout(this._throttleTimer);
            this._throttleTimer = null;
        }
    }

    _scheduleThrottledFlush() {
        const now = performance.now();
        const elapsed = now - this._lastSendTime;
        if (elapsed >= THROTTLE_MS && !this._isSending) {
            this._clearThrottleTimer();
            this.flushSettings();
            return;
        }
        if (this._throttleTimer === null) {
            const delay = Math.max(0, THROTTLE_MS - elapsed);
            this._throttleTimer = setTimeout(() => {
                this._throttleTimer = null;
                this.flushSettings();
            }, delay);
        }
    }

    handleSlider(e, track, id, min, max, shouldCommit) {
        if (!e.point) return;
        const local = track.worldToLocal(e.point.clone());
        const rawPercent = 0.5 + local.x;
        const percent = THREE.MathUtils.clamp(rawPercent, 0, 1);

        UIUtils.checkSliderLimitHaptic(e, rawPercent, `stream_${id}`, this.uiManager);

        const def = STREAM_SETTINGS_DEFS[id];
        const val = this._snapValue(min + (max - min) * percent, def);

        this._isDragging = !shouldCommit;
        if (this.values[id] !== val) {
            this.values[id] = val;
            this.pendingUpdates[id] = val;
            if (id === 'buffer_ms') {
                const pc = this.uiManager?.stereoPlayer?.playbackController;
                pc?.clearUrlBufferOverride();
                pc?.setStreamBufferMs(val);
            }
            this.uiManager.syncUI(false);
        } else if (shouldCommit && this._lastSentValues[id] !== val) {
            this.pendingUpdates[id] = val;
        }

        if (shouldCommit) {
            this._clearThrottleTimer();
            this.flushSettings();
        } else {
            this._scheduleThrottledFlush();
        }
    }

    async commitChanges() {
        this._isDragging = false;
        this._clearThrottleTimer();
        await this.flushSettings();
    }

    async flushSettings() {
        if (Object.keys(this.pendingUpdates).length === 0) return;
        if (this._isSending) {
            this._flushRequested = true;
            return;
        }

        this._isSending = true;
        try {
            while (Object.keys(this.pendingUpdates).length > 0) {
                this._flushRequested = false;
                const payload = { ...this.pendingUpdates };
                this.pendingUpdates = {};
                this._lastSendTime = performance.now();

                try {
                    const res = await fetch('/api/stream/settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        cache: 'no-store',
                        credentials: 'same-origin',
                        body: JSON.stringify(payload),
                    });
                    if (res.status === 401 && !this.uiManager.renderer?.xr?.isPresenting) {
                        window.location.reload();
                        return;
                    }
                    if (!res.ok) {
                        this.uiManager.debugLogInstance?.log(`[WARN] Stream settings update failed: HTTP ${res.status}`);
                        break;
                    }
                    const data = await res.json();
                    for (const [k, v] of Object.entries(payload)) {
                        this._lastSentValues[k] = v;
                    }
                    if (!this._isDragging && Object.keys(this.pendingUpdates).length === 0) {
                        this._applyServerSettings(data);
                        this.uiManager.syncUI(true);
                    }
                } catch (err) {
                    this.uiManager.debugLogInstance?.log(`[WARN] Stream settings update error: ${err.message}`);
                    break;
                }

                if (!this._flushRequested && Object.keys(this.pendingUpdates).length === 0) {
                    break;
                }
            }
        } finally {
            this._isSending = false;
        }
    }

    async fetchSettings() {
        if (!this.uiManager?.isStreamActive) return;
        if (this._isDragging || this._isSending || Object.keys(this.pendingUpdates).length > 0) {
            return;
        }
        try {
            const res = await fetch('/api/stream/settings', {
                cache: 'no-store',
                credentials: 'same-origin',
            });
            if (res.status === 401 && !this.uiManager.renderer?.xr?.isPresenting) {
                window.location.reload();
                return;
            }
            if (!res.ok) return;
            const data = await res.json();
            if (!this._isDragging && !this._isSending && Object.keys(this.pendingUpdates).length === 0) {
                this._applyServerSettings(data);
                this.uiManager.syncUI(true);
            }
        } catch (err) {
            this.uiManager.debugLogInstance?.log(`[WARN] Failed to fetch stream settings: ${err.message}`);
        }
    }

    async resetKeys(keys) {
        this._isDragging = false;
        this._clearThrottleTimer();
        for (const key of keys) {
            const def = STREAM_SETTINGS_DEFS[key];
            if (!def) continue;
            this.values[key] = def.default;
            this.pendingUpdates[key] = def.default;
            if (key === 'buffer_ms') {
                const pc = this.uiManager?.stereoPlayer?.playbackController;
                pc?.clearUrlBufferOverride();
                pc?.setStreamBufferMs(def.default);
            }
        }
        this.uiManager.syncUI(true);
        await this.flushSettings();
    }

    async resetSetting(key) {
        if (!STREAM_SETTINGS_DEFS[key]) return;
        await this.resetKeys([key]);
    }

    async reset() {
        await this.resetKeys(STREAM_SETTING_KEYS);
    }

    sync() {
        const trackWidth = 332;
        const thumbWidth = 24;
        for (const key of STREAM_SETTING_KEYS) {
            const sl = this.sliders[key];
            const def = STREAM_SETTINGS_DEFS[key];
            const val = this.values[key];
            const p = THREE.MathUtils.clamp((val - def.min) / (def.max - def.min), 0, 1);
            const margin = p * (trackWidth - thumbWidth);
            if (Math.abs((sl._lastMargin ?? -1) - margin) > 0.1) {
                sl.thumb.setProperties({ marginLeft: margin, backgroundColor: COLORS.accent });
                sl._lastMargin = margin;
            }
            if (sl.valueText) {
                let text;
                if (key === 'buffer_ms') {
                    const status = this.uiManager?.stereoPlayer?.playbackController?.getStreamBufferStatus?.();
                    if (status && !status.supported) {
                        text = 'N/A (MJPEG)';
                    } else if (val === 0) {
                        text = '0 ms';
                    } else if (status && status.isClamped) {
                        text = `${Math.round(val)} ms (eff ${status.effectiveMs})`;
                    } else {
                        text = `${Math.round(val)} ms`;
                    }
                } else {
                    text = this._formatValue(val, def);
                }
                if (sl._lastValueText !== text) {
                    sl.valueText.setProperties({ text });
                    sl._lastValueText = text;
                }
            }
        }
        this._lastAlignIcon = UIUtils.updateAlignIcon(this.headerBtns?.alignBtn, this.uiManager, this._lastAlignIcon);
    }
}

export { StreamSettingsMenu };
