import { Container, Image, Text } from '@pmndrs/uikit';
import { COLORS, UI_CONFIG, getSettingLimits, getSettingLabel, getSettingDefault, findSettingMeta } from './constants.js';

/**
 * Common UI building utilities to be used across different menu components.
 */
export const UIUtils = {
    vibratePointer(pointerId, intensity, duration, uiManager) {
        const index = uiManager.xrPointers.findIndex(p => p && p.id === pointerId);
        if (index !== -1) {
            const gamepad = uiManager.stereoPlayer.gamepads[index];
            if (gamepad?.hapticActuators?.[0]) {
                gamepad.hapticActuators[0].pulse(intensity, duration);
            }
        }
    },

    createButton(icon, onClick, uiManager, iconSize = 44, btnSize = 72, _id = null) {
        const b = new Container({
            width: btnSize,
            height: btnSize,
            backgroundColor: COLORS.button,
            borderRadius: 16,
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            hover: { backgroundColor: COLORS.hover },
            active: { backgroundColor: COLORS.accent },
            onPointerEnter: (e) => {
                // Prevent vibration if the UI is not visible
                if (!uiManager.visible) return;
                this.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, uiManager);
            },
            onClick: (e) => {
                if (!uiManager.visible) return;
                this.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, uiManager);
                onClick();
            }
        });
        const img = new Image({ src: icon, width: iconSize, height: iconSize, color: COLORS.text });
        b.add(img);
        
        let _lastSelected = null;
        const setSelected = (isSelected) => {
            if (_lastSelected === isSelected) return;
            _lastSelected = isSelected;

            b.setProperties({
                backgroundColor: isSelected ? COLORS.accent : COLORS.button,
                hover: { backgroundColor: isSelected ? COLORS.accentHover : COLORS.hover }
            });
            img.setProperties({ color: isSelected ? COLORS.text : COLORS.hover });
        };

        return { container: b, image: img, setSelected };
    },

    createSlider(container, label, id, min, max, uiManager, onReset = null, shortcut = null) {
        const group = new Container({
            flexDirection: 'column',
            width: '100%',
            gap: 6,
            marginTop: 12,
        });

        const labelRow = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '100%',
            paddingRight: 4
        });
        group.add(labelRow);

        const labelGroup = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            gap: 8
        });
        labelRow.add(labelGroup);

        const labelText = new Text({
            text: label,
            fontSize: 16,
            color: COLORS.textDim,
        });
        labelGroup.add(labelText);

        if (shortcut) {
            labelGroup.add(new Text({
                text: shortcut,
                fontSize: 12,
                color: COLORS.hint,
            }));
        }

        if (onReset) {
            const resetBtn = new Container({
                width: 24,
                height: 24,
                borderRadius: 4,
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                hover: { backgroundColor: COLORS.hover },
                onPointerEnter: (e) => {
                    if (!uiManager.visible) return;
                    this.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, uiManager);
                },
                onClick: (e) => {
                    if (!uiManager.visible) return;
                    this.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, uiManager);
                    onReset();
                }
            });
            resetBtn.add(new Image({ src: 'icons/restore.svg', width: 16, height: 16, color: COLORS.textDim }));
            labelRow.add(resetBtn);
        }

        const track = new Container({
            width: 332,
            height: 16,
            backgroundColor: COLORS.track,
            borderRadius: 8,
            cursor: 'pointer',
            flexDirection: 'row',
            justifyContent: 'flex-start',
            alignItems: 'center',
            alignSelf: 'center',
            overflow: 'visible',
            onPointerDown: (e) => {
                if (!uiManager.visible) return;
                this.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, uiManager);
                uiManager.handleSlider(e, track, id, min, max, false);
            },
            onClick: (e) => {
                if (!uiManager.visible) return;
                this.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, uiManager);
                uiManager.handleSlider(e, track, id, min, max, true);
            },
            onPointerMove: (e) => {
                if (!uiManager.visible) return;
                e.buttons > 0 && uiManager.handleSlider(e, track, id, min, max, false);
            },
            onPointerUp: async (_e) => {
                if (!uiManager.visible) return;
                await uiManager.saveSettings();
                uiManager.syncUI(true);
            },
            onPointerLeave: async (e) => {
                // If dragging and pointer leaves the track, save the last value
                if (!uiManager.visible) return;
                if (e.buttons > 0) {
                    await uiManager.saveSettings();
                    uiManager.syncUI(true);
                }
            },
        });
        
        const thumb = new Container({
            width: 24,
            height: 24,
            backgroundColor: COLORS.accent,
            borderRadius: 12,
            borderWidth: 1.5,
            borderColor: 0xffffff,
        });
        track.add(thumb);
        group.add(track);
        container.add(group);
        
        return { group, thumb, label: labelText, min, max };
    },

    createSlidersForKeys(container, keys, uiManager, sliders = {}) {
        const screenType = uiManager.screenType;
        keys.forEach(id => {
            const meta = findSettingMeta(id);
            const { min, max } = getSettingLimits(id, screenType, uiManager.activeScreen);
            const label = getSettingLabel(id, screenType);
            const shortcut = meta?.shortcut || null;

            sliders[id] = this.createSlider(
                container,
                label,
                id,
                min,
                max,
                uiManager,
                () => uiManager.onSliderChange(id, getSettingDefault(id, uiManager.screenType)),
                shortcut
            );
        });
        return sliders;
    },

    createCheckboxRow(parent, label, onClick, uiManager, options = {}) {
        const row = new Container({
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            width: '100%',
            height: options.height,
            marginTop: options.marginTop,
            marginBottom: options.marginBottom
        });
        row.add(new Text({ text: label, fontSize: 18, color: COLORS.textDim }));
        const toggle = this.createButton('icons/square.svg', onClick, uiManager, 22, options.btnSize ?? 44);
        row.add(toggle.container);
        if (parent) parent.add(row);

        const setChecked = (val) => {
            if (toggle._lastVal === val) return;
            toggle._lastVal = val;
            toggle.image.setProperties({
                src: val ? 'icons/square-check.svg' : 'icons/square.svg',
                color: val ? COLORS.text : COLORS.hover
            });
            toggle.setSelected(val);
        };

        return { row, toggle, setChecked };
    },

    createSeparator(width = 2, height = 40, props = {}) {
        return new Container({ 
            width: width, 
            height: height, 
            backgroundColor: COLORS.separator,
            ...props
        });
    },

    setupWindowHeader(container, title, onReset, uiManager) {
        const header = new Container({
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            width: '100%',
            marginBottom: 8
        });
        
        header.add(new Text({ text: title, fontSize: 24, color: COLORS.text }));

        const rightGroup = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            gap: 12
        });
        header.add(rightGroup);

        const alignBtn = this.createButton('icons/square-arrow-right.svg', () => uiManager.toggleMenuAlignment(), uiManager, 22, 48);
        rightGroup.add(alignBtn.container);
        
        const resetBtn = this.createButton('icons/restore.svg', onReset, uiManager, 22, 48);
        rightGroup.add(resetBtn.container);
        container.add(header);

        return { alignBtn, resetBtn };
    },

    updateAlignIcon(alignBtn, uiManager, lastIconState) {
        if (!alignBtn) return lastIconState;
        const isRight = uiManager.menuAlignment === 'right';
        const icon = isRight ? 'icons/square-arrow-left.svg' : 'icons/square-arrow-right.svg';
        if (lastIconState !== icon) {
            alignBtn.image.setProperties({ src: icon });
            return icon;
        }
        return lastIconState;
    },

    createSubMenuContainer(defaultFont, extraProps = {}) {
        const container = new Container({
            flexDirection: 'column',
            backgroundColor: COLORS.bg,
            backgroundOpacity: 0.9,
            borderRadius: 16,
            borderWidth: 2,
            borderColor: COLORS.border,
            padding: 24,
            gap: 0,
            width: 400,
            display: 'none',
            position: 'absolute',
            marginLeft: UI_CONFIG.menuMarginLeft,
            marginTop: UI_CONFIG.menuMarginTop,
            fontFamily: defaultFont,
            ...extraProps
        });
        container.visible = false;
        return container;
    },

    createSelector(parent, label, onDelta, uiManager, options = {}) {
        const group = new Container({
            flexDirection: 'column',
            width: '100%',
            gap: options.gap ?? 4,
            marginTop: options.marginTop ?? 12,
            marginBottom: options.marginBottom ?? 0
        });
        group.add(new Text({ text: label, fontSize: 16, color: COLORS.textDim }));

        const selector = new Container({
            flexDirection: 'row',
            width: '100%',
            height: 48,
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: COLORS.button,
            borderRadius: 8,
            padding: 4
        });
        group.add(selector);

        const createArrowBtn = (icon, delta) => {
            const btn = new Container({
                width: 40,
                height: 40,
                borderRadius: 6,
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                hover: { backgroundColor: COLORS.hover },
                onPointerEnter: (e) => {
                    if (!uiManager.visible) return;
                    this.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, uiManager);
                },
                onClick: (e) => {
                    if (!uiManager.visible) return;
                    this.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, uiManager);
                    onDelta(delta);
                }
            });
            btn.add(new Image({ src: icon, width: 32, height: 32, color: COLORS.text }));
            return btn;
        };

        selector.add(createArrowBtn('icons/chevron-left.svg', -1));
        const valueText = new Text({
            text: options.initialText ?? "None",
            fontSize: options.fontSize ?? 14,
            color: COLORS.text,
            width: 250,
            textAlign: 'center',
            wordBreak: options.wordBreak ?? 'break-all'
        });
        selector.add(valueText);
        selector.add(createArrowBtn('icons/chevron-right.svg', 1));

        if (parent) {
            parent.add(group);
        }

        return { group, valueText };
    },

    createDisabledNotice(parent, title, messages) {
        const notice = new Container({
            flexDirection: 'column',
            width: '100%',
            alignItems: 'center',
            justifyContent: 'center',
            paddingTop: 32,
            paddingBottom: 32,
            paddingLeft: 12,
            paddingRight: 12,
            gap: 12,
            display: 'none'
        });
        notice.add(new Text({
            text: title,
            fontSize: 18,
            color: COLORS.text,
            textAlign: 'center',
            fontWeight: 'bold'
        }));

        const msgArray = Array.isArray(messages) ? messages : [messages];
        msgArray.forEach((msg, idx) => {
            notice.add(new Text({
                text: msg,
                fontSize: idx === 0 ? 14 : 13,
                color: idx === 0 ? COLORS.textDim : COLORS.hint,
                textAlign: 'center',
                width: '100%'
            }));
        });

        if (parent) {
            parent.add(notice);
        }
        return notice;
    },

    syncMenuSupportState(isSupported, { disabledNotice, headerBtns, sliders, groups = [] }) {
        const displayMode = isSupported ? 'flex' : 'none';
        if (disabledNotice) {
            disabledNotice.setProperties({ display: isSupported ? 'none' : 'flex' });
        }
        if (headerBtns?.resetBtn?.container) {
            headerBtns.resetBtn.container.setProperties({ display: displayMode });
        }
        groups.forEach(g => {
            if (g) g.setProperties({ display: displayMode });
        });
        for (const id in sliders) {
            const sl = sliders[id];
            if (sl.group) {
                sl.group.setProperties({ display: displayMode });
            }
        }
    },

    syncSliders(sliders, settingsManager, screenType) {
        const trackWidth = 332;
        const thumbWidth = 24;
        for (const id in sliders) {
            const sl = sliders[id];
            const val = settingsManager.get(id, screenType);
            if (val === undefined) continue;

            const min = sl.min;
            const max = sl.max;
            const displayVal = (id === 'screen_height_index') ? Math.round(val) : val;
            const p = Math.max(0, Math.min(1, (displayVal - min) / (max - min)));
            const margin = p * (trackWidth - thumbWidth);
            if (Math.abs((sl._lastMargin || 0) - margin) > 0.1) {
                sl.thumb.setProperties({ marginLeft: margin, backgroundColor: COLORS.accent });
                sl._lastMargin = margin;
            }
        }
    }
};

export const ThreeUtils = {
    /**
     * Recursively disposes of a Three.js object and its children.
     * It handles geometries, materials, and textures.
     * @param {THREE.Object3D} obj - The object to dispose.
     * @param {THREE.Object3D} parent - Optional parent to remove the object from first.
     */
    disposeObject(obj, parent = null) {
        if (!obj) return;
        
        // 1. Remove from parent first to avoid rendering glitches
        if (parent) {
            parent.remove(obj);
        } else if (obj.parent) {
            obj.parent.remove(obj);
        }

        // 2. Recursively dispose resources
        obj.traverse((child) => {
            if (child.isMesh) {
                if (child.geometry) child.geometry.dispose();

                if (child.material) {
                    const materials = Array.isArray(child.material) ? child.material : [child.material];
                    materials.forEach(m => {
                        // Dispose all textures in the material
                        for (const key in m) {
                            if (m[key] && m[key].isTexture) {
                                m[key].dispose();
                            }
                        }
                        m.dispose();
                    });
                }
            }
        });
    },

    /**
     * Checks if a Three.js object and all its ancestors are visible.
     * @param {THREE.Object3D} obj - The object to test.
     * @returns {boolean}
     */
    isHierarchyVisible(obj) {
        let cur = obj;
        while (cur) {
            if (!cur.visible) return false;
            cur = cur.parent;
        }
        return true;
    }
};
