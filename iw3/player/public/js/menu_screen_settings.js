import { Container, Text } from '@pmndrs/uikit';
import { COLORS, UI_CONFIG, STEREO_FORMATS, getSettingLimits, getSettingLabel } from './constants.js';
import { UIUtils } from './ui_common.js';
import * as THREE from 'three';

const SCREEN_SLIDER_KEYS = [
    'screen_height_index', 'screen_tilt', 'screen_zoom', 'screen_eye_sep',
    'screen_tx', 'screen_ty', 'screen_roll', 'screen_curvature',
    'screen_edge_fade', 'screen_bg_color'
];

class ScreenSettingsMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = UIUtils.createSubMenuContainer(defaultFont);
        this.headerBtns = UIUtils.setupWindowHeader(this.container, "Screen Settings", () => this.reset(), this.uiManager);

        // Stereo Format Selection
        const formatGrid = new Container({
            flexDirection: 'row',
            flexWrap: 'wrap',
            gap: 8,
            marginTop: 16,
            marginBottom: 16
        });
        this.container.add(formatGrid);

        this.formatButtons = {};
        const formats = [
            { id: STEREO_FORMATS.SBS_FULL, label: "SBS Full" },
            { id: STEREO_FORMATS.SBS_HALF, label: "SBS Half" },
            { id: STEREO_FORMATS.SBS_FULL_CROSS, label: "Cross" },
            { id: STEREO_FORMATS.TB_FULL, label: "TB Full" },
            { id: STEREO_FORMATS.TB_HALF, label: "TB Half" },
            { id: STEREO_FORMATS.SBS_180, label: "VR180" },
            { id: STEREO_FORMATS.FLAT, label: "Flat" }
        ];

        formats.forEach(f => {
            const btn = new Container({
                width: 110,
                height: 40,
                backgroundColor: COLORS.button,
                borderRadius: 8,
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                hover: { backgroundColor: COLORS.hover },
                onPointerEnter: (e) => UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, this.uiManager),
                onClick: (e) => {
                    UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, this.uiManager);
                    this.uiManager.onFormatChange(f.id);
                }
            });
            const txt = new Text({ text: f.label, fontSize: 14, color: COLORS.text });
            btn.add(txt);
            formatGrid.add(btn);

            const setSelected = (isSelected) => {
                if (btn._lastSelected === isSelected) return;
                btn._lastSelected = isSelected;

                btn.setProperties({
                    backgroundColor: isSelected ? COLORS.accent : COLORS.button,
                    hover: { backgroundColor: isSelected ? COLORS.accentHover : COLORS.hover }
                });
                txt.setProperties({ color: COLORS.text });
            };
            this.formatButtons[f.id] = { container: btn, setSelected };
        });

        // Track Device Position Toggle & Reset
        const trackDeviceRow = new Container({
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            width: '100%',
            height: 44,
            marginBottom: 8
        });
        trackDeviceRow.add(new Text({ text: "Follow Device", fontSize: 18, color: COLORS.textDim }));

        const trackDeviceRightGroup = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            gap: 8
        });
        trackDeviceRow.add(trackDeviceRightGroup);

        this.trackDeviceResetBtn = UIUtils.createButton(
            'icons/restore.svg',
            () => {
                this.uiManager.stereoPlayer.resetScreenTransform();
            },
            this.uiManager,
            22,
            44
        );
        trackDeviceRightGroup.add(this.trackDeviceResetBtn.container);

        this.trackDeviceCheckbox = UIUtils.createButton(
            'icons/square.svg',
            () => {
                const current = this.uiManager.settingsManager.get('screen_track_device');
                this.uiManager.onSliderChange('screen_track_device', !current);
            },
            this.uiManager,
            22,
            44
        );
        trackDeviceRightGroup.add(this.trackDeviceCheckbox.container);

        this.setTrackDeviceChecked = (val) => {
            if (this.trackDeviceCheckbox._lastVal === val) return;
            this.trackDeviceCheckbox._lastVal = val;
            this.trackDeviceCheckbox.image.setProperties({
                src: val ? 'icons/square-check.svg' : 'icons/square.svg',
                color: val ? COLORS.text : COLORS.hover
            });
            this.trackDeviceCheckbox.setSelected(val);
        };

        this.container.add(trackDeviceRow);
        this.trackDeviceRow = trackDeviceRow;

        this.sliders = UIUtils.createSlidersForKeys(this.container, SCREEN_SLIDER_KEYS, this.uiManager);
    }

    async reset() {
        await this.uiManager.settingsManager.resetGroup('screen');
        this.uiManager.stereoPlayer.resetScreenTransform();
    }

    sync() {
        const trackWidth = 332, thumbWidth = 24;
        const activeScreen = this.uiManager.activeScreen;
        const screenType = activeScreen.screenType;

        const isTrackDevice = !!this.uiManager.settingsManager.get('screen_track_device');
        this.setTrackDeviceChecked(isTrackDevice);

        const resetDisplay = !isTrackDevice ? 'flex' : 'none';
        if (this.trackDeviceResetBtn._lastDisplay !== resetDisplay) {
            this.trackDeviceResetBtn.container.setProperties({ display: resetDisplay });
            this.trackDeviceResetBtn._lastDisplay = resetDisplay;
        }

        for (const id in this.sliders) {
            const sl = this.sliders[id];
            
            // Visibility & Label Sync
            let isSupported = activeScreen.supportedSettings.includes(id);
            if (id === 'screen_bg_color' && isSupported) {
                const isSkyboxNone = this.uiManager.skyboxes[this.uiManager.currentSkyboxIndex] === "None";
                if (this.uiManager.isARMode || !isSkyboxNone) {
                    isSupported = false;
                }
            }

            const displayMode = isSupported ? 'flex' : 'none';
            if (sl._lastDisplay !== displayMode) {
                sl.group.setProperties({ display: displayMode });
                sl._lastDisplay = displayMode;
            }

            if (isSupported) {
                if (sl.label) {
                    const newLabel = getSettingLabel(id, screenType);
                    if (sl._lastLabel !== newLabel) {
                        sl.label.setProperties({ text: newLabel });
                        sl._lastLabel = newLabel;
                    }
                }

                // Position Sync using effective value
                const val = this.uiManager.settingsManager.get(id, screenType);
                if (val === undefined) continue;

                const { min, max } = getSettingLimits(id, screenType, activeScreen);
                const displayVal = (id === 'screen_height_index') ? Math.round(val) : val;
                const p = THREE.MathUtils.clamp((displayVal - min) / (max - min), 0, 1);
                const margin = p * (trackWidth - thumbWidth);
                if (Math.abs((sl._lastMargin || 0) - margin) > 0.1) {
                    sl.thumb.setProperties({ marginLeft: margin, backgroundColor: COLORS.accent });
                    sl._lastMargin = margin;
                }
            }
        }

        const currentFormat = this.uiManager.currentStereoFormat;
        for (const id in this.formatButtons) {
            const btn = this.formatButtons[id];
            if (btn.setSelected) btn.setSelected(id === currentFormat);
        }

        this._lastAlignIcon = UIUtils.updateAlignIcon(this.headerBtns?.alignBtn, this.uiManager, this._lastAlignIcon);
    }
}

export { ScreenSettingsMenu };
