import { Container } from '@pmndrs/uikit';
import { UIUtils } from './ui_common.js';

const CHROMA_COLOR_PRESETS = ['Green', 'Blue'];

const SKYBOX_SLIDER_KEYS = ['environment_tilt', 'environment_rotation', 'environment_intensity'];
const MODEL_SLIDER_KEYS = ['environment_model_scale', 'environment_model_y'];
const CHROMA_SLIDER_KEYS = [
    'environment_chroma_threshold', 'environment_chroma_smoothness', 'environment_chroma_despill'
];

class EnvironmentSettingsMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = UIUtils.createSubMenuContainer(defaultFont);

        this.headerBtns = UIUtils.setupWindowHeader(this.container, "Env Settings", () => this.reset(), this.uiManager);
        this.sliders = {};

        // --- 1. Skybox Section ---
        this.skyboxSection = new Container({ flexDirection: 'column', width: '100%' });
        this.container.add(this.skyboxSection);

        const skySelector = UIUtils.createSelector(
            this.skyboxSection,
            "Skybox",
            (d) => this.uiManager.changeSkybox(d),
            this.uiManager
        );
        this.skyboxGroup = skySelector.group;
        this.skyboxNameText = skySelector.valueText;

        UIUtils.createSlidersForKeys(this.skyboxSection, SKYBOX_SLIDER_KEYS, this.uiManager, this.sliders);

        // --- Separator 1 (Between Skybox and 3D Model) ---
        this.sep1 = UIUtils.createSeparator(332, 1, { alignSelf: 'center', marginTop: 14, marginBottom: 8 });
        this.container.add(this.sep1);

        // --- 2. 3D Model Section ---
        this.modelSection = new Container({ flexDirection: 'column', width: '100%' });
        this.container.add(this.modelSection);

        const modelSelector = UIUtils.createSelector(
            this.modelSection,
            "3D Model",
            (d) => this.uiManager.changeModel(d),
            this.uiManager
        );
        this.modelGroup = modelSelector.group;
        this.modelNameText = modelSelector.valueText;

        UIUtils.createSlidersForKeys(this.modelSection, MODEL_SLIDER_KEYS, this.uiManager, this.sliders);

        // --- Separator 2 (Between 3D Model and Chroma Key) ---
        this.sep2 = UIUtils.createSeparator(332, 1, { alignSelf: 'center', marginTop: 14, marginBottom: 8 });
        this.container.add(this.sep2);

        // --- 3. Chroma Key Section ---
        this.chromaSection = new Container({ flexDirection: 'column', width: '100%' });
        this.container.add(this.chromaSection);

        this.chromaCheckbox = UIUtils.createCheckboxRow(
            this.chromaSection,
            "Transparent by Color",
            () => {
                const sm = this.uiManager.settingsManager;
                const next = !sm.get('environment_chroma_enabled');
                this.uiManager.onSliderChange('environment_chroma_enabled', next, true);
            },
            this.uiManager,
            { marginTop: 4, marginBottom: 4 }
        );

        this.chromaControls = new Container({
            flexDirection: 'column',
            width: '100%',
            display: 'none'
        });
        this.chromaSection.add(this.chromaControls);

        const colorSelector = UIUtils.createSelector(
            this.chromaControls,
            "Key Color",
            (d) => this.changeChromaColor(d),
            this.uiManager,
            { gap: 4, marginTop: 8, marginBottom: 4, initialText: "Green" }
        );
        this.chromaColorGroup = colorSelector.group;
        this.chromaColorText = colorSelector.valueText;

        this.chromaSliders = UIUtils.createSlidersForKeys(this.chromaControls, CHROMA_SLIDER_KEYS, this.uiManager);

        this.disabledNotice = UIUtils.createDisabledNotice(
            this.container,
            "Not Supported with XR Layers",
            [
                "Environment and background transparency settings cannot be applied in High Quality Video (XR Layers) mode.",
                "To adjust settings, disable High Quality Video in Render Settings."
            ]
        );
    }

    changeChromaColor(delta) {
        const sm = this.uiManager.settingsManager;
        const current = (sm.get('environment_chroma_color') || 'green').toLowerCase();
        let idx = CHROMA_COLOR_PRESETS.findIndex(p => p.toLowerCase() === current);
        if (idx === -1) idx = 0;
        idx = (idx + delta + CHROMA_COLOR_PRESETS.length) % CHROMA_COLOR_PRESETS.length;
        const next = CHROMA_COLOR_PRESETS[idx].toLowerCase();
        this.uiManager.onSliderChange('environment_chroma_color', next, true);
    }

    async reset() {
        const screenType = this.uiManager.screenType;
        if (screenType === 'dome_xr') return;
        await this.uiManager.settingsManager.resetGroup('environment');
    }

    sync() {
        const isAR = this.uiManager.isARMode;
        const screenType = this.uiManager.screenType;
        const isSupported = (screenType !== 'dome_xr');

        if (this.disabledNotice) {
            this.disabledNotice.setProperties({ display: isSupported ? 'none' : 'flex' });
        }
        if (this.headerBtns?.resetBtn) {
            this.headerBtns.resetBtn.container.setProperties({ display: isSupported ? 'flex' : 'none' });
        }

        if (isSupported) {
            const sm = this.uiManager.settingsManager;
            const isDome = (screenType === 'dome');

            const showSkybox = !isDome && !isAR;
            this.skyboxSection.setProperties({ display: showSkybox ? 'flex' : 'none' });

            const showModel = !isDome;
            this.modelSection.setProperties({ display: showModel ? 'flex' : 'none' });

            this.sep1.setProperties({ display: (showSkybox && showModel) ? 'flex' : 'none' });
            this.sep2.setProperties({ display: showModel ? 'flex' : 'none' });

            this.chromaSection.setProperties({ display: 'flex' });

            if (showSkybox || showModel) {
                UIUtils.syncSliders(this.sliders, sm);
            }

            if (showSkybox && this.skyboxNameText) {
                const skyText = sm.get('environment_name') || "None";
                if (this._lastSkyText !== skyText) {
                    this.skyboxNameText.setProperties({ text: skyText });
                    this._lastSkyText = skyText;
                }
            }

            if (showModel && this.modelNameText) {
                const modelText = sm.get('environment_model_name') || "None";
                if (this._lastModelText !== modelText) {
                    this.modelNameText.setProperties({ text: modelText });
                    this._lastModelText = modelText;
                }
            }

            const isChromaEnabled = !!sm.get('environment_chroma_enabled');
            this.chromaCheckbox.setChecked(isChromaEnabled);

            const targetDisplay = isChromaEnabled ? 'flex' : 'none';
            if (this._lastChromaDisplay !== targetDisplay) {
                this.chromaControls.setProperties({ display: targetDisplay });
                this._lastChromaDisplay = targetDisplay;
            }

            if (isChromaEnabled) {
                UIUtils.syncSliders(this.chromaSliders, sm);
                const colorVal = sm.get('environment_chroma_color') || 'green';
                const colorLabel = colorVal.charAt(0).toUpperCase() + colorVal.slice(1);
                if (this._lastChromaColor !== colorLabel) {
                    this.chromaColorText.setProperties({ text: colorLabel });
                    this._lastChromaColor = colorLabel;
                }
            }
        } else {
            this.skyboxSection.setProperties({ display: 'none' });
            this.sep1.setProperties({ display: 'none' });
            this.modelSection.setProperties({ display: 'none' });
            this.sep2.setProperties({ display: 'none' });
            this.chromaSection.setProperties({ display: 'none' });
        }

        this._lastAlignIcon = UIUtils.updateAlignIcon(this.headerBtns?.alignBtn, this.uiManager, this._lastAlignIcon);
    }
}

export { EnvironmentSettingsMenu };
