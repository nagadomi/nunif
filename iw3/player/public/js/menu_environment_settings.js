import { UIUtils } from './ui_common.js';

const SKYBOX_SLIDER_KEYS = ['environment_tilt', 'environment_rotation', 'environment_intensity'];
const MODEL_SLIDER_KEYS = ['environment_model_scale', 'environment_model_y'];

class EnvironmentSettingsMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = UIUtils.createSubMenuContainer(defaultFont);

        this.headerBtns = UIUtils.setupWindowHeader(this.container, "Env Settings", () => this.reset(), this.uiManager);
        this.sliders = {};

        // Skybox selector
        const skySelector = UIUtils.createSelector(
            this.container,
            "Skybox",
            (d) => this.uiManager.changeSkybox(d),
            this.uiManager
        );
        this.skyboxGroup = skySelector.group;
        this.skyboxNameText = skySelector.valueText;

        UIUtils.createSlidersForKeys(this.container, SKYBOX_SLIDER_KEYS, this.uiManager, this.sliders);

        // 3D Model selector
        const modelSelector = UIUtils.createSelector(
            this.container,
            "3D Model",
            (d) => this.uiManager.changeModel(d),
            this.uiManager
        );
        this.modelGroup = modelSelector.group;
        this.modelNameText = modelSelector.valueText;

        UIUtils.createSlidersForKeys(this.container, MODEL_SLIDER_KEYS, this.uiManager, this.sliders);

        this.disabledNotice = UIUtils.createDisabledNotice(
            this.container,
            "Not Supported in VR180",
            "Environment backgrounds and 3D models are not available in VR180 mode."
        );
    }

    async reset() {
        const screenType = this.uiManager.screenType;
        if (screenType === 'dome' || screenType === 'dome_xr') return;
        await this.uiManager.settingsManager.resetGroup('environment');
    }

    sync() {
        const screenType = this.uiManager.screenType;
        const isSupported = (screenType !== 'dome' && screenType !== 'dome_xr');
        UIUtils.syncMenuSupportState(isSupported, {
            disabledNotice: this.disabledNotice,
            headerBtns: this.headerBtns,
            sliders: this.sliders,
            groups: [this.skyboxGroup, this.modelGroup]
        });

        if (isSupported) {
            const sm = this.uiManager.settingsManager;
            UIUtils.syncSliders(this.sliders, sm);

            if (this.skyboxNameText) {
                const skyText = sm.get('environment_name') || "None";
                if (this._lastSkyText !== skyText) {
                    this.skyboxNameText.setProperties({ text: skyText });
                    this._lastSkyText = skyText;
                }
            }
            if (this.modelNameText) {
                const modelText = sm.get('environment_model_name') || "None";
                if (this._lastModelText !== modelText) {
                    this.modelNameText.setProperties({ text: modelText });
                    this._lastModelText = modelText;
                }
            }
        }
        this._lastAlignIcon = UIUtils.updateAlignIcon(this.headerBtns?.alignBtn, this.uiManager, this._lastAlignIcon);
    }
}

export { EnvironmentSettingsMenu };
