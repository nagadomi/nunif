import { UIUtils } from './ui_common.js';

const COLOR_SLIDER_KEYS = [
    'color_contrast', 'color_brightness', 'color_hue',
    'color_saturation', 'color_gamma'
];

class ColorSettingsMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = UIUtils.createSubMenuContainer(defaultFont);

        this.headerBtns = UIUtils.setupWindowHeader(this.container, "Color Settings", () => this.reset(), this.uiManager);

        const lutSelector = UIUtils.createSelector(
            this.container,
            "LUT (Color Profile)",
            (d) => this.uiManager.changeLut(d),
            this.uiManager,
            { marginBottom: 12 }
        );
        this.lutGroup = lutSelector.group;
        this.lutNameText = lutSelector.valueText;

        this.sliders = UIUtils.createSlidersForKeys(this.container, COLOR_SLIDER_KEYS, this.uiManager);

        this.disabledNotice = UIUtils.createDisabledNotice(
            this.container,
            "Not Supported with XR Layers",
            [
                "Color adjustments cannot be applied in High Quality Video (XR Layers) mode.",
                "To adjust colors, disable High Quality Video in Render Settings."
            ]
        );
    }

    async reset() {
        if (this.uiManager.screenType === 'dome_xr') return;
        await this.uiManager.settingsManager.resetGroup('color');
    }

    sync() {
        const isSupported = this.uiManager.screenType !== 'dome_xr';
        UIUtils.syncMenuSupportState(isSupported, {
            disabledNotice: this.disabledNotice,
            headerBtns: this.headerBtns,
            sliders: this.sliders,
            groups: [this.lutGroup]
        });

        if (isSupported) {
            const sm = this.uiManager.settingsManager;
            UIUtils.syncSliders(this.sliders, sm);

            if (this.lutNameText) {
                const lutText = sm.get('color_lut') || "None";
                if (this._lastLutText !== lutText) {
                    this.lutNameText.setProperties({ text: lutText });
                    this._lastLutText = lutText;
                }
            }
        }
        this._lastAlignIcon = UIUtils.updateAlignIcon(this.headerBtns?.alignBtn, this.uiManager, this._lastAlignIcon);
    }
}

export { ColorSettingsMenu };
