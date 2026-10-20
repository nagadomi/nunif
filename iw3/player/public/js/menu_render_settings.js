import { Container, Text } from '@pmndrs/uikit';
import { COLORS, UI_CONFIG } from './constants.js';
import { UIUtils } from './ui_common.js';

class RenderSettingsMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = UIUtils.createSubMenuContainer(defaultFont, { gap: 12 });

        this.headerBtns = UIUtils.setupWindowHeader(this.container, "Render Settings", () => this.reset(), this.uiManager);

        const group = new Container({
            flexDirection: 'column',
            width: '100%',
            gap: 8,
            marginTop: 8
        });
        this.container.add(group);

        group.add(new Text({ text: "Browser reload required", fontSize: 14, color: COLORS.hint, marginBottom: 4 }));
        
        // Font Selector
        const fontSelector = UIUtils.createSelector(
            group,
            "Font / Language",
            (d) => this.uiManager.changeFont(d),
            this.uiManager,
            { gap: 6, marginTop: 0, marginBottom: 8, initialText: "Auto", wordBreak: 'break-word' }
        );
        this.fontNameText = fontSelector.valueText;
        this.languagesText = new Text({
            text: "",
            fontSize: 12,
            color: COLORS.textDim,
            marginTop: 4,
            textAlign: 'center',
            width: '100%'
        });
        fontSelector.group.add(this.languagesText);

        // FPS Selector
        const fpsSelector = UIUtils.createSelector(
            group,
            "Target Frame Rate",
            (d) => this.uiManager.changeFPS(d),
            this.uiManager,
            { gap: 6, marginTop: 0, marginBottom: 8, initialText: "Auto", fontSize: 16, wordBreak: 'break-word' }
        );
        this.fpsValueText = fpsSelector.valueText;

        this.checkboxes = {};
        
        // Super Sampling (SS) Multi-choice
        const ssRow = new Container({
            flexDirection: 'column',
            width: '100%',
            gap: 6,
            marginBottom: 4
        });
        group.add(ssRow);
        ssRow.add(new Text({ text: "Super Sampling", fontSize: 16, color: COLORS.textDim }));
        
        const ssBtnGroup = new Container({
            flexDirection: 'row',
            width: '100%',
            gap: 12,
            height: 48,
            marginTop: 4
        });
        ssRow.add(ssBtnGroup);
        
        this.ssButtons = {};
        [1.0, 1.5, 2.0].forEach(factor => {
            const btn = new Container({
                flex: 1,
                height: '100%',
                borderRadius: 8,
                backgroundColor: COLORS.button,
                alignItems: 'center',
                justifyContent: 'center',
                paddingX: 12,
                cursor: 'pointer',
                hover: { backgroundColor: COLORS.hover },
                onPointerEnter: (e) => UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, this.uiManager),
                onClick: (e) => {
                    UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, this.uiManager);
                    this.uiManager.onSliderChange('render_ss', factor);
                }
            });
            btn.add(new Text({ text: `${factor.toFixed(1)}x`, fontSize: 16, color: COLORS.text, width: '100%', textAlign: 'center' }));
            ssBtnGroup.add(btn);
            this.ssButtons[factor] = btn;
        });

        const configs = [
            { id: 'render_antialias', label: 'Antialias' },
            { id: 'render_xr_layers', label: window.XRMediaBinding ? 'High Quality Video' : 'High Quality Video (Not supported)' }
        ];

        configs.forEach(cfg => {
            this.checkboxes[cfg.id] = UIUtils.createCheckboxRow(
                group,
                cfg.label,
                () => {
                    const curr = this.uiManager.settingsManager.get(cfg.id);
                    this.uiManager.onSliderChange(cfg.id, !curr);
                },
                this.uiManager,
                { height: 44, btnSize: 44 }
            );
        });
    }

    async reset() {
        await this.uiManager.settingsManager.resetGroup('render');
    }

    sync() {
        const sm = this.uiManager.settingsManager;
        for (const id in this.checkboxes) {
            this.checkboxes[id].setChecked(sm.get(id));
        }

        // Sync SS buttons
        const currentSS = sm.get('render_ss');
        for (let factor in this.ssButtons) {
            const isSelected = parseFloat(factor) === currentSS;
            const btn = this.ssButtons[factor];
            if (btn._lastSelected !== isSelected) {
                btn.setProperties({
                    backgroundColor: isSelected ? COLORS.accent : COLORS.button
                });
                btn._lastSelected = isSelected;
            }
        }

        const currentFontId = sm.get('render_font') || 'Auto';
        const currentFont = this.uiManager.fonts.find(f => f.id === currentFontId) || this.uiManager.fonts[0];

        if (this.fontNameText) {
            const fontLabel = currentFont.label || "Auto";
            if (this._lastFontLabel !== fontLabel) {
                this.fontNameText.setProperties({ text: fontLabel });
                this._lastFontLabel = fontLabel;
            }
        }
        if (this.languagesText) {
            const fontLangs = currentFont.languages || "";
            if (this._lastFontLangs !== fontLangs) {
                this.languagesText.setProperties({ text: fontLangs });
                this._lastFontLangs = fontLangs;
            }
        }
        if (this.fpsValueText) {
            const fpsVal = sm.get('render_fps') || 0;
            const fpsStr = fpsVal === 0 ? "Auto" : `${fpsVal} fps`;
            if (this._lastFpsStr !== fpsStr) {
                this.fpsValueText.setProperties({ text: fpsStr });
                this._lastFpsStr = fpsStr;
            }
        }
        this._lastAlignIcon = UIUtils.updateAlignIcon(this.headerBtns?.alignBtn, this.uiManager, this._lastAlignIcon);
    }
}

export { RenderSettingsMenu };
