import { UIUtils } from './ui_common.js';

const SUBTITLE_SLIDER_KEYS = ['subtitle_font_size', 'subtitle_z', 'subtitle_y', 'subtitle_eye_sep'];

class SubtitleSettingsMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = UIUtils.createSubMenuContainer(defaultFont, {
            borderRadius: 24,
            gap: 16
        });

        this.setupUI();
    }

    setupUI() {
        const u = this.uiManager;
        this.headerBtns = UIUtils.setupWindowHeader(this.container, "Subtitle Settings", () => this.reset(), u);

        // Subtitle Track Selector
        const trackSelector = UIUtils.createSelector(
            this.container,
            "Track",
            (d) => u.changeSubtitle(d),
            u,
            { gap: 6, marginTop: 4 }
        );
        this.trackGroup = trackSelector.group;
        this.trackNameText = trackSelector.valueText;

        // Visibility Toggle (Checkbox)
        this.showCheckbox = UIUtils.createCheckboxRow(
            this.container,
            "Show Subtitles",
            () => u.toggleSubtitles(),
            u,
            { btnSize: 48, marginTop: 10, marginBottom: 10 }
        );

        this.sliders = UIUtils.createSlidersForKeys(this.container, SUBTITLE_SLIDER_KEYS, u);

        this.disabledNotice = UIUtils.createDisabledNotice(
            this.container,
            "Not Supported in VR180",
            "Subtitles are not available in VR180 mode."
        );
    }

    async reset() {
        if (!this.uiManager.activeScreen?.supportsSubtitles) return;

        await this.uiManager.settingsManager.resetGroup('subtitle');
        
        // Reset track selection
        const hasTracks = this.uiManager.availableSubtitles.length > 0;
        this.uiManager.currentSubtitleIndex = hasTracks ? 0 : -1;
        this.uiManager.stereoPlayer?.setSubtitleTrack(this.uiManager.currentSubtitleIndex);

        this.uiManager.syncUI(true);
    }

    sync() {
        const isSupported = !!this.uiManager.activeScreen?.supportsSubtitles;
        UIUtils.syncMenuSupportState(isSupported, {
            disabledNotice: this.disabledNotice,
            headerBtns: this.headerBtns,
            sliders: this.sliders,
            groups: [this.trackGroup, this.showCheckbox.row]
        });

        if (isSupported) {
            const sm = this.uiManager.settingsManager;
            UIUtils.syncSliders(this.sliders, sm);
            this.showCheckbox.setChecked(sm.get('subtitle_visible'));

            if (this.trackNameText) {
                const trackText = this.uiManager.availableSubtitles[this.uiManager.currentSubtitleIndex]?.title || "None";
                if (this._lastTrackText !== trackText) {
                    this.trackNameText.setProperties({ text: trackText });
                    this._lastTrackText = trackText;
                }
            }
        }
        this._lastAlignIcon = UIUtils.updateAlignIcon(this.headerBtns?.alignBtn, this.uiManager, this._lastAlignIcon);
    }
}

export { SubtitleSettingsMenu };
