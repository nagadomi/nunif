import { Container, Image, Text } from '@pmndrs/uikit';
import { COLORS, UI_CONFIG } from './constants.js';
import { UIUtils } from './ui_common.js';

const VIDEO_SEEK_BAR_WIDTH = 370;
const VIDEO_THUMB_SIZE = 24;
const VOLUME_SEEK_BAR_WIDTH = 100;
const VOLUME_THUMB_SIZE = 24;

class MainMenu {
    constructor(uiManager, defaultFont) {
        this.uiManager = uiManager;
        this.container = new Container({
            flexDirection: 'column',
            backgroundColor: COLORS.bg,
            backgroundOpacity: 0.9,
            borderRadius: 24,
            borderWidth: 2,
            borderColor: COLORS.border,
            padding: 16,
            gap: 12,
            alignItems: 'center',
            marginTop: 500,
            width: 620, // Slightly wider to accommodate one more button
            fontFamily: defaultFont
        });

        this.setupSettingsRow();
        this.setupControlRow();
        this.setupSliderSection();
        this.setupTimeRow();
        this.setupFooterRow(); // Status info (FPS, Time, Battery)
    }

    setupFooterRow() {
        this.footerRow = new Container({
            flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-end',
            width: '100%', paddingLeft: 10, paddingRight: 10, marginTop: 4,
        });
        this.container.add(this.footerRow);

        const statusGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 10, });
        this.footerRow.add(statusGroup);
        this.fpsText = new Text({ text: "---f", fontSize: 16, color: COLORS.textDim });
        statusGroup.add(this.fpsText);
        statusGroup.add(new Image({ src: 'icons/clock.svg', width: 20, height: 20, color: COLORS.textDim }));
        this.timeText = new Text({ text: "00:00", fontSize: 16, color: COLORS.textDim });
        statusGroup.add(this.timeText);
        this.batteryIcon = new Image({ src: 'icons/battery-vertical-charging.svg', width: 20, height: 20, color: COLORS.textDim });
        statusGroup.add(this.batteryIcon);
        this.batteryText = new Text({ text: "--%", fontSize: 16, color: COLORS.textDim });
        statusGroup.add(this.batteryText);
        this._batteryInitialized = false;
        this._lastBatteryLevel = null;
        this._lastBatteryColor = null;
        this._lastBatteryIcon = null;
    }

    setupSettingsRow() {
        const row = new Container({
            flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
            width: '100%', paddingLeft: 10, paddingRight: 10,
        });
        this.container.add(row);

        this.recentButton = new Container({
            paddingLeft: 12, paddingRight: 12, height: 36,
            backgroundColor: COLORS.recent, borderRadius: 8,
            alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', display: 'none',
            hover: { backgroundColor: COLORS.recentHover },
            onPointerEnter: (e) => UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, this.uiManager),
            onClick: (e) => {
                UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, this.uiManager);
                this.uiManager.openRecentFile();
            }
        });
        this.recentButton.add(new Text({ text: "Open Recent File", fontSize: 14, color: COLORS.text }));
        row.add(this.recentButton);

        // Spacer to push settings buttons to the right
        row.add(new Container({ flexGrow: 1 }));

        const group = new Container({ flexDirection: 'row', alignItems: 'center', gap: 12 });
        row.add(group);
        const u = this.uiManager;
        
        // Order: Screen, Color, Env, Subtitle | Render, Shutdown
        this.screenButton = UIUtils.createButton('icons/device-desktop.svg', () => u.switchSubMenu(u.screenSettings), u, 22, 48);
        group.add(this.screenButton.container);

        this.colorButton = UIUtils.createButton('icons/color-filter.svg', () => u.switchSubMenu(u.colorSettings), u, 22, 48);
        group.add(this.colorButton.container);

        this.envButton = UIUtils.createButton('icons/planet.svg', () => u.switchSubMenu(u.environmentSettings), u, 22, 48);
        group.add(this.envButton.container);
        
        this.subtitleButton = UIUtils.createButton('icons/subtitles.svg', () => u.switchSubMenu(u.subtitleSettings), u, 22, 48);
        group.add(this.subtitleButton.container);

        group.add(UIUtils.createSeparator(2, 32, { marginLeft: 8, marginRight: 8 }));
        
        this.renderButton = UIUtils.createButton('icons/video.svg', () => u.switchSubMenu(u.renderSettings), u, 22, 48);
        group.add(this.renderButton.container);

        group.add(UIUtils.createButton('icons/power.svg', () => u.exitXR(), u, 22, 48).container);
    }

    setupControlRow() {
        const row = new Container({ flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-start', width: '100%', paddingLeft: 10, });
        this.container.add(row);
        const u = this.uiManager;
        
        this.explorerButton = UIUtils.createButton('icons/library-photo.svg', () => u.switchSubMenu(u.explorer), u);
        row.add(this.explorerButton.container);

        const playbackGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 20, marginLeft: 80 });
        row.add(playbackGroup);
        playbackGroup.add(UIUtils.createButton('icons/player-skip-back.svg', () => u.loadNextMedia(-1), u).container);
        const playPause = UIUtils.createButton('icons/player-play.svg', () => u.togglePlayPause(), u, 44, 80, 'play_pause');
        this.playPauseButton = playPause.container; this.playPauseIcon = playPause.image;
        playbackGroup.add(this.playPauseButton);
        playbackGroup.add(UIUtils.createButton('icons/player-skip-forward.svg', () => u.loadNextMedia(1), u).container);
    }

    setupSliderSection() {
        const group = new Container({
            flexDirection: 'column',
            alignItems: 'center',
            gap: 6,
            width: '100%',
            overflow: 'visible'
        });
        this.container.add(group);

        this.setupSeekBarRow(group);
        this.setupVolumeRow(group);
    }

    setupSeekBarRow(parent) {
        const row = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            gap: 12,
            width: '100%',
            backgroundColor: 0x000000,
            backgroundOpacity: 0.2,
            borderRadius: 16,
            paddingTop: 6,
            paddingBottom: 6,
            paddingLeft: 12,
            paddingRight: 12,
            overflow: 'visible'
        });
        parent.add(row);
        const u = this.uiManager;

        const repeat = UIUtils.createButton('icons/repeat.svg', () => u.toggleVideoRepeat(), u, 24, 40);
        this.repeatButton = repeat;
        row.add(this.repeatButton.container);

        this.videoSeekBar = new Container({
            width: VIDEO_SEEK_BAR_WIDTH,
            height: 16,
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'flex-start',
            backgroundColor: COLORS.track,
            borderRadius: 8,
            cursor: 'pointer',
            overflow: 'visible',
            onClick: (e) => u.handleVideoSeek(e, this.videoSeekBar, true),
            onPointerMove: (e) => { if (e.buttons > 0) u.handleVideoSeek(e, this.videoSeekBar, false); },
            onPointerUp: (_e) => { if (u.visible) u.savePlaybackPosition(); },
            onPointerLeave: (e) => { if (u.visible && e.buttons > 0) u.savePlaybackPosition(); }
        });
        this.videoThumb = new Container({
            width: VIDEO_THUMB_SIZE,
            height: VIDEO_THUMB_SIZE,
            backgroundColor: COLORS.thumb,
            borderRadius: VIDEO_THUMB_SIZE / 2,
            borderWidth: 1.5,
            borderColor: 0xffffff
        });
        this.videoSeekBar.add(this.videoThumb);
        row.add(this.videoSeekBar);

        this.videoTimeText = new Text({
            text: "",
            fontSize: 14,
            color: COLORS.textDim,
            width: 130,
            textAlign: 'right',
            wordBreak: 'keep-all'
        });
        row.add(this.videoTimeText);
    }

    setupVolumeRow(parent) {
        const row = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'flex-start',
            gap: 12,
            width: '100%',
            backgroundColor: 0x000000,
            backgroundOpacity: 0.2,
            borderRadius: 16,
            paddingTop: 6,
            paddingBottom: 6,
            paddingLeft: 12,
            paddingRight: 12,
            overflow: 'visible'
        });
        parent.add(row);
        const u = this.uiManager;

        // --- Volume Group ---
        const volGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 12 });
        row.add(volGroup);

        const volBtn = UIUtils.createButton('icons/volume.svg', () => u.toggleMute(), u, 24, 40);
        this.volumeButton = volBtn;
        this.volumeIcon = volBtn.image;
        volGroup.add(this.volumeButton.container);

        this.volumeSeekBar = new Container({
            width: VOLUME_SEEK_BAR_WIDTH,
            height: 16,
            flexDirection: 'row',
            alignItems: 'center',
            backgroundColor: COLORS.track,
            borderRadius: 8,
            cursor: 'pointer',
            overflow: 'visible',
            onPointerDown: (e) => UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, u),
            onClick: (e) => { 
                UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, u); 
                u.handleVolumeChange(e, this.volumeSeekBar, true); 
            },
            onPointerMove: (e) => { 
                if (e.buttons > 0) u.handleVolumeChange(e, this.volumeSeekBar, false); 
            },
            onPointerUp: (_e) => { 
                if (u.visible) u.saveSettings(); 
            },
            onPointerLeave: (e) => {
                if (u.visible && e.buttons > 0) u.saveSettings();
            },
        });
        this.volumeThumb = new Container({
            width: VOLUME_THUMB_SIZE,
            height: VOLUME_THUMB_SIZE,
            backgroundColor: COLORS.thumb,
            borderRadius: VOLUME_THUMB_SIZE / 2,
            borderWidth: 1.5,
            borderColor: 0xffffff
        });
        this.volumeSeekBar.add(this.volumeThumb);
        volGroup.add(this.volumeSeekBar);

        // --- Separator ---
        row.add(UIUtils.createSeparator(2, 28, { marginLeft: 4, marginRight: 4 }));

        // --- Speed Group ---
        const speedGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 6 });
        row.add(speedGroup);

        const speedBtn = UIUtils.createButton('icons/player-speed.svg', () => u.resetPlaybackRate(), u, 24, 40);
        this.speedButton = speedBtn;
        speedGroup.add(this.speedButton.container);

        this.minusButton = UIUtils.createButton('icons/minus.svg', () => u.changePlaybackRate(-0.25), u, 20, 40);
        this.minusButton.container.setProperties({ marginLeft: 8 });
        speedGroup.add(this.minusButton.container);

        this.speedValueText = new Text({
            text: "1.00x",
            fontSize: 14,
            color: COLORS.textDim,
            width: 52,
            textAlign: 'center',
            wordBreak: 'keep-all'
        });
        speedGroup.add(this.speedValueText);

        this.plusButton = UIUtils.createButton('icons/plus.svg', () => u.changePlaybackRate(0.25), u, 20, 40);
        speedGroup.add(this.plusButton.container);
    }

    setupTimeRow() {
        this.timeRow = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'flex-start',
            width: '100%',
            height: 24,
            paddingLeft: 10,
            paddingRight: 10,
            marginTop: -8, // Tighten gap with slider row
            marginBottom: 0,
            overflow: 'hidden'
        });
        this.container.add(this.timeRow);

        // Clip filename horizontally and vertically
        const fileClip = new Container({ width: 568, height: 24, overflow: 'hidden' });
        this.fileNameText = new Text({ 
            text: "", 
            fontSize: 14, 
            color: COLORS.textDim,
            width: 2000, // Large width to prevent wrapping
            wordBreak: 'keep-all'
        });
        fileClip.add(this.fileNameText);
        this.timeRow.add(fileClip);
    }

    formatTime(seconds) {
        if (isNaN(seconds) || seconds === null) return "00:00:00";
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    sync(params) {
        if (params.slowMetricsUpdated) {
            const now = new Date();
            const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
            if (this._lastTimeStr !== timeStr) {
                this.timeText.setProperties({ text: timeStr });
                this._lastTimeStr = timeStr;
            }

            if (!this._batteryInitialized && navigator.getBattery) {
                this._batteryInitialized = true;
                navigator.getBattery().then(battery => {
                    const update = () => { 
                        const levelStr = `${Math.round(battery.level * 100)}%`;
                        const chargingColor = battery.charging ? 0x00ff00 : COLORS.textDim;

                        // Select icon based on level and charging state
                        let iconSrc = 'icons/battery-vertical-charging.svg';
                        if (!battery.charging) {
                            if (battery.level > 0.75) iconSrc = 'icons/battery-vertical-4.svg';
                            else if (battery.level > 0.50) iconSrc = 'icons/battery-vertical-3.svg';
                            else if (battery.level > 0.25) iconSrc = 'icons/battery-vertical-2.svg';
                            else iconSrc = 'icons/battery-vertical-1.svg';
                        }

                        if (this._lastBatteryLevel !== levelStr) {
                            this.batteryText.setProperties({ text: levelStr });
                            this._lastBatteryLevel = levelStr;
                        }
                        if (this._lastBatteryColor !== chargingColor) {
                            this.batteryIcon.setProperties({ color: chargingColor });
                            this._lastBatteryColor = chargingColor;
                        }
                        if (this._lastBatteryIcon !== iconSrc) {
                            this.batteryIcon.setProperties({ src: iconSrc });
                            this._lastBatteryIcon = iconSrc;
                        }
                    };
                    update(); battery.addEventListener('levelchange', update); battery.addEventListener('chargingchange', update);
                });
            }
            if (params.fps !== undefined) {
                const fpsStr = `${Math.round(params.fps)}f`;
                if (this._lastFpsStr !== fpsStr) {
                    this.fpsText.setProperties({ text: fpsStr });
                    this._lastFpsStr = fpsStr;
                }
            }
        }

        if (params.currentFileName !== undefined) {
            if (this._lastFileName !== params.currentFileName) {
                this.fileNameText.setProperties({ text: params.currentFileName || "" });
                this._lastFileName = params.currentFileName;
            }
        }

        if (params.isVideoActive !== undefined || params.isPlaying !== undefined || params.videoProgress !== undefined) {
            const isVideo = params.isVideoActive ?? (!!this.uiManager.videoElement);
            if (isVideo) {
                if (params.currentTime !== undefined && params.duration !== undefined) {
                    const timeText = `${this.formatTime(params.currentTime)} / ${this.formatTime(params.duration)}`;
                    if (this._lastVideoTimeText !== timeText) {
                        this.videoTimeText.setProperties({ text: timeText });
                        this._lastVideoTimeText = timeText;
                    }
                }
                if (params.isPlaying !== undefined) {
                    const iconSrc = params.isPlaying ? 'icons/player-pause.svg' : 'icons/player-play.svg';
                    const bgColor = params.isPlaying ? COLORS.accent : COLORS.track;
                    if (this._lastPlayIcon !== iconSrc) {
                        this.playPauseIcon.setProperties({ src: iconSrc });
                        this._lastPlayIcon = iconSrc;
                    }
                    if (this._lastPlayBg !== bgColor) {
                        this.playPauseButton.setProperties({ backgroundColor: bgColor });
                        this._lastPlayBg = bgColor;
                    }
                }
                if (params.videoProgress !== undefined) {
                    const margin = params.videoProgress * (VIDEO_SEEK_BAR_WIDTH - VIDEO_THUMB_SIZE);
                    if (Math.abs((this._lastVideoMargin || 0) - margin) > 0.1) {
                        this.videoThumb.setProperties({ marginLeft: margin });
                        this._lastVideoMargin = margin;
                    }
                }
                if (this._lastVideoCursor !== 'pointer') {
                    this.videoSeekBar.setProperties({ cursor: 'pointer' });
                    this._lastVideoCursor = 'pointer';
                }
            } else {
                if (this._lastVideoTimeText !== "") {
                    this.videoTimeText.setProperties({ text: "" });
                    this._lastVideoTimeText = "";
                }
                if (this._lastVideoCursor !== 'default') {
                    this.videoSeekBar.setProperties({ cursor: 'default' });
                    this._lastVideoCursor = 'default';
                }
                if (this._lastVideoMargin !== 0) {
                    this.videoThumb.setProperties({ marginLeft: 0 });
                    this._lastVideoMargin = 0;
                }
                const disabledBg = COLORS.buttonDisabled;
                if (this._lastPlayBg !== disabledBg) {
                    this.playPauseButton.setProperties({ backgroundColor: disabledBg });
                    this._lastPlayBg = disabledBg;
                }
            }
        }

        if (this.repeatButton) {
            this.repeatButton.setSelected(this.uiManager.videoRepeat);
        }

        if (this.volumeButton) {
            const isMuted = params.isMuted ?? this.uiManager.isMuted;
            const currentVol = params.volume !== undefined ? params.volume : this.uiManager.currentVolume;
            const iconSrc = (isMuted || currentVol === 0) ? 'icons/volume-off.svg' : 'icons/volume.svg';
            if (this._lastVolumeIcon !== iconSrc) {
                this.volumeIcon.setProperties({ src: iconSrc });
                this._lastVolumeIcon = iconSrc;
            }
            this.volumeButton.setSelected(isMuted);
        }

        if (params.volume !== undefined) {
            const volMargin = params.volume * (VOLUME_SEEK_BAR_WIDTH - VOLUME_THUMB_SIZE);
            if (Math.abs((this._lastVolMargin || 0) - volMargin) > 0.1) {
                this.volumeThumb.setProperties({ marginLeft: volMargin });
                this._lastVolMargin = volMargin;
            }
        }

        if (params.speed !== undefined) {
            const speedText = `${params.speed.toFixed(2)}x`;
            if (this._lastSpeedText !== speedText) {
                this.speedValueText.setProperties({ text: speedText });
                this._lastSpeedText = speedText;
            }
        }

        // --- Sub-menu selection highlight ---
        const activeMenu = this.uiManager.activeSubMenu;
        const menus = [
            { btn: this.screenButton, target: this.uiManager.screenSettings },
            { btn: this.colorButton, target: this.uiManager.colorSettings },
            { btn: this.envButton, target: this.uiManager.environmentSettings },
            { btn: this.subtitleButton, target: this.uiManager.subtitleSettings },
            { btn: this.renderButton, target: this.uiManager.renderSettings },
            { btn: this.explorerButton, target: this.uiManager.explorer }
        ];

        menus.forEach(m => {
            if (m.btn?.setSelected) {
                m.btn.setSelected(activeMenu === m.target);
            }
        });
    }
}

export { MainMenu };
