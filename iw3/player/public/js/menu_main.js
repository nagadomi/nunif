import { Container, Image, Text } from '@pmndrs/uikit';
import { COLORS, UI_CONFIG } from './constants.js';
import { UIUtils } from './ui_common.js';

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
        this.setupSliderRow();
        this.setupTimeRow();
        this.setupSpeedMenu(defaultFont);
        this.setupFooterRow(); // Status info (FPS, Time, Battery)
    }

    setupFooterRow() {
        this.footerRow = new Container({
            flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-end',
            width: '100%', paddingLeft: 10, paddingRight: 10, marginTop: 4,
        });
        this.container.add(this.footerRow);

        const u = this.uiManager;
        const toggleSpeed = (e) => {
            if (!u.visible) return;
            e.stopPropagation?.();
            UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, u);
            u.switchSubMenu(this.speedMenu);
        };
        this.speedReadout = new Container({
            width: 56, height: 24, borderRadius: 6, cursor: 'pointer',
            alignItems: 'center', justifyContent: 'center',
            hover: { backgroundColor: COLORS.hover },
            onPointerDown: (e) => { if (e.button === 2) toggleSpeed(e); },
            onClick: (e) => { if (e.button !== 2) toggleSpeed(e); }
        });
        this.speedReadoutText = new Text({ text: '1.0x', fontSize: 16, color: COLORS.textDim });
        this.speedReadout.add(this.speedReadoutText);
        this.footerRow.add(this.speedReadout);
        this.footerRow.add(new Container({ flexGrow: 1 }));

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
        this.screenButton = UIUtils.createButton('icons/device-desktop.svg', () => u.toggleScreenSettingsVisibility(), u, 22, 48);
        group.add(this.screenButton.container);

        this.colorButton = UIUtils.createButton('icons/color-filter.svg', () => u.toggleColorSettingsVisibility(), u, 22, 48);
        group.add(this.colorButton.container);

        this.envButton = UIUtils.createButton('icons/planet.svg', () => u.toggleEnvironmentSettingsVisibility(), u, 22, 48);
        group.add(this.envButton.container);
        
        this.subtitleButton = UIUtils.createButton('icons/subtitles.svg', () => u.toggleSubtitleSettingsVisibility(), u, 22, 48);
        group.add(this.subtitleButton.container);

        group.add(UIUtils.createSeparator(2, 32, { marginLeft: 8, marginRight: 8 }));
        
        this.renderButton = UIUtils.createButton('icons/video.svg', () => u.toggleRenderSettingsVisibility(), u, 22, 48);
        group.add(this.renderButton.container);

        group.add(UIUtils.createButton('icons/power.svg', () => u.stereoPlayer.exitXR(), u, 22, 48).container);
    }

    setupControlRow() {
        const row = new Container({ flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-start', width: '100%', paddingLeft: 10, });
        this.container.add(row);
        const u = this.uiManager;
        
        this.explorerButton = UIUtils.createButton('icons/library-photo.svg', () => u.toggleExplorerVisibility(), u);
        row.add(this.explorerButton.container);

        const playbackGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 20, marginLeft: 80 });
        row.add(playbackGroup);
        playbackGroup.add(UIUtils.createButton('icons/player-skip-back.svg', () => u.stereoPlayer.loadNextImage(-1), u).container);
        const playPause = UIUtils.createButton('icons/player-play.svg', () => u.stereoPlayer.togglePlayPause(), u, 44, 80, 'play_pause');
        this.playPauseButton = playPause.container; this.playPauseIcon = playPause.image;
        playbackGroup.add(this.playPauseButton);
        playbackGroup.add(UIUtils.createButton('icons/player-skip-forward.svg', () => u.stereoPlayer.loadNextImage(1), u).container);
    }

    setupSliderRow() {
        const row = new Container({ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 12, width: '100%', backgroundColor: 0x000000, backgroundOpacity: 0.2, borderRadius: 16, padding: 12, });
        this.container.add(row);
        const u = this.uiManager;

        const videoGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 12, overflow: 'visible' });
        row.add(videoGroup);

        const repeat = UIUtils.createButton('icons/repeat.svg', () => u.toggleVideoRepeat(), u, 24, 40);
        this.repeatButton = repeat; // Store full object
        videoGroup.add(this.repeatButton.container);

        this.videoSeekBar = new Container({
            width: 350, height: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-start', backgroundColor: COLORS.track, borderRadius: 8, cursor: 'pointer', overflow: 'visible',
            onClick: (e) => u.handleVideoSeek(e, this.videoSeekBar, true),
            onPointerMove: (e) => { if (e.buttons > 0) u.handleVideoSeek(e, this.videoSeekBar, false); },
            onPointerUp: (_e) => { if (u.visible) u.stereoPlayer.savePlaybackPosition(); },
            onPointerLeave: (e) => { if (u.visible && e.buttons > 0) u.stereoPlayer.savePlaybackPosition(); }
        });
        this.videoThumb = new Container({ width: 24, height: 24, backgroundColor: COLORS.thumb, borderRadius: 12, borderWidth: 1.5, borderColor: 0xffffff });
        this.videoSeekBar.add(this.videoThumb);
        videoGroup.add(this.videoSeekBar);

        const volGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 8 });
        row.add(volGroup);
        volGroup.add(new Image({ src: 'icons/volume.svg', width: 32, height: 32, color: COLORS.text }));
        this.volumeSeekBar = new Container({
            width: 100, height: 16, flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.track, borderRadius: 8, cursor: 'pointer', overflow: 'visible',
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
        this.volumeThumb = new Container({ width: 24, height: 24, backgroundColor: COLORS.thumb, borderRadius: 12, borderWidth: 1.5, borderColor: 0xffffff });
        this.volumeSeekBar.add(this.volumeThumb);
        volGroup.add(this.volumeSeekBar);
    }

    setupSpeedMenu(defaultFont) {
        const u = this.uiManager;
        const popup = new Container({
            display: 'none', position: 'absolute',
            marginLeft: UI_CONFIG.menuMarginLeft, marginTop: UI_CONFIG.menuMarginTop,
            width: 400, padding: 16, gap: 16, flexDirection: 'column',
            backgroundColor: COLORS.bg, backgroundOpacity: 0.95,
            borderRadius: 16, borderWidth: 2, borderColor: COLORS.border,
            fontFamily: defaultFont
        });
        this.speedMenu = { container: popup, onClose: () => this.cancelSpeedBoundEdit() };
        const row = new Container({
            flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
            gap: 8, width: '100%', height: 48
        });

        const button = (text, action, width = 48, label = null) => {
            const control = new Container({
                width, height: 48, borderRadius: 12, backgroundColor: COLORS.button,
                alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
                hover: { backgroundColor: COLORS.hover }, active: { backgroundColor: COLORS.accent },
                onPointerEnter: (e) => {
                    if (u.visible) UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, u);
                },
                onClick: (e) => {
                    if (!this.isSpeedMenuActive()) return;
                    UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, u);
                    return action();
                }
            });
            control.add(label || new Text({ text, fontSize: 22, color: COLORS.text }));
            return control;
        };
        const header = new Container({ flexDirection: 'row', alignItems: 'center', gap: 8, width: '100%' });
        this.speedValueText = new Text({ text: 'Speed 1.0x', fontSize: 20, color: COLORS.text });
        header.add(this.speedValueText);
        header.add(new Container({ flexGrow: 1 }));
        this.speedDefaults = button('', () => {
            this.cancelSpeedBoundEdit();
            return u.restorePlaybackSpeedDefaults();
        }, 48, new Image({ src: 'icons/restore.svg', width: 22, height: 22, color: COLORS.text }));
        header.add(this.speedDefaults);
        this.speedReset = button('1x', () => u.resetPlaybackSpeed(), 48);
        header.add(this.speedReset);
        this.speedClose = button('X', () => u.switchSubMenu(null), 40);
        header.add(this.speedClose);
        popup.add(header);
        popup.add(row);
        const step = (direction) => {
            u.playbackSpeed.step(direction);
            return u.setPlaybackSpeed(u.playbackSpeed.rate);
        };
        this.speedMinus = button('-', () => step(-1));
        row.add(this.speedMinus);
        this.speedTrack = new Container({
            width: 240, height: 16, flexDirection: 'row', alignItems: 'center',
            backgroundColor: COLORS.track, borderRadius: 8, cursor: 'pointer', overflow: 'visible',
            onPointerDown: (e) => {
                if (u.visible) UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, u);
            },
            onClick: (e) => {
                if (!this.isSpeedMenuActive()) return;
                UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, u);
                u.handlePlaybackSpeed(e, this.speedTrack, true);
            },
            onPointerMove: (e) => { if (this.isSpeedMenuActive() && e.buttons > 0) u.handlePlaybackSpeed(e, this.speedTrack, false); },
            onPointerUp: () => { if (this.isSpeedMenuActive()) u.saveSettings(); },
            onPointerLeave: (e) => { if (this.isSpeedMenuActive() && e.buttons > 0) u.saveSettings(); }
        });
        this.speedThumb = new Container({ width: 24, height: 24, backgroundColor: COLORS.thumb, borderRadius: 12, borderWidth: 1.5, borderColor: 0xffffff });
        this.speedTrack.add(this.speedThumb);
        row.add(this.speedTrack);
        this.speedPlus = button('+', () => step(1));
        row.add(this.speedPlus);

        const boundsRow = new Container({ flexDirection: 'row', gap: 8, width: '100%' });
        popup.add(boundsRow);
        this.speedBoundFields = {};
        this.speedBoundTexts = {};
        for (const bound of ['min', 'max']) {
            const group = new Container({ flexDirection: 'column', gap: 4, width: 172 });
            group.add(new Text({ text: bound === 'min' ? 'Min' : 'Max', fontSize: 16, color: COLORS.textDim }));
            const label = new Text({ text: String(u.playbackSpeed[bound]), fontSize: 20, color: COLORS.text });
            this.speedBoundTexts[bound] = label;
            this.speedBoundFields[bound] = button('', () => this.editSpeedBound(bound), 172, label);
            group.add(this.speedBoundFields[bound]);
            boundsRow.add(group);
        }

        this.speedKeypad = new Container({ display: 'none', flexDirection: 'column', gap: 8, width: '100%' });
        popup.add(this.speedKeypad);
        this.speedInputText = new Text({ text: '', fontSize: 20, color: COLORS.text, width: '100%' });
        this.speedKeypad.add(this.speedInputText);
        this.speedKeyButtons = {};
        for (const keys of [['7', '8', '9', 'Back'], ['4', '5', '6', 'Clear'], ['1', '2', '3', 'Cancel'], ['0', '.', 'OK']]) {
            const keyRow = new Container({ flexDirection: 'row', gap: 8, width: '100%' });
            this.speedKeypad.add(keyRow);
            for (const key of keys) {
                const keyButton = button(key, () => this.handleSpeedKey(key), key === 'OK' ? 172 : 82);
                this.speedKeyButtons[key] = keyButton;
                keyRow.add(keyButton);
            }
        }
        if (typeof window !== 'undefined') {
            // Capture before InputManager's bubbling shortcut listeners, including in XR.
            const options = { capture: true, signal: u.abortController?.signal };
            for (const type of ['keydown', 'keyup']) {
                window.addEventListener(type, (e) => this.handleSpeedKeyboard(e), options);
            }
        }
    }

    handleSpeedKeyboard(e) {
        if (!this.isSpeedMenuActive() || !this.speedBoundEdit) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        if (e.type !== 'keydown') return;
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
            this.speedBoundEdit.replace = true;
            return;
        }
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        const key = { Enter: 'OK', Escape: 'Cancel', Backspace: 'Back', Delete: 'Clear' }[e.key] || e.key;
        return this.handleSpeedKey(key);
    }

    isSpeedMenuActive() {
        return this.uiManager.visible && this.uiManager.activeSubMenu === this.speedMenu;
    }

    editSpeedBound(bound) {
        if (!this.isSpeedMenuActive()) return;
        const applied = this.uiManager.setPlaybackSpeed(this.uiManager.playbackSpeed[bound]);
        this.speedBoundEdit = { bound, text: String(this.uiManager.playbackSpeed[bound]), replace: true };
        this.speedKeypad.setProperties({ display: 'flex' });
        this.updateSpeedInputText();
        return applied;
    }

    updateSpeedInputText() {
        const edit = this.speedBoundEdit;
        this.speedInputText.setProperties({ text: `${edit.bound === 'min' ? 'Min' : 'Max'}: ${edit.text || '_'}` });
    }

    cancelSpeedBoundEdit() {
        this.speedBoundEdit = null;
        this.speedKeypad.setProperties({ display: 'none' });
    }

    async handleSpeedKey(key) {
        if (!this.isSpeedMenuActive() || !this.speedBoundEdit) return;
        const edit = this.speedBoundEdit;
        if (key === 'Cancel') {
            this.cancelSpeedBoundEdit();
            return;
        }
        if (key === 'OK') {
            const value = /^(?:\d+(?:\.\d*)?|\.\d+)$/.test(edit.text) ? Number(edit.text) : NaN;
            const speed = this.uiManager.playbackSpeed;
            if (await this.uiManager.setPlaybackSpeedBounds(edit.bound === 'min' ? value : speed.min,
                edit.bound === 'max' ? value : speed.max)) {
                if (this.speedBoundEdit === edit) this.cancelSpeedBoundEdit();
                this.sync({});
            } else {
                this.uiManager.showNotification('Use 0.01 to 100; Min < Max');
            }
            return;
        }
        if (key === 'Clear') edit.text = '';
        else if (key === 'Back') edit.text = edit.text.slice(0, -1);
        else if (/^[0-9.]$/.test(key)) {
            if (edit.replace) edit.text = '';
            if (key !== '.' || !edit.text.includes('.')) edit.text += key;
        } else return;
        edit.replace = false;
        this.updateSpeedInputText();
    }

    setupTimeRow() {
        this.timeRow = new Container({
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'space-between',
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
        const fileClip = new Container({ width: 380, height: 24, overflow: 'hidden' });
        this.fileNameText = new Text({ 
            text: "", 
            fontSize: 14, 
            color: COLORS.textDim,
            width: 2000, // Large width to prevent wrapping
            wordBreak: 'keep-all'
        });
        fileClip.add(this.fileNameText);
        this.timeRow.add(fileClip);

        this.videoTimeText = new Text({ text: "", fontSize: 14, color: COLORS.textDim, width: 200, textAlign: 'right' });
        this.timeRow.add(this.videoTimeText);
    }

    formatTime(seconds) {
        if (isNaN(seconds) || seconds === null) return "00:00:00";
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    sync(params) {
        const speed = this.uiManager.playbackSpeed;
        const speedKey = `${speed.rate}/${speed.min}/${speed.max}`;
        if (this._lastSpeedKey !== speedKey) {
            const displayRate = Number(speed.rate.toPrecision(4));
            const rateText = Number.isInteger(displayRate) ? displayRate.toFixed(1) : String(displayRate);
            this.speedValueText.setProperties({ text: `Speed ${rateText}x` });
            this.speedReadoutText.setProperties({ text: `${rateText}x` });
            this.speedBoundTexts.min.setProperties({ text: String(speed.min) });
            this.speedBoundTexts.max.setProperties({ text: String(speed.max) });
            const percent = Math.max(0, Math.min(1, (speed.rate - speed.min) / (speed.max - speed.min)));
            this.speedThumb.setProperties({ marginLeft: percent * (240 - 24) });
            this._lastSpeedKey = speedKey;
        }
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
            const isVideo = params.isVideoActive ?? (!!this.uiManager.stereoPlayer.videoElement);
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
                    const margin = params.videoProgress * (350 - 24);
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

        if (params.volume !== undefined) {
            const volMargin = params.volume * (100 - 24);
            if (Math.abs((this._lastVolMargin || 0) - volMargin) > 0.1) {
                this.volumeThumb.setProperties({ marginLeft: volMargin });
                this._lastVolMargin = volMargin;
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
