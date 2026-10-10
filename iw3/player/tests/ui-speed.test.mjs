import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, event, Widget, load } from './ui-fixture.mjs';

test('main VR controls provide minus/plus/reset, live slider and hidden-menu guards', async () => {
    const { u, cache, UIUtils } = await fixture();
    const MainMenu = await load('menu_main', 'MainMenu', { UIUtils });
    const menu = new MainMenu(u, 'NotoSans');
    menu.speedReadout.props.onClick(event());
    menu.speedPlus.props.onClick(event());
    assert.equal(u.playbackSpeed.rate, 1.1);
    menu.speedMinus.props.onClick(event());
    assert.equal(u.playbackSpeed.rate, 1);
    await u.setPlaybackSpeedBounds(.25, 3);
    menu.speedTrack.props.onPointerMove(event(.5, 1));
    assert.equal(u.playbackSpeed.rate, 3);
    menu.speedTrack.props.onPointerUp(event());
    assert.equal(cache.playback_speed.rate, 3);
    menu.sync({});
    assert.equal(menu.speedValueText.props.text, 'Speed 3.0x');
    assert.equal(menu.speedThumb.props.marginLeft, 216);
    menu.speedReset.props.onClick(event());
    assert.equal(u.playbackSpeed.rate, 1);
    menu.speedTrack.props.onClick(event(-.5));
    assert.equal(u.playbackSpeed.rate, .25);
    u.visible = false;
    menu.speedPlus.props.onClick(event());
    menu.speedReset.props.onClick(event());
    menu.speedTrack.props.onPointerMove(event(.5, 1));
    assert.equal(u.playbackSpeed.rate, .25);
});

test('Render Settings has no speed options and reset/sync never touches speed', async () => {
    const { u, UIUtils, storage } = await fixture();
    const RenderSettingsMenu = await load('menu_render_settings', 'RenderSettingsMenu', { UIUtils, storage });
    const menu = new RenderSettingsMenu(u, 'NotoSans');
    assert.equal(menu.speedSliders, undefined);
    assert.equal(menu.speedBoundsText, undefined);
    const calls = [];
    u.onSliderChange = (id, value) => calls.push([id, value]);
    delete u.playbackSpeed;
    menu.sync();
    menu.reset();
    assert.equal(calls.length, 5);
    assert.ok(calls.every(([id]) => id.startsWith('render_')));
    const text = widget => [widget.props.text || '', ...widget.children.map(text)].join(' ');
    assert.doesNotMatch(text(menu.container), /speed|audio support/i);
});

test('unsupported native rate preserves previous actual speed outside bounds without retry loops', async () => {
    const { u, cache } = await fixture();
    let actualRate = 1.7;
    let attempts = 0;
    u.stereoPlayer.videoElement = {
        get playbackRate() { return actualRate; },
        set playbackRate(rate) { attempts++; if (rate < .0625 || rate > 16) throw new Error('Unsupported'); actualRate = rate; },
        defaultPlaybackRate: 1.7
    };
    await u.setPlaybackSpeed(1.7);
    await u.setPlaybackSpeedBounds(32, 100);
    assert.equal(u.playbackSpeed.rate, 1.7);
    assert.equal(actualRate, 1.7);
    assert.deepEqual(cache.playback_speed, {min: 32, max: 100, rate: 1.7});
    assert.match(u.notification, /unsupported/i);
    const before = attempts;
    u.applyPlaybackSpeed();
    assert.equal(attempts, before + 1, 'one application, no fallback/clamp recursion');
    assert.equal(actualRate, 1.7);
    await u.resetPlaybackSpeed();
    assert.deepEqual(cache.playback_speed, {min: 1, max: 100, rate: 1});
});

test('manager persists exact live bounds, rounds slider only, reloads settings and expands 1x reset', async () => {
    const { u, cache } = await fixture();
    await u.setPlaybackSpeed(1.6);
    assert.equal(u.stereoPlayer.videoElement.playbackRate, 1.6);
    assert.deepEqual(cache.playback_speed, {min: .1, max: 16, rate: 1.6});
    await u.handlePlaybackSpeed(event(-.5), new Widget(), false);
    assert.equal(u.playbackSpeed.rate, .1);
    assert.equal(cache.playback_speed.rate, 1.6);
    await u.setPlaybackSpeedBounds(.0625, 1.2345);
    await u.handlePlaybackSpeed(event(-.5), new Widget());
    assert.equal(u.playbackSpeed.rate, .0625, 'exact minimum even when slider rounds');
    await u.handlePlaybackSpeed(event(.5), new Widget());
    assert.equal(u.playbackSpeed.rate, 1.2345, 'exact maximum even when slider rounds');
    await u.handlePlaybackSpeed(event(0), new Widget());
    assert.equal(u.playbackSpeed.rate, .65);
    assert.equal(await u.setPlaybackSpeedBounds(0, 2), false);
    assert.equal(u.playbackSpeed.min, .0625);
    await u.setPlaybackSpeedBounds(.01, .25);
    await u.resetPlaybackSpeed();
    assert.deepEqual(cache.playback_speed, {min: .01, max: 1, rate: 1});
    await u.setPlaybackSpeedBounds(2.12345, 32);
    await u.resetPlaybackSpeed();
    assert.deepEqual(cache.playback_speed, {min: 1, max: 32, rate: 1});
    u.playbackSpeed.setRate(8);
    await u.loadSettings();
    assert.equal(u.playbackSpeed.rate, 1);
    assert.equal(u.stereoPlayer.videoElement.playbackRate, 1);
    u.visible = false;
    await u.handlePlaybackSpeed(event(.5), new Widget());
    assert.equal(u.playbackSpeed.rate, 1);
    await u.saveSettings();
    assert.deepEqual(cache.playback_speed, {min: 1, max: 32, rate: 1});
});
