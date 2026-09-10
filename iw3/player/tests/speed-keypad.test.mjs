import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture, event, load} from './ui-fixture.mjs';

async function setup(extra = {}) {
    const f = await fixture();
    f.u.abortController = new AbortController();
    const MainMenu = await load('menu_main', 'MainMenu', {UIUtils: f.UIUtils, ...extra});
    const menu = new MainMenu(f.u, 'NotoSans');
    f.u.mainMenu = menu;
    f.u.speedMenu = menu.speedMenu;
    menu.speedReadout.props.onClick(event());
    return {...f, menu};
}
const press = (menu, key) => menu.speedKeyButtons[key].props.onClick(event());
async function enter(menu, bound, text) {
    menu.speedBoundFields[bound].props.onClick(event());
    await press(menu, 'Clear');
    for (const key of text) await press(menu, key);
    await press(menu, 'OK');
}

test('popup min/max fields use an in-world keypad, apply exact bounds live and persist', async () => {
    const {u, menu, cache} = await setup();
    assert.ok(menu.speedBoundFields?.min, 'popup has clickable Min');
    assert.ok(menu.speedBoundFields.max, 'popup has clickable Max');
    assert.equal(menu.speedKeypad.props.display, 'none');
    assert.ok(menu.speedMenu.container.children.includes(menu.speedKeypad));
    assert.deepEqual(Object.keys(menu.speedKeyButtons).sort(), [...'0123456789', '.', 'Back', 'Clear', 'Cancel', 'OK'].sort());
    menu.sync({});
    assert.equal(menu.speedBoundTexts.min.props.text, '0.1');
    assert.equal(menu.speedBoundTexts.max.props.text, '16');
    await enter(menu, 'min', '.0625');
    assert.equal(u.playbackSpeed.min, .0625);
    assert.equal(cache.playback_speed.min, .0625);
    assert.equal(menu.speedKeypad.props.display, 'none');
    await enter(menu, 'max', '100');
    assert.equal(cache.playback_speed.max, 100);
    await u.setPlaybackSpeed(.0625);
    menu.sync({});
    assert.equal(menu.speedReadoutText.props.text, '0.0625x');
    assert.equal(menu.speedBoundTexts.min.props.text, '0.0625');
    await enter(menu, 'min', '.01');
    await u.setPlaybackSpeed(.01);
    menu.sync({});
    assert.equal(menu.speedReadoutText.props.text, '0.01x');
    menu.speedPlus.props.onClick(event());
    assert.equal(u.playbackSpeed.rate, .011);
    menu.speedMinus.props.onClick(event());
    assert.equal(u.playbackSpeed.rate, .01);
});

test('keypad rejects invalid bounds, edits decimal/backspace, cancels and ignores hidden controls', async () => {
    const {u, menu, cache} = await setup();
    assert.ok(menu.speedBoundFields?.min);
    for (const text of ['', '.', '0', '.009', '16', '100', '101']) {
        await enter(menu, 'min', text);
        assert.equal(u.playbackSpeed.min, .1, text);
        assert.equal(menu.speedKeypad.props.display, 'flex', 'invalid draft stays editable');
        assert.equal(cache.playback_speed.min, .1, 'tap saves selected bound but invalid draft does not change bounds');
        assert.match(u.notification, /0.01.*100|min.*max/i);
        await press(menu, 'Cancel');
    }
    menu.speedBoundFields.min.props.onClick(event());
    await press(menu, 'Clear');
    for (const key of ['.', '0', '2', '.', '5', 'Back', '6', 'OK']) await press(menu, key);
    assert.equal(u.playbackSpeed.min, .026);
    menu.speedBoundFields.max.props.onClick(event());
    await press(menu, '3');
    await press(menu, 'Cancel');
    assert.equal(u.playbackSpeed.max, 16);
    menu.speedBoundFields.max.props.onClick(event());
    const beforeHide = u.playbackSpeed.rate;
    menu.speedClose.props.onClick(event());
    assert.equal(menu.speedKeypad.props.display, 'none');
    menu.speedPlus.props.onClick(event());
    menu.speedBoundFields.min.props.onClick(event());
    assert.equal(u.playbackSpeed.rate, beforeHide);
    assert.equal(menu.speedKeypad.props.display, 'none');
    menu.speedReadout.props.onClick(event());
    menu.speedBoundFields.max.props.onClick(event());
    u.switchSubMenu({container: menu.container});
    assert.equal(menu.speedKeypad.props.display, 'none');
});

test('physical keyboard edits only active popup, captures shortcuts and cleans up on abort', async () => {
    const registrations = [];
    const window = new class extends EventTarget {
        addEventListener(type, listener, options) {
            registrations.push({type, options});
            super.addEventListener(type, listener, options);
        }
    }();
    const {u, menu, cache} = await setup({window});
    assert.equal(registrations.length, 2, 'keydown/keyup capture listeners are installed');
    for (const {options} of registrations) {
        assert.equal(options.capture, true);
        assert.equal(options.signal, u.abortController.signal);
    }
    let shortcutCalls = 0;
    window.addEventListener('keydown', () => shortcutCalls++);
    const send = (key, type = 'keydown', extra = {}) => {
        const e = new Event(type, {cancelable: true});
        Object.assign(e, {key, ...extra});
        window.dispatchEvent(e);
        return e.defaultPrevented;
    };
    assert.equal(send(' '), false, 'visible popup without active field leaves shortcuts alone');
    menu.speedBoundFields.max.props.onClick(event());
    for (const key of [' ', 'ArrowLeft', 'q', 'Shift', 'Tab']) {
        assert.equal(send(key), true);
        assert.equal(send(key, 'keyup'), true);
    }
    assert.equal(shortcutCalls, 1);
    send('3'); send('2'); send('Enter');
    await new Promise(setImmediate);
    assert.equal(cache.playback_speed.max, 32);
    assert.equal(menu.speedKeypad.props.display, 'none');
    menu.speedBoundFields.min.props.onClick(event());
    send('a', 'keydown', {ctrlKey: true});
    for (const key of ['.', '0', '6', '2', '6', 'Backspace', '5', 'Enter']) send(key);
    await new Promise(setImmediate);
    assert.equal(cache.playback_speed.min, .0625);
    menu.speedBoundFields.min.props.onClick(event());
    send('Delete'); send('0'); send('Escape');
    assert.equal(menu.speedKeypad.props.display, 'none');
    assert.equal(u.playbackSpeed.min, .0625);
    menu.speedBoundFields.min.props.onClick(event());
    u.visible = false;
    assert.equal(send('1'), false);
    u.visible = true;
    u.switchSubMenu(null);
    assert.equal(send('1'), false);
    menu.speedReadout.props.onClick(event());
    menu.speedBoundFields.min.props.onClick(event());
    u.abortController.abort();
    assert.equal(send('1'), false, 'abort removed the capture listener');
});
