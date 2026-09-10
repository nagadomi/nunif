import { readFile } from 'node:fs/promises';
import * as constants from '../public/js/constants.js';
import { PlaybackSpeed } from '../public/js/playback_speed.js';

// Only GPU, DOM and IndexedDB boundaries are doubled. Real menu/helper/manager
// bodies are executed; no replacement implementations of playback behavior.
export class Widget {
    constructor(props = {}) { this.props = props; this.children = []; }
    add(child) { this.children.push(child); return this; }
    setProperties(props) { Object.assign(this.props, props); }
    worldToLocal(point) { return point; }
}
export async function load(name, symbol, extra = {}) {
    const source = (await readFile(new URL(`../public/js/${name}.js`, import.meta.url), 'utf8'))
        .replace(/^import .*;\r?\n/gm, '')
        .replace(/^export \{.*\};?\r?$/gm, '').replace(/^export const /gm, 'const ');
    const deps = { ...constants, PlaybackSpeed, Container: Widget, Text: Widget, Image: Widget,
        THREE: { Group: class {}, MathUtils: { clamp: (x, min, max) => Math.max(min, Math.min(max, x)) } }, ...extra };
    return new Function(...Object.keys(deps), `${source}\nreturn ${symbol};`)(...Object.values(deps));
}
export async function fixture() {
    const cache = {};
    const storage = { get: (k, d = null) => cache[k] ?? d, set: async (k, v) => { cache[k] = structuredClone(v); } };
    const UIUtils = await load('ui_common', 'UIUtils');
    const UIManager = await load('ui_manager', 'UIManager', { UIUtils, storage });
    const u = Object.create(UIManager.prototype);
    Object.assign(u, { playbackSpeed: new PlaybackSpeed(), visible: true, xrPointers: [], _atLimit: {},
        stereoPlayer: { videoElement: { playbackRate: 1, defaultPlaybackRate: 1, preservesPitch: false }, galleryManager: { getCurrentItem: () => null }, stereoScreen: {} },
        syncUI() {}, showNotification(message) { this.notification = message; }, dirtyGroups: new Set(),
        allSettingsMenus: [], menuAlignment: 'right', videoRepeat: false });
    for (const k of ['screenSettings', 'colorSettings', 'environmentSettings', 'renderSettings', 'subtitleSettings']) u[k] = { load() {}, save() {} };
    return { u, cache, storage, UIUtils };
}
export const event = (x = 0, buttons = 0) => ({ pointerId: 1, buttons, point: { x, clone() { return { x }; } } });
