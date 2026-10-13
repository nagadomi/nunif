import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, load } from './ui-fixture.mjs';

test('loadMedia applies selected rate to new videos, metadata reloads and forced-format reuse', async () => {
    const { u, storage } = await fixture();
    const made = [];
    const document = { createElement() {
        const video = { playbackRate: 1, defaultPlaybackRate: 1, preservesPitch: false, videoWidth: 1920, videoHeight: 1080,
            pause() {}, load() {}, currentTime: 0 };
        made.push(video);
        return video;
    } };
    const THREE = { Group: class {}, VideoTexture: class {}, SRGBColorSpace: 'srgb' };
    const StereoPlayer = await load('stereo_player', 'StereoPlayer', { document, THREE, storage });
    const p = Object.create(StereoPlayer.prototype);
    const files = [{ type: 'video', name: 'one.mp4', path: '/one.mp4' }, { type: 'video', name: 'two.mp4', path: '/two.mp4' }, { type: 'image', name: 'still.jpg', path: '/still.jpg' }];
    Object.assign(p, { uiManager: u, currentLoadId: 0,
        galleryManager: { playbackGallery: files, setPlayingItem() {}, getCurrentItem: () => null },
        stereoScreen: { displayScreenLeft: { material: {} }, displayScreenRight: { material: {} }, updateTexture() {} },
        debugLogInstance: { log() {} }, renderer: { capabilities: { getMaxAnisotropy: () => 1 } },
        loadSubtitles() {}, textureCache: new Map([['/still.jpg', { texture: {}, aspectRatio: 1 }]]), prefetchImages() {} });
    u.stereoPlayer = p;
    Object.assign(u, { updateSubtitleText() {}, getFileConfig: async () => null, updateFileConfig() {}, hideRecentButton() {} });
    u.renderSettings.values = {};
    await u.setPlaybackSpeed(1.8);
    await p.loadMedia(0);
    assert.equal(p.videoElement.playbackRate, 1.8);
    assert.equal(p.videoElement.preservesPitch, true);
    p.videoElement.playbackRate = 1;
    p.videoElement.onloadedmetadata();
    assert.equal(p.videoElement.playbackRate, 1.8);
    await p.loadMedia(1);
    assert.equal(made.length, 2);
    assert.equal(p.videoElement.playbackRate, 1.8);
    await p.loadMedia(1, 'sbs');
    assert.equal(made.length, 2);
    assert.equal(p.videoElement.playbackRate, 1.8);
    await p.loadMedia(2);
    assert.equal(p.videoElement, null);
    assert.equal(u.playbackSpeed.rate, 1.8);
    await p.loadMedia(0);
    assert.equal(p.videoElement.playbackRate, 1.8);
});
