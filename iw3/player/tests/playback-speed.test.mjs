import test from 'node:test';
import assert from 'node:assert/strict';
import { PlaybackSpeed } from '../public/js/playback_speed.js';

test('bounds accept exact finite decimals from .01 to 100 without requiring 1x', () => {
    const speed = new PlaybackSpeed();
    assert.deepEqual(speed.toJSON(), { min: 0.1, max: 16, rate: 1 });
    for (const [min, max] of [[0, 2], [-1, 2], [NaN, 2], [.5, Infinity], [2, .5], [1, 1], [.009, 2], [.5, 101], ['.5', 2]]) {
        assert.equal(speed.setBounds(min, max), false);
        assert.deepEqual(speed.toJSON(), { min: .1, max: 16, rate: 1 });
    }
    for (const [min, max] of [[.01, 100], [.0625, .99], [2.12345, 32]]) {
        assert.equal(speed.setBounds(min, max), true);
        assert.equal(speed.min, min);
        assert.equal(speed.max, max);
        assert.ok(speed.rate >= min && speed.rate <= max);
        assert.deepEqual(new PlaybackSpeed(speed.toJSON()).toJSON(), speed.toJSON());
    }
    assert.deepEqual(new PlaybackSpeed(null).toJSON(), { min: .1, max: 16, rate: 1 });
    assert.deepEqual(new PlaybackSpeed({min: .5, max: 2, rate: 1.7}).toJSON(), {min: .1, max: 16, rate: 1.7});
    assert.deepEqual(new PlaybackSpeed({min: .5, max: 3, rate: 1.7}).toJSON(), {min: .5, max: 3, rate: 1.7});
});

test('deliberately choosing the old default pair survives a save/reload', () => {
    const speed = new PlaybackSpeed();
    speed.setBounds(0.5, 2);
    const restored = new PlaybackSpeed(speed.toJSON());
    assert.equal(restored.min, 0.5);
    assert.equal(restored.max, 2);
});

test('rate retains precision, clamps, and steps proportionally without decimal drift', () => {
    const speed = new PlaybackSpeed();
    speed.setRate(1.26);
    assert.equal(speed.rate, 1.26);
    speed.step(-1);
    assert.ok(Math.abs(speed.rate - 1.26 / 1.1) < 1e-12);
    speed.setRate(99);
    assert.equal(speed.rate, 16);
    speed.setRate(-5);
    assert.equal(speed.rate, .1);
    speed.setRate(NaN);
    assert.equal(speed.rate, .1);
    speed.setBounds(.01, 100);
    speed.setRate(.0625);
    assert.equal(speed.rate, .0625);
    speed.step(-1);
    assert.ok(Math.abs(speed.rate - .0625 / 1.1) < 1e-12);
    speed.step(1);
    assert.ok(Math.abs(speed.rate - .0625) < 1e-12);
});

test('applies rate and pitch preservation without playing or seeking', () => {
    const speed = new PlaybackSpeed({ rate: 1.7 });
    assert.equal(speed.applyTo(null), true);
    for (const pitchKey of ['preservesPitch', 'mozPreservesPitch', 'webkitPreservesPitch']) {
        const video = { playbackRate: 1, defaultPlaybackRate: 1, [pitchKey]: false, paused: true, currentTime: 23 };
        assert.equal(speed.applyTo(video), true);
        assert.equal(video.playbackRate, 1.7);
        assert.equal(video.defaultPlaybackRate, 1.7);
        assert.equal(video[pitchKey], true);
        assert.equal(video.paused, true);
        assert.equal(video.currentTime, 23);
    }
});
