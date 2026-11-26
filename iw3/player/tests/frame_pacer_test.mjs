// Run: node iw3/player/tests/frame_pacer_test.mjs
import assert from "node:assert/strict";
import { FramePacer } from "../public/js/frame_pacer.js";

const frame = (tsMs) => ({ timestamp: tsMs * 1000, closed: false, close() { this.closed = true; } });

// 24fps sender; frames 2 and 3 arrive together after a 60 ms Wi-Fi stall.
const pacer = new FramePacer(80);
const f = [0, 1, 2, 3, 4].map((i) => frame(1000 + i * 41.7));
const arrival = [10, 51.7, 153.4, 153.5, 176.8]; // transit 10 ms, except the stalled pair
const shown = [];
let next = 0;
for (let now = 0; now < 400; now += 11.1) { // ~90Hz headset
    while (next < f.length && arrival[next] <= now) {
        pacer.push(f[next], arrival[next] + 1000); // receiver clock: sender ts 1000 at local 0
        next++;
    }
    const out = pacer.pop(now + 1000);
    if (out) shown.push({ i: f.indexOf(out), now });
}
// Every frame is shown once, none dropped despite the burst.
assert.deepEqual(shown.map((s) => s.i), [0, 1, 2, 3, 4]);
assert.ok(f.every((x) => !x.closed));
// And they come out evenly spaced (within one 11.1 ms refresh of 41.7 ms).
for (let k = 1; k < shown.length; k++) {
    const gap = shown[k].now - shown[k - 1].now;
    assert.ok(Math.abs(gap - 41.7) <= 11.2, `gap ${gap}`);
}

// Without enough buffer, a late frame is skipped in favour of the newer one, and the skipped one is closed.
const tight = new FramePacer(0);
const a = frame(0), b = frame(41.7);
tight.push(a, 100); tight.push(b, 100);
assert.equal(tight.pop(200), b);
assert.ok(a.closed);

// clear() releases held frames.
const c = frame(0);
tight.push(c, 0);
tight.clear();
assert.ok(c.closed);
assert.equal(tight.pop(1e9), null);

// 60fps + bufferMs=200: automatically clamps effective buffer to ~100ms without freezing.
{
    const pacer60 = new FramePacer(200);
    const interval = 1000 / 60;
    const frames = Array.from({ length: 40 }, (_, i) => frame(i * interval));
    const shownIndices = [];
    let pushIdx = 0;
    for (let now = 0; now < 900; now += 1000 / 90) {
        while (pushIdx < frames.length && pushIdx * interval <= now) {
            pacer60.push(frames[pushIdx], pushIdx * interval + 5);
            pushIdx++;
        }
        const out = pacer60.pop(now + 5);
        if (out) {
            shownIndices.push(frames.indexOf(out));
        }
    }
    assert.ok(shownIndices.length >= 30, `expected smooth playback, got ${shownIndices.length} frames`);
    assert.equal(shownIndices[shownIndices.length - 1], frames.length - 1);
    const status = pacer60.getStatus();
    assert.equal(status.configuredMs, 200);
    assert.equal(status.isClamped, true);
    assert.ok(status.effectiveMs >= 95 && status.effectiveMs <= 105, `effectiveMs=${status.effectiveMs}`);
}

// 240fps + bufferMs=100: clamps effective buffer to ~25ms and keeps advancing.
{
    const pacer240 = new FramePacer(100);
    const interval = 1000 / 240;
    const frames = Array.from({ length: 80 }, (_, i) => frame(i * interval));
    const shownIndices = [];
    let pushIdx = 0;
    for (let now = 0; now < 450; now += 1000 / 90) {
        while (pushIdx < frames.length && pushIdx * interval <= now) {
            pacer240.push(frames[pushIdx], pushIdx * interval + 2);
            pushIdx++;
        }
        const out = pacer240.pop(now + 2);
        if (out) {
            shownIndices.push(frames.indexOf(out));
        }
    }
    assert.ok(shownIndices.length >= 20);
    assert.equal(shownIndices[shownIndices.length - 1], frames.length - 1);
    const status = pacer240.getStatus();
    assert.equal(status.isClamped, true);
    assert.ok(status.effectiveMs >= 20 && status.effectiveMs <= 30, `effectiveMs=${status.effectiveMs}`);
}

// 60fps + bufferMs=100 with minor capture jitter does not flicker into isClamped=true.
{
    const pacerJitter = new FramePacer(100);
    let ts = 0;
    const dts = [16.4, 16.9, 16.3, 16.8, 16.5, 16.7, 16.4, 16.9];
    for (const dt of dts) {
        ts += dt;
        pacerJitter.push(frame(ts), ts + 5);
    }
    const status = pacerJitter.getStatus();
    assert.equal(status.effectiveMs, 100);
    assert.equal(status.isClamped, false);
}

// Sudden FPS jump (24fps -> 120fps): queue capacity safety valve prevents stall and EMA converges.
{
    const pacerJump = new FramePacer(200);
    let ts = 0;
    for (let i = 0; i < 6; i++) {
        pacerJump.push(frame(ts), ts + 5);
        pacerJump.pop(ts + 5);
        ts += 41.7;
    }
    const shownHighFps = [];
    for (let i = 0; i < 40; i++) {
        const fHigh = frame(ts);
        pacerJump.push(fHigh, ts + 5);
        const out = pacerJump.pop(ts + 5);
        if (out) shownHighFps.push(out);
        ts += 8.33;
    }
    assert.ok(shownHighFps.length >= 25, `expected continued output after FPS jump, got ${shownHighFps.length}`);
    assert.ok(pacerJump.getEffectiveBufferMs() < 80, `expected clamped effective buffer, got ${pacerJump.getEffectiveBufferMs()}`);
}

// Dynamic setBufferMs() and takeLatestAndClear().
{
    const dyn = new FramePacer(0);
    assert.equal(dyn.getStatus().configuredMs, 0);
    dyn.setBufferMs(100);
    assert.equal(dyn.getStatus().configuredMs, 100);
    dyn.setBufferMs(50);
    assert.equal(dyn.getStatus().configuredMs, 50);
    const f1 = frame(10);
    const f2 = frame(50);
    dyn.push(f1, 15);
    dyn.push(f2, 55);
    const latest = dyn.takeLatestAndClear();
    assert.equal(latest, f2);
    assert.ok(f1.closed);
    assert.ok(!f2.closed);
    assert.equal(dyn.getStatus().queueLength, 0);
}

// clear() resets clock offset so server restart (timestamp reset near 0) recovers immediately.
{
    const reconnectPacer = new FramePacer(50);
    reconnectPacer.push(frame(100000), 100010);
    reconnectPacer.clear();
    const fresh = frame(10);
    reconnectPacer.push(fresh, 50000);
    assert.equal(reconnectPacer.pop(50040), null);
    assert.equal(reconnectPacer.pop(50060), fresh);
}

console.log("frame_pacer OK");
