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

console.log("frame_pacer OK");
