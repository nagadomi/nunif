// Jitter buffer for decoded VideoFrames.
// Frames are released at their capture spacing (frame.timestamp, microseconds on the sender clock),
// delayed by bufferMs, so a network burst does not collapse into a skipped frame.

// Decoders have a small output frame pool; holding too many frames stalls decoding.
const MAX_QUEUE = 8;
// Lets the clock offset rise slowly (ms per pushed frame) so sender/receiver clock drift
// cannot stall playback. 0.05 ms/frame at 24fps tolerates ~1200 ppm drift.
const OFFSET_CREEP_MS = 0.05;

export class FramePacer {
    constructor(bufferMs) {
        this.bufferMs = bufferMs;
        this.queue = [];
        this.offsetMs = null;
    }

    push(frame, nowMs) {
        // Fastest observed transit = clock offset + minimum network delay.
        const transitMs = nowMs - frame.timestamp / 1000;
        this.offsetMs = this.offsetMs === null ? transitMs : Math.min(transitMs, this.offsetMs + OFFSET_CREEP_MS);
        this.queue.push(frame);
        while (this.queue.length > MAX_QUEUE) {
            this.queue.shift().close();
        }
    }

    // Returns the newest frame that is due, closing older due frames, or null if none is due yet.
    pop(nowMs) {
        let due = null;
        while (this.queue.length > 0 && this.queue[0].timestamp / 1000 + this.offsetMs + this.bufferMs <= nowMs) {
            if (due) {
                due.close();
            }
            due = this.queue.shift();
        }
        return due;
    }

    clear() {
        for (const frame of this.queue) {
            frame.close();
        }
        this.queue = [];
        this.offsetMs = null;
    }
}
