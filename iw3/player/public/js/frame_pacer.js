// Jitter buffer for decoded VideoFrames.
// Frames are released at their capture spacing (frame.timestamp, microseconds on the sender clock),
// delayed by bufferMs, so a network burst does not collapse into a skipped frame.

// Decoders have a small output frame pool; holding too many frames stalls decoding.
const MAX_QUEUE = 8;
// Reserve headroom for minor bursts so high-FPS streams clamp before hitting MAX_QUEUE eviction.
const QUEUE_HEADROOM = 2;
// Lets the clock offset rise slowly (ms per pushed frame) so sender/receiver clock drift
// cannot stall playback. 0.05 ms/frame at 24fps tolerates ~1200 ppm drift.
const OFFSET_CREEP_MS = 0.05;
const INTERVAL_EMA_ALPHA = 0.2;
const MAX_VALID_DT_MS = 500;
const EFFECTIVE_ROUND_MS = 5;

export class FramePacer {
    constructor(bufferMs) {
        this.bufferMs = Math.max(0, Number(bufferMs) || 0);
        this.queue = [];
        this.offsetMs = null;
        this._lastTimestampUs = null;
        this.frameIntervalMs = null;
    }

    setBufferMs(bufferMs) {
        this.bufferMs = Math.max(0, Number(bufferMs) || 0);
    }

    getEffectiveBufferMs() {
        if (this.bufferMs <= 0) {
            return 0;
        }
        if (this.frameIntervalMs === null) {
            return this.bufferMs;
        }
        const maxSafeMs = Math.max(0, (MAX_QUEUE - QUEUE_HEADROOM) * this.frameIntervalMs);
        return Math.min(this.bufferMs, maxSafeMs);
    }

    getStatus() {
        const configuredMs = Math.round(this.bufferMs);
        if (configuredMs <= 0) {
            return {
                configuredMs: 0,
                effectiveMs: 0,
                isClamped: false,
                queueLength: this.queue.length,
                frameIntervalMs: this.frameIntervalMs,
            };
        }
        const rawEffectiveMs = this.getEffectiveBufferMs();
        const roundedMs = Math.round(rawEffectiveMs / EFFECTIVE_ROUND_MS) * EFFECTIVE_ROUND_MS;
        const effectiveMs = Math.min(configuredMs, Math.max(0, roundedMs));
        return {
            configuredMs,
            effectiveMs,
            isClamped: effectiveMs <= configuredMs - EFFECTIVE_ROUND_MS,
            queueLength: this.queue.length,
            frameIntervalMs: this.frameIntervalMs,
        };
    }

    push(frame, nowMs) {
        if (this._lastTimestampUs !== null) {
            const dtMs = (frame.timestamp - this._lastTimestampUs) / 1000;
            if (dtMs > 0 && dtMs <= MAX_VALID_DT_MS) {
                this.frameIntervalMs =
                    this.frameIntervalMs === null
                        ? dtMs
                        : this.frameIntervalMs + INTERVAL_EMA_ALPHA * (dtMs - this.frameIntervalMs);
            }
        }
        this._lastTimestampUs = frame.timestamp;

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
        const effectiveBufferMs = this.getEffectiveBufferMs();
        let due = null;
        while (
            this.queue.length > 0 &&
            this.queue[0].timestamp / 1000 + this.offsetMs + effectiveBufferMs <= nowMs
        ) {
            if (due) {
                due.close();
            }
            due = this.queue.shift();
        }
        // Safety valve if queue reaches capacity before EMA catches up to a sudden FPS jump.
        if (!due && this.queue.length >= MAX_QUEUE) {
            due = this.queue.shift();
        }
        return due;
    }

    takeLatestAndClear() {
        let latest = null;
        if (this.queue.length > 0) {
            latest = this.queue.pop();
        }
        this.clear();
        return latest;
    }

    clear() {
        for (const frame of this.queue) {
            frame.close();
        }
        this.queue = [];
        this.offsetMs = null;
        this._lastTimestampUs = null;
        this.frameIntervalMs = null;
    }
}
