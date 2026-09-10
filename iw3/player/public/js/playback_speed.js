// Shared playback state: deliberately independent of file-specific settings.
export class PlaybackSpeed {
    // Configurable UI bounds, not a promise of browser/codec support.
    static MIN = 0.01;
    static MAX = 100;

    constructor(saved = {}) {
        this.min = 0.1;
        this.max = 16;
        this.rate = 1;
        const legacyDefaults = (saved?.min === 0.5 && saved?.max === 2) ||
            (saved?.min === 0.25 && saved?.max === 16);
        if (!(legacyDefaults && !saved?.customBounds)) {
            this.setBounds(saved?.min, saved?.max);
        }
        this.setRate(saved?.rate);
    }

    setBounds(min, max) {
        if (!Number.isFinite(min) || !Number.isFinite(max) ||
            min < PlaybackSpeed.MIN || max > PlaybackSpeed.MAX ||
            min >= max) return false;
        this.min = min;
        this.max = max;
        this.setRate(this.rate);
        return true;
    }

    setRate(rate) {
        if (!Number.isFinite(rate)) return;
        this.rate = Math.max(this.min, Math.min(this.max, rate));
    }

    step(direction) {
        this.setRate(Number((this.rate * Math.pow(1.1, direction)).toPrecision(15)));
    }

    applyTo(video) {
        if (!video) return true;
        const previousRate = video.playbackRate;
        const previousDefault = video.defaultPlaybackRate;
        try {
            video.playbackRate = this.rate;
            video.defaultPlaybackRate = this.rate;
            for (const key of ['preservesPitch', 'mozPreservesPitch', 'webkitPreservesPitch']) {
                if (key in video) video[key] = true;
            }
            this.rate = video.playbackRate;
            return true;
        } catch (_error) {
            // A rejected assignment must not turn into a clamped retry loop.
            // The actual rate may remain outside the user's configured bounds.
            try {
                if (video.playbackRate !== previousRate) video.playbackRate = previousRate;
                if (video.defaultPlaybackRate !== previousDefault) video.defaultPlaybackRate = previousDefault;
            } catch (_restoreError) { /* Read back actual state even if restoration fails. */ }
            if (Number.isFinite(video.playbackRate)) this.rate = video.playbackRate;
            return false;
        }
    }

    toJSON() {
        const saved = { min: this.min, max: this.max, rate: this.rate };
        // Distinguish an intentional new choice from the legacy default pair.
        if ((this.min === 0.5 && this.max === 2) || (this.min === 0.25 && this.max === 16)) {
            saved.customBounds = true;
        }
        return saved;
    }
}
