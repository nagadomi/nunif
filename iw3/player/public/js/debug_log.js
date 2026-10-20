import * as THREE from "three";


class DebugLog {
    constructor(scene, maxLogLines = 20) {
        this.scene = scene;
        this.maxLogLines = maxLogLines;
        this._debugLog = [];

        this.canvasWidth = 512;
        this.canvasHeight = this.maxLogLines * 22;

        this.debugCanvas = document.createElement('canvas');
        this.debugCanvas.width = this.canvasWidth;
        this.debugCanvas.height = this.canvasHeight;
        this.debugContext = this.debugCanvas.getContext('2d');

        this.debugTexture = new THREE.CanvasTexture(this.debugCanvas);
        const debugMaterial = new THREE.SpriteMaterial({ map: this.debugTexture, depthTest: false, depthWrite: false });
        this.debugSprite = new THREE.Sprite(debugMaterial);

        this.debugSprite.scale.set(this.canvasWidth / 250, this.canvasHeight / 250, 1);
        this.debugSprite.position.set(0, 1.2, -2);
        this.debugSprite.renderOrder = 99999;

        this.debugSprite.visible = true;
        this.debugSprite.layers.enable(1); // Left Eye
        this.debugSprite.layers.enable(2); // Right Eye
        this.scene.add(this.debugSprite);
        this.update();
    }

    setVisible(val) {
        this.debugSprite.visible = !!val;
    }

    log(s) {
        console.log(s);
        this._debugLog.push(s);
        this.update();
    }

    update() {
        if (!this.debugContext) return;

        this.debugContext.clearRect(0, 0, this.debugCanvas.width, this.debugCanvas.height);
        this.debugContext.font = '14px monospace';
        this.debugContext.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.debugContext.fillRect(0, 0, this.canvasWidth, this.canvasHeight);
        this.debugContext.fillStyle = 'white';
        this.debugContext.textAlign = 'left';
        this.debugContext.textBaseline = 'top';

        const startIndex = Math.max(0, this._debugLog.length - this.maxLogLines);
        for (let i = 0; i < this.maxLogLines; i++) {
            if (startIndex + i < this._debugLog.length) {
                const line = this._debugLog[startIndex + i];
                // Simple wrapping or truncation if too long
                const displayLine = line.length > 60 ? line.substring(0, 57) + "..." : line;
                this.debugContext.fillText(displayLine, 5, 5 + (i * 22));
            }
        }
        this.debugTexture.needsUpdate = true;
    }
}

export { DebugLog };
