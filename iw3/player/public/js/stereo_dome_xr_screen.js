/* global XRMediaBinding */
import * as THREE from "three";
import { StereoDomeScreen } from "./stereo_dome_screen.js";

/**
 * Experimental Dome Screen using WebXR Layers API (XRMediaBinding).
 * Provides high-quality video playback by bypassing the Three.js fragment shader.
 */
class StereoDomeXRScreen extends StereoDomeScreen {
    constructor(options = {}) {
        super(null, 1.0, { ...options, screenType: 'dome_xr', domeRadius: 10.0 });

        this.debugLogInstance = options.debugLogInstance || null;
        this.supportedSettings = [
            'screen_zoom', 'screen_tx', 'screen_ty', 
            'screen_roll', 'screen_tilt'
        ];

        this.xrLayer = null;
        this.videoElement = null;
        this.xrSession = null;
        this.binding = null;
    }

    /**
     * Overrides dome mesh creation since XRMediaBinding manages its own layer rendering.
     */
    _initDomeGeometry() {}

    /**
     * Initializes the XR Layer. Must be called after XRSession starts.
     * @param {XRSession} session 
     * @param {HTMLVideoElement} video 
     * @param {XRReferenceSpace} [refSpace]
     */
    async initXRLayer(session, video, refSpace = null) {
        if (!session || !video || !window.XRMediaBinding) {
            this.debugLogInstance?.log(`[XR-Layers] Missing dependency: session=${!!session}, video=${!!video}, binding=${!!window.XRMediaBinding}`);
            return;
        }

        this.xrSession = session;
        this.videoElement = video;
        this.binding = new XRMediaBinding(session);

        try {
            this.debugLogInstance?.log(`[XR-Layers] Creating Equirect Layer: ${video.videoWidth}x${video.videoHeight}`);
            
            const space = refSpace || await session.requestReferenceSpace('local');

            this.xrLayer = this.binding.createEquirectLayer(video, {
                space: space,
                layout: "stereo-left-right",
                radius: this.DOME_RADIUS,
                centralHorizontalAngle: Math.PI,
                upperVerticalAngle: Math.PI / 2,
                lowerVerticalAngle: -Math.PI / 2
            });

            this.debugLogInstance?.log(`[XR-Layers] Layer created successfully: ${!!this.xrLayer}`);
        } catch (e) {
            this.debugLogInstance?.log(`[XR-Layers] ERROR: ${e.message}`);
            console.error("Failed to create XRMediaBinding layer:", e);
        }
    }

    destroyXRLayer() {
        if (this.xrLayer) {
            try {
                this.xrLayer.destroy();
            } catch {
                // Ignore cleanup errors
            }
            this.xrLayer = null;
        }
        this.videoElement = null;
        this.binding = null;
        this.xrSession = null;
    }

    setEyeSeparation(_val) { return false; }
    slideEyeSeparation(_direction, _delta = 16) { return false; }

    updatePositions(leftOffset = null, rightOffset = null) {
        if (leftOffset) this.lastKnownLeftOffset.copy(leftOffset);
        if (rightOffset) this.lastKnownRightOffset.copy(rightOffset);

        this._applyDomeTransform();
        this.scale.set(-1, 1, 1);

        if (this.xrLayer && this.xrSession && window.XRRigidTransform) {
            try {
                this.updateMatrixWorld(true);
                
                const position = new THREE.Vector3();
                const quaternion = new THREE.Quaternion();
                this.matrixWorld.decompose(position, quaternion, new THREE.Vector3());

                this.xrLayer.transform = new XRRigidTransform(
                    { x: position.x, y: position.y, z: position.z },
                    { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w }
                );
            } catch {
                // Silently fail during sync to avoid crashing the render loop during transitions
            }
        }
    }

    updateEyeVisibility(_isXR) {}
    updateTexture(_texture, _aspectRatio, _format) {}
    clearScreen() {}
}

export { StereoDomeXRScreen };
