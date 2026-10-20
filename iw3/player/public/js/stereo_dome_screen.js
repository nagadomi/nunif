import * as THREE from "three";
import { LIMITS, STEREO_FORMATS, VR_CONFIG, getSettingDefault } from "./constants.js";
import { StereoBaseScreen, COMMON_FRAGMENT_HEADER, COLOR_CORRECTION_SNIPPET } from "./stereo_base_screen.js";

// --- Dome Specific Shader Snippets ---

const VERTEX_HEADER = `
    varying vec2 vFadeUv;
    uniform vec4 uTiling;
`;

const VERTEX_UV_PATCH = `
    #include <uv_vertex>
    vFadeUv = uv;
    vMapUv = uv * uTiling.xy + uTiling.zw;
`;

const FRAGMENT_HEADER = `
    varying vec2 vFadeUv;
` + COMMON_FRAGMENT_HEADER;

// Custom map patch for Dome: Blackout the back half and handle clipping
const DOME_FRAGMENT_MAP_PATCH = `
    #include <map_fragment>

    // --- VR180 Clipping Logic ---
    if (vFadeUv.x > 0.5) {
        diffuseColor.rgb = vec3(0.0);
    }
    
    if (vFadeUv.y < 0.001 || vFadeUv.y > 0.999) {
        diffuseColor.rgb = vec3(0.0);
    }
` + COLOR_CORRECTION_SNIPPET;

class StereoDomeScreen extends StereoBaseScreen {
    constructor(texture, physicalScreenHeight, options = {}) {
        super(options);

        this.screenType = options.screenType || 'dome';
        this.currentStereoFormat = STEREO_FORMATS.SBS_180;
        this.supportedSettings = [
            'screen_zoom', 'screen_eye_sep', 'screen_tx', 'screen_ty', 
            'screen_roll', 'screen_tilt'
        ];
        this.canOccludeUI = false;
        this.supportsSubtitles = false;
        this.physicalScreenHeight = 1.0; 
        
        this.currentEyeSeparation = getSettingDefault('screen_eye_sep', this.screenType);
        this.currentZoom = getSettingDefault('screen_zoom', this.screenType);
        this.currentScreenX = getSettingDefault('screen_tx', this.screenType); 
        this.currentScreenY = getSettingDefault('screen_ty', this.screenType);
        this.currentRoll = getSettingDefault('screen_roll', this.screenType);

        this.FRONT_PAN_OFFSET = 0; 
        this.DOME_RADIUS = options.domeRadius || 100.0;

        this._initDomeGeometry(texture);

        this.setZoom(this.currentZoom);
        this.updatePositions();
    }

    _initDomeGeometry(texture) {
        const geometry = new THREE.SphereGeometry(
            this.DOME_RADIUS, 
            VR_CONFIG.screenSegments * 2, 
            VR_CONFIG.screenSegments, 
            Math.PI, 
            Math.PI * 2, 
            0, 
            Math.PI
        );

        this._initMeshes(geometry);
        
        this.displayScreenLeft.scale.set(-1, 1, 1);
        this.displayScreenRight.scale.set(-1, 1, 1);

        if (texture) {
            this.updateTexture(texture, 1.0, STEREO_FORMATS.SBS_180);
        }
    }

    patchShader(shader, eyeUniforms) {
        Object.assign(shader.uniforms, this.commonUniforms, eyeUniforms);
        shader.vertexShader = VERTEX_HEADER + shader.vertexShader.replace('#include <uv_vertex>', VERTEX_UV_PATCH);
        shader.fragmentShader = FRAGMENT_HEADER + shader.fragmentShader.replace('#include <map_fragment>', DOME_FRAGMENT_MAP_PATCH);
    }

    _unitToAngle(unit) { return unit * 180; }

    setPhysicalScreenHeight(_val) {
        this.physicalScreenHeight = 1.0;
        if (this.onSizeChange) this.onSizeChange(this.physicalScreenHeight);
    }

    setZoom(z) {
        const nextZoom = THREE.MathUtils.clamp(z, -LIMITS.screen_zoomMax, -LIMITS.screen_zoomMin);
        const changed = Math.abs(nextZoom - this.currentZoom) > 1e-4;
        this.currentZoom = nextZoom;
        this.updatePositions();
        return changed;
    }

    zoomScreen(direction, delta = 16) {
        const sensitivity = 2.0 * (delta / 1000);
        return this.setZoom(this.currentZoom + direction * sensitivity);
    }

    saveInteractionStart() {
        super.saveInteractionStart();
        this.startPan = this.currentScreenX;
        this.startTilt = this.currentScreenY;
    }

    handleDrag(deltaPos) {
        const panSensitivity = 2.0;
        const tySensitivity = 2.0; 

        const tx = this.startPan - deltaPos.x * panSensitivity;
        const ty = this.startTilt + deltaPos.y * tySensitivity; 
        if (this.onTransformChange) {
            this.onTransformChange('screen_tx', tx);
            this.onTransformChange('screen_ty', ty);
        }
    }

    translateScreen(dx, dy) {
        const tx = THREE.MathUtils.clamp(this.currentScreenX - dx * 0.5, -0.5, 0.5);
        const ty = THREE.MathUtils.clamp(this.currentScreenY + dy * 2.0, -0.5, 0.5);
        if (this.onTransformChange) {
            this.onTransformChange('screen_tx', tx);
            this.onTransformChange('screen_ty', ty);
        } else {
            this.setTranslation(tx, ty);
        }
    }

    setTranslation(x, y) {
        this.currentScreenX = THREE.MathUtils.clamp(x, -0.5, 0.5); 
        this.currentScreenY = THREE.MathUtils.clamp(y, -0.5, 0.5); 
        this.updatePositions();
    }

    setRoll(val) {
        this.currentRoll = THREE.MathUtils.clamp(val, LIMITS.screen_rollMin, LIMITS.screen_rollMax);
        this.updatePositions();
    }

    _applyDomeTransform() {
        const panDeg = this._unitToAngle(this.currentScreenX) + this.FRONT_PAN_OFFSET;
        const rollDeg = this.currentRoll; 

        this.rotation.set(0, THREE.MathUtils.degToRad(panDeg), THREE.MathUtils.degToRad(rollDeg), 'YXZ');

        let zShift = (this.currentZoom + 3.0);
        zShift *= (zShift > 0) ? (this.DOME_RADIUS * 0.2) : (this.DOME_RADIUS * 0.1);
        
        const yPos = this.currentScreenY * this.DOME_RADIUS;
        this.position.set(0, yPos, zShift);
    }

    updatePositions(leftOffset = null, rightOffset = null) {
        super.updatePositions(leftOffset, rightOffset);
        this._applyDomeTransform();

        const eyeOffsetRad = THREE.MathUtils.degToRad(this.currentEyeSeparation * 10);
        this.displayScreenLeft.rotation.y = eyeOffsetRad;
        this.displayScreenRight.rotation.y = -eyeOffsetRad;

        if (leftOffset && rightOffset) {
            this.displayScreenLeft.position.copy(leftOffset);
            this.displayScreenRight.position.copy(rightOffset);
        } else {
            this.displayScreenLeft.position.set(0, 0, 0);
            this.displayScreenRight.position.set(0, 0, 0);
        }
    }

    updateStereoTiling(totalAspectRatio, format) {
        this.currentStereoFormat = format;
        [this.displayScreenLeft, this.displayScreenRight].forEach((m, i) => {
            m.material.userData.eyeUniforms.uTiling.value.set(1.0, 1.0, i === 1 ? 0.5 : 0.0, 0);
        });
        this.aspectRatio = 1.0; 
        return 1.0;
    }

    updateTexture(texture, totalAspectRatio, format = STEREO_FORMATS.SBS_180) {
        this.updateStereoTiling(totalAspectRatio, format);
        this._applyTextureToMeshes(texture);
    }
}

export { StereoDomeScreen };
