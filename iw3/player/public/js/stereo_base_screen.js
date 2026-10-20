import * as THREE from "three";
import { STEREO_FORMATS, DEFAULTS, LIMITS, UI_CONFIG } from "./constants.js";

// --- Shared Shader Snippets ---

export const COMMON_FRAGMENT_HEADER = `
    uniform float uBrightness, uContrast, uGamma, uSaturation, uHue;
    uniform sampler3D uLutMap;
    uniform float uLutEnabled;

    vec3 rgb2hsv(vec3 c) {
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }
`;

export const COLOR_CORRECTION_SNIPPET = `
    // 1. Brightness & Contrast (Linear space)
    diffuseColor.rgb *= uBrightness;
    diffuseColor.rgb = (diffuseColor.rgb - 0.18) * uContrast + 0.18;

    bool hasLut = uLutEnabled > 0.5;
    bool hasHue = abs(uHue) > 0.001;
    bool hasSat = abs(uSaturation - 1.0) > 0.01;

    if ( hasLut || hasHue || hasSat ) {
        vec3 srgb = pow(max(diffuseColor.rgb, 0.0), vec3(1.0/2.2));
        if ( hasLut ) {
            srgb = texture(uLutMap, clamp(srgb, 0.0, 1.0)).rgb;
        }
        if ( hasHue ) {
            vec3 hsv = rgb2hsv(srgb);
            hsv.x = fract(hsv.x + uHue / 360.0);
            srgb = hsv2rgb(hsv);
        }
        if ( hasSat ) {
            float luma = dot(srgb, vec3(0.2126, 0.7152, 0.0722));
            srgb = mix(vec3(luma), srgb, uSaturation);
        }
        diffuseColor.rgb = pow(max(srgb, 0.0), vec3(2.2));
    }

    if ( abs(uGamma - 1.0) > 0.001 ) {
        diffuseColor.rgb = pow(max(diffuseColor.rgb, 0.0), vec3(1.0 / uGamma));
    }
`;

export const COMMON_FRAGMENT_MAP_PATCH = `
    #include <map_fragment>
` + COLOR_CORRECTION_SNIPPET;

const COLOR_UNIFORM_MAP = {
    color_brightness: 'uBrightness',
    color_contrast: 'uContrast',
    color_gamma: 'uGamma',
    color_saturation: 'uSaturation',
    color_hue: 'uHue'
};

class StereoBaseScreen extends THREE.Group {
    constructor(options = {}) {
        super();
        
        this.onTransformChange = options.onTransformChange || null;
        this.onSizeChange = options.onSizeChange || null;
        this.screenType = 'base'; // Overridden by subclasses
        this.aspectRatio = 1.0;
        this.currentStereoFormat = STEREO_FORMATS.SBS_FULL;
        this.isXR = false;
        this.canOccludeUI = true;
        this.supportsSubtitles = true;

        this.supportedSettings = [
            'screen_height_index', 'screen_zoom', 'screen_eye_sep', 'screen_tx', 'screen_ty', 
            'screen_roll', 'screen_tilt', 'screen_bg_color'
        ];

        // Interaction State
        this.currentEyeSeparation = DEFAULTS.screen_eye_sep;
        this.currentScreenX = 0;
        this.currentScreenY = 0;
        this.currentZoom = DEFAULTS.screen_zoom;

        // Cached Eye Offsets (Local to StereoPlayer)
        this.lastKnownLeftOffset = new THREE.Vector3(-0.032, 0, 0);
        this.lastKnownRightOffset = new THREE.Vector3(0.032, 0, 0);

        // Interaction Start State (to be captured at the beginning of a drag)
        this.startScreenX = 0;
        this.startScreenY = 0;
        this.startZoom = 0;
        this.startPan = 0;
        this.startTilt = 0;
        this.startRoll = 0;

        // Common Uniforms
        const blackData = new Uint8Array([0, 0, 0, 255]);
        const blackTex = new THREE.Data3DTexture(blackData, 1, 1, 1);
        blackTex.format = THREE.RGBAFormat;
        blackTex.needsUpdate = true;

        this.commonUniforms = {
            uBrightness: { value: DEFAULTS.color_brightness },
            uContrast: { value: DEFAULTS.color_contrast },
            uGamma: { value: DEFAULTS.color_gamma },
            uSaturation: { value: DEFAULTS.color_saturation },
            uHue: { value: DEFAULTS.color_hue },
            uLutMap: { value: blackTex },
            uLutEnabled: { value: 0.0 }
        };

        this.displayScreenLeft = null;
        this.displayScreenRight = null;
    }

    _initMeshes(geometry) {
        const blackData = new Uint8Array([0, 0, 0, 255]);
        const dummyTex = new THREE.DataTexture(blackData, 1, 1, THREE.RGBAFormat);
        dummyTex.needsUpdate = true;
        this.dummyTexture = dummyTex;

        const createMaterial = (isRight) => {
            const eyeUniforms = {
                uTiling: { value: new THREE.Vector4(0.5, 1, isRight ? 0.5 : 0.0, 0) }
            };
            const mat = new THREE.MeshBasicMaterial({
                map: dummyTex,
                side: THREE.DoubleSide,
                transparent: true,
            });
            mat.userData.eyeUniforms = eyeUniforms; 
            
            // Unique key to prevent shader program leakage between classes
            mat.customProgramCacheKey = () => this.screenType;

            mat.onBeforeCompile = (shader) => this.patchShader(shader, eyeUniforms);
            return mat;
        };

        this.displayScreenLeft = new THREE.Mesh(geometry, createMaterial(false));
        this.displayScreenLeft.layers.set(1);
        this.displayScreenLeft.frustumCulled = false;
        this.add(this.displayScreenLeft);

        this.displayScreenRight = new THREE.Mesh(geometry, createMaterial(true));
        this.displayScreenRight.layers.set(2);
        this.displayScreenRight.frustumCulled = false;
        this.add(this.displayScreenRight);
    }

    /**
     * Resets screen materials to the safe black dummy texture to prevent
     * WebGL errors when an underlying video element or texture is destroyed.
     */
    resetTexture() {
        if (!this.displayScreenLeft || !this.displayScreenRight || !this.dummyTexture) return;
        this.displayScreenLeft.material.map = this.dummyTexture;
        this.displayScreenLeft.material.needsUpdate = true;
        this.displayScreenRight.material.map = this.dummyTexture;
        this.displayScreenRight.material.needsUpdate = true;
    }

    _applyTextureToMeshes(texture, newGeometry = null) {
        if (!this.displayScreenLeft || !this.displayScreenRight) return;
        this.displayScreenLeft.visible = true;
        this.displayScreenRight.visible = this.isXR;
        if (newGeometry && this.displayScreenLeft.geometry) {
            this.displayScreenLeft.geometry.dispose();
        }
        [this.displayScreenLeft, this.displayScreenRight].forEach((m) => {
            if (newGeometry) m.geometry = newGeometry;
            m.material.map = texture;
            m.material.needsUpdate = true;
        });
        if (texture?.isVideoTexture && texture.image?.readyState >= 2) {
            texture.needsUpdate = true;
        }
    }

    clearScreen() {
        this.resetTexture();
        if (this.displayScreenLeft) this.displayScreenLeft.visible = false;
        if (this.displayScreenRight) this.displayScreenRight.visible = false;
    }

    patchShader(shader, eyeUniforms) {
        Object.assign(shader.uniforms, this.commonUniforms, eyeUniforms);
        shader.fragmentShader = COMMON_FRAGMENT_HEADER + shader.fragmentShader
            .replace('#include <map_fragment>', COMMON_FRAGMENT_MAP_PATCH);
    }

    updateEyeVisibility(isXR) {
        this.isXR = isXR;
        if (isXR) {
            // In XR, screens are rendered by Layer 1/2. Disable Layer 0 (PC camera).
            this.displayScreenLeft.layers.disable(0);
            this.displayScreenRight.layers.disable(0);
            this.displayScreenRight.visible = this.displayScreenLeft.visible;
        } else {
            // In PC (non-XR), show only the left eye screen on Layer 0.
            this.displayScreenLeft.layers.enable(0);
            this.displayScreenRight.layers.disable(0);
            this.displayScreenRight.visible = false;
        }
    }

    updateStereoTiling(totalAspectRatio, format) {
        this.currentStereoFormat = format;
        let displayAspectRatio = totalAspectRatio;
        let repeatX = 0.5, repeatY = 1.0;
        let leftOffset = new THREE.Vector2(0, 0), rightOffset = new THREE.Vector2(0.5, 0);

        if (format === STEREO_FORMATS.SBS_FULL) { displayAspectRatio = totalAspectRatio / 2.0; } 
        else if (format === STEREO_FORMATS.SBS_HALF) { displayAspectRatio = totalAspectRatio; } 
        else if (format === STEREO_FORMATS.SBS_FULL_CROSS) { displayAspectRatio = totalAspectRatio / 2.0; leftOffset.set(0.5, 0); rightOffset.set(0, 0); } 
        else if (format === STEREO_FORMATS.TB_FULL) { displayAspectRatio = totalAspectRatio * 2.0; repeatX = 1.0; repeatY = 0.5; leftOffset.set(0, 0.5); rightOffset.set(0, 0); } 
        else if (format === STEREO_FORMATS.TB_HALF) { displayAspectRatio = totalAspectRatio; repeatX = 1.0; repeatY = 0.5; leftOffset.set(0, 0.5); rightOffset.set(0, 0); } 
        else if (format === STEREO_FORMATS.FLAT) { displayAspectRatio = totalAspectRatio; repeatX = 1.0; repeatY = 1.0; leftOffset.set(0, 0); rightOffset.set(0, 0); }

        this.aspectRatio = displayAspectRatio;

        [this.displayScreenLeft, this.displayScreenRight].forEach((m, i) => {
            const off = (i === 1) ? rightOffset : leftOffset;
            m.material.userData.eyeUniforms.uTiling.value.set(repeatX, repeatY, off.x, off.y);
        });

        return displayAspectRatio;
    }

    setColorParameter(key, val) {
        const uniformName = COLOR_UNIFORM_MAP[key];
        if (uniformName && this.commonUniforms[uniformName]) {
            this.commonUniforms[uniformName].value = val;
        }
    }
    
    setEyeSeparation(sep) {
        this.currentEyeSeparation = THREE.MathUtils.clamp(sep, LIMITS.screen_eye_sepMin, LIMITS.screen_eye_sepMax);
        this.updatePositions();
        return true;
    }

    slideEyeSeparation(direction, delta = 16) {
        const maxSep = LIMITS.screen_eye_sepMax;
        const ratio = Math.abs(this.currentEyeSeparation) / maxSep;
        const baseSens = (UI_CONFIG.eyeSeparationSensitivity * (1.0 - ratio) + UI_CONFIG.eyeSeparationSensitivityMin) * (delta / 1000);
        return this.setEyeSeparation(this.currentEyeSeparation + direction * baseSens);
    }

    setLutTexture(texture) {
        if (this.commonUniforms.uLutMap.value && this.commonUniforms.uLutMap.value instanceof THREE.Texture && (this.commonUniforms.uLutMap.value.image?.width > 1)) {
            this.commonUniforms.uLutMap.value.dispose();
        }
        if (texture) {
            this.commonUniforms.uLutMap.value = texture;
            this.commonUniforms.uLutEnabled.value = 1.0;
        } else {
            this.commonUniforms.uLutEnabled.value = 0.0;
        }
        // Force shader recompile to reflect uLutEnabled change in FRAGMENT_MAP_PATCH
        if (this.displayScreenLeft) {
            this.displayScreenLeft.material.needsUpdate = true;
            this.displayScreenRight.material.needsUpdate = true;
        }
    }

    saveInteractionStart() {
        this.startScreenX = this.currentScreenX;
        this.startScreenY = this.currentScreenY;
        this.startZoom = this.currentZoom;
    }

    handleDrag(_deltaPos) {}

    // Default implementations to avoid crashes (can be overridden)
    setCurvature(_val) {}
    setEdgeFade(_val) {}
    setPhysicalScreenHeight(_val) {}
    setTranslation(_x, _y) {}
    translateScreen(_dx, _dy) {}
    setRoll(_val) {}
    setZoom(_z) { return false; }
    zoomScreen(_direction, _delta = 16) { return false; }
    updatePositions(leftOffset = null, rightOffset = null) {
        if (leftOffset) this.lastKnownLeftOffset.copy(leftOffset);
        if (rightOffset) this.lastKnownRightOffset.copy(rightOffset);
    }
}

export { StereoBaseScreen };
