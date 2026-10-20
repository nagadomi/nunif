import * as THREE from "three";
import { LIMITS, DEFAULTS, STEREO_FORMATS, UI_CONFIG, VR_CONFIG } from "./constants.js";
import { StereoBaseScreen, COMMON_FRAGMENT_HEADER, COMMON_FRAGMENT_MAP_PATCH } from "./stereo_base_screen.js";

// --- Plane Specific Shader Snippets ---

const VERTEX_HEADER = `
    varying vec2 vFadeUv;
    uniform float uCurvature, uWidth;
    uniform vec4 uTiling;
    #ifndef PI
    #define PI 3.141592653589793
    #endif
`;

const VERTEX_UV_PATCH = `
    #include <uv_vertex>
    vFadeUv = uv;
    vMapUv = uv * uTiling.xy + uTiling.zw;
`;

const VERTEX_BEGIN_PATCH = `
    vec3 transformed = vec3( position );
    if ( uCurvature > 0.001 ) {
        float theta = uCurvature * PI;
        float r = uWidth / theta;
        float phi = (position.x / uWidth) * theta;
        transformed.x = r * sin(phi);
        transformed.z = r - r * cos(phi);
    }
`;

const FRAGMENT_HEADER = `
    varying vec2 vFadeUv;
    uniform float uEdgeFade, uWidth, uHeight;
` + COMMON_FRAGMENT_HEADER;

const FRAGMENT_OPAQUE_PATCH = `
    #include <opaque_fragment>
    if ( uEdgeFade > 0.001 ) {
        float aspect = uHeight / (uWidth * 0.5); 
        float fx = smoothstep(0.0, uEdgeFade * aspect, min(vFadeUv.x, 1.0 - vFadeUv.x) - 0.01);
        float fy = smoothstep(0.0, uEdgeFade, min(vFadeUv.y, 1.0 - vFadeUv.y) - 0.01);
        gl_FragColor.a *= fx * fy;
        if (gl_FragColor.a < 0.001) discard;
    }
`;

class StereoScreen extends StereoBaseScreen {
    constructor(texture, physicalScreenHeight, options = {}) {
        super(options);

        this.screenType = 'plane';
        this.supportedSettings = [
            'screen_height_index', 'screen_zoom', 'screen_eye_sep', 'screen_tx', 'screen_ty', 
            'screen_tilt', 'screen_curvature', 'screen_edge_fade', 'screen_bg_color'
        ];
        this.physicalScreenHeight = physicalScreenHeight;
        this.currentCurvature = DEFAULTS.screen_curvature;
        this.currentEdgeFade = DEFAULTS.screen_edge_fade;
        
        const geometry = new THREE.PlaneGeometry(this.aspectRatio, 1.0, VR_CONFIG.screenSegments, 1);
        
        this.planeUniforms = {
            uCurvature: { value: DEFAULTS.screen_curvature },
            uWidth: { value: this.aspectRatio },
            uHeight: { value: 1.0 },
            uEdgeFade: { value: DEFAULTS.screen_edge_fade }
        };

        this._initMeshes(geometry);
        
        // Initial setup
        this.setTranslation(DEFAULTS.screen_tx, DEFAULTS.screen_ty);
        this.setZoom(this.currentZoom);
        this.updateTexture(texture, 1.0, STEREO_FORMATS.SBS_FULL);
    }

    patchShader(shader, eyeUniforms) {
        Object.assign(shader.uniforms, this.commonUniforms);
        Object.assign(shader.uniforms, this.planeUniforms);
        Object.assign(shader.uniforms, eyeUniforms);

        shader.vertexShader = VERTEX_HEADER + shader.vertexShader
            .replace('#include <uv_vertex>', VERTEX_UV_PATCH)
            .replace('#include <begin_vertex>', VERTEX_BEGIN_PATCH);

        shader.fragmentShader = FRAGMENT_HEADER + shader.fragmentShader
            .replace('#include <map_fragment>', COMMON_FRAGMENT_MAP_PATCH)
            .replace('#include <opaque_fragment>', FRAGMENT_OPAQUE_PATCH);
    }

    setCurvature(val) {
        this.currentCurvature = THREE.MathUtils.clamp(val, LIMITS.screen_curvatureMin, LIMITS.screen_curvatureMax);
        this.planeUniforms.uCurvature.value = this.currentCurvature;
    }

    setPhysicalScreenHeight(val) {
        this.physicalScreenHeight = THREE.MathUtils.clamp(val, LIMITS.screen_scaleMin, LIMITS.screen_scaleMax);
        this.displayScreenLeft.scale.set(this.physicalScreenHeight, this.physicalScreenHeight, this.physicalScreenHeight);
        this.displayScreenRight.scale.set(this.physicalScreenHeight, this.physicalScreenHeight, this.physicalScreenHeight);
        this.planeUniforms.uWidth.value = this.aspectRatio;
        this.planeUniforms.uHeight.value = 1.0;
        this.setTranslation(this.currentScreenX, this.currentScreenY);
        if (this.onSizeChange) this.onSizeChange(this.physicalScreenHeight);
    }

    setEdgeFade(val) {
        this.currentEdgeFade = THREE.MathUtils.clamp(val, LIMITS.screen_edge_fadeMin, LIMITS.screen_edge_fadeMax);
        this.planeUniforms.uEdgeFade.value = this.currentEdgeFade;
    }

    setZoom(z) {
        this.currentZoom = THREE.MathUtils.clamp(z, -LIMITS.screen_zoomMax, -LIMITS.screen_zoomMin);
        this.updatePositions();
        return true;
    }

    zoomScreen(direction, delta = 16) {
        const rootPos = this.getWorldPosition(new THREE.Vector3());
        const currentPos = this.localToWorld(new THREE.Vector3(0, 0, this.currentZoom));
        const currentDist = rootPos.distanceTo(currentPos);
        const sensitivity = (currentDist * 0.9) * (delta / 1000);
        return this.setZoom(this.currentZoom + sensitivity * direction);
    }

    handleDrag(deltaPos) {
        const sensitivity = UI_CONFIG.translationSensitivity;
        const scale = this.physicalScreenHeight;
        if (this.onTransformChange) {
            this.onTransformChange('screen_tx', this.startScreenX + deltaPos.x * sensitivity * scale);
            this.onTransformChange('screen_ty', this.startScreenY + deltaPos.y * sensitivity * scale);
        }
    }

    translateScreen(dx, dy) {
        if (this.onTransformChange) {
            this.onTransformChange('screen_tx', this.currentScreenX + dx);
            this.onTransformChange('screen_ty', this.currentScreenY + dy);
        }
    }

    setTranslation(x, y) {
        const xLimit = LIMITS.screen_txMax * this.physicalScreenHeight * this.aspectRatio;
        const yLimit = LIMITS.screen_tyMax * this.physicalScreenHeight;
        this.currentScreenX = THREE.MathUtils.clamp(x, -xLimit, xLimit);
        this.currentScreenY = THREE.MathUtils.clamp(y, -yLimit, yLimit);
        this.updatePositions();
    }

    updatePositions(_leftOffset = null, _rightOffset = null) {
        this.position.set(0, 0, 0);
        this.displayScreenLeft.position.set(this.currentScreenX - this.currentEyeSeparation / 2, this.currentScreenY, this.currentZoom);
        this.displayScreenRight.position.set(this.currentScreenX + this.currentEyeSeparation / 2, this.currentScreenY, this.currentZoom);
    }

    updateTexture(texture, totalAspectRatio, format = STEREO_FORMATS.SBS_FULL) {
        const screenWidth = this.updateStereoTiling(totalAspectRatio, format);
        const newGeo = new THREE.PlaneGeometry(screenWidth, 1.0, VR_CONFIG.screenSegments, 1);
        this.planeUniforms.uWidth.value = screenWidth;
        this.planeUniforms.uHeight.value = 1.0;
        this._applyTextureToMeshes(texture, newGeo);
        this.updatePositions();
    }
}

export { StereoScreen };
