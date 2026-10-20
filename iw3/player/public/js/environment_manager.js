import * as THREE from "three";
import { HDRLoader, EXRLoader, GLTFLoader, DRACOLoader } from 'three';
import { LIMITS, DEFAULTS, COLORS } from "./constants.js";
import { ThreeUtils } from "./ui_common.js";

class EnvironmentManager extends THREE.Group {
    constructor(scene, debugLog) {
        super();
        this.scene = scene;
        this.debugLog = debugLog;

        this.currentEnvironment = "None"; // Skybox (Texture)
        this.currentModelName = "None";   // 3D Model

        this.currentRotation = DEFAULTS.environment_rotation;
        this.currentTilt = DEFAULTS.environment_tilt;
        this.currentIntensity = DEFAULTS.environment_intensity;
        this.currentModelScale = DEFAULTS.environment_model_scale;
        this.currentModelY = DEFAULTS.environment_model_y;
        this.backgroundLevel = DEFAULTS.screen_bg_color;

        this.hdrLoader = new HDRLoader();
        this.exrLoader = new EXRLoader();
        
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('/lib/draco/');
        this.glbLoader = new GLTFLoader();
        this.glbLoader.setDRACOLoader(dracoLoader);
        
        this._pendingTexture = undefined;
        this._pendingModel = undefined;
        this._pendingMixer = undefined;

        this._skyboxLoadId = 0;
        this._modelLoadId = 0;
        this._loadingSkyboxName = "None";
        this._loadingModelName = "None";

        this.currentModel = null;
        this.mixer = null;

        this.fadePhase = 'none'; // 'none', 'out', 'in'
        this.fadeValue = 1.0;
        this._isSkyboxLoading = false;
        this._isModelLoading = false;
        this.isBackgroundLocked = false;

        // Ambient Light (synchronized with background color)
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.0);
        this.add(this.ambientLight);

        // Color caches to avoid allocations
        this._targetColor = new THREE.Color();
        this._bgColor = new THREE.Color();
        this._blackColor = new THREE.Color(0x000000);
        this._fadedColor = new THREE.Color();

        this._fadeTimer = 0;
        this._lastFadeValue = -1;

        // Ensure scene properties are ready
        this.scene.backgroundRotation.order = 'XYZ';
        this.scene.environmentRotation.order = 'XYZ';

        // Initial update
        this.updateEnvironmentState();
    }

    setBackgroundLevel(val) {
        this.backgroundLevel = val;
        if (!this.isBackgroundLocked) {
            this.updateEnvironmentState();
        }
    }

    setRotation(degrees) {
        this.currentRotation = THREE.MathUtils.clamp(degrees, LIMITS.environment_rotationMin, LIMITS.environment_rotationMax);
        this.updateEnvironmentState();
    }

    setTilt(degrees) {
        this.currentTilt = THREE.MathUtils.clamp(degrees, LIMITS.environment_tiltMin, LIMITS.environment_tiltMax);
        this.updateEnvironmentState();
    }

    setIntensity(val) {
        this.currentIntensity = val;
        this.updateEnvironmentState();
    }

    setModelScale(val) {
        this.currentModelScale = THREE.MathUtils.clamp(val, LIMITS.environment_model_scaleMin, LIMITS.environment_model_scaleMax);
        if (this.currentModel) {
            const scale = Math.pow(10, this.currentModelScale);
            this.currentModel.scale.setScalar(scale);
        }
    }

    setModelY(val) {
        this.currentModelY = THREE.MathUtils.clamp(val, LIMITS.environment_model_yMin, LIMITS.environment_model_yMax);
        if (this.currentModel) {
            this.currentModel.position.y = this.currentModelY;
        }
    }

    async loadSkybox(filename) {
        if (this.isBackgroundLocked) return;
        if (filename === this.currentEnvironment && this._pendingTexture === undefined) return;
        if (this._isSkyboxLoading && filename === this._loadingSkyboxName) return;
        this._isSkyboxLoading = true;
        this.fadePhase = 'out';
        this._fadeTimer = 0;
        this._performLoadSkybox(filename);
    }

    async loadModel(filename) {
        if (this.isBackgroundLocked) return;
        if (filename === this.currentModelName && this._pendingModel === undefined) return;
        if (this._isModelLoading && filename === this._loadingModelName) return;
        this._isModelLoading = true;
        this.fadePhase = 'out';
        this._fadeTimer = 0;
        this._performLoadModel(filename);
    }

    // Actual internal loading methods
    _performLoadSkybox(filename) {
        const loadId = ++this._skyboxLoadId;
        this._loadingSkyboxName = filename;
        this._pendingTexture = undefined;
        this._isSkyboxLoading = true;

        if (filename === "None") {
            this._pendingTexture = null;
            this._isSkyboxLoading = false;
            return;
        }

        const url = `/environments/${filename}`;
        const loader = filename.toLowerCase().endsWith('.exr') ? this.exrLoader : this.hdrLoader;

        loader.load(url, (texture) => {
            if (loadId !== this._skyboxLoadId) {
                texture.dispose();
                return;
            }
            texture.mapping = THREE.EquirectangularReflectionMapping;
            texture.magFilter = THREE.LinearFilter;
            texture.minFilter = THREE.LinearFilter;
            texture.colorSpace = THREE.LinearSRGBColorSpace;
            this._pendingTexture = texture;
            this._isSkyboxLoading = false;
        }, undefined, (err) => {
            if (this.debugLog) this.debugLog.log(`Skybox Load Error: ${err}`);
            if (loadId === this._skyboxLoadId) {
                this._isSkyboxLoading = false;
                if (this.fadePhase === 'out') {
                    this._pendingTexture = null;
                }
            }
        });
    }

    _performLoadModel(filename) {
        const loadId = ++this._modelLoadId;
        this._loadingModelName = filename;
        this._pendingModel = undefined;
        this._pendingMixer = undefined;
        this._isModelLoading = true;

        if (filename === "None") {
            this._pendingModel = null;
            this._isModelLoading = false;
            return;
        }

        const url = `/environments/${filename}`;
        this.glbLoader.load(url, (gltf) => {
            if (loadId !== this._modelLoadId) {
                return;
            }
            let mixer = null;
            if (gltf.animations?.length > 0) {
                mixer = new THREE.AnimationMixer(gltf.scene);
                gltf.animations.forEach(clip => mixer.clipAction(clip).play());
            }
            this._pendingModel = gltf.scene;
            this._pendingMixer = mixer;
            this._isModelLoading = false;
        }, undefined, (err) => {
            if (this.debugLog) this.debugLog.log(`Model Load Error: ${err}`);
            if (loadId === this._modelLoadId) {
                this._isModelLoading = false;
                if (this.fadePhase === 'out') {
                    this._pendingModel = null;
                }
            }
        });
    }

    update(delta) {
        // Update animation mixer (delta is in ms)
        if (this.mixer) {
            this.mixer.update(delta / 1000);
        }

        if (this.fadePhase === 'none') return;

        // Reset delta after a long freeze (e.g. GPU texture upload) to prevent animation jumps.
        // If delta is huge (>200ms), we treat it as a frame skip and use a nominal 16ms.
        const dt = (delta > 200) ? 16 : delta;
        this._fadeTimer += dt;

        // Step-based update: advance fadeValue at most every 10ms to keep it smooth.
        // With 1500ms duration, this gives 150 steps, which is very fine for VR.
        const stepThreshold = 10; 
        if (this._fadeTimer >= stepThreshold) {
            const steps = Math.floor(this._fadeTimer / stepThreshold);
            this._fadeTimer -= steps * stepThreshold;
            
            const duration = 1500; // 1.5 seconds for a smooth, premium feel
            const stepAmount = (steps * stepThreshold) / duration;

            if (this.fadePhase === 'out') {
                this.fadeValue = Math.max(0, this.fadeValue - stepAmount);
                if (this.fadeValue <= 0) {
                    if (!this._isSkyboxLoading && !this._isModelLoading) {
                        this.applyPendingChange();
                        this.fadePhase = 'waiting'; 
                    }
                }
            } else if (this.fadePhase === 'waiting') {
                // One frame in pure black state
                this.fadePhase = 'in';
            } else if (this.fadePhase === 'in') {
                this.fadeValue = Math.min(1, this.fadeValue + stepAmount);
                if (this.fadeValue >= 1) {
                    this.fadePhase = 'none';
                }
            }
        }

        // Only call expensive update if values actually changed
        if (this.fadeValue !== this._lastFadeValue) {
            this.updateEnvironmentState();
            this._lastFadeValue = this.fadeValue;
        }
    }

    applyPendingChange() {
        // Handle Texture Change
        // undefined means no change requested, null means 'None' selected
        if (this._pendingTexture !== undefined) {
            if (this.scene.background && this.scene.background instanceof THREE.Texture) {
                const oldTex = this.scene.background;
                setTimeout(() => oldTex.dispose(), 2000);
            }
            if (this.scene.environment && this.scene.environment instanceof THREE.Texture && this.scene.environment !== this.scene.background) {
                const oldEnv = this.scene.environment;
                setTimeout(() => oldEnv.dispose(), 2000);
            }
            
            if (this.isBackgroundLocked) {
                this.scene.background = null;
                this.scene.environment = null;
            } else if (this._pendingTexture) {
                this.scene.background = this._pendingTexture;
                this.scene.environment = this._pendingTexture; // Share reference
            } else {
                this.scene.background = null; 
                this.scene.environment = null;
            }
            this.currentEnvironment = this._loadingSkyboxName;
            this._pendingTexture = undefined;
        }

        // Handle Model Change
        if (this._pendingModel !== undefined) {
            if (this.mixer) {
                this.mixer.stopAllAction();
                this.mixer = null;
            }
            if (this.currentModel) {
                const oldModel = this.currentModel;
                this.remove(oldModel);
                setTimeout(() => ThreeUtils.disposeObject(oldModel), 2000);
                this.currentModel = null;
            }

            if (this._pendingModel) {
                this.currentModel = this._pendingModel;
                const scale = Math.pow(10, this.currentModelScale);
                this.currentModel.scale.setScalar(scale);
                this.currentModel.position.y = this.currentModelY;
                this.mixer = this._pendingMixer;
                this.add(this.currentModel);
            }
            this.currentModelName = this._loadingModelName;
            this._pendingModel = undefined;
            this._pendingMixer = undefined;
        }
    }

    updateEnvironmentState() {
        // Pre-calculate exposure to save power calls
        const exposureMultiplier = Math.pow(2, this.currentIntensity);
        
        // Perceptual easing: Use squared value for smoother transition in linear space.
        // This makes the brightness change feel more linear to the human eye.
        const easedFade = this.fadeValue * this.fadeValue;
        const totalIntensity = exposureMultiplier * easedFade;
        
        const tiltRad = this.currentTilt * Math.PI / 180;
        const rotRad = this.currentRotation * Math.PI / 180;

        // Apply rotations
        this.scene.backgroundRotation.set(tiltRad, rotRad, 0, 'YXZ');
        this.scene.environmentRotation.set(tiltRad, rotRad, 0, 'YXZ');
        
        // Intensity control (only affects skybox/background texture)
        this.scene.backgroundIntensity = totalIntensity;
        this.scene.environmentIntensity = totalIntensity;

        // Handle Background and Ambient Light based on ACTUAL scene background state
        const ambientIntensityMultiplier = 1.0;

        if (!this.isBackgroundLocked && (this.scene.background instanceof THREE.Color || this.scene.background === null)) {
            // Handle Solid Background (or null which we treat as black/color transition)
            if (!(this.scene.background instanceof THREE.Color)) {
                this.scene.background = new THREE.Color(0x000000);
            }

            this._targetColor.set(COLORS.bgTarget);
            this._bgColor.copy(this._blackColor).lerp(this._targetColor, this.backgroundLevel);
            
            // Perceptually smooth fade for the background color using cached colors
            this._fadedColor.copy(this._blackColor).lerp(this._bgColor, easedFade);
            this.scene.background.copy(this._fadedColor);

            // Synchronize Ambient Light
            this.ambientLight.color.copy(this._targetColor);
            this.ambientLight.intensity = Math.max(DEFAULTS.environment_ambient_min * easedFade, 
                                                  this.backgroundLevel * ambientIntensityMultiplier * easedFade);
        } else {
            // Background is a Texture (Skybox)
            // Ensure ambient light is off when skybox is dominant
            this.ambientLight.intensity = 0;
        }

        // Model visibility (only affects opacity during transition)
        // Optimization: Only traverse if we are in a transition and have a model
        if (this.currentModel) {
            if (this.fadeValue <= 0) {
                if (this.currentModel.visible) this.currentModel.visible = false;
            } else {
                if (!this.currentModel.visible) this.currentModel.visible = true;
                
                if (this.fadeValue < 1.0) {
                    this.currentModel.traverse((child) => {
                        if (child.isMesh) {
                            const materials = Array.isArray(child.material) ? child.material : [child.material];
                            materials.forEach(m => {
                                if (!m.transparent) m.transparent = true;
                                if (m.opacity !== easedFade) m.opacity = easedFade;
                            });
                        }
                    });
                    this.currentModel.userData.isTransparent = true;
                } else if (this.currentModel.userData.isTransparent) {
                    // Reset to opaque once transition is finished
                    this.currentModel.traverse((child) => {
                        if (child.isMesh) {
                            const materials = Array.isArray(child.material) ? child.material : [child.material];
                            materials.forEach(m => {
                                m.transparent = false;
                                m.opacity = 1.0;
                            });
                        }
                    });
                    this.currentModel.userData.isTransparent = false;
                }
            }
        }
    }

    dispose() {
        if (this.scene.background && this.scene.background instanceof THREE.Texture) {
            this.scene.background.dispose();
        }
        if (this.scene.environment) {
            this.scene.environment.dispose();
        }
        if (this.currentModel) {
            this.remove(this.currentModel);
        }
    }

    isGLBActive() {
        return !!this.currentModel;
    }
}

export { EnvironmentManager };
