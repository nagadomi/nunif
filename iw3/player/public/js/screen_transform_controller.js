import * as THREE from "three";
import { LIMITS, UI_CONFIG } from "./constants.js";

/**
 * Controller responsible for 3D gesture manipulation of the screen:
 * - Rotation (Tilt via controller pitch, Roll via hand X translation)
 * - Spatial dragging for translation (Pan / X-Y) and axis locking
 * - Mouse drag integration
 */
class ScreenTransformController {
    constructor(options = {}) {
        this.onTransformChange = options.onTransformChange || null;
        this.onSaveSettings = options.onSaveSettings || null;
        this.onVibrate = options.onVibrate || null;
        this.onDragStateChange = options.onDragStateChange || null;
        this.debugLogInstance = options.debugLogInstance || null;

        // Rotation tracking state
        this.isRotating = false;
        this.activeRotationController = null;
        this.initialRotationX = 0;
        this.initialControllerRotX = 0;
        this.initialRotationRoll = 0;
        this.initialRotationControllerPosition = new THREE.Vector3();

        // Translation tracking state
        this.isTranslating = false;
        this.activeTranslationController = null;
        this.initialControllerPosition = new THREE.Vector3();
        this.lockedTranslationAxis = null;
        this._atLimit = {};
    }

    startRotation(controller, currentTiltDeg, currentRollDeg = 0) {
        if (this.isRotating) return;

        controller.updateMatrixWorld(true);
        this.activeRotationController = controller;
        this.isRotating = true;
        this.initialRotationX = currentTiltDeg * Math.PI / 180;
        this.initialRotationRoll = currentRollDeg;

        const quat = new THREE.Quaternion();
        controller.getWorldQuaternion(quat);
        this.initialControllerRotX = new THREE.Euler().setFromQuaternion(quat, 'YXZ').x;

        controller.getWorldPosition(this.initialRotationControllerPosition);

        if (this.onDragStateChange) this.onDragStateChange(true);
    }

    updateRotation(controller, activeScreen = null) {
        if (!this.isRotating || controller !== this.activeRotationController) return;

        // 1. Tilt: controller pitch rotation
        const quat = new THREE.Quaternion();
        controller.getWorldQuaternion(quat);
        const currRotX = new THREE.Euler().setFromQuaternion(quat, 'YXZ').x;

        let deltaX = currRotX - this.initialControllerRotX;
        if (deltaX > Math.PI) deltaX -= Math.PI * 2;
        if (deltaX < -Math.PI) deltaX += Math.PI * 2;

        const nextTilt = (this.initialRotationX + deltaX) * 180 / Math.PI;

        if (this.onTransformChange) {
            this.onTransformChange('screen_tilt', nextTilt);
        }

        // 2. Roll: controller hand X translation (Dome screens only)
        const isDome = activeScreen && (activeScreen.screenType === 'dome' || activeScreen.screenType === 'dome_xr');
        if (isDome) {
            const pos = new THREE.Vector3();
            controller.getWorldPosition(pos);
            const deltaPosX = pos.x - this.initialRotationControllerPosition.x;

            // Sensitivity: Hand translation sensitivity for Roll (degrees per meter)
            const ROLL_SENSITIVITY = 60.0;
            const targetRoll = this.initialRotationRoll - deltaPosX * ROLL_SENSITIVITY;
            const clampedRoll = THREE.MathUtils.clamp(targetRoll, LIMITS.screen_rollMin, LIMITS.screen_rollMax);

            if (this.onTransformChange) {
                this.onTransformChange('screen_roll', clampedRoll);
            }
        }
    }

    endRotation(controller) {
        if (controller !== this.activeRotationController) return;
        this.isRotating = false;
        this.activeRotationController = null;
        if (this.onDragStateChange && !this.isTranslating) this.onDragStateChange(false);
        if (this.onSaveSettings) this.onSaveSettings();
    }

    cancelRotation() {
        this.isRotating = false;
        this.activeRotationController = null;
        if (this.onDragStateChange && !this.isTranslating) this.onDragStateChange(false);
    }

    startTranslation(controller, activeScreen) {
        if (this.isTranslating) return;

        this.activeTranslationController = controller;
        this.isTranslating = true;
        this.lockedTranslationAxis = null;

        controller.updateMatrixWorld(true);
        controller.getWorldPosition(this.initialControllerPosition);
        if (activeScreen) {
            activeScreen.saveInteractionStart();
        }
        if (this.onDragStateChange) this.onDragStateChange(true);
    }

    updateTranslation(controller, activeScreen, controllerIndex = -1) {
        if (!this.isTranslating || controller !== this.activeTranslationController || !activeScreen) return;

        const pos = new THREE.Vector3();
        controller.getWorldPosition(pos);
        const deltaPos = pos.clone().sub(this.initialControllerPosition);
        const screenType = activeScreen.screenType;

        // Axis Lock for Translation (Dome only)
        if (screenType === 'dome' || screenType === 'dome_xr') {
            if (!this.lockedTranslationAxis) {
                const threshold = 0.03; // 3cm deadzone
                if (Math.abs(deltaPos.x) > threshold || Math.abs(deltaPos.y) > threshold) {
                    this.lockedTranslationAxis = (Math.abs(deltaPos.x) > Math.abs(deltaPos.y)) ? 'pan' : 'posY';
                    this.initialControllerPosition.copy(pos);
                    activeScreen.saveInteractionStart();
                    return;
                } else {
                    return;
                }
            }
            if (this.lockedTranslationAxis === 'pan') deltaPos.y = 0;
            else if (this.lockedTranslationAxis === 'posY') deltaPos.x = 0;
        }

        // Delegate movement handling to the active screen
        activeScreen.handleDrag(deltaPos);

        // Limit checks for translation haptics
        const isAtLimit = (Math.abs(activeScreen.currentScreenX) >= 0.499 || Math.abs(activeScreen.currentScreenY) >= 0.499);
        if (isAtLimit) {
            if (controllerIndex !== -1 && !this._atLimit['screen_pos']) {
                if (this.onVibrate) {
                    this.onVibrate(controllerIndex, UI_CONFIG.hapticLimitIntensity, UI_CONFIG.hapticLimitDuration);
                }
                this._atLimit['screen_pos'] = true;
            }
        } else {
            this._atLimit['screen_pos'] = false;
        }
    }

    endTranslation(controller) {
        if (controller !== this.activeTranslationController) return;
        this.isTranslating = false;
        this.activeTranslationController = null;
        this.lockedTranslationAxis = null;
        if (this.onDragStateChange && !this.isRotating) this.onDragStateChange(false);
        if (this.onSaveSettings) this.onSaveSettings();
    }

    handleMouseRotate(dy, currentTiltRad) {
        const nextRad = currentTiltRad - dy * 0.01;
        const clampedRad = THREE.MathUtils.clamp(
            nextRad, 
            LIMITS.screen_tiltMin * Math.PI / 180, 
            LIMITS.screen_tiltMax * Math.PI / 180
        );
        const nextTiltDeg = clampedRad * 180 / Math.PI;
        if (this.onTransformChange) {
            this.onTransformChange('screen_tilt', nextTiltDeg);
        }
        return clampedRad;
    }

    handleMouseTranslate(dx, dy, activeScreen) {
        if (!activeScreen) return;
        activeScreen.translateScreen(dx * 0.005, -dy * 0.005);
    }
}

export { ScreenTransformController };
