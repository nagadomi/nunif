import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory.js';
import { DEFAULTS, INPUT_ACTIONS, UI_CONFIG, getSettingLimits } from './constants.js';
import { UIUtils } from './ui_common.js';

/**
 * Normalizes inputs from VR controllers, Mouse, and Keyboard into semantic Actions.
 */
class InputManager {
    constructor(stereoPlayer, uiManager) {
        this.player = stereoPlayer;
        this.ui = uiManager;

        // VR Controller Hardware & State
        this.controllers = [null, null];
        this.gamepads = [null, null];
        this.controllerRays = [null, null];
        this.controllerGrips = [null, null];
        this.selectStartTime = [0, 0];
        this.initialRotationX = DEFAULTS.screen_tilt * Math.PI / 180;

        this.isStickReset = [true, true];
        this.activeStickAxis = [null, null]; // 'x' or 'y'
        this.isMenuButtonPressed = [false, false];
        this.isAButtonPressed = [false, false];

        // Raycasting helpers
        this.raycaster = new THREE.Raycaster();
        this.tempMatrix = new THREE.Matrix4();

        // Feedback State
        this._atLimit = {}; // Tracks limit feedback for joysticks and translation

        // Keyboard State
        this.isShiftDown = false;

        // Mouse State
        this.isRotating = false;
        this.isTranslating = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.setupXRControllers();
        this.setupKeyboardEvents();
        this.setupMouseEvents();
    }

    setupXRControllers() {
        const renderer = this.player.renderer;
        const scene = this.player.scene;
        const controllerModelFactory = new XRControllerModelFactory();
        const rayWidth = 0.003, rayLength = 5.0;
        const rayGeometry = new THREE.CylinderGeometry(rayWidth, rayWidth, rayLength, 8);
        rayGeometry.rotateX(-Math.PI / 2);
        rayGeometry.translate(0, 0, -rayLength / 2);
        const rayMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });

        for (let i = 0; i < 2; i++) {
            this.controllers[i] = renderer.xr.getController(i);
            scene.add(this.controllers[i]);
            this.controllers[i].addEventListener('connected', (event) => {
                this.gamepads[i] = event.data.gamepad;
                const grip = renderer.xr.getControllerGrip(i);
                const model = controllerModelFactory.createControllerModel(grip);
                model.visible = this.ui.isMainMenuVisible;
                grip.add(model);
                scene.add(grip);
                this.controllerGrips[i] = model;
                const ray = new THREE.Mesh(rayGeometry, rayMaterial);
                ray.visible = this.ui.isMainMenuVisible;
                this.controllers[i].add(ray);
                this.controllerRays[i] = ray;
            });
            this.controllers[i].addEventListener('selectstart', (e) => this.onSelectStart(e));
            this.controllers[i].addEventListener('selectend', (e) => this.onSelectEnd(e));
            this.controllers[i].addEventListener('squeezestart', (e) => this.onSqueezeStart(e));
            this.controllers[i].addEventListener('squeezeend', (e) => this.onSqueezeEnd(e));
        }
    }

    checkUIHit(controller) {
        this.tempMatrix.identity().extractRotation(controller.matrixWorld);
        this.raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
        this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(this.tempMatrix);
        this.raycaster.near = 0;
        this.raycaster.far = Infinity;
        return this.raycaster.intersectObjects(this.ui.getInteractableMeshes(), true);
    }

    onSelectStart(event) {
        const controller = event.target;
        if (this.checkUIHit(controller).length > 0) return;
        if (this.player.transformController.isRotating) return;

        const index = this.controllers.indexOf(controller);
        if (index !== -1) this.selectStartTime[index] = performance.now();
        this.initialRotationX = this.player.rotation.x;

        this.player.transformController.startRotation(
            controller,
            this.player.rotation.x * 180 / Math.PI,
            this.player.activeScreen.currentRoll || 0
        );
    }

    onSelectEnd(event) {
        const controller = event.target;
        const tc = this.player.transformController;
        if (controller !== tc.activeRotationController) return;
        const index = this.controllers.indexOf(controller);
        const duration = performance.now() - this.selectStartTime[index];
        const rotDelta = Math.abs(this.player.rotation.x - this.initialRotationX);

        if (duration < 500 && rotDelta < 0.05) {
            if (this.ui.isAnyMenuVisible() || this.ui.activeSubMenu === this.ui.explorer) {
                if (this.checkUIHit(controller).length === 0) {
                    this.ui.hideMainMenu(true);
                    this.toggleControllerUIVisibility(false);
                }
            } else {
                this.ui.showMainMenu(index);
                this.toggleControllerUIVisibility(true);
            }
            tc.cancelRotation();
        } else {
            tc.endRotation(controller);
        }
    }

    onSqueezeStart(event) {
        const controller = event.target;
        if (this.player.transformController.isTranslating) return;
        this.player.transformController.startTranslation(controller, this.player.activeScreen);
    }

    onSqueezeEnd(event) {
        if (event.target === this.player.transformController.activeTranslationController) {
            this.player.transformController.endTranslation(event.target);
        }
    }

    toggleControllerUIVisibility(isVisible) {
        for (let i = 0; i < 2; i++) {
            if (this.controllerRays[i]) this.controllerRays[i].visible = isVisible;
            if (this.controllerGrips[i]) this.controllerGrips[i].visible = isVisible;
        }
    }

    setupMouseEvents() {
        const el = this.player.renderer.domElement;
        el.addEventListener('contextmenu', (e) => e.preventDefault());
        el.addEventListener('mousedown', (e) => this.onMouseDown(e));
        el.addEventListener('mousemove', (e) => this.onMouseMove(e));
        el.addEventListener('mouseup', () => { 
            if (this.isRotating || this.isTranslating) {
                if (this.ui) this.ui.saveSettings();
                this.player.setAxesGuideVisible(false);
            }
            this.isRotating = this.isTranslating = false; 
        });
    }

    onMouseDown(event) {
        const mouse = new THREE.Vector2((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
        this.raycaster.setFromCamera(mouse, this.player.camera);
        if (this.raycaster.intersectObjects(this.ui.getInteractableMeshes(), true).length > 0) return;
        
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;
        if (event.button === 0) { 
            if (this.isShiftDown || event.shiftKey) this.isTranslating = true; 
            else this.isRotating = true; 
            this.player.setAxesGuideVisible(true);
        }
    }

    onMouseMove(event) {
        const dx = event.clientX - this.lastMouseX;
        const dy = event.clientY - this.lastMouseY;
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;

        if (this.isRotating) {
            this.player.transformController.handleMouseRotate(dy, this.player.rotation.x);
        } else if (this.isTranslating) {
            this.player.transformController.handleMouseTranslate(dx, dy, this.player.activeScreen);
        }
    }

    setupKeyboardEvents() {
        window.addEventListener('keydown', (e) => {
            if (e.repeat) return;
            switch (e.code) {
                case 'Space': e.preventDefault(); this.dispatch(INPUT_ACTIONS.MENU_TOGGLE, { autoExplorer: false }); break;
                case 'ArrowLeft': this.dispatch(this.player.videoElement ? INPUT_ACTIONS.SEEK_BWD : INPUT_ACTIONS.NAV_PREV); break;
                case 'ArrowRight': this.dispatch(this.player.videoElement ? INPUT_ACTIONS.SEEK_FWD : INPUT_ACTIONS.NAV_NEXT); break;
                case 'ArrowUp': this.dispatch(INPUT_ACTIONS.ZOOM_IN); break;
                case 'ArrowDown': this.dispatch(INPUT_ACTIONS.ZOOM_OUT); break;
                case 'KeyQ': this.dispatch(INPUT_ACTIONS.EYE_SEP_DEC); break;
                case 'KeyE': this.dispatch(INPUT_ACTIONS.EYE_SEP_INC); break;
                case 'ShiftLeft': case 'ShiftRight': this.isShiftDown = true; break;
            }
            if (this.ui) this.ui.saveSettings();
        });
        window.addEventListener('keyup', (e) => {
            if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') this.isShiftDown = false;
        });
        window.addEventListener('wheel', (e) => {
            this.dispatch(e.deltaY > 0 ? INPUT_ACTIONS.ZOOM_OUT : INPUT_ACTIONS.ZOOM_IN);
            if (this.ui) this.ui.saveSettings();
        }, { passive: true });
    }

    update(delta) {
        if (this.player.isXRActive) {
            this.handleXRUpdate(delta);
        }
    }

    handleXRUpdate(delta) {
        const tc = this.player.transformController;

        // Stick & Button Input Processing
        for (let i = 0; i < 2; i++) {
            const controller = this.controllers[i];
            const gamepad = this.gamepads[i];
            if (!controller || !gamepad) continue;

            let stickX = gamepad.axes[2];
            let stickY = gamepad.axes[3];
            const absX = Math.abs(stickX);
            const absY = Math.abs(stickY);

            if (this.activeStickAxis[i] === null) {
                if (absX > 0.5 || absY > 0.5) this.activeStickAxis[i] = (absX > absY) ? 'x' : 'y';
            } else {
                if (absX < 0.1 && absY < 0.1) { 
                    this.activeStickAxis[i] = null; 
                    this.isStickReset[i] = true; 
                    // Save settings when stick is released
                    if (this.ui) this.ui.saveSettings();
                }
            }

            const hitInfo = this.getUIContext(controller);
            
            if (this.activeStickAxis[i] === 'y') {
                if (hitInfo.isExplorer) {
                    if (this.isStickReset[i]) {
                        this.dispatch(stickY > 0 ? INPUT_ACTIONS.PAGE_NEXT : INPUT_ACTIONS.PAGE_PREV, { controllerIndex: i });
                        this.isStickReset[i] = false;
                    }
                } else if (!hitInfo.isUI) {
                    const params = { delta, controllerIndex: i };
                    if (tc.isTranslating && tc.activeTranslationController === controller) {
                        this.dispatch(-stickY > 0 ? INPUT_ACTIONS.EYE_SEP_INC : INPUT_ACTIONS.EYE_SEP_DEC, params);
                    } else {
                        this.dispatch(-stickY > 0 ? INPUT_ACTIONS.ZOOM_IN : INPUT_ACTIONS.ZOOM_OUT, params);
                    }
                }
            }

            if (this.activeStickAxis[i] === 'x') {
                if (tc.isTranslating && tc.activeTranslationController === controller) {
                    if (this.isStickReset[i]) {
                        this.dispatch(stickX > 0 ? INPUT_ACTIONS.SCREEN_SIZE_INC : INPUT_ACTIONS.SCREEN_SIZE_DEC, { controllerIndex: i });
                        this.isStickReset[i] = false;
                    }
                } else if (!hitInfo.isUI && !gamepad.buttons[0].pressed && !gamepad.buttons[1].pressed) {
                    if (this.isStickReset[i]) {
                        const params = { controllerIndex: i };
                        if (this.player.videoElement) this.dispatch(stickX > 0 ? INPUT_ACTIONS.SEEK_FWD : INPUT_ACTIONS.SEEK_BWD, params);
                        else this.dispatch(stickX > 0 ? INPUT_ACTIONS.NAV_NEXT : INPUT_ACTIONS.NAV_PREV, params);
                        this.isStickReset[i] = false;
                    }
                }
            }

            const menuPressed = gamepad.buttons[5]?.value > 0.5;
            if (menuPressed && !this.isMenuButtonPressed[i]) this.dispatch(INPUT_ACTIONS.MENU_TOGGLE, { autoExplorer: true });
            this.isMenuButtonPressed[i] = menuPressed;

            const aPressed = gamepad.buttons[4]?.pressed || gamepad.buttons[4]?.value > 0.5;
            if (aPressed && !this.isAButtonPressed[i]) this.dispatch(INPUT_ACTIONS.PLAY_PAUSE, { controllerIndex: i });
            this.isAButtonPressed[i] = aPressed;

            if (this.controllerRays[i]?.visible) {
                this.controllerRays[i].scale.z = hitInfo.isUI ? (hitInfo.distance / 5.0) : 1.0;
            }
        }

        // Controller Rotation (Tilt & Roll)
        if (tc.isRotating && tc.activeRotationController && this.player.renderer.xr.isPresenting) {
            tc.updateRotation(
                tc.activeRotationController,
                this.player.activeScreen
            );
        }

        // Controller Translation (Squeeze + Movement)
        if (tc.isTranslating && tc.activeTranslationController && this.player.renderer.xr.isPresenting) {
            const cIdx = this.controllers.indexOf(tc.activeTranslationController);
            tc.updateTranslation(
                tc.activeTranslationController,
                this.player.activeScreen,
                cIdx
            );
        }
    }

    getUIContext(controller) {
        if (!this.ui.visible) return { isUI: false, isExplorer: false, distance: 0 };
        
        // Primary check: raycaster
        const intersects = this.checkUIHit(controller);
        let isUI = intersects.length > 0;
        let distance = isUI ? intersects[0].distance : 0;

        // Fallback: Check uikit's own pointer intersections
        const cIdx = this.controllers.indexOf(controller);
        if (!isUI && cIdx !== -1 && this.ui.xrPointers[cIdx]) {
            const uiIntersects = this.ui.xrPointers[cIdx].intersections;
            if (uiIntersects && uiIntersects.length > 0) {
                isUI = true;
                distance = uiIntersects[0].distance;
            }
        }

        let isExplorer = false;
        if (isUI && this.ui.activeSubMenu === this.ui.explorer) {
            const pointer = (cIdx !== -1) ? this.ui.xrPointers[cIdx] : null;
            const targets = intersects.length > 0 ? intersects : (pointer ? pointer.intersections : null);
            isExplorer = targets && targets.some(ins => {
                let o = ins.object;
                while (o) { if (o === this.ui.explorer.container) return true; o = o.parent; }
                return false;
            });
        }
        return { isUI, isExplorer, distance };
    }

    dispatch(action, params = {}) {
        const delta = params.delta || 16;
        const s = this.player.activeScreen;
        const cIdx = params.controllerIndex;
        const u = this.ui;
        
        let changed = true;
        let id = null;

        switch (action) {
            case INPUT_ACTIONS.MENU_TOGGLE: {
                const autoExplorer = params.autoExplorer ?? false;
                const cIdx = params.controllerIndex ?? -1;
                if (!u.isAnyMenuVisible()) {
                    u.showMainMenu(cIdx);
                    if (autoExplorer) u.switchSubMenu(u.explorer);
                } else {
                    if (autoExplorer) {
                        if (u.activeSubMenu === u.explorer) {
                            u.hideMainMenu();
                        } else {
                            u.switchSubMenu(u.explorer);
                        }
                    } else {
                        u.toggleMainMenuVisibility();
                    }
                }
                this.player.toggleControllerUIVisibility(u.isAnyMenuVisible());
                break;
            }
            case INPUT_ACTIONS.NAV_NEXT:
            case INPUT_ACTIONS.NAV_PREV: {
                const dir = (action === INPUT_ACTIONS.NAV_NEXT) ? 1 : -1;
                const next = this.player.galleryManager.getNextIndex(dir);
                if (next !== -1) {
                    this.player.loadMedia(next);
                } else if (cIdx !== undefined) {
                    const pointer = u.xrPointers[cIdx];
                    if (pointer) UIUtils.vibratePointer(pointer.id, UI_CONFIG.hapticLimitIntensity, UI_CONFIG.hapticLimitDuration, u);
                }
                break;
            }
            case INPUT_ACTIONS.PAGE_NEXT:
            case INPUT_ACTIONS.PAGE_PREV: {
                const pDir = (action === INPUT_ACTIONS.PAGE_NEXT) ? 1 : -1;
                const prevPage = u.currentPage;
                u.changePage(pDir);
                if (u.currentPage === prevPage && cIdx !== undefined) {
                    const pointer = u.xrPointers[cIdx];
                    if (pointer) UIUtils.vibratePointer(pointer.id, UI_CONFIG.hapticLimitIntensity, UI_CONFIG.hapticLimitDuration, u);
                }
                break;
            }
            case INPUT_ACTIONS.ZOOM_IN: 
                changed = s.zoomScreen(1, delta); id = 'screen_zoom';
                u.onSliderChange(id, s.currentZoom, false);
                break;
            case INPUT_ACTIONS.ZOOM_OUT: 
                changed = s.zoomScreen(-1, delta); id = 'screen_zoom';
                u.onSliderChange(id, s.currentZoom, false);
                break;
            case INPUT_ACTIONS.EYE_SEP_INC: 
            case INPUT_ACTIONS.EYE_SEP_DEC: {
                if (!s.supportedSettings.includes('screen_eye_sep')) {
                    changed = false; id = 'screen_eye_sep';
                    break;
                }
                const dir = (action === INPUT_ACTIONS.EYE_SEP_INC) ? 1 : -1;
                changed = s.slideEyeSeparation(dir, delta); id = 'screen_eye_sep';
                u.onSliderChange(id, s.currentEyeSeparation, false);
                break;
            }
            case INPUT_ACTIONS.SCREEN_SIZE_INC:
            case INPUT_ACTIONS.SCREEN_SIZE_DEC: {
                const screenType = s.screenType;
                if (screenType === 'dome' || screenType === 'dome_xr') {
                    // Dome has no physical screen size to change via Stick X
                    changed = false; id = 'screen_height_index';
                    break;
                }

                const sizeDir = (action === INPUT_ACTIONS.SCREEN_SIZE_INC) ? 1 : -1;
                const { min, max } = getSettingLimits('screen_height_index', screenType, s);

                // Round to ensure we start from an integer step
                const currentSizeIdx = Math.round(u.settingsManager.get('screen_height_index'));
                const nextSizeIdx = THREE.MathUtils.clamp(currentSizeIdx + sizeDir, min, max);
                if (nextSizeIdx !== currentSizeIdx) {
                    u.onSliderChange('screen_height_index', nextSizeIdx, true);
                    u.showNotification(`Screen Size: ${nextSizeIdx}`, 1000);
                } else {
                    changed = false; id = 'screen_height_index';
                }
                break;
            }
            case INPUT_ACTIONS.SEEK_FWD: 
                if (this.player.videoElement) {
                    this.player.seek(this.player.videoElement.currentTime + 10);
                    this.player.savePlaybackPosition();
                }
                break;
            case INPUT_ACTIONS.SEEK_BWD: 
                if (this.player.videoElement) {
                    this.player.seek(this.player.videoElement.currentTime - 10);
                    this.player.savePlaybackPosition();
                }
                break;
            case INPUT_ACTIONS.PLAY_PAUSE:
                this.player.togglePlayPause();
                break;
        }

        if (id && !changed && cIdx !== undefined) {
            if (!this._atLimit[id]) {
                const pointer = u.xrPointers[cIdx];
                if (pointer) UIUtils.vibratePointer(pointer.id, UI_CONFIG.hapticLimitIntensity, UI_CONFIG.hapticLimitDuration, u);
                this._atLimit[id] = true;
            }
        } else if (id && changed) {
            this._atLimit[id] = false;
        }
    }
}

export { InputManager };
