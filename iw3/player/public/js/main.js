import * as THREE from "three";
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { ARButton } from 'three/examples/jsm/webxr/ARButton.js';
import { StereoPlayer } from './stereo_player.js';
import { DebugLog } from './debug_log.js';
import { DEFAULTS } from './constants.js';
import { storage } from './storage.js';

// Suppress "Missing glyph info" warnings from uikit/three-msdf-text-utils
const originalWarn = console.warn;
console.warn = function(...args) {
    if (typeof args[0] === 'string' && args[0].includes('Missing glyph info')) {
        return;
    }
    originalWarn.apply(console, args);
};

let camera, scene, renderer, stereoPlayer;

init();

async function init() {
    scene = new THREE.Scene();

    // Recommended near/far for WebXR to maintain high Z-buffer precision and prevent stereo flickering
    camera = new THREE.PerspectiveCamera(90, window.innerWidth / window.innerHeight, 0.05, 500);
    camera.position.set(0, 0, 0);

    // Ensure storage is ready before accessing settings
    await storage.ready;

    // Load persistent rendering settings early (Sync)
    const renderSettings = storage.get('render', {});
    const useAntialias = renderSettings.render_antialias ?? DEFAULTS.render_antialias;
    const ssFactor = renderSettings.render_ss ?? DEFAULTS.render_ss;

    renderer = new THREE.WebGLRenderer({ antialias: useAntialias, alpha: true });
    renderer.setClearAlpha(0);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    
    // Super Sampling (Scale Factor)
    renderer.xr.setFramebufferScaleFactor(ssFactor);

    // ARButton hardcodes 'local', override to maintain 'local-floor' for consistent floor height
    const origSetReferenceSpaceType = renderer.xr.setReferenceSpaceType.bind(renderer.xr);
    renderer.xr.setReferenceSpaceType = (type) => {
        return origSetReferenceSpaceType(type === 'local' ? 'local-floor' : type);
    };
    
    renderer.localClippingEnabled = true;
    document.body.appendChild(renderer.domElement);

    const xrButtons = document.createElement('div');
    xrButtons.id = 'xrButtons';
    document.body.appendChild(xrButtons);

    const vrButton = VRButton.createButton(renderer, { 
        optionalFeatures: ['local-floor', 'bounded-floor', 'layers'] 
    });
    xrButtons.appendChild(vrButton);

    const arButton = ARButton.createButton(renderer, {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['bounded-floor', 'layers']
    });
    xrButtons.appendChild(arButton);

    let isARSupported = false;
    if ('xr' in navigator) {
        navigator.xr.isSessionSupported('immersive-ar').then((supported) => {
            isARSupported = supported;
            if (!supported && arButton) {
                arButton.style.display = 'none';
            }
        }).catch(() => {
            isARSupported = false;
            if (arButton) arButton.style.display = 'none';
        });
    }

    const debugLog = new DebugLog(scene);

    // High quality VR settings
    renderer.xr.addEventListener('sessionstart', () => {
        const resetBtn = document.getElementById('factoryResetButton');
        if (resetBtn) resetBtn.style.display = 'none';

        const session = renderer.xr.getSession();
        const blendMode = renderer.xr.getEnvironmentBlendMode?.() ?? session?.environmentBlendMode;
        const isAR = (blendMode === 'alpha-blend' || blendMode === 'additive');

        debugLog.log(`[XR] sessionstart: isAR=${isAR}, blendMode=${blendMode}`);

        if (isAR) {
            if (vrButton) vrButton.style.display = 'none';
        } else if (blendMode === 'opaque') {
            if (arButton) arButton.style.display = 'none';
        }

        // Ensure high quality rendering without foveation artifacts
        renderer.xr.setFoveation(0.0);

        // Frame Rate control
        if (session && session.supportedFrameRates) {
            const supported = Array.from(session.supportedFrameRates).sort((a, b) => a - b);
            stereoPlayer.uiManager.availableFrameRates = [0, ...supported];
            stereoPlayer.uiManager.updateTargetFPS(renderSettings.render_fps || 0);
        }
    });

    renderer.xr.addEventListener('sessionend', () => {
        const resetBtn = document.getElementById('factoryResetButton');
        if (resetBtn) resetBtn.style.display = '';
        if (vrButton) vrButton.style.display = '';
        if (arButton) arButton.style.display = isARSupported ? '' : 'none';
    });

    stereoPlayer = new StereoPlayer(renderer, camera, scene, debugLog);
    scene.add(stereoPlayer);

    const clock = new THREE.Clock();
    renderer.setAnimationLoop(() => {
        const delta = clock.getDelta() * 1000; // ms
        stereoPlayer.handleAnimation(delta);
        renderer.render(scene, camera);
    });

    window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
    if (!renderer.xr.isPresenting) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
}
