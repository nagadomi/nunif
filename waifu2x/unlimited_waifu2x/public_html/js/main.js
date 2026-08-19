import { AppUI } from "./ui.js";

document.addEventListener("DOMContentLoaded", () => {
    // onnxruntime-web configuration
    if (typeof ort !== "undefined") {
        ort.env.wasm.proxy = true;
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    }

    // Initialize UI
    new AppUI();
});
