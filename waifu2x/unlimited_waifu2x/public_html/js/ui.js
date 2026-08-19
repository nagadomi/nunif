import { CONFIG } from "./models.js";
import { onnxRunner } from "./runner.js";
import { decodeImage, checkClipboard, readFromClipboard, uuid, onnxSession } from "./utils.js";

class Settings {
    static get(name, defaultValue = null) {
        const val = localStorage.getItem(name);
        return val !== null ? val : defaultValue;
    }

    static set(name, value) {
        localStorage.setItem(name, value);
    }
}

export class AppUI {
    constructor() {
        this.dom = {
            file: document.getElementById("file"),
            model: document.querySelector("select[name=model]"),
            scale: document.querySelector("select[name=scale]"),
            noiseLevel: document.querySelector("select[name=noise_level]"),
            tileSize: document.querySelector("select[name=tile_size]"),
            tileRandom: document.querySelector("input[name=tile_random]"),
            tta: document.querySelector("select[name=tta]"),
            alpha: document.querySelector("select[name=alpha]"),
            backend: document.querySelector("select[name=backend]"),
            start: document.getElementById("start"),
            stop: document.getElementById("stop"),
            src: document.getElementById("src"),
            dest: document.getElementById("dest"),
            message: document.getElementById("message"),
            scaleComment: document.getElementById("scale-comment"),
            tileComment: document.getElementById("tile-comment"),
            pasteButton: document.getElementById("paste-button"),
            ttaRow: document.getElementById("tta-row")
        };

        this.init();
    }

    init() {
        this.dom.start.addEventListener("click", () => this.handleStart());
        this.dom.stop.addEventListener("click", () => { onnxRunner.stop_flag = true; });
        this.dom.file.addEventListener("change", () => this.handleFileChange());
        
        this.dom.src.addEventListener("click", () => this.toggleSrcSize());
        this.dom.dest.addEventListener("click", () => this.toggleDestSize());

        // Model change handler
        this.dom.model.addEventListener("change", () => {
            const model = this.dom.model.value;
            const [arch, style] = model.split(".");
            Settings.set("model", model);

            if (arch === "swin_unet") {
                this.dom.scale.querySelector('option[value="4"]').style.display = "";
                this.dom.scaleComment.style.display = "none";
            } else {
                this.dom.scale.querySelector('option[value="4"]').style.display = "none";
                this.dom.scaleComment.style.display = "";
                if (this.dom.scale.value === "4") {
                    this.dom.scale.value = "2";
                }
            }
            this.updateTileComment(style);
        });

        // Settings persistence
        ["noise_level", "scale", "tile_size", "tta", "alpha", "backend"].forEach(name => {
            const el = document.querySelector(`select[name=${name}]`);
            el.addEventListener("change", () => {
                Settings.set(name, el.value);
                if (name === "tile_size") {
                    const style = this.dom.model.value.split(".")[1];
                    this.updateTileComment(style);
                }
                if (name === "backend") {
                    onnxSession.backend = el.value;
                    onnxSession.sessions = {}; // clear sessions to switch backend
                    this.updateTTAVisibility();
                }
            });
        });

        this.dom.tileRandom.addEventListener("change", () => {
            Settings.set("tile_random", this.dom.tileRandom.checked);
        });

        this.setupDragAndDrop();
        this.setupClipboard();
        this.restoreSettings();

        // Initial trigger
        this.dom.model.dispatchEvent(new Event("change"));

        window.addEventListener("unhandledrejection", (e) => {
            this.setMessage(`(-_-) Error: ${e.reason}`, -1);
            onnxRunner.running = false;
            onnxRunner.stop_flag = false;
        });
    }

    restoreSettings() {
        const savedModel = Settings.get("model");
        if (savedModel) {
            const oldValue = this.dom.model.value;
            this.dom.model.value = savedModel;
            if (!this.dom.model.value) {
                this.dom.model.value = oldValue || "swin_unet.art";
            }
        }
        ["noise_level", "scale", "tile_size", "tta", "alpha", "backend"].forEach(name => {
            const val = Settings.get(name);
            if (val) {
                const el = document.querySelector(`select[name=${name}]`);
                const oldValue = el.value;
                el.value = val;
                if (!el.value) {
                    el.value = oldValue;
                }
                if (name === "backend") {
                    onnxSession.backend = el.value;
                }
            }
        });
        this.updateTTAVisibility();
        if (Settings.get("tile_random") === "true") {
            this.dom.tileRandom.checked = true;
        }
    }

    updateTileComment(style) {
        if ((style === "photo" || style === "photo_gan" || style === "art_scan") && parseInt(this.dom.tileSize.value) < 256) {
            this.dom.tileComment.style.display = "";
        } else {
            this.dom.tileComment.style.display = "none";
        }
    }

    updateTTAVisibility() {
        const backend = this.dom.backend.value;
        if (backend === "webgpu" || backend === "auto") {
            this.dom.ttaRow.style.display = "none";
        } else {
            this.dom.ttaRow.style.display = "";
        }
    }

    setMessage(text, seconds = 2, isHtml = false) {
        if (isHtml) {
            this.dom.message.innerHTML = text;
        } else {
            this.dom.message.textContent = text;
        }

        if (seconds > 0) {
            setTimeout(() => {
                if (this.dom.message.textContent === text) {
                    this.dom.message.textContent = "( ・∀・)";
                }
            }, seconds * 1000);
        }
    }

    loopMessage(texts, interval = 0.5) {
        let i = 0;
        this.dom.message.textContent = texts[i];
        const id = setInterval(() => {
            const prevMessage = texts[i % texts.length];
            i++;
            const nextMessage = texts[i % texts.length];
            if (this.dom.message.textContent === prevMessage) {
                this.dom.message.textContent = nextMessage;
            } else {
                clearInterval(id);
            }
        }, interval * 1000);
    }

    async handleStart() {
        const file = this.dom.file.files[0];
        if (file && file.type.match(/image/)) {
            await this.process(file);
        } else {
            this.setMessage("(ﾟ∀ﾟ) No Image Found");
        }
    }

    async process(file) {
        if (onnxRunner.running) {
            console.log("Already running");
            return;
        }

        const modelName = this.dom.model.value;
        const [arch, style] = modelName.split(".");
        const scale = parseInt(this.dom.scale.value);
        const noiseLevel = parseInt(this.dom.noiseLevel.value);
        let method;

        if (scale === 1) {
            if (noiseLevel === -1) {
                this.setMessage("(・A・) No Noise Reduction selected!");
                return;
            }
            method = `noise${noiseLevel}`;
        } else {
            method = (noiseLevel === -1) ? `scale${scale}x` : `noise${noiseLevel}_scale${scale}x`;
        }

        const config = CONFIG.get_config(arch, style, method);
        if (!config) {
            this.setMessage("(・A・) Model Not found!");
            return;
        }

        const tileSize = config.calc_tile_size(parseInt(this.dom.tileSize.value), config);
        const tileRandom = this.dom.tileRandom.checked;
        const ttaLevel = parseInt(this.dom.tta.value);

        const imageData = decodeImage(this.dom.src);
        this.dom.dest.style.width = "auto";
        this.dom.dest.style.height = "auto";

        const alphaEnabled = parseInt(this.dom.alpha.value) === 1;
        const hasAlpha = alphaEnabled && onnxRunner.check_alpha_channel(imageData.data);
        let alphaConfig = null;

        if (hasAlpha) {
            const alphaMethod = method.includes("scale2x") ? "scale2x" : (method.includes("scale4x") ? "scale4x" : "scale1x");
            alphaConfig = CONFIG.get_config(arch, style, alphaMethod);
            if (!alphaConfig) {
                this.setMessage("(・A・) Model Not found!");
                return;
            }
        }

        this.setMessage("(・∀・)φ ... ", -1);

        await onnxRunner.tiled_render(
            imageData, config, alphaConfig,
            ttaLevel,
            tileSize, tileRandom,
            this.dom.dest, (progress, maxProgress, processing) => {
                if (processing) {
                    const progressMessage = `(${progress}/${maxProgress})`;
                    this.loopMessage([
                        "( ・∀・)" + (progress % 2 === 0 ? "φ　 " : " φ　") + progressMessage,
                        "( ・∀・)" + (progress % 2 !== 0 ? "φ　 " : " φ　") + progressMessage
                    ], 0.5);
                } else {
                    this.setMessage("(・A・)!!", 1);
                }
            }
        );

        if (!onnxRunner.stop_flag) {
            this.dom.dest.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const baseName = file.name.split(/(?=\.[^.]+$)/)[0];
                const filename = `${baseName}_waifu2x_${method}.png`;
                this.setMessage(`( ・∀・)つ　<a href="${url}" download="${filename}">Download</a>`, -1, true);
            }, "image/png");
        }
    }

    handleFileChange() {
        if (onnxRunner.running) {
            console.log("Already running");
            return;
        }
        const file = this.dom.file.files[0];
        if (file && file.type.match(/image/)) {
            this.setInputImage(file);
            this.setMessage("( ・∀・)b");
        } else {
            this.clearInputImage();
            this.setMessage("(ﾟ∀ﾟ)", 1);
        }
    }

    setInputImage(file) {
        const reader = new FileReader();
        reader.onload = () => {
            this.dom.src.src = reader.result;
            this.dom.src.onload = () => {
                const hScale = 128 / this.dom.src.naturalHeight;
                this.dom.src.style.width = Math.floor(hScale * this.dom.src.naturalWidth) + "px";
                this.dom.src.style.height = "128px";

                this.dom.dest.width = 128;
                this.dom.dest.height = 128;
                const ctx = this.dom.dest.getContext("2d", { willReadFrequently: true });
                ctx.clearRect(0, 0, 128, 128);
                this.dom.dest.style.width = "128px";
                this.dom.dest.style.height = "128px";
                this.dom.start.disabled = false;
            };
        };
        this.dom.start.disabled = true;
        reader.readAsDataURL(file);
    }

    clearInputImage() {
        this.dom.src.src = "blank.png";
        this.dom.src.style.width = "128px";
        this.dom.src.style.height = "128px";

        this.dom.dest.width = 128;
        this.dom.dest.height = 128;
        const ctx = this.dom.dest.getContext("2d", { willReadFrequently: true });
        ctx.clearRect(0, 0, 128, 128);
        this.dom.dest.style.width = "auto";
        this.dom.dest.style.height = "auto";
    }

    toggleSrcSize() {
        const currentWidth = parseInt(this.dom.src.style.width);
        if (currentWidth !== this.dom.src.naturalWidth) {
            this.dom.src.style.width = this.dom.src.naturalWidth + "px";
            this.dom.src.style.height = this.dom.src.naturalHeight + "px";
        } else {
            const height = 128;
            const width = Math.floor((height / this.dom.src.naturalHeight) * this.dom.src.naturalWidth);
            this.dom.src.style.width = width + "px";
            this.dom.src.style.height = height + "px";
        }
    }

    toggleDestSize() {
        const width = this.dom.dest.style.width;
        if (width === "auto" || parseInt(width) === this.dom.dest.width) {
            this.dom.dest.style.width = "60%";
            this.dom.dest.style.height = "auto";
        } else {
            this.dom.dest.style.width = "auto";
            this.dom.dest.style.height = "auto";
        }
    }

    setupDragAndDrop() {
        document.addEventListener("dragover", (e) => e.preventDefault());
        document.addEventListener("drop", (e) => {
            e.preventDefault();
            if (onnxRunner.running) return;

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.match(/image/)) {
                this.dom.file.files = files;
                this.handleFileChange();
            }
        });
    }

    setupClipboard() {
        if (checkClipboard()) {
            const a = document.createElement("a");
            a.id = "paste";
            a.href = "#";
            a.title = "→paste image";
            a.textContent = "📋";
            a.addEventListener("click", async (e) => {
                e.preventDefault();
                const blob = await readFromClipboard();
                if (blob) {
                    const file = new File([blob], uuid(), { type: blob.type });
                    const container = new DataTransfer();
                    container.items.add(file);
                    this.dom.file.files = container.files;
                    this.handleFileChange();
                }
            });
            this.dom.pasteButton.appendChild(document.createTextNode(", "));
            this.dom.pasteButton.appendChild(a);
        }
    }
}
