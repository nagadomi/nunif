export const onnxSession = {
    sessions: {},
    backend: "auto",
    async getSession(onnx_path) {
        if (!(onnx_path in this.sessions)) {
            let ep;
            if (this.backend === "webgpu") {
                ep = ["webgpu"];
            } else if (this.backend === "wasm") {
                ep = ["wasm"];
            } else {
                ep = ["webgpu", "wasm"];
            }
            try {
                this.sessions[onnx_path] = await ort.InferenceSession.create(
                    onnx_path,
                    // webgl provider does not work due to various problems
                    {
                        logSeverityLevel: 3,
                        executionProviders: ep,
                    }
                );
            } catch (error) {
                console.error(error);
                return null;
            }
        }
        return this.sessions[onnx_path];
    }
};

export function decodeImage(image) {
    const [width, height] = [image.naturalWidth, image.naturalHeight];
    const canvas = new OffscreenCanvas(width, height);
    const gl = canvas.getContext("webgl");
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    const image_data = new ImageData(width, height);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, image_data.data);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(framebuffer);

    return image_data;
}

export function checkClipboard() {
    return ("clipboard" in navigator) && ("read" in navigator.clipboard);
}

export async function readFromClipboard() {
    try {
        const items = await navigator.clipboard.read();
        for (const item of items) {
            const mime = item.types.find(type => type.startsWith("image/"));
            if (mime) {
                const blob = await item.getType(mime);
                return blob;
            }
        }
    } catch (_e) {
        console.error(_e);
    }
    return null;
}

export function uuid() {
    // ref: http://stackoverflow.com/a/2117523
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0, v = c == "x" ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

export function debugPrintImageData(image_data) {
    const canvas = document.createElement("canvas");
    canvas.width = image_data.width;
    canvas.height = image_data.height;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(image_data, 0, 0);
    document.body.append(canvas);
}
