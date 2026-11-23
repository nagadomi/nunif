function disable_buttons() {
    const btn = document.getElementById("submit-button");
    if (btn) {
        btn.disabled = true;
    }
}

function enable_buttons() {
    const btn = document.getElementById("submit-button");
    if (btn) {
        btn.disabled = false;
    }
}

function on_recap_checked() {
    enable_buttons();
}

function on_turnstile_success(token) {
    const input = document.getElementById("turnstile_response");
    if (input) {
        input.value = token;
    }
    enable_buttons();
}

document.addEventListener("DOMContentLoaded", () => {
    const g_expires = 365;
    const recaptcha_js = "https://www.recaptcha.net/recaptcha/api.js";
    let turnstile_widgetid;
    let turnstile_is_enabled = false;
    let g_last_fingerprint = "";
    let g_error_count = 0;

    function set_cookie(name, value, days) {
        const date = new Date();
        date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
        document.cookie = `${encodeURIComponent(name)}=${encodeURIComponent(value)}; expires=${date.toUTCString()}; path=/; SameSite=Lax`;
    }

    function get_cookie(name) {
        const prefix = `${encodeURIComponent(name)}=`;
        const parts = document.cookie ? document.cookie.split(";") : [];
        for (const rawPart of parts) {
            const part = rawPart.trimStart();
            if (part.startsWith(prefix)) {
                try {
                    return decodeURIComponent(part.slice(prefix.length));
                } catch {
                    return "";
                }
            }
        }
        return "";
    }

    function get_radio_value(name) {
        const checked = document.querySelector(`input[name="${name}"]:checked`);
        return checked ? checked.value : "";
    }

    function set_radio_value(name, value) {
        if (!value) {
            return;
        }
        const inputs = document.querySelectorAll(`input[name="${name}"]`);
        for (const input of inputs) {
            if (input.value === value) {
                input.checked = true;
                break;
            }
        }
    }

    function get_fingerprint() {
        const urlInput = document.getElementById("url");
        const fileInput = document.getElementById("file");
        const obj = {
            style: get_radio_value("style"),
            noise: get_radio_value("noise"),
            scale: get_radio_value("scale"),
            format: get_radio_value("format"),
            url: urlInput ? urlInput.value : "",
        };
        const file = fileInput && fileInput.files ? fileInput.files[0] : null;
        if (file) {
            obj.file = { name: file.name, size: file.size, lastModified: file.lastModified };
        }
        return JSON.stringify(obj);
    }

    function parse_error_html(html) {
        const title_match = /<title\b[^>]*>([\s\S]*?)<\/title>/i.exec(html);
        const pre_match = /<body\b[^>]*>[\s\S]*?<pre\b[^>]*>([\s\S]*?)<\/pre>/i.exec(html);
        const title = title_match ? title_match[1].trim() : "";
        const body_pre = pre_match ? pre_match[1].trim() : "";
        let text = "";
        if (title) {
            text += title;
        }
        if (body_pre) {
            text += (text ? "\n" : "") + body_pre;
        }
        return text;
    }

    function set_error_message(text, second, is_html, status) {
        const errorEl = document.getElementById("error_message");
        if (!errorEl) {
            return;
        }
        const count = ++g_error_count;
        const timeout_sec = second || 2;
        const display_text = is_html ? parse_error_html(text) : text;

        if (display_text) {
            errorEl.textContent = display_text;
        } else if (status) {
            errorEl.textContent = `HTTP Error (${status})`;
        } else {
            errorEl.textContent = text;
        }
        errorEl.style.display = "block";

        setTimeout(() => {
            if (count === g_error_count) {
                errorEl.style.display = "none";
            }
        }, timeout_sec * 1000);
    }

    function reset_turnstile() {
        if (typeof turnstile !== "undefined" && turnstile_widgetid !== undefined) {
            turnstile.reset(turnstile_widgetid);
        }
    }

    function clear_file() {
        const fileInput = document.getElementById("file");
        if (fileInput) {
            fileInput.value = "";
        }
    }

    function clear_url() {
        const urlInput = document.getElementById("url");
        if (urlInput) {
            urlInput.value = "";
        }
    }

    function update_title() {
        const style = get_radio_value("style");
        const titles = document.querySelectorAll(".main-title");
        for (const el of titles) {
            if (style !== "photo") {
                el.textContent = "waifu2x";
            } else {
                el.innerHTML = "w<s>/a/</s>ifu2x";
            }
        }
    }

    function on_change_style() {
        update_title();
        const style = get_radio_value("style");
        if (style) {
            set_cookie("style", style, g_expires);
        }
    }

    function on_change_noise_level() {
        const noise = get_radio_value("noise");
        if (noise) {
            set_cookie("noise", noise, g_expires);
        }
    }

    function on_change_scale_factor() {
        const scale = get_radio_value("scale");
        if (scale) {
            set_cookie("scale", scale, g_expires);
        }
    }

    function on_change_format() {
        const format = get_radio_value("format");
        if (format) {
            set_cookie("format", format, g_expires);
        }
    }

    function commit_recap_response(reset = true) {
        if (typeof grecaptcha !== "undefined") {
            console.log("recaptcha: enabled");
            const recapInput = document.getElementById("recap_response");
            if (recapInput) {
                recapInput.value = grecaptcha.getResponse();
            }
            if (reset) {
                grecaptcha.reset();
                disable_buttons();
            }
        } else {
            console.log("recaptcha: disabled");
        }
    }

    function commit_turnstile_response(reset = true) {
        if (turnstile_is_enabled) {
            console.log("turnstile: enabled");
            if (reset) {
                reset_turnstile();
                disable_buttons();
            }
        } else {
            console.log("turnstile: disabled");
        }
    }

    function restore_from_cookie() {
        set_radio_value("style", get_cookie("style"));
        set_radio_value("noise", get_cookie("noise"));
        set_radio_value("scale", get_cookie("scale"));
        set_radio_value("format", get_cookie("format"));
    }

    function uuid() {
        // ref: http://stackoverflow.com/a/2117523
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = (Math.random() * 16) | 0;
            const v = c === "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    }

    function extract_filename(disposition, default_val) {
        if (disposition && disposition.indexOf("filename") !== -1) {
            const reg = /filename[^;=\n]*=(?:(\\?['"])(.*?)\1|(?:[^\s]+'.*?')?([^;\n]*))/i;
            const matches = reg.exec(disposition);
            if (matches && matches.length >= 4 && matches[3]) {
                try {
                    return decodeURI(matches[3]);
                } catch {
                    return default_val;
                }
            }
        }
        return default_val;
    }

    function webp_to_png(blob, callback) {
        console.log("Use webp_to_png");
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
            const width = img.width;
            const height = img.height;
            let canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const gl = canvas.getContext("webgl");
            if (gl) {
                gl.activeTexture(gl.TEXTURE0);
                const texture = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                const framebuffer = gl.createFramebuffer();
                gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
                const image_data = new ImageData(width, height);
                gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, image_data.data);

                let has_alpha = false;
                for (let i = 3; i < image_data.data.length; i += 4) {
                    if (image_data.data[i] < 255) {
                        has_alpha = true;
                        break;
                    }
                }

                const out_canvas = document.createElement("canvas");
                out_canvas.width = width;
                out_canvas.height = height;
                const out_ctx = out_canvas.getContext("2d", { alpha: has_alpha });
                out_ctx.putImageData(image_data, 0, 0);
                canvas = out_canvas;

                gl.deleteTexture(texture);
                gl.deleteFramebuffer(framebuffer);
            } else {
                canvas.getContext("2d").drawImage(img, 0, 0);
            }
            canvas.toBlob((pngBlob) => {
                URL.revokeObjectURL(url);
                callback(pngBlob);
            }, "image/png");
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            callback(blob);
        };
        img.src = url;
    }

    async function request_image() {
        if (
            typeof window.URL.createObjectURL === "undefined" ||
            typeof window.Blob === "undefined" ||
            typeof window.fetch === "undefined" ||
            typeof window.URL.revokeObjectURL === "undefined"
        ) {
            return;
        }

        const current_fingerprint = get_fingerprint();
        if (current_fingerprint === g_last_fingerprint) {
            set_error_message("Same request.");
            return;
        }

        const fileInput = document.getElementById("file");
        const urlInput = document.getElementById("url");
        const file = fileInput && fileInput.files ? fileInput.files[0] : null;
        const url = urlInput ? urlInput.value : "";
        if (!file && !url) {
            set_error_message("Select a file or enter a URL.");
            return;
        }

        const errorEl = document.getElementById("error_message");
        if (errorEl) {
            errorEl.style.display = "none";
        }

        const item = document.createElement("div");
        item.className = "result-item";
        const loading = document.createElement("div");
        loading.className = "item-loading";
        const loadingImg = document.createElement("img");
        loadingImg.src = "loading.webp";
        loading.appendChild(loadingImg);
        item.appendChild(loading);

        const resultEl = document.getElementById("result");
        const resultWrapper = document.getElementById("result_wrapper");
        if (resultEl) {
            resultEl.prepend(item);
        }
        if (resultWrapper) {
            resultWrapper.style.display = "block";
        }

        const form = document.querySelector("form");
        if (!form) {
            loading.remove();
            return;
        }
        commit_recap_response(false);
        const formData = new FormData(form);
        commit_recap_response(true);
        commit_turnstile_response(true);
        disable_buttons();

        try {
            const response = await fetch("/api", {
                method: "POST",
                body: formData,
            });
            loading.remove();
            if (!turnstile_is_enabled && typeof grecaptcha === "undefined") {
                enable_buttons();
            }

            if (response.status === 200) {
                g_last_fingerprint = current_fingerprint;
                const contentType = response.headers.get("Content-Type");
                const blob = await response.blob();
                const format = get_radio_value("format");
                const disposition = response.headers.get("Content-Disposition");

                const proceed = (finalBlob) => {
                    const finalType = finalBlob.type;
                    const ext = finalType === "image/webp" ? "webp" : "png";
                    let filename = extract_filename(disposition, `${uuid()}.${ext}`);
                    if (ext === "png") {
                        filename = filename.replace(/\.webp$/i, ".png");
                        if (!filename.match(/\.png$/i)) {
                            filename += ".png";
                        }
                    } else {
                        filename = filename.replace(/\.png$/i, ".webp");
                        if (!filename.match(/\.webp$/i)) {
                            filename += ".webp";
                        }
                    }

                    const objUrl = URL.createObjectURL(finalBlob);
                    const img = document.createElement("img");
                    img.src = objUrl;
                    img.alt = filename;
                    img.title = filename;
                    img.addEventListener("click", () => {
                        img.classList.toggle("zoomed");
                    });

                    const link = document.createElement("a");
                    link.href = objUrl;
                    link.download = filename;
                    link.className = "download-link";
                    link.textContent = `Download ${filename}`;

                    item.appendChild(link);
                    item.appendChild(img);
                };

                if (format === "0" && contentType && contentType.toLowerCase().startsWith("image/webp")) {
                    webp_to_png(blob, proceed);
                } else {
                    proceed(blob);
                }
            } else {
                const contentType = response.headers.get("Content-Type");
                let error_text = `HTTP Error (${response.status})`;
                const is_html = Boolean(contentType && contentType.indexOf("text/html") !== -1);
                try {
                    const decoded = await response.text();
                    if (decoded) {
                        error_text = is_html ? parse_error_html(decoded) : decoded;
                        if (!error_text) {
                            error_text = `HTTP Error (${response.status})`;
                        }
                    }
                } catch (err) {}
                const errorDiv = document.createElement("div");
                errorDiv.className = "item-error";
                errorDiv.textContent = error_text;
                item.appendChild(errorDiv);
            }
        } catch (err) {
            loading.remove();
            if (!turnstile_is_enabled && typeof grecaptcha === "undefined") {
                enable_buttons();
            }
            const errorDiv = document.createElement("div");
            errorDiv.className = "item-error";
            errorDiv.textContent = "Network Error";
            item.appendChild(errorDiv);
        }
    }

    async function load_recaptcha() {
        try {
            const response = await fetch("/recaptcha_state.json");
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            if (data.recaptcha_enabled) {
                console.log("recaptcha is enabled");
                const container = document.getElementById("recap_container");
                if (container) {
                    const div = document.createElement("div");
                    div.className = "g-recaptcha";
                    div.setAttribute("data-sitekey", data.recaptcha_site_key);
                    div.setAttribute("data-callback", "on_recap_checked");
                    container.appendChild(div);
                }
                const script = document.createElement("script");
                script.type = "text/javascript";
                script.src = recaptcha_js;
                document.head.appendChild(script);
                disable_buttons();
            } else if (data.turnstile_enabled) {
                if (typeof turnstile !== "undefined") {
                    console.log("turnstile is enabled");
                    turnstile_is_enabled = true;
                    turnstile_widgetid = turnstile.render("#turnstile_container", {
                        sitekey: data.turnstile_site_key,
                        callback: on_turnstile_success,
                    });
                    disable_buttons();
                } else {
                    console.log("turnstile is enabled but script not loaded");
                }
            } else {
                console.log("recaptcha/turnstile is disabled");
            }
        } catch (e) {
            console.log(e);
        }
    }

    function set_param() {
        const params = new URLSearchParams(window.location.search);
        const url = params.get("url");
        const style = params.get("style");
        const noise = params.get("noise");
        const scale = params.get("scale");
        const format = params.get("format");

        if (url) {
            const urlInput = document.querySelector('input[name="url"]');
            if (urlInput) {
                urlInput.value = url;
            }
        }
        set_radio_value("style", style);
        set_radio_value("noise", noise);
        set_radio_value("scale", scale);
        set_radio_value("format", format);
    }

    const urlInput = document.getElementById("url");
    if (urlInput) {
        urlInput.addEventListener("change", clear_file);
    }
    const fileInput = document.getElementById("file");
    if (fileInput) {
        fileInput.addEventListener("change", clear_url);
    }

    for (const el of document.querySelectorAll('input[name="style"]')) {
        el.addEventListener("change", on_change_style);
    }
    for (const el of document.querySelectorAll('input[name="noise"]')) {
        el.addEventListener("change", on_change_noise_level);
    }
    for (const el of document.querySelectorAll('input[name="scale"]')) {
        el.addEventListener("change", on_change_scale_factor);
    }
    for (const el of document.querySelectorAll('input[name="format"]')) {
        el.addEventListener("change", on_change_format);
    }

    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", (e) => {
            if (document.getElementById("result")) {
                e.preventDefault();
                request_image();
            } else {
                commit_recap_response();
                commit_turnstile_response();
            }
        });
    }

    const clearBtn = document.getElementById("clear_results");
    if (clearBtn) {
        clearBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            location.reload();
        });
    }

    document.addEventListener("dragover", (e) => {
        e.preventDefault();
    });
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) {
            return;
        }
        const dropped = e.dataTransfer.files[0];
        if (dropped && dropped.type.match(/image/) && fileInput) {
            const files = new DataTransfer();
            files.items.add(dropped);
            fileInput.files = files.files;
            fileInput.dispatchEvent(new Event("change"));
        }
    });

    restore_from_cookie();
    on_change_style();
    on_change_scale_factor();
    on_change_noise_level();
    on_change_format();
    set_param();
    update_title();
    load_recaptcha();
});
