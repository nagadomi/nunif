var g_expires = 365;

function gen_arch_config()
{
    var config = {};
    config["swin_unet"] = {art: {}}
    config["swin_unet"]["art"] = {
        scale2x: {scale: 2, offset: 16},
        scale4x: {scale: 4, offset: 32}};
    for (var i = 0; i < 4; ++i) {
        config["swin_unet"]["art"]["noise" + i + "_scale2x"] = {scale: 2, offset: 16};
        config["swin_unet"]["art"]["noise" + i + "_scale4x"] = {scale: 4, offset: 32};
        config["swin_unet"]["art"]["noise" + i] = {scale: 1, offset: 8};
    }
    return config;
}


const CONFIG = {
    arch: gen_arch_config(),
    get_config: function(arch, style, method) {
        if ((arch in this.arch) && (style in this.arch[arch]) && (method in this.arch[arch][style])) {
            config = this.arch[arch][style][method];
            config["path"] = `models/${arch}/${style}/${method}.onnx`;
            return config;
        } else {
            return null;
        }
    },
    get_helper_model_path: function(name) {
        return `models/utils/${name}.onnx`;
    }
};

const onnx_runner = {
    sessions: {},
    stop_flag: false,
    running: false,
    scanline_effect: function (data) {
        for (var y = 0; y < data.height; ++y) {
            if (y % 2 == 0) {
                continue;
            }
            for (var x = 0; x < data.width; ++x) {
                for (var c = 0; c < 3; ++c) {
                    var i = (y * data.width * 4) + (x * 4) + c;
                    data.data[i] = data.data[i] / 1.5;
                }
            }
        }
        return data;
    },
    to_input: function(rgba, width, height) {
        // HWC -> CHW
        // 0-255 -> 0.0-1.0
        const rgb = new Float32Array(height * width * 3);
        const bg_color = 1.0;
        for (var y = 0; y < height; ++y) {
            for (var x = 0; x < width; ++x) {
                var alpha = rgba[(y * width * 4) + (x * 4) + 3] / 255.0;
                for (var c = 0; c < 3; ++c) {
                    var i = (y * width * 4) + (x * 4) + c;
                    var j = (y * width + x) + c * (height * width);
                    rgb[j] = alpha * (rgba[i] / 255.0) + (1 - alpha) * bg_color;
                }
            }
        }
        return new ort.Tensor('float32', rgb, [1, 3, height, width])
    },
    to_image_data: function(z, width, height) {
        // CHW -> HWC
        // 0.0-1.0 -> 0-255
        const rgba = new Uint8ClampedArray(height * width * 4);
        // fill alpha
        rgba.fill(255);
        for (var y = 0; y < height; ++y) {
            for (var x = 0; x < width; ++x) {
                for (var c = 0; c < 3; ++c) {
                    var i = (y * width * 4) + (x * 4) + c;
                    var j = (y * width + x) + c * (height * width);
                    rgba[i] = (z[j] * 255.0) + 0.49999;
                }
            }
        }
        return new ImageData(rgba, width, height);
    },
    check_single_color: function(rgba) {
        const bg_color = 1.0;
        var r = rgba[0];
        var g = rgba[1];
        var b = rgba[2];
        var a = rgba[3];
        for (var i = 0; i < rgba.length; i += 4) {
            if (r != rgba[i + 0] || g != rgba[i + 1] || b != rgba[i + 2] || a != rgba[i + 3]) {
                return null;
            }
        }
        a = a / 255.0;
        r = a * (r / 255.0) + (1 - a) * bg_color;
        g = a * (g / 255.0) + (1 - a) * bg_color;
        b = a * (b / 255.0) + (1 - a) * bg_color;
        return [r, g, b];
    },
    create_single_color_tensor: function(rgb, size) {
        // CHW
        const data = new Float32Array(size * size * 3);
        for (var c = 0; c < 3; c += 1) {
            const v = rgb[c];
            for (var i = 0; i < size * size; ++i) {
                data[c * size * size + i] = v;
            }
        }
        return new ort.Tensor("float32", data, [1, 3, size, size]);
    },
    calc_parameters: function(x, scale, offset, tile_size, blend_size) {
        // from nunif/utils/seam_blending.py
        var p = {};
        const x_h = x.dims[2];
        const x_w = x.dims[3];

        p.y_h = x_h * scale;
        p.y_w = x_w * scale;

        p.input_offset = Math.ceil(offset / scale);
        p.input_blend_size = Math.ceil(blend_size / scale);
        p.input_tile_step = tile_size - (p.input_offset * 2 + p.input_blend_size);
        p.output_tile_step = p.input_tile_step * scale;

        let [h_blocks, w_blocks, input_h, input_w] = [0, 0, 0, 0];
        while (input_h < x_h + p.input_offset * 2) {
            input_h = h_blocks * p.input_tile_step + tile_size;
            ++h_blocks;
        }
        while (input_w < x_w + p.input_offset * 2) {
            input_w = w_blocks * p.input_tile_step + tile_size;
            ++w_blocks;
        }
        p.h_blocks = h_blocks;
        p.w_blocks = w_blocks;
        p.y_buffer_h = input_h * scale;
        p.y_buffer_w = input_w * scale;
        p.pad = [
            p.input_offset,
            input_w - (x_w + p.input_offset),
            p.input_offset,
            input_h - (x_h + p.input_offset)
        ];
        return p
    },
    shuffleArray: (array) => {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    },
    tiled_render: async function(image_data, config,
                                 tta_level,
                                 tile_size, tile_random,
                                 output_canvas, block_callback)
    {
        // NOTE: allowed tile_size = 64, 112, 160, 256, 400, 1024, ...
        // tile_size must be `((tile_size - 16) % 12 == 0 && (tile_size - 16) % 16 == 0)`
        this.stop_flag = false; // reset flag
        if (this.running) {
            console.log("Already running");
            return;
        }
        this.running = true;
        console.log(`tile size = ${tile_size}`);

        // setup output canvas
        output_canvas.width = image_data.width * config.scale;
        output_canvas.height = image_data.height * config.scale;
        var output_ctx = output_canvas.getContext("2d", {willReadFrequently: true});

        // load model
        const model = await this.get_session(config.path);

        // preprocessing, padding
        const blend_size = 4; // for SwinUNet models
        var x = this.to_input(image_data.data, image_data.width, image_data.height);
        var p = this.calc_parameters(x, config.scale, config.offset, tile_size, blend_size);
        x = await this.padding(x, BigInt(p.pad[0]), BigInt(p.pad[1]), BigInt(p.pad[2]), BigInt(p.pad[3]));
        var ch, h, w;
        [ch, h, w] = [x.dims[1], x.dims[2], x.dims[3]];

        // create temporary canvas for tile input
        image_data = this.to_image_data(x.data, x.dims[3], x.dims[2]);
        var input_canvas = document.createElement("canvas");
        input_canvas.width = w;
        input_canvas.height = h;
        var input_ctx = input_canvas.getContext("2d", {willReadFrequently: true});
        input_ctx.putImageData(image_data, 0, 0);

        // seam blending resources
        var all_blocks = p.h_blocks * p.w_blocks;
        var seam_blending_filter = await this.create_seam_blending_filter(
            BigInt(config.scale), BigInt(config.offset), BigInt(tile_size));
        var seam_blending_buffer = {
            weights: new ort.Tensor(
                'float32',
                new Float32Array(p.y_buffer_h * p.y_buffer_w * 3),
                [3, p.y_buffer_h, p.y_buffer_w]), // initialized by 0
            pixels: new ort.Tensor(
                'float32',
                new Float32Array(p.y_buffer_h * p.y_buffer_w * 3),
                [3, p.y_buffer_h, p.y_buffer_w]),
        };
        var seam_blending_y = new ort.Tensor(
            'float32',
            new Float32Array(seam_blending_filter.data.length),
            seam_blending_filter.dims);

        // tiled rendering
        var progress = 0;
        console.time("render");
        // create index list
        tiles = [];
        for (var h_i = 0; h_i < p.h_blocks; ++h_i) {
            for (var w_i = 0; w_i < p.w_blocks; ++w_i) {
                const i = h_i * p.input_tile_step;
                const j = w_i * p.input_tile_step;
                const ii = h_i * p.output_tile_step;
                const jj = w_i * p.output_tile_step;
                tiles.push([i, j, ii, jj, h_i, w_i])
            }
        }
        if (tile_random) {
            // shuffle tiled rendering
            this.shuffleArray(tiles);
        }
        block_callback(0, all_blocks, true);
        for (var k = 0; k < tiles.length; ++k) {
            const [i, j, ii, jj, h_i, w_i] = tiles[k];
            var tile_image_data = input_ctx.getImageData(j, i, tile_size, tile_size);
            var single_color = this.check_single_color(tile_image_data.data);
            if (single_color == null) {
                var tile_x = this.to_input(tile_image_data.data,
                                           tile_image_data.width, tile_image_data.height);
                if (tta_level > 0) {
                    tile_x = await this.tta_split(tile_x, BigInt(tta_level));
                }
                var tile_output = await model.run({x: tile_x});
                var tile_y = tile_output.y;
                if (tta_level > 0) {
                    tile_y = await this.tta_merge(tile_y, BigInt(tta_level));
                }
            } else {
                // no need waifu2x, tile is single color image
                var tile_y = this.create_single_color_tensor(
                    single_color, tile_size * config.scale - config.offset * 2);
            }
            this.seam_blending(
                seam_blending_y,
                tile_y, seam_blending_filter,
                seam_blending_buffer.pixels, seam_blending_buffer.weights,
                p.output_tile_step, h_i, w_i);
            var output_image_data = this.to_image_data(seam_blending_y.data, tile_y.dims[3], tile_y.dims[2]);
            output_ctx.putImageData(output_image_data, jj, ii);
            ++progress;
            if (this.stop_flag) {
                block_callback(progress, all_blocks, false);
                this.running = false;
                console.timeEnd("render");
                return;
            } else {
                block_callback(progress, all_blocks, true);
            }
        }
        console.timeEnd("render");
        this.running = false;
    },
    padding: async function(x, left, right, top, bottom) {
        const ses = await this.get_session(CONFIG.get_helper_model_path("pad"));
        left = new ort.Tensor('int64', BigInt64Array.from([left]), []);
        right = new ort.Tensor('int64', BigInt64Array.from([right]), []);
        top = new ort.Tensor('int64', BigInt64Array.from([top]), []);
        bottom = new ort.Tensor('int64', BigInt64Array.from([bottom]), []);
        var out = await ses.run({
            "x": x,
            "left": left, "right": right,
            "top": top, "bottom": bottom});
        return out.y;
    },
    tta_split: async function(x, tta_level) {
        const ses = await this.get_session(CONFIG.get_helper_model_path("tta_split"));
        tta_level = new ort.Tensor('int64', BigInt64Array.from([tta_level]), []);
        var out = await ses.run({
            "x": x,
            "tta_level": tta_level});
        return out.y;
    },
    tta_merge: async function(x, tta_level) {
        const ses = await this.get_session(CONFIG.get_helper_model_path("tta_merge"));
        tta_level = new ort.Tensor('int64', BigInt64Array.from([tta_level]), []);
        var out = await ses.run({
            "x": x,
            "tta_level": tta_level});
        return out.y;
    },
    create_seam_blending_filter: async function(scale, offset, tile_size) {
        const ses = await this.get_session(CONFIG.get_helper_model_path("create_seam_blending_filter"));
        scale = new ort.Tensor('int64', BigInt64Array.from([scale]), []);
        offset = new ort.Tensor('int64', BigInt64Array.from([offset]), []);
        tile_size = new ort.Tensor('int64', BigInt64Array.from([tile_size]), []);
        var out = await ses.run({
            "scale": scale,
            "offset": offset,
            "tile_size": tile_size,
        });
        return out.y;
    },
    seam_blending: function(output, x, blend_filter, pixels, weights, step_size, i, j) {
        // Cumulative Tile Seam/Border Blending
        // This function requires large buffers and does not work with onnxruntime's web-worker.
        // So this function is implemented in non-async pure javascript.
        const [C, H, W] = blend_filter.dims;
        const HW = H * W;
        const buffer_h = pixels.dims[1];
        const buffer_w = pixels.dims[2];
        const buffer_hw = buffer_h * buffer_w;
        const h_i = step_size * i;
        const w_i = step_size * j;

        var old_weight, next_weight, new_weight;
        for (var c = 0; c < 3; ++c) {
            for (var i = 0; i < H; ++i) {
                for (var j = 0; j < W; ++j) {
                    var tile_index = c * HW + i * W + j;
                    var buffer_index = c * buffer_hw + (h_i + i) * buffer_w + (w_i + j);
                    old_weight = weights.data[buffer_index];
                    next_weight = old_weight + blend_filter.data[tile_index];
                    old_weight = old_weight / next_weight;
                    new_weight = 1.0 - old_weight;
                    pixels.data[buffer_index] = pixels.data[buffer_index] * old_weight + x.data[tile_index] * new_weight;
                    weights.data[buffer_index] += blend_filter.data[tile_index];
                    output.data[tile_index] = pixels.data[buffer_index];
                }
            }
        }
    },
    get_session: async function(onnx_path) {
        if (!(onnx_path in this.sessions)) {
            try {
                this.sessions[onnx_path] = await ort.InferenceSession.create(
                    onnx_path,
                    // webgl provider does not work due to various problems
                    { executionProviders: ["wasm"] });
            } catch (error) {
                console.log(error);
                return null;
            }
        }
        return this.sessions[onnx_path];
    },
};

/* UI */
$(function () {
    /* init */
    ort.env.wasm.proxy = true;

    function removeAlpha(blob)
    {
        // TODO: I want to remove alpha channel (PNG24, not PNG32) but can't find a way.
        return blob;
    }

    async function process(file) {
        if (onnx_runner.running) {
            console.log("Already running");
            return;
        }
        var style = "art";
        var scale = parseInt($("select[name=scale]").val());
        var noise_level = parseInt($("select[name=noise_level]").val());
        var method;
        if (scale == 1) {
            if (noise_level == -1) {
                set_message("(・A・) No Noise Reduction selected!");
                return;
            }
            method = "noise" + noise_level;
            arch = "swin_unet";
        } else if (scale == 2) {
            if (noise_level == -1) {
                method = "scale2x";
            } else {
                method = "noise" + noise_level + "_scale2x";
            }
            arch = "swin_unet";
        } else if (scale == 4) {
            if (noise_level == -1) {
                method = "scale4x";
            } else {
                method = "noise" + noise_level + "_scale4x";
            }
            arch = "swin_unet";
        }
        const config = CONFIG.get_config(arch, style, method);
        if (config == null) {
            set_message("(・A・) Model Not found!");
            return;
        }
        const tile_size = parseInt($("select[name=tile_size]").val());
        const tile_random = $("input[name=tile_random]").prop("checked");
        const tta_level = parseInt($("select[name=tta]").val());

        var canvas = $("#src").get(0);
        var ctx = canvas.getContext("2d", {willReadFrequently: true});
        $("#dest").css({width: "auto", height: "auto"});
        var output_canvas = $("#dest").get(0);
        var image_data = ctx.getImageData(0, 0, canvas.width, canvas.height);

        set_message("(・∀・)φ ... ", -1);
        await onnx_runner.tiled_render(
            image_data, config,
            tta_level,
            tile_size, tile_random,
            output_canvas, (progress, max_progress, processing) => {
                if (processing) {
                    //progress_message = "(" + progress + "/" + max_progress + ")";
                    progress_message = "(" + progress + "/" + max_progress + ")";
                    loop_message(["( ・∀・)" + (progress % 2 == 0 ? "φ　 ":" φ　") + progress_message,
                                  "( ・∀・)" + (progress % 2 != 0 ? "φ　 ":" φ　") + progress_message], 0.5);
                } else {
                    set_message("(ﾟ∀ﾟ)!!", 1);
                }
            });
        if (!onnx_runner.stop_flag) {
            var output_canvas = $("#dest").get(0);
            output_canvas.toBlob((blob) => {
                // TODO: removeAlpha is not implemented
                var url = URL.createObjectURL(removeAlpha(blob));
                var filename = (file.name.split(/(?=\.[^.]+$)/))[0] + "_waifu2x_" + method + ".png";
                set_message('( ・∀・)つ　<a href="' + url +
                            '" download="' + filename  +
                            '">Download</a>', -1, true);
            }, "image/png");
        }
    };
    function set_input_image(file) {
        var reader = new FileReader();
        reader.addEventListener("load", function() {
            var img = new Image();
            img.src = reader.result;
            img.onload = () => {
                // set input canvas
                var canvas = $("#src").get(0);
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                var ctx = canvas.getContext("2d", {willReadFrequently: true});
                ctx.drawImage(img, 0, 0);
                // set input preview size
                var h_scale = 128 / img.naturalHeight;
                $("#src").css({width: Math.floor(h_scale * img.naturalWidth), height: 128});

                // clear output canvas
                var canvas = $("#dest").get(0);
                canvas.width = 128;
                canvas.height = 128;
                var ctx = canvas.getContext("2d", {willReadFrequently: true});
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                $("#dest").css({width: 128, height: 128});
                $("#start").prop("disabled", false);
            };
        });
        $("#start").prop("disabled", true);
        reader.readAsDataURL(file);
    };
    function clear_input_image(file) {
        var canvas = $("#src").get(0);
        canvas.width = 128;
        canvas.height = 128;
        var ctx = canvas.getContext("2d", {willReadFrequently: true});
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        $("#src").css({width: 128, height: 128});

        var canvas = $("#dest").get(0);
        canvas.width = 128;
        canvas.height = 128;
        var ctx = canvas.getContext("2d", {willReadFrequently: true});
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        $("#dest").css({width: "auto", height: "auto"});
    };

    function set_message(text, second=2, html=false) {
        if (html) {
            $("#message").html(text);
        } else {
            $("#message").text(text);
        }
        if (second > 0) {
            setTimeout(() => {
                if ($("#message").text() == text) {
                    $("#message").text("( ・∀・)");
                }
            }, second * 1000);
        }
    };
    function loop_message(texts, second=0.5) {
        var i = 0;
        $("#message").text(texts[i]);
        var id = setInterval(() => {
            var prev_message = texts[i % texts.length];
            ++i;
            var next_message = texts[i % texts.length];
            if ($("#message").text() == prev_message) {
                $("#message").text(next_message);
            } else {
                clearInterval(id);
            }
        }, second * 1000);
    };
    $("#start").click(async () => {
        var file = $("#file").get(0);
        if (file.files.length > 0 && file.files[0].type.match(/image/)) {
            await process(file.files[0]);
        } else {
            set_message("(ﾟ∀ﾟ) No Image Found");
        }
    });
    $("#file").change(() => {
        if (onnx_runner.running) {
            console.log("Already running");
            return;
        }
        if (file.files.length > 0 && file.files[0].type.match(/image/)) {
            set_input_image(file.files[0]);
            set_message("( ・∀・)b");
        } else {
            clear_input_image();
            set_message("(ﾟ∀ﾟ)", 1);
        }
    });
    $("#stop").click(() => {
        onnx_runner.stop_flag = true;
    });
    $("#src").click(() => {
        var canvas = $("#src").get(0);
        var css_width = parseInt($("#src").css("width"));
        if (css_width != canvas.width) {
            $("#src").css({width: canvas.width, height: canvas.height});
        } else {
            var height = 128;
            var width = Math.floor((height / canvas.height) * canvas.width);
            $("#src").css({width: width, height: height});
        }
    });
    $("#dest").click(() => {
        var width = $("#dest").css("width");
        var canvas = $("#dest").get(0);
        if (width == "auto" || parseInt(width) == canvas.width) {
            $("#dest").css({"width": "60%", "height": "auto"});
        } else {
            $("#dest").css({"width": "auto", "height": "auto"});
        }
    });

    function restore_from_cookie()
    {
        if ($.cookie("noise_level")) {
            $("select[name=noise_level]").val($.cookie("noise_level"));
        }
        if ($.cookie("scale")) {
            $("select[name=scale]").val($.cookie("scale"));
        }
        if ($.cookie("tta")) {
            $("select[name=tta]").val($.cookie("tta"));
        }
        if ($.cookie("tile_size")) {
            $("select[name=tile_size]").val($.cookie("tile_size"));
        }
        if ($.cookie("tile_random") == "true") {
            $("input[name=tile_random]").prop("checked", true);
        }
    };
    restore_from_cookie();

    $("select[name=noise_level]").change(() => {
        $.cookie("noise_level", $("select[name=noise_level]").val(), {expires: g_expires});
    });
    $("select[name=scale]").change(() => {
        $.cookie("scale", $("select[name=scale]").val(), {expires: g_expires});
    });
    $("select[name=tta]").change(() => {
        $.cookie("tta", $("select[name=tta]").val(), {expires: g_expires});
    });
    $("select[name=tile_size]").change(() => {
        $.cookie("tile_size", $("select[name=tile_size]").val(), {expires: g_expires});
    });
    $("input[name=tile_random]").change(() => {
        $.cookie("tile_random", $("input[name=tile_random]").prop("checked"), {expires: g_expires});
    });
    window.addEventListener("unhandledrejection", function(e) {
        set_message("Error: " + e.toString(), -1);
        console.error(e);
    });
});
