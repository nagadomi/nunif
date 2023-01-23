
function gen_arch_config()
{
    var config = {};
    // [scale, offset]
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
            config["path"] = "models/" + arch + "/" + style + "/" + method + ".onnx";
            return config;
        } else {
            return null;
        }
    },
    get_helper_path: function(name) {
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
        const rgb = new Float32Array(height * width * 3);
        const bg_color = 1.0;
        // HWC -> CHW
        // 0-255 -> 0.0-1.0
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
    create_single_color_image_data: function(rgb, size) {
        const rgba = new Uint8ClampedArray(size * size * 4);
        var r = rgb[0] * 255;
        var g = rgb[1] * 255;
        var b = rgb[2] * 255;
        for (var i = 0; i < rgba.length; i += 4) {
            rgba[i + 0] = r;
            rgba[i + 1] = g;
            rgba[i + 2] = b;
            rgba[i + 3] = 255;
        }
        return new ImageData(rgba, size, size);
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
            h_blocks += 1;
        }
        while (input_w < x_w + p.input_offset * 2) {
            input_w = w_blocks * p.input_tile_step + tile_size;
            w_blocks += 1;
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
    find_tile_size: (tile_size, image_data, config) => {
        const min_size = Math.max(Math.max(image_data.width, image_data.height) + config.offset * 2, 64);
        if (min_size > tile_size) {
            return tile_size;
        }
        for (var i = min_size; i < min_size * 4; ++i) {
            if ((i - 16) % 12 == 0 && (i - 16) % 16 == 0) {
                if (i < tile_size) {
                    tile_size = i;
                }
                return tile_size;
            }
        }
        return 64; // default
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
        tile_size = this.find_tile_size(tile_size, image_data, config);
        console.log(`tile size = ${tile_size}`);

        // setup output canvas
        output_canvas.width = image_data.width * config.scale;
        output_canvas.height = image_data.height * config.scale;
        var output_ctx = output_canvas.getContext("2d", {willReadFrequently: true});

        // load model
        const model = await this.get_session(config.path);

        // preprocessing, padding
        var x = this.to_input(image_data.data, image_data.width, image_data.height);
        var p = this.calc_parameters(x, config.scale, config.offset, tile_size, 0);
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

        // tiled rendering
        var all_blocks = p.h_blocks * p.w_blocks;
        var progress = 0;

        console.time("render");
        // create index list
        tiles = [];
        for (var h_i = 0; h_i < p.h_blocks; h_i += 1) {
            for (var w_i = 0; w_i < p.w_blocks; w_i += 1) {
                const i = h_i * p.input_tile_step;
                const j = w_i * p.input_tile_step;
                const ii = h_i * p.output_tile_step;
                const jj = w_i * p.output_tile_step;
                tiles.push([i, j, ii, jj])
            }
        }
        if (tile_random) {
            this.shuffleArray(tiles);
        }
        block_callback(0, all_blocks, true);
        for (var k = 0; k < tiles.length; ++k) {
            const [i, j, ii, jj] = tiles[k];
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
                var output_image_data = this.to_image_data(tile_y.data, tile_y.dims[3], tile_y.dims[2]);
            } else {
                var output_image_data = this.create_single_color_image_data(
                    single_color, tile_size * config.scale - config.offset * 2);
            }
            output_ctx.putImageData(output_image_data, jj, ii);
            progress += 1;
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
        const ses = await this.get_session(CONFIG.get_helper_path("pad"));
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
        const ses = await this.get_session(CONFIG.get_helper_path("tta_split"));
        tta_level = new ort.Tensor('int64', BigInt64Array.from([tta_level]), []);
        var out = await ses.run({
            "x": x,
            "tta_level": tta_level});
        return out.y;
    },
    tta_merge: async function(x, tta_level) {
        const ses = await this.get_session(CONFIG.get_helper_path("tta_merge"));
        tta_level = new ort.Tensor('int64', BigInt64Array.from([tta_level]), []);
        var out = await ses.run({
            "x": x,
            "tta_level": tta_level});
        return out.y;
    },
    create_seam_blending_filter: async function(scale, offset, tile_size) {
        const ses = await this.get_session(CONFIG.get_helper_path("create_seam_blending_filter"));
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

    get_session: async function(onnx_path) {
        if (!(onnx_path in this.sessions)) {
            try {
                this.sessions[onnx_path] = await ort.InferenceSession.create(
                    onnx_path,
                    // webgl provider does not work due to various problems
                    {executionProviders: ["wasm"]});
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
            i += 1;
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
            var min_size = Math.min(canvas.width, canvas.height);
            if (min_size > 480) {
                var scale = 480 / min_size;
                $("#dest").css({"width": Math.floor(scale * canvas.width),
                                "height": Math.floor(scale * canvas.height)});
            }
        } else {
            $("#dest").css({"width": "auto", "height": "auto"});
        }
    });
});
