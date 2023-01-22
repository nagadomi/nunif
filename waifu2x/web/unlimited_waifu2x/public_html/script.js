
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
        if (name == "pad") {
            return "models/utils/pad.onnx";
        } else {
            return null;
        }
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
    calc_padding_param: function(x, scale, offset, tile_size) {
        // from nunif/utils/render.py
        var p = {};
        p.x_h = x.dims[2];
        p.x_w = x.dims[3];
        input_offset = Math.ceil(offset / scale);
        process_size = tile_size - input_offset * 2;
        p.h_blocks = Math.floor(p.x_h / process_size) + (p.x_h % process_size == 0 ? 0:1);
        p.w_blocks = Math.floor(p.x_w / process_size) + (p.x_w % process_size == 0 ? 0:1);
        h = (p.h_blocks * process_size) + input_offset * 2;
        w = (p.w_blocks * process_size) + input_offset * 2;
        p.pad = [input_offset, (w - input_offset) - p.x_w, input_offset, (h - input_offset) - p.x_h];
        p.z_h = Math.floor(p.x_h * scale);
        p.z_w = Math.floor(p.x_w * scale);
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
    tiled_render: async function(image_data, config, tile_size, tile_random,
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
        var p = this.calc_padding_param(x, config.scale, config.offset, tile_size);
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
        var output_size = tile_size * config.scale - config.offset * 2;
        var output_size_in_input = tile_size - Math.ceil(config.offset / config.scale) * 2;
        var all_blocks = p.h_blocks * p.w_blocks;
        var progress = 0;

        console.time("render");
        // create index list
        tiles = [];
        for (var i = 0; i < h; i += output_size_in_input) {
            for (var j = 0; j < w; j += output_size_in_input) {
                if (!(i + tile_size <= h && j + tile_size <= w)) {
                    continue;
                }
                var ii = i * config.scale;
                var jj = j * config.scale;
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
            var tile_x = this.to_input(tile_image_data.data,
                                       tile_image_data.width, tile_image_data.height);
            var tile_output = await model.run({x: tile_x});
            var tile_y = tile_output.y;
            var output_image_data = this.to_image_data(tile_y.data, tile_y.dims[3], tile_y.dims[2]);
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
    ort.env.debug = true;
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
                set_message("(ﾟдﾟ) No Noise Reduction selected!");
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
            set_message("(ﾟдﾟ) Model Not found!");
            return;
        }
        const tile_size = parseInt($("select[name=tile_size]").val());
        const tile_random = $("input[name=tile_random]").prop("checked");

        var canvas = $("#src").get(0);
        var ctx = canvas.getContext("2d", {willReadFrequently: true});
        $("#dest").css({"width": "auto", "height": "auto"});
        var output_canvas = $("#dest").get(0);
        var image_data = ctx.getImageData(0, 0, canvas.width, canvas.height);

        set_message("(・∀・)φ ... ", -1);
        await onnx_runner.tiled_render(
            image_data, config, tile_size, tile_random,
            output_canvas, (progress, max_progress, processing) => {
                if (processing) {
                    progress_message = "(" + progress + "/" + max_progress + ")";
                    loop_message(["( ・∀・)" + (progress % 2 == 0 ? "φ　 ":" φ　") + progress_message,
                                  "( ・∀・)" + (progress % 2 != 0 ? "φ　 ":" φ　") + progress_message], 0.5);
                } else {
                    set_message("(ﾟдﾟ) !!", 1);
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
            img.onload = async () => {
                // set input canvas
                var canvas = $("#src").get(0);
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                var ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0);
                // set input preview size
                var h_scale = 128 / img.naturalHeight;
                $("#src").css({width: Math.floor(h_scale * img.naturalWidth), height: 128});

                // clear output canvas
                var canvas = $("#dest").get(0);
                canvas.width = 128;
                canvas.height = 128;
                var ctx = canvas.getContext("2d");
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                $("#dest").css({width: 128, height: 128});
            };
        });
        reader.readAsDataURL(file);
    };
    function clear_input_image(file) {
        var canvas = $("#src").get(0);
        canvas.width = 128;
        canvas.height = 128;
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        $("#src").css({width: 128, height: 128});

        var canvas = $("#dest").get(0);
        canvas.width = 128;
        canvas.height = 128;
        var ctx = canvas.getContext("2d");
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
            set_message("( ﾟДﾟ) No Image Found");
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
            set_message("(・A・)", 1);
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
            if (min_size > 320) {
                var scale = 320 / min_size;
                $("#dest").css({"width": Math.floor(scale * canvas.width),
                                "height": Math.floor(scale * canvas.height)});
            }
        } else {
            $("#dest").css({"width": "auto", "height": "auto"});
        }
    });
});
