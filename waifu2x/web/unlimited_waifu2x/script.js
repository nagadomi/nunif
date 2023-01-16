
function gen_arch_config()
{
    var config = {};
    // [scale, offset]
    config["cunet"] = {art: {}}
    config["cunet"]["art"] = {
        scale2x: {scale: 2, offset: 36}
    };
    for (var i = 0; i < 4; ++i) {
        config["cunet"]["art"]["noise" + i + "_scale2x"] = {scale: 2, offset: 36};
        config["cunet"]["art"]["noise" + i] = {scale: 1, offset: 28};
    }
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

const onnx_utils = {
    sessions: {},
    stop_flag: false,
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
    tiled_render: async function(image_data, arch, style, method,
                                 output_canvas, block_callback)
    {
        const TILE_SIZE = 256;
        const config = CONFIG.get_config(arch, style, method);
        this.stop_flag = false; // reset flag
        if (config == null) {
            return null;
        }
        const model = await this.get_session(config.path);

        // preprocessing, padding
        var x = this.to_input(image_data.data, image_data.width, image_data.height);
        var p = this.calc_padding_param(x, config.scale, config.offset, TILE_SIZE);
        x = await this.padding(x, BigInt(p.pad[0]), BigInt(p.pad[1]), BigInt(p.pad[2]), BigInt(p.pad[3]));
        var ch, h, w;
        [ch, h, w] = [x.dims[1], x.dims[2], x.dims[3]];
        // create temporary canvas for tile input
        image_data = this.to_image_data(x.data, x.dims[3], x.dims[2]);
        var input_canvas = document.createElement("canvas");
        input_canvas.width = w;
        input_canvas.height = h;
        var input_ctx = input_canvas.getContext("2d");
        input_ctx.putImageData(image_data, 0, 0);
        // setup output canvas
        // todo
        var output_ctx = output_canvas.getContext("2d");

        // tiled rendering
        var output_size = TILE_SIZE * config.scale - config.offset * 2;
        var output_size_in_input = TILE_SIZE - Math.ceil(config.offset / config.scale) * 2;
        var all_blocks = (h / output_size_in_input) * (w / output_size_in_input);
        var processed_blocks = 0;
        for (var i = 0; i < h; i += output_size_in_input) {
            for (var j = 0; j < w; j += output_size_in_input) {
                if (!(i + TILE_SIZE <= h && j + TILE_SIZE <= w)) {
                    continue;
                }
                var ii = i * config.scale;
                var jj = j * config.scale;
                var tile_image_data = input_ctx.getImageData(j, i, j + TILE_SIZE, i + TILE_SIZE);
                var tile_x = this.to_input(tile_image_data.data,
                                           tile_image_data.width, tile_image_data.height);
                var tile_output = await model.run({x: tile_x});
                var tile_y = tile_output.y;
                var output_image_data = this.to_image_data(tile_y.data, tile_y.dims[3], tile_y.dims[2]);
                output_ctx.putImageData(output_image_data, jj, ii);
                processed_blocks += 1;
                if (this.stop_flag) {
                    block_callback(processed_blocks, all_blocks, false);
                    return;
                } else {
                    block_callback(processed_blocks, all_blocks, true);
                }
            }
        }
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
    console.log(ort.env.wasm);
    console.log(ort.env.webgl);

    var SCALE = 2;
    var OFFSET = 36;
    var reader = new FileReader();
    reader.addEventListener("load", function() {
        var img = new Image();
        img.src = reader.result;
        img.onload = async () => {
            var canvas = $("#src").get(0);
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            var ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            setup_canvas(img, 2);
            var output_canvas = $("#dest").get(0);
            var image_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
            var y = await onnx_utils.tiled_render(
                image_data, "cunet", "art", "scale2x",
                output_canvas, (progress, max_progress, stop) => {
                    console.log(progress, max_progress, stop); 
                });
            //draw_result(onnx_utils.to_image_data(y.data, y.dims[3], y.dims[2]));
        };
    });
    $("#start").click(function() {
        var file = $("#file").get(0);
        if (file.files.length > 0 && file.files[0].type.match(/image/)) {
            reader.readAsDataURL(file.files[0]);
        } else {
            set_message("( ﾟДﾟ) No Image Found");
        }
    });
    $("#stop").click(function() {
        onnx_utils.stop_flag = true;
    });
    function setup_canvas(img, scale_factor) {
        var canvas = $("#dest").get(0);
        canvas.width = img.naturalWidth * scale_factor;
        canvas.height = img.naturalHeight * scale_factor;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img,
                      0, 0, img.naturalWidth, img.naturalHeight,
                      0, 0, canvas.width, canvas.height);
        var init_image = onnx_utils.scanline_effect(ctx.getImageData(0, 0, canvas.width, canvas.height));
        ctx.putImageData(init_image, 0, 0);
    };
    function draw_result(image_data) {
        var canvas = $("#dest").get(0);
        canvas.width = image_data.height;
        canvas.height = image_data.width;
        var ctx = canvas.getContext("2d");
        ctx.putImageData(image_data, 0, 0);
    };
    function set_message(text, second=2) {
        $("#message").text(text);
        setTimeout(() => { $("#message").text("( ・∀・)") }, second * 1000);
    };
});
