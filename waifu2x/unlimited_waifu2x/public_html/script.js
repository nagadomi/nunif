var g_expires = 365;

async function check_webgpu()
{
    try {
        if (!navigator.gpu) {
            return false;
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return false;
        }
        const adapter_info = await adapter.requestAdapterInfo();
        if (!adapter_info) {
            return false;
        }
        console.log(adapter_info);
        const device = await adapter.requestDevice();
        return device ? true: false;
    } catch (e) {
        return false;
    }
};

function gen_arch_config()
{
    var config = {};

    /* swin_unet */
    config["swin_unet"] = {
        art: {color_stability: true},
        art_scan: {color_stability: false},
        photo: {color_stability: false}};
    var swin = config["swin_unet"];
    const calc_tile_size_swin_unet = function (tile_size, config) {
        while (true) {
            if ((tile_size - 16) % 12 == 0 && (tile_size - 16) % 16 == 0) {
                break;
            }
            tile_size += 1;
        }
        return tile_size;
    };
    for (const domain of ["art", "art_scan", "photo"]) {
        var base_config = {
            ...swin[domain],
            arch: "swin_unet", domain: domain, calc_tile_size: calc_tile_size_swin_unet};
        swin[domain] = {
            scale2x: {...base_config, scale: 2, offset: 16},
            scale4x: {...base_config, scale: 4, offset: 32},
            scale1x: {...base_config, scale: 1, offset: 8}, // bypass for alpha denoise
        };
        for (var i = 0; i < 4; ++i) {
            swin[domain]["noise" + i + "_scale2x"] = {...base_config, scale: 2, offset: 16};
            swin[domain]["noise" + i + "_scale4x"] = {...base_config, scale: 4, offset: 32};
            swin[domain]["noise" + i] = {...base_config, scale: 1, offset: 8};
        }
    }
    /* cunet */
    config["cunet"] = {art: {}};
    const calc_tile_size_cunet = function (tile_size, config) {
        var adj = config.scale == 1 ? 16:32;
        tile_size = ((tile_size * config.scale + config.offset * 2) - adj) / config.scale;
        tile_size -= tile_size % 4;
        return tile_size;
    };
    var base_config = {
        arch: "cunet", domain: "art", calc_tile_size: calc_tile_size_cunet,
        color_stability: true
    };
    config["cunet"]["art"] = {
        scale2x: {...base_config, scale: 2, offset: 36},
        scale1x: {...base_config, scale: 1, offset: 28}, // bypass for alpha denoise
    };
    var base = config["cunet"];
    for (var i = 0; i < 4; ++i) {
        base["art"]["noise" + i + "_scale2x"] = {...base_config, scale: 2, offset: 36};
        base["art"]["noise" + i] = {...base_config, scale: 1, offset: 28};
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

const onnx_session = {
    sessions: {},
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
    }
};

const BLEND_SIZE = 16;
const SeamBlending = class {
    // Cumulative Tile Seam/Border Blending
    // This function requires large buffers and does not work with onnxruntime's web-worker.
    // So this function is implemented in non-async pure javascript.
    // original code: nunif/utils/seam_blending.py
    constructor(x_size, scale, offset, tile_size, blend_size = BLEND_SIZE) {
        this.x_size = x_size;
        this.scale = scale;
        this.offset = offset;
        this.tile_size = tile_size;
        this.blend_size = blend_size;
    }
    async build() {
        // constructor() cannot be `async` so build members with this method
        this.param = SeamBlending.calc_parameters(
            this.x_size, this.scale, this.offset, this.tile_size, this.blend_size);
        // NOTE: Float32Array is initialized by 0
        this.pixels = new ort.Tensor(
            'float32',
            new Float32Array(this.param.y_buffer_h * this.param.y_buffer_w * 3),
            [3, this.param.y_buffer_h, this.param.y_buffer_w]);
        this.weights = new ort.Tensor(
            'float32',
            new Float32Array(this.param.y_buffer_h * this.param.y_buffer_w * 3),
            [3, this.param.y_buffer_h, this.param.y_buffer_w]);
        this.blend_filter = await this.create_seam_blending_filter();
        this.output = new ort.Tensor(
            'float32',
            new Float32Array(this.blend_filter.data.length),
            this.blend_filter.dims);
    }
    update(x, tile_i, tile_j) {
        const step_size = this.param.output_tile_step;
        const [C, H, W] = this.blend_filter.dims;
        const HW = H * W;
        const buffer_h = this.pixels.dims[1];
        const buffer_w = this.pixels.dims[2];
        const buffer_hw = buffer_h * buffer_w;
        const h_i = step_size * tile_i;
        const w_i = step_size * tile_j;

        var old_weight, next_weight, new_weight;
        for (var c = 0; c < 3; ++c) {
            for (var i = 0; i < H; ++i) {
                for (var j = 0; j < W; ++j) {
                    var tile_index = c * HW + i * W + j;
                    var buffer_index = c * buffer_hw + (h_i + i) * buffer_w + (w_i + j);
                    old_weight = this.weights.data[buffer_index];
                    next_weight = old_weight + this.blend_filter.data[tile_index];
                    old_weight = old_weight / next_weight;
                    new_weight = 1.0 - old_weight;
                    this.pixels.data[buffer_index] = (this.pixels.data[buffer_index] * old_weight +
                                                      x.data[tile_index] * new_weight);
                    this.weights.data[buffer_index] += this.blend_filter.data[tile_index];
                    this.output.data[tile_index] = this.pixels.data[buffer_index];
                }
            }
        }
        return this.output;
    }
    get_rendering_config() {
        return this.param;
    }
    static calc_parameters(x_size, scale, offset, tile_size, blend_size) {
        // from nunif/utils/seam_blending.py
        let p = {};
        const x_h = x_size[2];
        const x_w = x_size[3];

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
        return p;
    }
    async create_seam_blending_filter() {
        const ses = await onnx_session.get_session(CONFIG.get_helper_model_path("create_seam_blending_filter"));
        let scale = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.scale)]), []);
        let offset = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.offset)]), []);
        let tile_size = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.tile_size)]), []);
        let out = await ses.run({
            "scale": scale,
            "offset": offset,
            "tile_size": tile_size,
        });
        return out.y;
    }
};

const onnx_runner = {
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
    to_input: function(rgba, width, height, keep_alpha = false) {
        // HWC -> CHW
        // 0-255 -> 0.0-1.0
        if (keep_alpha) {
            const rgb = new Float32Array(height * width * 3);
            const alpha1 = new Float32Array(height * width * 1);
            const alpha3 = new Float32Array(height * width * 3);
            for (var y = 0; y < height; ++y) {
                for (var x = 0; x < width; ++x) {
                    var i = (y * width * 4) + (x * 4);
                    var j = (y * width + x);
                    rgb[j] = rgba[i + 0] / 255.0;
                    rgb[j + 1 * (height * width)] = rgba[i + 1] / 255.0;
                    rgb[j + 2 * (height * width)] = rgba[i + 2] / 255.0;
                    var alpha = rgba[i + 3] / 255.0;
                    alpha1[j] = alpha;
                    alpha3[j] = alpha;
                    alpha3[j + 1 * (height * width)] = alpha;
                    alpha3[j + 2 * (height * width)] = alpha;
                }
            }
            return [
                new ort.Tensor('float32', rgb, [1, 3, height, width]),
                new ort.Tensor('float32', alpha1, [1, 1, height, width]), // for mask
                new ort.Tensor('float32', alpha3, [1, 3, height, width])  // for upscaling with rgb input
            ];
        } else {
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
            return [new ort.Tensor('float32', rgb, [1, 3, height, width])];
        }
    },
    to_image_data: function(z, alpha3, width, height) {
        // CHW -> HWC
        // 0.0-1.0 -> 0-255
        const rgba = new Uint8ClampedArray(height * width * 4);
        if (alpha3 != null) {
            for (var y = 0; y < height; ++y) {
                for (var x = 0; x < width; ++x) {
                    var alpha_v = 0.0;
                    for (var c = 0; c < 3; ++c) {
                        var i = (y * width * 4) + (x * 4) + c;
                        var j = (y * width + x) + c * (height * width);
                        rgba[i] = (z[j] * 255.0) + 0.49999;
                        alpha_v += alpha3[j] * (1.0 / 3.0);
                    }
                    rgba[(y * width * 4) + (x * 4) + 3] = (alpha_v * 255.0) + 0.49999;
                }
            }
        } else {
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
        }
        return new ImageData(rgba, width, height);
    },
    crop_image_data: function(image_data, x, y, width, height)
    {
        const roi = new Uint8ClampedArray(height * width * 4);
        const ey = y + height;
        let i = 0;
        for (let yy = y; yy < ey; ++yy) {
            const sx = image_data.width * 4 * yy + x * 4;
            const ex = image_data.width * 4 * yy + (x + width) * 4;
            for (let j = sx; j < ex; ++j) {
                roi[i++] = image_data.data[j];
            }
        }
        return new ImageData(roi, width, height);
    },
    crop_tensor: function(bchw, x, y, width, height)
    {
        const [B, C, H, W] = bchw.dims;
        const ex = x + width;
        const ey = y + height;
        let roi = new Float32Array(B * C * height * width);
        let i = 0;
        for (let b = 0; b < B; ++b) {
            const bi = b * C * H * W;
            for (let c = 0; c < C; ++c) {
                const ci = bi + c * H * W;
                for (let h = y; h < ey; ++h) {
                    const hi = ci + h * W;
                    for (let w = x; w < ex; ++w) {
                        roi[i++] = bchw.data[hi + w];
                    }
                }
            }
        }
        return new ort.Tensor('float32', roi, [B, C, height, width]);
    },
    check_single_color: function(x, alpha3, keep_alpha=false) {
        const [B, C, H, W] = x.dims;
        let [r, g, b] = [x.data[0], x.data[1 * (H * W)], x.data[2 * (H * W)]];
        let a = 1.0;
        for (let bi = 0; bi < B; ++bi) {
            for (let h = 0; h < H; ++h) {
                for (let w = 0; w < W; ++w) {
                    let i = bi * (C * H * W) + h * W + w;
                    if (r != x.data[i + 0 * (H * W)]
                        || g != x.data[i + 1 * (H * W)]
                        || b != x.data[i + 2 * (H * W)])
                    {
                        return null;
                    }
                }
            }
        }
        if (alpha3 != null) {
            a = alpha3.data[0];
            const n = alpha3.dims[0] * alpha3.dims[1] * alpha3.dims[2] * alpha3.dims[3];
            for (let i = 0; i < n; ++i) {
                if (a != alpha3.data[i]) {
                    return null;
                }
            }
        }
        if (keep_alpha) {
            return [r, g, b, a];
        } else {
            const bg_color = 1.0;
            r = a * r + (1 - a) * bg_color;
            g = a * g + (1 - a) * bg_color;
            b = a * b + (1 - a) * bg_color;
            return [r, g, b, 1.0];
        }
    },
    check_alpha_channel: function(rgba) {
        for (var i = 0; i < rgba.length; i += 4) {
            var alpha = rgba[i + 3];
            if (alpha != 255) {
                return true;
            }
        }
        return false;
    },
    create_single_color_tensor: function(rgba, size) {
        // CHW
        var rgb = new Float32Array(size * size * 3);
        var alpha3 = new Float32Array(size * size * 3);
        alpha3.fill(rgba[3]);
        for (var c = 0; c < 3; ++c) {
            const v = rgba[c];
            for (var i = 0; i < size * size; ++i) {
                rgb[c * size * size + i] = v;
            }
        }
        return [new ort.Tensor("float32", rgb, [1, 3, size, size]),
                new ort.Tensor("float32", alpha3, [1, 3, size, size])];
    },
    shuffleArray: (array) => {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    },
    tiled_render: async function(image_data, config, alpha_config,
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
        var has_alpha = alpha_config != null;
        const model = await onnx_session.get_session(config.path);
        var alpha_model = null;
        if (has_alpha) {
            alpha_model = await onnx_session.get_session(alpha_config.path);
        }
        // preprocessing, padding
        var x = this.to_input(image_data.data, image_data.width, image_data.height, has_alpha);
        if (has_alpha) {
            var [rgb, alpha1, alpha3] = x;
            var seam_blending = new SeamBlending(rgb.dims, config.scale, config.offset, tile_size);
            var seam_blending_alpha = new SeamBlending(alpha3.dims, config.scale, config.offset, tile_size);
            await seam_blending_alpha.build();
            await seam_blending.build();

            var p = seam_blending.get_rendering_config();
            x = await this.alpha_border_padding(rgb, alpha1, BigInt(config.offset));
            // _debug_print_image_data(this.to_image_data(x.data, null, x.dims[3], x.dims[2]));
            x = await this.padding(x, BigInt(p.pad[0]), BigInt(p.pad[1]),
                                   BigInt(p.pad[2]), BigInt(p.pad[3]));
            alpha3 = await this.padding(alpha3, BigInt(p.pad[0]), BigInt(p.pad[1]),
                                        BigInt(p.pad[2]), BigInt(p.pad[3]));
            alpha1 = null;
        } else {
            var alpha3 = {data: null};
            x = x[0];
            var seam_blending = new SeamBlending(x.dims, config.scale, config.offset, tile_size);
            await seam_blending.build();
            var p = seam_blending.get_rendering_config();
            x = await this.padding(x, BigInt(p.pad[0]), BigInt(p.pad[1]),
                                   BigInt(p.pad[2]), BigInt(p.pad[3]));
        }
        var ch, h, w;
        [ch, h, w] = [x.dims[1], x.dims[2], x.dims[3]];
        var all_blocks = p.h_blocks * p.w_blocks;

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

            let tile_x = this.crop_tensor(x, j, i, tile_size, tile_size);
            let tile_alpha3 = null;
            if (has_alpha) {
                tile_alpha3 = this.crop_tensor(alpha3, j, i, tile_size, tile_size);
            }
            let single_color = (config.color_stability ?
                                this.check_single_color(tile_x, tile_alpha3, has_alpha) : null);
            if (single_color == null) {
                if (has_alpha) {
                    if (tta_level > 0) {
                        tile_x = await this.tta_split(tile_x, BigInt(tta_level));
                    }
                    var output = await model.run({x: tile_x});
                    var tile_y = output.y;
                    if (tta_level > 0) {
                        tile_y = await this.tta_merge(tile_y, BigInt(tta_level));
                    }
                    var alpha_output = await alpha_model.run({x: tile_alpha3});
                    var tile_alpha_y = alpha_output.y;
                } else {
                    if (tta_level > 0) {
                        tile_x = await this.tta_split(tile_x, BigInt(tta_level));
                    }
                    var tile_output = await model.run({x: tile_x});
                    var tile_y = tile_output.y;
                    if (tta_level > 0) {
                        tile_y = await this.tta_merge(tile_y, BigInt(tta_level));
                    }
                }
            } else {
                // no need waifu2x, tile is single color image
                var [tile_y, tile_alpha_y] = this.create_single_color_tensor(
                    single_color, tile_size * config.scale - config.offset * 2);
            }
            if (has_alpha) {
                var rgb = seam_blending.update(tile_y, h_i, w_i);
                var alpha = seam_blending_alpha.update(tile_alpha_y, h_i, w_i);
                var output_image_data = this.to_image_data(rgb.data, alpha.data,
                                                           tile_y.dims[3], tile_y.dims[2]);
            } else {
                var rgb = seam_blending.update(tile_y, h_i, w_i);
                var output_image_data = this.to_image_data(rgb.data, null,
                                                           tile_y.dims[3], tile_y.dims[2]);
            }
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
        const ses = await onnx_session.get_session(CONFIG.get_helper_model_path("pad"));
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
        const ses = await onnx_session.get_session(CONFIG.get_helper_model_path("tta_split"));
        tta_level = new ort.Tensor('int64', BigInt64Array.from([tta_level]), []);
        var out = await ses.run({
            "x": x,
            "tta_level": tta_level});
        return out.y;
    },
    tta_merge: async function(x, tta_level) {
        const ses = await onnx_session.get_session(CONFIG.get_helper_model_path("tta_merge"));
        tta_level = new ort.Tensor('int64', BigInt64Array.from([tta_level]), []);
        var out = await ses.run({
            "x": x,
            "tta_level": tta_level});
        return out.y;
    },
    alpha_border_padding: async function(rgb, alpha, offset) {
        const ses = await onnx_session.get_session(CONFIG.get_helper_model_path("alpha_border_padding"));
        // unsqueeze
        rgb = new ort.Tensor('float32', rgb.data, [rgb.dims[1], rgb.dims[2], rgb.dims[3]]);
        alpha = new ort.Tensor('float32', alpha.data, [alpha.dims[1], alpha.dims[2], alpha.dims[3]]);
        offset = new ort.Tensor('int64', BigInt64Array.from([offset]), []);
        var out = await ses.run({
            "rgb": rgb,
            "alpha": alpha,
            "offset": offset,
        });
        // squeeze
        return new ort.Tensor("float32", out.y.data, [1, out.y.dims[0], out.y.dims[1], out.y.dims[2]]);
    },
    antialias: async function(x) {
        const ses = await onnx_session.get_session(CONFIG.get_helper_model_path("antialias"));
        var out = await ses.run({"x": x});
        return out.y;
    },
};

function decode_image(image)
{
    const [width, height] = [image.naturalWidth, image.naturalHeight];
    const canvas = new OffscreenCanvas(width, height);
    const gl = canvas.getContext("webgl");
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
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
};

function _debug_print_image_data(image_data)
{
    var canvas = document.createElement("canvas");
    canvas.width = image_data.width;
    canvas.height = image_data.height;
    var ctx = canvas.getContext("2d");
    ctx.putImageData(image_data, 0, 0);
    document.body.append(canvas);
};

function check_clipboard()
{
    return ("clipboard" in navigator) && ("read" in navigator.clipboard);
}

async function read_from_clipboard()
{
    try {
        const items = await navigator.clipboard.read();
        for (const item of items) {
            const mime = item.types.find(type => type.startsWith("image/"));
            if (mime) {
                const blob = await item.getType(mime);
                return blob;
            }
        }
    } catch (e) {
        console.log(e);
    }
    return null;
}

function uuid()
{
    // ref: http://stackoverflow.com/a/2117523
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
        return v.toString(16);
    });
}

/* UI */
$(function () {
    /* init */
    ort.env.wasm.proxy = true;
    ort.env.wasm.numThreads = navigator.hardwareConcurrency;

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
        var model_name = $("select[name=model]").val();
        var [arch, style] = model_name.split(".");
        var scale = parseInt($("select[name=scale]").val());
        var noise_level = parseInt($("select[name=noise_level]").val());
        var method;
        if (scale == 1) {
            if (noise_level == -1) {
                set_message("(・A・) No Noise Reduction selected!");
                return;
            }
            method = "noise" + noise_level;
        } else if (scale == 2) {
            if (noise_level == -1) {
                method = "scale2x";
            } else {
                method = "noise" + noise_level + "_scale2x";
            }
        } else if (scale == 4) {
            if (noise_level == -1) {
                method = "scale4x";
            } else {
                method = "noise" + noise_level + "_scale4x";
            }
        }
        const config = CONFIG.get_config(arch, style, method);
        if (config == null) {
            set_message("(・A・) Model Not found!");
            return;
        }
        const tile_size = config.calc_tile_size(parseInt($("select[name=tile_size]").val()), config);
        const tile_random = $("input[name=tile_random]").prop("checked");
        const tta_level = parseInt($("select[name=tta]").val());

        var img = $("#src").get(0);
        var image_data = decode_image(img);
        $("#dest").css({width: "auto", height: "auto"});
        var output_canvas = $("#dest").get(0);
        const alpha_enabled = parseInt($("select[name=alpha]").val()) == 1;
        const has_alpha = !alpha_enabled ? false: onnx_runner.check_alpha_channel(image_data.data);
        var alpha_config = null;
        if (has_alpha) {
            var alpha_method;
            if (method.includes("scale2x")) {
                alpha_method = "scale2x";
            } else if (method.includes("scale4x")) {
                alpha_method = "scale4x";
            } else {
                alpha_method = "scale1x";
            }
            alpha_config = CONFIG.get_config(arch, style, alpha_method);
            if (alpha_config == null) {
                set_message("(・A・) Model Not found!");
                return;
            }
        }
        set_message("(・∀・)φ ... ", -1);

        await onnx_runner.tiled_render(
            image_data, config, alpha_config,
            tta_level,
            tile_size, tile_random,
            output_canvas, (progress, max_progress, processing) => {
                if (processing) {
                    progress_message = "(" + progress + "/" + max_progress + ")";
                    loop_message(["( ・∀・)" + (progress % 2 == 0 ? "φ　 ":" φ　") + progress_message,
                                  "( ・∀・)" + (progress % 2 != 0 ? "φ　 ":" φ　") + progress_message], 0.5);
                } else {
                    set_message("(・A・)!!", 1);
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
            var img = $("#src").get(0);
            img.src = reader.result;
            img.onload = () => {
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
        $("#src").get(0).src = "blank.png";
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
    $(document).on({
        dragover: function() { return false; },
        drop: function(e) {
            if (!(e.originalEvent.dataTransfer && e.originalEvent.dataTransfer.files.length)) {
                return false;
            }
            if (onnx_runner.running) {
                console.log("Already running");
                return false;
            }
            var file = e.originalEvent.dataTransfer;
            if (file.files.length > 0 && file.files[0].type.match(/image/)) {
                var files = new DataTransfer();
                files.items.add(file.files[0]);
                $("#file").get(0).files = files.files;
                $("#file").trigger("change");
                return false;
            } else {
                return false;
            }
        }
    });
    if (check_clipboard()) {
        let a = $("<a>", {id: "paste", href: "#", title: "→paste image", text: "📋"});
        a.click(async () => {
            const blob = await read_from_clipboard();
            if (blob != null) {
                const file = new File([blob], uuid(), {type: blob.type});
                let files = new DataTransfer();
                files.items.add(file);
                $("#file").get(0).files = files.files;
                $("#file").trigger("change");
            }
            return false;
        });
        $("#paste-button").append(", ");
        $("#paste-button").append(a);
    }
    $("#stop").click(() => {
        onnx_runner.stop_flag = true;
    });
    $("#src").click(() => {
        var img = $("#src").get(0);
        var css_width = parseInt($("#src").css("width"));
        if (css_width != img.naturalWidth) {
            $("#src").css({width: img.naturalWidth, height: img.naturalHeight});
        } else {
            var height = 128;
            var width = Math.floor((height / img.naturalHeight) * img.naturalWidth);
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
        if ($.cookie("model")) {
            $("select[name=model]").val($.cookie("model"));
            if (!$("select[name=model]").val()) {
                $("select[name=model]").val("swin_unet.art");
            }
        }
        if ($.cookie("noise_level")) {
            $("select[name=noise_level]").val($.cookie("noise_level"));
        }
        if ($.cookie("scale")) {
            $("select[name=scale]").val($.cookie("scale"));
        }
        if ($.cookie("tile_size")) {
            $("select[name=tile_size]").val($.cookie("tile_size"));
        }
        if ($.cookie("tile_random") == "true") {
            $("input[name=tile_random]").prop("checked", true);
        }
        if ($.cookie("tta")) {
            $("select[name=tta]").val($.cookie("tta"));
        }
        if ($.cookie("alpha")) {
            $("select[name=alpha]").val($.cookie("alpha"));
        }
    };
    restore_from_cookie();

    $("select[name=model]").change(() => {
        var model = $("select[name=model]").val();
        var [arch, style] = model.split(".");
        $.cookie("model", model, {expires: g_expires});
        if (arch == "swin_unet") {
            $("select[name=scale]").children("option[value=4]").show();
            $("#scale-comment").hide();
        } else {
            var scale = $("select[name=scale]").val();
            $("select[name=scale]").children("option[value=4]").hide();
            $("#scale-comment").show();
            if (scale == "4") {
                $("select[name=scale]").val("2");
            }
        }
        if ((style == "photo" || style == "photo_gan" || style == "art_scan") && $("select[name=tile_size]").val() < 256) {
            $("#tile-comment").show();
        } else {
            $("#tile-comment").hide();
        }
    });
    $("select[name=model]").trigger("change");
    $("select[name=noise_level]").change(() => {
        $.cookie("noise_level", $("select[name=noise_level]").val(), {expires: g_expires});
    });
    $("select[name=scale]").change(() => {
        $.cookie("scale", $("select[name=scale]").val(), {expires: g_expires});
    });
    $("select[name=tile_size]").change(() => {
        $.cookie("tile_size", $("select[name=tile_size]").val(), {expires: g_expires});

        var model = $("select[name=model]").val();
        var [arch, style] = model.split(".");
        if ((style == "photo" || style == "photo_gan" || style == "art_scan") && $("select[name=tile_size]").val() < 256) {
            $("#tile-comment").show();
        } else {
            $("#tile-comment").hide();
        }
    });
    $("input[name=tile_random]").change(() => {
        $.cookie("tile_random", $("input[name=tile_random]").prop("checked"), {expires: g_expires});
    });
    $("select[name=tta]").change(() => {
        $.cookie("tta", $("select[name=tta]").val(), {expires: g_expires});
    });
    $("select[name=alpha]").change(() => {
        $.cookie("alpha", $("select[name=alpha]").val(), {expires: g_expires});
    });
    window.addEventListener("unhandledrejection", function(e) {
        set_message("(-_-) Error: " + e.reason, -1);
        // reset running flags
        onnx_runner.running = false;
        onnx_runner.stop_flag = false;
    });
});
