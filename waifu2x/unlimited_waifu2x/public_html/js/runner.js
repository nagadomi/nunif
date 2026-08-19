import { CONFIG } from "./models.js";
import { onnxSession } from "./utils.js";
import { SeamBlending } from "./seam_blending.js";

export const onnxRunner = {
    stop_flag: false,
    running: false,

    scanline_effect(data) {
        for (let y = 0; y < data.height; ++y) {
            if (y % 2 == 0) {
                continue;
            }
            for (let x = 0; x < data.width; ++x) {
                for (let c = 0; c < 3; ++c) {
                    const i = (y * data.width * 4) + (x * 4) + c;
                    data.data[i] = data.data[i] / 1.5;
                }
            }
        }
        return data;
    },

    to_input(rgba, width, height, keep_alpha = false) {
        // HWC -> CHW
        // 0-255 -> 0.0-1.0
        if (keep_alpha) {
            const rgb = new Float32Array(height * width * 3);
            const alpha1 = new Float32Array(height * width * 1);
            const alpha3 = new Float32Array(height * width * 3);
            for (let y = 0; y < height; ++y) {
                for (let x = 0; x < width; ++x) {
                    const i = (y * width * 4) + (x * 4);
                    const j = (y * width + x);
                    rgb[j] = rgba[i + 0] / 255.0;
                    rgb[j + 1 * (height * width)] = rgba[i + 1] / 255.0;
                    rgb[j + 2 * (height * width)] = rgba[i + 2] / 255.0;
                    const alpha = rgba[i + 3] / 255.0;
                    alpha1[j] = alpha;
                    alpha3[j] = alpha;
                    alpha3[j + 1 * (height * width)] = alpha;
                    alpha3[j + 2 * (height * width)] = alpha;
                }
            }
            return [
                new ort.Tensor("float32", rgb, [1, 3, height, width]),
                new ort.Tensor("float32", alpha1, [1, 1, height, width]), // for mask
                new ort.Tensor("float32", alpha3, [1, 3, height, width])  // for upscaling with rgb input
            ];
        } else {
            const rgb = new Float32Array(height * width * 3);
            const bg_color = 1.0;
            for (let y = 0; y < height; ++y) {
                for (let x = 0; x < width; ++x) {
                    const alpha = rgba[(y * width * 4) + (x * 4) + 3] / 255.0;
                    for (let c = 0; c < 3; ++c) {
                        const i = (y * width * 4) + (x * 4) + c;
                        const j = (y * width + x) + c * (height * width);
                        rgb[j] = alpha * (rgba[i] / 255.0) + (1 - alpha) * bg_color;
                    }
                }
            }
            return [new ort.Tensor("float32", rgb, [1, 3, height, width])];
        }
    },

    to_image_data(z, alpha3, width, height) {
        // CHW -> HWC
        // 0.0-1.0 -> 0-255
        const rgba = new Uint8ClampedArray(height * width * 4);
        if (alpha3 != null) {
            for (let y = 0; y < height; ++y) {
                for (let x = 0; x < width; ++x) {
                    let alpha_v = 0.0;
                    for (let c = 0; c < 3; ++c) {
                        const i = (y * width * 4) + (x * 4) + c;
                        const j = (y * width + x) + c * (height * width);
                        rgba[i] = (z[j] * 255.0) + 0.49999;
                        alpha_v += alpha3[j] * (1.0 / 3.0);
                    }
                    rgba[(y * width * 4) + (x * 4) + 3] = (alpha_v * 255.0) + 0.49999;
                }
            }
        } else {
            rgba.fill(255);
            for (let y = 0; y < height; ++y) {
                for (let x = 0; x < width; ++x) {
                    for (let c = 0; c < 3; ++c) {
                        const i = (y * width * 4) + (x * 4) + c;
                        const j = (y * width + x) + c * (height * width);
                        rgba[i] = (z[j] * 255.0) + 0.49999;
                    }
                }
            }
        }
        return new ImageData(rgba, width, height);
    },

    crop_image_data(image_data, x, y, width, height) {
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

    crop_tensor(bchw, x, y, width, height) {
        const [B, C, H, W] = bchw.dims;
        const ex = x + width;
        const ey = y + height;
        const roi = new Float32Array(B * C * height * width);
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
        return new ort.Tensor("float32", roi, [B, C, height, width]);
    },

    check_single_color(x, alpha3, keep_alpha = false) {
        const [B, C, H, W] = x.dims;
        let [r, g, b] = [x.data[0], x.data[1 * (H * W)], x.data[2 * (H * W)]];
        let a = 1.0;
        for (let bi = 0; bi < B; ++bi) {
            for (let h = 0; h < H; ++h) {
                for (let w = 0; w < W; ++w) {
                    const i = bi * (C * H * W) + h * W + w;
                    if (r != x.data[i + 0 * (H * W)]
                        || g != x.data[i + 1 * (H * W)]
                        || b != x.data[i + 2 * (H * W)]) {
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

    check_alpha_channel(rgba) {
        for (let i = 0; i < rgba.length; i += 4) {
            const alpha = rgba[i + 3];
            if (alpha != 255) {
                return true;
            }
        }
        return false;
    },

    create_single_color_tensor(rgba, size) {
        // CHW
        const rgb = new Float32Array(size * size * 3);
        const alpha3 = new Float32Array(size * size * 3);
        alpha3.fill(rgba[3]);
        for (let c = 0; c < 3; ++c) {
            const v = rgba[c];
            for (let i = 0; i < size * size; ++i) {
                rgb[c * size * size + i] = v;
            }
        }
        return [new ort.Tensor("float32", rgb, [1, 3, size, size]),
        new ort.Tensor("float32", alpha3, [1, 3, size, size])];
    },

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    },

    async tiled_render(image_data, config, alpha_config,
        tta_level,
        tile_size, tile_random,
        output_canvas, block_callback) {
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
        const output_ctx = output_canvas.getContext("2d", { willReadFrequently: true });

        // load model
        const has_alpha = alpha_config != null;
        const model = await onnxSession.getSession(config.path);
        // TTA does not work with WebGPU, so disable it when WebGPU is selected or in auto mode.
        const is_webgpu_potential = (onnxSession.backend === "webgpu" || onnxSession.backend === "auto");
        const effective_tta_level = is_webgpu_potential ? 0 : tta_level;

        let alpha_model = null;
        if (has_alpha) {
            alpha_model = await onnxSession.getSession(alpha_config.path);
        }
        // preprocessing, padding
        let x = this.to_input(image_data.data, image_data.width, image_data.height, has_alpha);
        let alpha3;
        let seam_blending, seam_blending_alpha;
        let p;

        if (has_alpha) {
            let rgb, alpha1;
            [rgb, alpha1, alpha3] = x;
            seam_blending = new SeamBlending(rgb.dims, config.scale, config.offset, tile_size);
            seam_blending_alpha = new SeamBlending(alpha3.dims, config.scale, config.offset, tile_size);
            await seam_blending_alpha.build();
            await seam_blending.build();

            p = seam_blending.get_rendering_config();
            x = await this.alpha_border_padding(rgb, alpha1, config.offset);
            x = await this.padding(x, BigInt(p.pad[0]), BigInt(p.pad[1]),
                BigInt(p.pad[2]), BigInt(p.pad[3]), config.padding);
            alpha3 = await this.padding(alpha3, BigInt(p.pad[0]), BigInt(p.pad[1]),
                BigInt(p.pad[2]), BigInt(p.pad[3]), config.padding);
        } else {
            alpha3 = { data: null };
            x = x[0];
            seam_blending = new SeamBlending(x.dims, config.scale, config.offset, tile_size);
            await seam_blending.build();
            p = seam_blending.get_rendering_config();
            x = await this.padding(x, BigInt(p.pad[0]), BigInt(p.pad[1]),
                BigInt(p.pad[2]), BigInt(p.pad[3]), config.padding);
        }

        const all_blocks = p.h_blocks * p.w_blocks;

        // tiled rendering
        let progress = 0;
        console.time("render");
        // create index list
        const tiles = [];
        for (let h_i = 0; h_i < p.h_blocks; ++h_i) {
            for (let w_i = 0; w_i < p.w_blocks; ++w_i) {
                const i = h_i * p.input_tile_step;
                const j = w_i * p.input_tile_step;
                const ii = h_i * p.output_tile_step;
                const jj = w_i * p.output_tile_step;
                tiles.push([i, j, ii, jj, h_i, w_i]);
            }
        }
        if (tile_random) {
            // shuffle tiled rendering
            this.shuffleArray(tiles);
        }
        block_callback(0, all_blocks, true);
        for (let k = 0; k < tiles.length; ++k) {
            const [i, j, ii, jj, h_i, w_i] = tiles[k];
            let tile_x = this.crop_tensor(x, j, i, tile_size, tile_size);
            let tile_alpha3 = null;
            if (has_alpha) {
                tile_alpha3 = this.crop_tensor(alpha3, j, i, tile_size, tile_size);
            }
            const single_color = (config.color_stability ?
                this.check_single_color(tile_x, tile_alpha3, has_alpha) : null);
            
            let tile_y, tile_alpha_y;
            if (single_color == null) {
                if (has_alpha) {
                    if (effective_tta_level > 0) {
                        tile_x = await this.tta_split(tile_x, effective_tta_level);
                    }
                    const output = await model.run({ x: tile_x });
                    tile_y = output.y;
                    if (effective_tta_level > 0) {
                        tile_y = await this.tta_merge(tile_y, effective_tta_level);
                    }
                    const alpha_output = await alpha_model.run({ x: tile_alpha3 });
                    tile_alpha_y = alpha_output.y;
                } else {
                    if (effective_tta_level > 0) {
                        tile_x = await this.tta_split(tile_x, effective_tta_level);
                    }
                    const tile_output = await model.run({ x: tile_x });
                    tile_y = tile_output.y;
                    if (effective_tta_level > 0) {
                        tile_y = await this.tta_merge(tile_y, effective_tta_level);
                    }
                }
            } else {
                // no need waifu2x, tile is single color image
                [tile_y, tile_alpha_y] = this.create_single_color_tensor(
                    single_color, tile_size * config.scale - config.offset * 2);
            }

            let output_image_data;
            if (has_alpha) {
                const rgb = seam_blending.update(tile_y, h_i, w_i);
                const alpha = seam_blending_alpha.update(tile_alpha_y, h_i, w_i);
                output_image_data = this.to_image_data(rgb.data, alpha.data,
                    tile_y.dims[3], tile_y.dims[2]);
            } else {
                const rgb = seam_blending.update(tile_y, h_i, w_i);
                output_image_data = this.to_image_data(rgb.data, null,
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

    async padding(x, left, right, top, bottom, mode) {
        const ses = await onnxSession.getSession(CONFIG.get_helper_model_path(mode + "_pad"));
        const left_t = new ort.Tensor("int64", BigInt64Array.from([BigInt(left)]), []);
        const right_t = new ort.Tensor("int64", BigInt64Array.from([BigInt(right)]), []);
        const top_t = new ort.Tensor("int64", BigInt64Array.from([BigInt(top)]), []);
        const bottom_t = new ort.Tensor("int64", BigInt64Array.from([BigInt(bottom)]), []);
        const out = await ses.run({
            "x": x,
            "left": left_t, "right": right_t,
            "top": top_t, "bottom": bottom_t
        });
        return out.y;
    },

    async tta_split(x, tta_level) {
        const ses = await onnxSession.getSession(CONFIG.get_helper_model_path("tta_split"));
        const tta_level_tensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(tta_level)]), []);
        const out = await ses.run({
            "x": x,
            "tta_level": tta_level_tensor
        });
        return out.y;
    },

    async tta_merge(x, tta_level) {
        const ses = await onnxSession.getSession(CONFIG.get_helper_model_path("tta_merge"));
        const tta_level_tensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(tta_level)]), []);
        const out = await ses.run({
            "x": x,
            "tta_level": tta_level_tensor
        });
        return out.y;
    },

    async alpha_border_padding(rgb, alpha, offset) {
        const ses = await onnxSession.getSession(CONFIG.get_helper_model_path("alpha_border_padding"));
        // unsqueeze
        rgb = new ort.Tensor("float32", rgb.data, [rgb.dims[1], rgb.dims[2], rgb.dims[3]]);
        alpha = new ort.Tensor("float32", alpha.data, [alpha.dims[1], alpha.dims[2], alpha.dims[3]]);
        const offset_t = new ort.Tensor("int64", BigInt64Array.from([BigInt(offset)]), []);
        const out = await ses.run({
            "rgb": rgb,
            "alpha": alpha,
            "offset": offset_t,
        });
        // squeeze
        return new ort.Tensor("float32", out.y.data, [1, out.y.dims[0], out.y.dims[1], out.y.dims[2]]);
    },
};
