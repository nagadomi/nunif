import { CONFIG } from "./models.js";
import { onnxSession } from "./utils.js";

const BLEND_SIZE = 16;

export class SeamBlending {
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
            "float32",
            new Float32Array(this.param.y_buffer_h * this.param.y_buffer_w * 3),
            [3, this.param.y_buffer_h, this.param.y_buffer_w]);
        this.weights = new ort.Tensor(
            "float32",
            new Float32Array(this.param.y_buffer_h * this.param.y_buffer_w * 3),
            [3, this.param.y_buffer_h, this.param.y_buffer_w]);
        this.blend_filter = await this.create_seam_blending_filter();
        this.output = new ort.Tensor(
            "float32",
            new Float32Array(this.blend_filter.data.length),
            this.blend_filter.dims);
    }

    update(x, tile_i, tile_j) {
        const step_size = this.param.output_tile_step;
        const [_, H, W] = this.blend_filter.dims;
        const HW = H * W;
        const buffer_h = this.pixels.dims[1];
        const buffer_w = this.pixels.dims[2];
        const buffer_hw = buffer_h * buffer_w;
        const h_i = step_size * tile_i;
        const w_i = step_size * tile_j;

        let old_weight, next_weight, new_weight;
        for (let c = 0; c < 3; ++c) {
            for (let i = 0; i < H; ++i) {
                for (let j = 0; j < W; ++j) {
                    const tile_index = c * HW + i * W + j;
                    const buffer_index = c * buffer_hw + (h_i + i) * buffer_w + (w_i + j);
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
        const ses = await onnxSession.getSession(CONFIG.get_helper_model_path("create_seam_blending_filter"));
        const scale = new ort.Tensor("int64", BigInt64Array.from([BigInt(this.scale)]), []);
        const offset = new ort.Tensor("int64", BigInt64Array.from([BigInt(this.offset)]), []);
        const tile_size = new ort.Tensor("int64", BigInt64Array.from([BigInt(this.tile_size)]), []);
        const out = await ses.run({
            "scale": scale,
            "offset": offset,
            "tile_size": tile_size,
        });
        return out.y;
    }
}
