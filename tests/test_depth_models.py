import argparse
import os
import sys
from os import path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from iw3.base_depth_model import BaseDepthModel
from iw3.depth_anything_model import MODEL_FILES as DA_MODELS
from iw3.depth_anything_v3_model import MODEL_FILES as DA3_MODELS
from iw3.depth_model_factory import create_depth_model
from iw3.depth_pro_model import MODEL_FILES as DP_MODELS
from iw3.mapper import get_mapper, resolve_mapper_name
from iw3.video_depth_anything_model import MODEL_FILES as VDA_MODELS
from iw3.video_depth_anything_streaming_model import MODEL_FILES as VDA_STREAM_MODELS
from iw3.zoedepth_model import MODEL_FILES as ZOE_MODELS

DEFAULT_TEST_IMAGE = "tests/images/dog_448.png"
DEFAULT_OUTPUT_DIR = "tests/data/test_depth_models"

ALL_MODEL_TYPES = (
    ["NULL"]
    + list(ZOE_MODELS.keys())
    + list(DA_MODELS.keys())
    + list(DA3_MODELS.keys())
    + list(DP_MODELS.keys())
    + list(VDA_STREAM_MODELS.keys())
    + list(VDA_MODELS.keys())
)


def run_model(
    model_type: str,
    input_image_path: str,
    output_dir: str,
    gpu: int = 0,
    mapper: str | None = None,
    foreground_scale: float = 0,
) -> str:
    im = Image.open(input_image_path).convert("RGB")
    model = create_depth_model(model_type)

    if not model.has_checkpoint_file(model_type):
        return f"[SKIP] Checkpoint file for `{model_type}` is not found locally."

    model.load(gpu=gpu)

    if model.get_name() == "VideoDepthAnything":
        # VideoDepthAnything is streaming-based and requires infer_with_normalize
        x = TF.to_tensor(im).unsqueeze(0).to(model.device)
        out_list = model.infer_with_normalize(x, pts=[0], reset_pts=[0])
        model.reset()
        if len(out_list) == 0:
            return f"[FAIL] `{model_type}` produced no output frames."
        depth = out_list[0]
    else:
        # Standard BaseDepthModel.infer
        depth = model.infer(im, tta=False, enable_amp=True)
        depth = model.minmax_normalize_chw(depth)

    output_filename = path.join(output_dir, f"{model_type}.png")

    mapper_name = resolve_mapper_name(mapper=mapper, foreground_scale=foreground_scale, metric_depth=model.is_metric())
    depth = get_mapper(mapper_name)(depth)

    BaseDepthModel.save_normalized_depth(depth, output_filename)

    return f"[OK] `{model_type}` (mapper={mapper_name}) -> {output_filename} (shape={tuple(depth.shape)})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test all depth models and export depth maps")
    parser.add_argument("-i", "--input", type=str, default=DEFAULT_TEST_IMAGE, help="Input test image")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0 if torch.cuda.is_available() else -1, help="GPU device ID")
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        help="Model type name to test, or 'all' to test all models",
    )
    parser.add_argument("--mapper", type=str, default=None, help="Depth mapper function (default: auto)")
    parser.add_argument("--foreground-scale", type=float, default=0, help="Foreground scale for mapper (-3 to 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_type == "all":
        target_models = ALL_MODEL_TYPES
    else:
        target_models = [args.model_type]

    print(f"Testing depth models on input: {args.input} (gpu={args.gpu})")
    print(f"Output directory: {args.output_dir}")
    print(f"Total target models: {len(target_models)}")
    print("-" * 60)

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for model_type in target_models:
        try:
            msg = run_model(
                model_type,
                args.input,
                args.output_dir,
                gpu=args.gpu,
                mapper=args.mapper,
                foreground_scale=args.foreground_scale,
            )
            print(msg)
            if msg.startswith("[OK]"):
                ok_count += 1
            elif msg.startswith("[SKIP]"):
                skip_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"[ERROR] `{model_type}` failed with: {e}", file=sys.stderr)
            fail_count += 1

    print("-" * 60)
    print(f"Results: {ok_count} succeeded, {skip_count} skipped, {fail_count} failed.")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
