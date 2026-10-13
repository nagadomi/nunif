import os

# Disable XET transfer to avoid network instability
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_EMERGENCY_RETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple

MODEL_ID = "Cloudflare/clef-flash"
TORCH_DTYPE = torch.bfloat16

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


class _SlicedEmbeddingWeight:
    """Wrapper that holds the 2GB output embedding table in CPU RAM and transfers

    only the small token slices requested by JointSchemaHead on GPU.
    """

    def __init__(self, weight_cpu: torch.Tensor, device: torch.device, dtype: torch.dtype) -> None:
        self.weight_cpu = weight_cpu
        self.device = device
        self.dtype = dtype

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.weight_cpu[idx.to("cpu")].to(device=self.device, dtype=self.dtype)


# The official `ClefModel` and `load_release_model` hardcode the entire model to a single device
# without quantization support, causing GPU OOM on consumer hardware.
# `ClefFinder` bypasses the official classes to enable 4-bit / 8-bit bitsandbytes quantization
# and offloading, and interfaces with the underlying language model and joint schema head directly.
class ClefFinder:
    def __init__(
        self,
        text_model: Any,
        head: Any,
        processor: Any,
        encode_record: Any,
        collate_records: Any,
        input_device: torch.device,
        output_embeddings_cpu: torch.Tensor,
    ) -> None:
        self.text_model = text_model
        self.head = head
        self.processor = processor
        self.encode_record = encode_record
        self.collate_records = collate_records
        self.input_device = input_device
        self.output_embeddings_cpu = output_embeddings_cpu
        self.emb_weight = _SlicedEmbeddingWeight(
            weight_cpu=output_embeddings_cpu,
            device=input_device,
            dtype=TORCH_DTYPE,
        )

    def predict_match(
        self,
        image: Image.Image,
        prompt: str,
        max_size: int = 512,
    ) -> float:
        if max_size > 0 and max(image.size) > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)

        record = {
            "state": "",
            "images": [image],
            "questions": {
                "match": {
                    "type": "noul",
                    "instructions": prompt,
                }
            },
            "id": "query",
        }

        encoded = self.encode_record(
            self.processor.tokenizer,
            record,
            max_length=16384,
            processor=self.processor,
        )
        batch = self.collate_records(
            [encoded],
            self.processor.tokenizer.pad_token_id,
            self.input_device,
        )

        media = batch.get("media") or {}
        with torch.inference_mode():
            outputs = self.text_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
                **media,
            )

            head_device = outputs.last_hidden_state.device
            head_dtype = outputs.last_hidden_state.dtype

            input_ids = batch["input_ids"].to(head_device)
            attention_mask = batch["attention_mask"].to(head_device)
            self.emb_weight.device = head_device
            self.emb_weight.dtype = head_dtype
            emb_weight = self.emb_weight

            logits = self.head(
                outputs.last_hidden_state,
                input_ids,
                attention_mask,
                batch["records"],
                emb_weight,
            )

        q_logits = logits[0][0]
        q_encoded = encoded.questions[0]
        probs = dict(
            zip(
                q_encoded.option_ids,
                q_logits.float().softmax(-1).tolist(),
                strict=False,
            )
        )
        return float(probs.get("true", 0.0))


def list_images(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not input_path.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_finder(device: str | torch.device = "cuda", q8: bool = False) -> ClefFinder:
    model_path = Path(snapshot_download(MODEL_ID))
    path_str = str(model_path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    from joint_schema_model import (  # type: ignore[import-not-found]
        JointSchemaHead,
        collate_records,
        encode_record,
    )

    if q8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    device_str = str(device)
    if device_str == "cpu":
        device_map: Any = "cpu"
    else:
        # Offload lm_head (~2GB) directly to CPU during initial loading so it never touches VRAM.
        device_map = {"lm_head": "cpu", "": "cuda:0"}

    # Load lm_head weight directly into CPU RAM from safetensors.
    # JointSchemaHead only reads a few token vectors from it during inference.
    with open(model_path / "model.safetensors.index.json") as fp:
        idx = json.load(fp)
    lm_head_file = idx["weight_map"]["lm_head.weight"]
    with safe_open(model_path / lm_head_file, framework="pt", device="cpu") as f:
        output_embeddings_cpu = f.get_tensor("lm_head.weight")

    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    backbone.config.use_cache = False

    head_config = json.loads((model_path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(model_path / "joint_head.safetensors"), strict=True)

    input_device = next(backbone.parameters()).device
    head = head.to(device=input_device, dtype=TORCH_DTYPE).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ClefFinder(
        text_model=backbone.model,
        head=head,
        processor=processor,
        encode_record=encode_record,
        collate_records=collate_records,
        input_device=input_device,
        output_embeddings_cpu=output_embeddings_cpu,
    )


def save_matched_image(src_path: Path, input_root: Path, output_dir: Path, action: str) -> None:
    if input_root.is_file():
        target_path = output_dir / src_path.name
    else:
        target_path = output_dir / src_path.relative_to(input_root)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if action == "copy":
        shutil.copy2(src_path, target_path)
    elif action == "symlink":
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        target_path.symlink_to(src_path.resolve())


def process_images(
    image_paths: list[Path],
    input_root: Path,
    output_dir: Path,
    prompt: str,
    finder: ClefFinder,
    threshold: float = 0.5,
    action: str = "copy",
    max_size: int = 512,
    quiet: bool = False,
) -> tuple[int, int]:
    matched_count = 0
    skipped_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=len(image_paths), desc="Finding", ncols=80)
    print_lock = Lock()

    def handle_result(
        src_path: Path,
        prob_true: float | None,
        error_msg: str | None,
    ) -> bool:
        if error_msg is not None:
            with print_lock:
                tqdm.write(f"[ERROR] Failed to process {src_path}: {error_msg}")
                progress.update(1)
            return False

        assert prob_true is not None
        is_match = prob_true >= threshold
        if not quiet:
            status_label = "MATCH" if is_match else "SKIP"
            with print_lock:
                tqdm.write(f"[{status_label}] (prob={prob_true:.4f}) {src_path}")

        if is_match:
            save_matched_image(src_path, input_root, output_dir, action)

        with print_lock:
            progress.update(1)
        return is_match

    # Thread 1: ImageLoader background thread with fast simple loader
    loader = ImageLoader(
        files=[str(p) for p in image_paths],
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"},
        max_queue_size=16,
    )

    # Thread 3: ThreadPoolExecutor for Disk I/O (copy/symlink) and synchronized console logging
    with ThreadPoolExecutor(max_workers=2) as executor, loader:
        futures = []
        for img, meta in loader:
            p = Path(meta["filename"])
            if img is None:
                fut = executor.submit(handle_result, p, None, "Failed to load image")
                futures.append((fut, p))
                continue

            try:
                # Thread 2 (Main): Synchronous GPU inference
                prob_true = finder.predict_match(
                    image=img,
                    prompt=prompt,
                    max_size=max_size,
                )
                fut = executor.submit(handle_result, p, prob_true, None)
                futures.append((fut, p))
            except Exception as e:
                fut = executor.submit(handle_result, p, None, str(e))
                futures.append((fut, p))

        for fut, _ in futures:
            try:
                matched = fut.result()
                if matched:
                    matched_count += 1
                else:
                    skipped_count += 1
            except Exception:
                skipped_count += 1

    progress.close()
    return matched_count, skipped_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find matching images using Cloudflare/clef-flash decision model.")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Question or prompt to decide on (e.g. 'Is there a dog in the picture?').",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to an input image file or directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output directory where matched images will be placed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for matching (default: 0.5).",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["copy", "symlink"],
        default="copy",
        help="Action to perform on matched images (default: copy).",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search images recursively in subdirectories.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for inference (default: 'cuda' if available else 'cpu').",
    )
    parser.add_argument(
        "--q8",
        action="store_true",
        help="Use 8-bit quantization instead of the default 4-bit.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Max image size (longest edge) for model input. 0 disables resizing (default: 512).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-file logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: input path '{input_path}' does not exist.")

    images = list_images(input_path, recursive=args.recursive)
    if not images:
        print(f"No supported images found in '{input_path}'.")
        return

    output_dir = Path(args.output)
    device = torch.device(args.device)

    if not args.quiet:
        bit_str = "8-bit" if args.q8 else "4-bit"
        print(f"Found {len(images)} image(s) to process.")
        print(f"Loading model '{MODEL_ID}' (device={device}, {bit_str})...")

    finder = load_finder(device=device, q8=args.q8)

    matched, skipped = process_images(
        image_paths=images,
        input_root=input_path,
        output_dir=output_dir,
        prompt=args.prompt,
        finder=finder,
        threshold=args.threshold,
        action=args.action,
        max_size=args.max_size,
        quiet=args.quiet,
    )

    if not args.quiet:
        print(
            f"Finished. Matched: {matched}, Skipped: {skipped}, Total: {len(images)}. Output saved to '{output_dir}'."
        )


if __name__ == "__main__":
    main()
