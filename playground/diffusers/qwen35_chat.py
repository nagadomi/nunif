# Qwen3.5 9B Abliterated Chat pipeline with 4-bit quantization and multimodal support
#
# requires ~7-8GB VRAM with 4-bit quantization

import os

# Disable XET transfer because my home router keeps freezing
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_EMERGENCY_RETRY"] = "1"

import argparse
import copy
import re
import sys
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    Qwen3_5ForConditionalGeneration,
    TextStreamer,
    utils,
)

utils.logging.set_verbosity_error()

DEFAULT_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
TORCH_DTYPE = torch.bfloat16


def extract_images_and_text(user_input: str) -> tuple[list[Image.Image], str]:
    """Extract image paths prefixed with @ from text and return loaded images and cleaned text."""
    pattern = r'@(?:<([^>]+)>|"([^"]+)"|\'([^\']+)\'|([^\s,;:　、。！？!?]+))'
    images: list[Image.Image] = []

    def replacer(match: re.Match[str]) -> str:
        raw_path = match.group(1) or match.group(2) or match.group(3) or match.group(4)
        if not raw_path:
            return ""
        expanded_path = os.path.expanduser(raw_path.strip())
        if os.path.isfile(expanded_path):
            try:
                img = Image.open(expanded_path).convert("RGB")
                images.append(img)
                print(f"[Loaded image: {expanded_path}]")
            except Exception as e:
                print(f"[Warning: Failed to load image '{expanded_path}': {e}]")
        else:
            print(f"[Warning: File not found: '{expanded_path}']")
        return ""

    clean_text = re.sub(pattern, replacer, user_input)
    clean_text = re.sub(r"[ \t]+", " ", clean_text).strip()
    return images, clean_text


def normalize_images(
    images: Sequence[Image.Image | str] | Image.Image | str | None,
) -> list[Image.Image]:
    """Normalize image argument into a list of PIL Images."""
    if images is None:
        return []
    if isinstance(images, (Image.Image, str)):
        images = [images]

    result: list[Image.Image] = []
    for item in images:
        if isinstance(item, Image.Image):
            result.append(item.convert("RGB"))
        elif isinstance(item, str):
            expanded = os.path.expanduser(item)
            if os.path.isfile(expanded):
                try:
                    result.append(Image.open(expanded).convert("RGB"))
                except Exception as e:
                    print(f"[Warning: Failed to load image '{expanded}': {e}]")
            else:
                print(f"[Warning: Image file not found: '{expanded}']")
    return result


def format_user_message(text: str, images: list[Image.Image]) -> dict[str, Any]:
    """Format user message into multimodal or text-only structure."""
    if not images:
        return {"role": "user", "content": text}

    content: list[dict[str, Any]] = []
    for img in images:
        content.append({"type": "image", "image": img})
    if text:
        content.append({"type": "text", "text": text})

    return {"role": "user", "content": content}


class Qwen35Chat:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        q8: bool = False,
        system_prompt: str | None = None,
        thinking: bool = False,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.thinking = thinking
        self.system_prompt = system_prompt

        self.model, self.processor = self._load_model_and_processor(
            model_id=model_id,
            q8=q8,
            device=device,
        )
        self.tokenizer = self.processor.tokenizer

        tok = self.tokenizer
        pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.messages: list[dict[str, Any]] = []
        self.reset()

    @staticmethod
    def _load_model_and_processor(
        model_id: str = DEFAULT_MODEL_ID,
        q8: bool = False,
        device: str = "cuda",
    ) -> tuple[Any, Any]:
        processor = AutoProcessor.from_pretrained(model_id)

        target_device = device if device else "cuda"
        device_map = {"": target_device}

        if q8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=TORCH_DTYPE,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=TORCH_DTYPE,
            device_map=device_map,
            quantization_config=quantization_config,
        )
        model.eval()

        return model, processor

    def reset(self) -> None:
        """Reset conversation context to initial system prompt if set, else empty."""
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []

    def generate(
        self,
        prompt_text: str,
        images: Sequence[Image.Image | str] | Image.Image | str | None = None,
        stream: bool = False,
        save_history: bool = True,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """Non-interactive generation interface.

        Args:
            prompt_text: Input prompt text (can also contain @<path> syntax).
            images: Optional PIL Image or image file path(s).
            stream: Whether to stream tokens to stdout in real-time.
            save_history: Whether to append this interaction to conversation context.
            max_new_tokens: Override max_new_tokens if provided.
            temperature: Override temperature if provided.
            top_p: Override top_p if provided.

        Returns:
            The generated assistant response string.
        """
        extracted_images, clean_text = extract_images_and_text(prompt_text)
        explicit_images = normalize_images(images)
        all_images = explicit_images + extracted_images

        user_message = format_user_message(clean_text, all_images)

        if save_history:
            self.messages.append(user_message)
            context = self.messages
        else:
            context = copy.deepcopy(self.messages)
            context.append(user_message)

        model_inputs = self.processor.apply_chat_template(
            context,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        gen_config = copy.deepcopy(self.generation_config)
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=gen_config,
                streamer=streamer,
            )

        input_length = model_inputs["input_ids"].shape[1]
        response_ids = generated_ids[0, input_length:]
        ai_response: str = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        if save_history:
            self.messages.append({"role": "assistant", "content": ai_response})

        return ai_response

    def interactive_loop(
        self,
        initial_images: Sequence[Image.Image | str] | Image.Image | str | None = None,
    ) -> None:
        """Run terminal interactive chat loop."""
        pending_images = normalize_images(initial_images)
        memory_history = InMemoryHistory()

        print("> Commands: /q or quit: exit, /reset or /clear: reset conversation history")
        print("> Image input: specify via -i CLI option, or use @<path_to_image> in chat")
        while True:
            try:
                user_input = prompt("You > ", history=memory_history).strip()
                if user_input.startswith("/q") or user_input == "quit":
                    sys.exit(0)
                if user_input in ("/reset", "/clear"):
                    self.reset()
                    pending_images = []
                    print("Conversation history reset.")
                    continue
                if not user_input:
                    continue

                print("AI : ", end="", flush=True)
                self.generate(
                    prompt_text=user_input,
                    images=pending_images,
                    stream=True,
                    save_history=True,
                )
                pending_images = []

            except (EOFError, KeyboardInterrupt):
                sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B Chat CLI (Multimodal supported)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model ID or local path")
    parser.add_argument("--image", "-i", type=str, nargs="*", default=None, help="Input image file(s)")
    parser.add_argument("--q8", action="store_true", help="Load model in 8-bit quantization (default is 4-bit)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (chain-of-thought)")
    parser.add_argument("--system-prompt", type=str, default=None, help="Custom system prompt (default: None)")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Single prompt to execute and exit")
    args = parser.parse_args()

    precision = "8bit" if args.q8 else "4bit"
    print(f"Loading {args.model} ({precision})...")
    chat = Qwen35Chat(
        model_id=args.model,
        q8=args.q8,
        system_prompt=args.system_prompt,
        thinking=args.thinking,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.prompt is not None:
        chat.generate(
            prompt_text=args.prompt,
            images=args.image,
            stream=True,
            save_history=False,
        )
        sys.exit(0)

    chat.interactive_loop(initial_images=args.image)


if __name__ == "__main__":
    main()
