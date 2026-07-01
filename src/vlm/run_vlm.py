"""Qwen2.5-VL-3B inference for DocVQA under BOPS preprocessing.

Loads the vision-language model once (4-bit quantization by default for limited
VRAM) and supports:
    - Single-image QA (resize baseline)
    - Overview + multi-patch QA (BOPS layout)

Requires GPU, ``transformers``, and ``bitsandbytes`` for 4-bit loading.
On 4 GB GPUs (e.g. RTX 3050), images are downscaled before inference and patch
counts should stay low (≤2) to avoid OOM.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from src.vlm.parse_answers import parse_answer
from src.vlm.prompt_templates import format_overview_patches, format_single

_model = None
_processor = None

# Cap longest image side before VLM encode (saves VRAM on 4 GB cards)
_VLM_MAX_SIDE = 768


def _resize_for_vlm(image: Image.Image, max_side: int = _VLM_MAX_SIDE) -> Image.Image:
    """Downscale an image so its longest side is at most ``max_side`` pixels."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def load_vlm(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", load_in_4bit: bool = True):
    """Load or return cached Qwen2.5-VL model and processor.

    Args:
        model_name: Hugging Face model id.
        load_in_4bit: Use 4-bit quantization (recommended for 3B on 4–8 GB VRAM).

    Returns:
        Tuple of (model, processor).
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    _processor = AutoProcessor.from_pretrained(model_name)
    return _model, _processor


def _generate(images: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> tuple[str, str]:
    """Run chat-template generation with one or more images.

    Args:
        images: PIL images in prompt order.
        prompt: Text prompt (from :mod:`prompt_templates`).
        max_new_tokens: Generation length cap.

    Returns:
        Tuple of (raw decoded string, parsed answer).
    """
    model, processor = load_vlm()
    images = [_resize_for_vlm(im) for im in images]
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    return raw, parse_answer(raw)


def run_vlm_single(image: Image.Image, question: str) -> tuple[str, str]:
    """Answer a DocVQA question given a single preprocessed image.

    Args:
        image: Visual input (e.g. resized full page).
        question: Document question.

    Returns:
        Tuple of (raw model output, parsed answer).
    """
    prompt = format_single(question)
    return _generate([image], prompt)


def run_vlm_overview_patches(
    overview: Image.Image,
    patches: list[Image.Image],
    question: str,
) -> tuple[str, str]:
    """Answer a DocVQA question using overview + high-res patches (BOPS).

    Args:
        overview: Low-resolution full-page image.
        patches: List of high-resolution patch crops.
        question: Document question.

    Returns:
        Tuple of (raw model output, parsed answer).
    """
    prompt = format_overview_patches(question)
    images = [overview] + patches
    return _generate(images, prompt)
