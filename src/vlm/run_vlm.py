"""Run Qwen2.5-VL-3B for DocVQA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.vlm.parse_answers import parse_answer
from src.vlm.prompt_templates import format_overview_patches, format_single

_model = None
_processor = None


def load_vlm(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", load_in_4bit: bool = True):
    global _model, _processor
    if _model is not None:
        return _model, _processor
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.float16}
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    _processor = AutoProcessor.from_pretrained(model_name)
    return _model, _processor


def _generate(images: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> str:
    model, processor = load_vlm()
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return parse_answer(decoded)


def run_vlm_single(image: Image.Image, question: str) -> str:
    prompt = format_single(question)
    return _generate([image], prompt)


def run_vlm_overview_patches(
    overview: Image.Image,
    patches: list[Image.Image],
    question: str,
) -> str:
    prompt = format_overview_patches(question)
    images = [overview] + patches
    return _generate(images, prompt)
