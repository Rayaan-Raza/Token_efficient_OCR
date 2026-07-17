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

import gc
from typing import Any

import torch
from PIL import Image

from src.vlm.parse_answers import parse_answer
from src.vlm.prompt_templates import format_overview_patches, format_overview_patches_with_ocr, format_single

_model = None
_processor = None
_loaded_model_name: str | None = None

# Cap longest image side before VLM encode (saves VRAM on 4 GB cards)
_VLM_MAX_SIDE = 768
_VLM_MIN_SIDE = 32


def _resize_for_vlm(image: Image.Image, max_side: int = _VLM_MAX_SIDE) -> Image.Image:
    """Resize for VLM encode: cap longest side and enforce a minimum dimension."""
    w, h = image.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    w, h = image.size
    shortest = min(w, h)
    if shortest < _VLM_MIN_SIDE and shortest > 0:
        scale = _VLM_MIN_SIDE / shortest
        image = image.resize(
            (max(_VLM_MIN_SIDE, int(w * scale)), max(_VLM_MIN_SIDE, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def reset_vlm() -> None:
    """Drop cached model (e.g. when switching Hugging Face ids)."""
    global _model, _processor, _loaded_model_name
    _model = None
    _processor = None
    _loaded_model_name = None
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        # CUDA context may already be poisoned after a device-side assert.
        pass


def is_cuda_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "cuda" in msg or "cublas" in msg or "device-side assert" in msg:
        return True
    return exc.__class__.__name__ in {"AcceleratorError", "OutOfMemoryError"}


def load_vlm(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", load_in_4bit: bool = True):
    """Load or return cached VL model and processor.

    Supports Qwen2.5-VL and Qwen2-VL families (4-bit by default for 4 GB VRAM).
    """
    global _model, _processor, _loaded_model_name
    if _model is not None and _loaded_model_name == model_name:
        return _model, _processor
    if _model is not None and _loaded_model_name != model_name:
        reset_vlm()

    from transformers import AutoProcessor, BitsAndBytesConfig

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

    name_l = model_name.lower()
    if "qwen2.5-vl" in name_l or "qwen2_5_vl" in name_l:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    elif "qwen2-vl" in name_l:
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
    else:
        # Fallback: try Qwen2.5 class first.
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls

    _model = ModelCls.from_pretrained(model_name, **kwargs)
    _processor = AutoProcessor.from_pretrained(model_name)
    _loaded_model_name = model_name
    return _model, _processor


def set_vlm_model(model_name: str) -> None:
    """Force the active model id used by subsequent ``_generate`` calls."""
    load_vlm(model_name)

def _generate_once(images: list[Image.Image], prompt: str, max_new_tokens: int) -> tuple[str, str]:
    model, processor = load_vlm(_loaded_model_name or "Qwen/Qwen2.5-VL-3B-Instruct")
    images = [_resize_for_vlm(im) for im in images]
    messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    return raw, parse_answer(raw)


def _generate(images: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> tuple[str, str]:
    """Run chat-template generation with one or more images."""
    return _generate_once(images, prompt, max_new_tokens)


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
    *,
    ocr_evidence_lines: list[str] | None = None,
) -> tuple[str, str]:
    """Answer a DocVQA question using overview + high-res patches (BOPS).

    Args:
        overview: Low-resolution full-page image.
        patches: List of high-resolution patch crops.
        question: Document question.
        ocr_evidence_lines: Optional OCR text lines appended to the prompt (RAG-style).

    Returns:
        Tuple of (raw model output, parsed answer).
    """
    if ocr_evidence_lines:
        prompt = format_overview_patches_with_ocr(question, ocr_evidence_lines)
    else:
        prompt = format_overview_patches(question)
    images = [overview] + patches
    return _generate(images, prompt)
