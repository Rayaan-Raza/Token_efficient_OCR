"""OCR inference with PaddleOCR (preferred) or EasyOCR (fallback).

Lazy-loads a single global backend on first use. PaddlePaddle may be unavailable
on some Python versions; EasyOCR provides a portable fallback for development.

Used by:
    - OCR evaluation (:mod:`scripts.run_ocr_eval`)
    - BOPS patch scoring (:mod:`src.preprocessing.patch_scoring`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sys

_ocr_backend = None
_backend_name = None


def _paddle_available() -> bool:
    """Return False on Python versions where PaddlePaddle is typically unavailable."""
    return sys.version_info < (3, 14)


def _init_paddle():
    """Initialize PaddleOCR English detector+recognizer."""
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False), "paddle"


def _init_easyocr(use_gpu: bool):
    """Initialize EasyOCR English reader."""
    import easyocr

    print(f"[OCR] Initializing EasyOCR (gpu={use_gpu}) ...")
    return easyocr.Reader(["en"], gpu=use_gpu, verbose=False), "easyocr"


def reset_ocr(*, force_cpu: bool = False) -> None:
    """Drop the global OCR backend (e.g. after CUDA OOM) and optionally force CPU."""
    global _ocr_backend, _backend_name
    _ocr_backend = None
    _backend_name = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if force_cpu:
        _ocr_backend, _backend_name = _init_easyocr(False)
        print("[OCR] Using backend: easyocr (forced CPU)")


def get_ocr():
    """Return the shared OCR backend, initializing on first call.

    Tries PaddleOCR (if Python < 3.14), then EasyOCR with GPU, then EasyOCR CPU.

    Returns:
        Tuple of (backend instance, backend name ``"paddle"`` or ``"easyocr"``).

    Raises:
        RuntimeError: If no backend can be initialized.
    """
    global _ocr_backend, _backend_name
    if _ocr_backend is not None:
        return _ocr_backend, _backend_name

    errors: list[str] = []
    init_fns: list[tuple] = []
    if _paddle_available():
        init_fns.append((_init_paddle, "paddle"))
    else:
        print("[OCR] Skipping PaddleOCR on Python 3.14+")

    import torch
    init_fns.append((lambda: _init_easyocr(torch.cuda.is_available()), "easyocr-gpu"))
    init_fns.append((lambda: _init_easyocr(False), "easyocr-cpu"))

    for init_fn, label in init_fns:
        try:
            _ocr_backend, _backend_name = init_fn()
            print(f"[OCR] Using backend: {_backend_name}")
            return _ocr_backend, _backend_name
        except Exception as e:
            errors.append(f"{label}: {e}")
            print(f"[OCR] {label} failed: {e}")

    raise RuntimeError(
        "No OCR backend available. Install easyocr or paddleocr.\n" + "\n".join(errors)
    )


def run_ocr_on_image(image_path: str | Path) -> str:
    """Run OCR on an image file and return concatenated line text.

    Args:
        image_path: Path to image (PNG/JPEG).

    Returns:
        Space-joined recognized text lines.
    """
    ocr, backend = get_ocr()
    if backend == "paddle":
        result = ocr.ocr(str(image_path), cls=True)
        lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    lines.append(line[1][0])
        return " ".join(lines)
    result = ocr.readtext(str(image_path))
    return " ".join(t[1] for t in result)


def run_ocr_with_boxes(image_path: str | Path) -> list[dict[str, Any]]:
    """Run OCR and return per-line boxes for patch scoring.

    Args:
        image_path: Path to image file.

    Returns:
        List of dicts with ``box`` (polygon), ``text``, and ``confidence``.
    """
    ocr, backend = get_ocr()
    boxes = []
    if backend == "paddle":
        result = ocr.ocr(str(image_path), cls=True)
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    boxes.append({
                        "box": line[0],
                        "text": line[1][0],
                        "confidence": float(line[1][1]),
                    })
    else:
        for item in ocr.readtext(str(image_path)):
            boxes.append({"box": item[0], "text": item[1], "confidence": float(item[2])})
    return boxes
