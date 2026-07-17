"""OCR inference with PaddleOCR (preferred) or EasyOCR (fallback).

Lazy-loads a single global backend on first use. PaddlePaddle may be unavailable
on some Python versions; EasyOCR provides a portable fallback for development.

Used by:
    - OCR evaluation (:mod:`scripts.run_ocr_eval`)
    - BOPS patch scoring (:mod:`src.preprocessing.patch_scoring`)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ocr_backend = None
_backend_name = None
_forced_engine: str | None = None


def set_ocr_engine(engine: str | None) -> None:
    """Force a specific OCR backend on next ``get_ocr()`` call."""
    global _ocr_backend, _backend_name, _forced_engine
    _ocr_backend = None
    _backend_name = None
    _forced_engine = engine


def _find_tesseract_cmd() -> str | None:
    """Resolve tesseract executable from PATH or common install locations."""
    import shutil

    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    candidates = [
        Path.home() / "scoop" / "shims" / "tesseract.exe",
        Path.home() / "scoop" / "apps" / "tesseract" / "current" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _find_tessdata_prefix() -> str | None:
    """Resolve tessdata directory containing ``eng.traineddata``."""
    import os

    env = os.environ.get("TESSDATA_PREFIX")
    if env and Path(env, "eng.traineddata").exists():
        return env
    if env and Path(env, "tessdata", "eng.traineddata").exists():
        return str(Path(env, "tessdata"))
    candidates = [
        Path.home() / "scoop" / "apps" / "tesseract-languages" / "current",
        Path.home() / "scoop" / "apps" / "tesseract" / "current" / "tessdata",
        repo_tools := Path(__file__).resolve().parents[2] / "tools" / "tesseract" / "tessdata",
    ]
    for p in candidates:
        if Path(p, "eng.traineddata").exists():
            return str(p)
    return None


def _init_tesseract():
    """Initialize Tesseract via pytesseract (requires system tesseract binary)."""
    import os

    import pytesseract

    cmd = _find_tesseract_cmd()
    if not cmd:
        raise RuntimeError("tesseract binary not found on PATH")
    pytesseract.pytesseract.tesseract_cmd = cmd
    tessdata = _find_tessdata_prefix()
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(f"tesseract init failed: {exc}") from exc
    return pytesseract, "tesseract"


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


def reset_ocr(*, force_cpu: bool = False, engine: str | None = None) -> None:
    """Drop the global OCR backend (e.g. after CUDA OOM) and optionally force CPU."""
    global _ocr_backend, _backend_name, _forced_engine
    _ocr_backend = None
    _backend_name = None
    if engine is not None:
        _forced_engine = engine
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
    When ``set_ocr_engine`` was called, only that backend is attempted.

    Returns:
        Tuple of (backend instance, backend name ``"paddle"``, ``"easyocr"``, or ``"tesseract"``).

    Raises:
        RuntimeError: If no backend can be initialized.
    """
    global _ocr_backend, _backend_name
    if _ocr_backend is not None:
        return _ocr_backend, _backend_name

    errors: list[str] = []
    init_fns: list[tuple] = []

    if _forced_engine == "tesseract":
        init_fns.append((_init_tesseract, "tesseract"))
    elif _forced_engine == "paddle":
        if _paddle_available():
            init_fns.append((_init_paddle, "paddle"))
        else:
            raise RuntimeError("PaddleOCR unavailable on Python 3.14+")
    elif _forced_engine == "easyocr":
        import torch
        init_fns.append((lambda: _init_easyocr(torch.cuda.is_available()), "easyocr-gpu"))
        init_fns.append((lambda: _init_easyocr(False), "easyocr-cpu"))
    else:
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
    if backend == "tesseract":
        boxes = _tesseract_boxes(image_path)
        return " ".join(b["text"] for b in boxes)
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


def _tesseract_boxes(image_path: str | Path) -> list[dict[str, Any]]:
    """Run Tesseract and return word-level boxes."""
    import pytesseract
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes: list[dict[str, Any]] = []
    n = len(data.get("text") or [])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        boxes.append({"box": poly, "text": text, "confidence": conf / 100.0})
    return boxes


def run_ocr_with_boxes(image_path: str | Path) -> list[dict[str, Any]]:
    """Run OCR and return per-line boxes for patch scoring.

    Args:
        image_path: Path to image file.

    Returns:
        List of dicts with ``box`` (polygon), ``text``, and ``confidence``.
    """
    ocr, backend = get_ocr()
    boxes = []
    if backend == "tesseract":
        return _tesseract_boxes(image_path)
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
