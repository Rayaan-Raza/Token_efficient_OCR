"""Run OCR on images (PaddleOCR preferred, EasyOCR fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_ocr_backend = None
_backend_name = None


def _init_paddle():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False), "paddle"


def _init_easyocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False), "easyocr"


def get_ocr():
    global _ocr_backend, _backend_name
    if _ocr_backend is not None:
        return _ocr_backend, _backend_name
    for init_fn, name in ((_init_paddle, "paddle"), (_init_easyocr, "easyocr")):
        try:
            _ocr_backend, _backend_name = init_fn()
            return _ocr_backend, _backend_name
        except Exception:
            continue
    raise RuntimeError("No OCR backend available. Install paddleocr or easyocr.")


def run_ocr_on_image(image_path: str | Path) -> str:
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
