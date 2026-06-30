"""OCR backends, text normalization, and evaluation metrics.

Submodules:
    normalize_text: Lowercasing and tokenization for metric computation
    ocr_metrics: Canonical CER, WER, and word recall v1 definitions
    run_ocr: PaddleOCR with EasyOCR fallback
    merge_patch_ocr: Combine OCR outputs from multiple BOPS patches
"""
