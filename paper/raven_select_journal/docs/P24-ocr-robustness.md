# P24 — OCR Engine Robustness (EasyOCR vs Tesseract)

## Scope

Hold VLM reader CSVs fixed (DocVQA n=500, Qwen2.5-VL-3B) and recompute
RAVEN-Select grounding with Tesseract full-page OCR instead of EasyOCR.

## Results (method 1.0.0)

| Grounding engine | ANLS | EM |
|------------------|------|-----|
| EasyOCR (production) | 0.8053 | 0.706 |
| Tesseract | 0.8053 | 0.706 |
| Δ (Easy − Tess) | **0.0000** | 0.000 |

Gate thresholds: PASS if drop ≤ 0.015; PARTIAL if ≤ 0.03.

Status: **PASS** — selection outcomes unchanged under Tesseract grounding on this split.
Artifacts: `outputs/metrics/raven_select_ocr_presence_n500_tesseract.parquet`,
ablation JSON from `run_raven_select_ocr_engine_ablation.py`.

## Commit

- TBD
