# 04 — Status

**Status:** PILOT COMPLETE · n=1000 RUNNING

Restore parent study: git `0faa3c3` / `paper/raven_select_journal/RESTORE_POINT.md`.

## Pilot DocVQA n=100 (full-page alone)

| Full-page reader | ANLS | EM |
|------------------|------|----|
| Resize | 0.7876 | 0.680 |
| MarginCrop-Resize | 0.7957 | 0.680 |
| WhitespaceCompress-Resize | **0.8302** | **0.710** |
| OCR-SeamResize | 0.8055 | 0.660 |

## Pilot selector (frozen RAVEN-Select v1.0.0)

Original RAVEN-Select: 0.8198 / 0.690

| Setting | ANLS | EM | vs resize CI>0 | vs orig CI>0 | Gate |
|---------|------|----|----------------|--------------|------|
| + MarginCrop | **0.8522** | 0.730 | yes | yes | **PASS** |
| + WhitespaceCompress | **0.8659** | 0.760 | no | yes | FAIL* |
| + OCR-SeamResize | 0.8507 | 0.720 | no | no | FAIL |

\*Highest mean; resize CI crosses zero at n=100.

Artifact: `outputs/metrics/ocr_seam_resize_selector_n100.json`

## n=1000

VLM driver running (margin_crop → ws_compress → ocr_seam). BM25/LER already cached.

Gate on n=1000 remains **PENDING** until CSVs + selector eval finish.
