# 04 — Status

**Status:** n=1000 IN PROGRESS

Restore parent study: git `0faa3c3` / `paper/raven_select_journal/RESTORE_POINT.md`.

## Pilot DocVQA n=100 (full-page alone)

| Full-page reader | ANLS | EM |
|------------------|------|----|
| Resize | 0.7876 | 0.680 |
| MarginCrop-Resize | 0.7957 | 0.680 |
| WhitespaceCompress-Resize | **0.8302** | **0.710** |
| OCR-SeamResize | 0.8055 | 0.660 |

Pilot selector (n=100): MarginCrop **PASS** on the pilot gate (small-n; not definitive).

## DocVQA n=1000 — full-page alone

| Full-page reader | ANLS | EM | Status |
|------------------|------|----|--------|
| Resize | 0.8149 | 0.706 | done |
| MarginCrop-Resize | 0.8197 | 0.718 | done |
| WhitespaceCompress-Resize | — | — | **running** (~120/1000) |
| OCR-SeamResize | — | — | queued |

## DocVQA n=1000 — frozen selector (corrected vs true resize)

Original RAVEN-Select: 0.8234 / 0.723 · true resize: 0.8149 / 0.706

| Setting | ANLS | EM | vs resize CI>0 | vs orig CI>0 | Gate |
|---------|------|----|----------------|--------------|------|
| + MarginCrop | 0.8278 | 0.734 | no `[-0.0019,+0.0288]` | no | **FAIL** |
| + WhitespaceCompress | — | — | — | — | pending |
| + OCR-SeamResize | — | — | — | — | pending |

Note: eval now compares against the **true** resize CSV (bugfix: previously compared against the swapped full-page reader).

## Next

1. Finish WS Compress + OCR-Seam VLM n=1000  
2. Re-run `run_ocr_seam_resize_eval.py --n 1000 --write-gates`  
3. Record P26 PASS/PARTIAL/FAIL
