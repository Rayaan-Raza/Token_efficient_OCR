# 04 — Status

**Status:** SCAFFOLDED

Restore parent study: git `0faa3c3` / `paper/raven_select_journal/RESTORE_POINT.md`.

| Variant | Code | Pilot | n=1000 full-page | Selector |
|---------|------|-------|------------------|----------|
| MarginCrop-Resize | `src/preprocessing/ocr_page_compress.py` | not run | — | — |
| WhitespaceCompress-Resize | same | not run | — | — |
| OCR-SeamResize | same | not run | — | — |

## How to run

```text
# Smoke / pilot full-page readers
python scripts/run_vlm_eval.py --manifest Data/manifests/docvqa_100.jsonl \
  --method margin_crop_resize --limit 100 --checkpoint-every 5
python scripts/run_vlm_eval.py --manifest Data/manifests/docvqa_100.jsonl \
  --method ws_compress_resize --limit 100 --checkpoint-every 5
python scripts/run_vlm_eval.py --manifest Data/manifests/docvqa_100.jsonl \
  --method ocr_seam_resize --limit 100 --checkpoint-every 5

# After n=1000 full-page CSVs exist (BM25/LER already cached):
python scripts/run_ocr_seam_resize_eval.py --n 1000 --write-gates --skip-missing
```

Gate: **PENDING** (see `docs/03-gates.md`)

## Constraints

- Frozen RAVEN-Select v1.0.0 selector is not retuned.
- No gold answers in compression.
- Classic unprotected seam carving is not used.
