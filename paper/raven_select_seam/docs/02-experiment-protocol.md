# 02 — Experiment protocol

## Dataset

DocVQA nested validation subset **n=1000** (`Data/manifests/docvqa_1000.jsonl`).
Pilot smoke: n=50 or n=100 before full scale.

## A. Full-page readers alone (1 VLM call)

| Method | Tag |
|--------|-----|
| Resize | `resize` (existing) |
| MarginCrop-Resize | `margin_crop_resize` |
| WhitespaceCompress-Resize | `ws_compress_resize` |
| OCR-SeamResize | `ocr_seam_resize` |

Report ANLS, EM, paired bootstrap CI vs resize, median runtime.

## B. Selector (3 VLM calls, frozen RAVEN-Select)

Reader set for each row: `{full_page_variant} + BM25 + LER-BOPS`.

Reuse cached BM25 / LER-BOPS CSVs from the journal n=1000 run. Only re-run the
full-page variant VLM eval, then rebuild OCR-presence for the new route and
evaluate the frozen rule.

| Selector setting | Full-page reader |
|------------------|------------------|
| Original RAVEN-Select | resize |
| RAVEN-Select + MarginCrop | margin_crop_resize |
| RAVEN-Select + WhitespaceCompress | ws_compress_resize |
| RAVEN-Select + OCR-SeamResize | ocr_seam_resize |

Report:

- ANLS / EM
- paired CI vs resize
- paired CI vs original RAVEN-Select
- route counts
- runtime
- qualitative failures (damaged text/layout)

## Commands (sketch)

```text
# Full-page variants
python scripts/run_vlm_eval.py --manifest Data/manifests/docvqa_1000.jsonl \
  --method margin_crop_resize --limit 1000 --checkpoint-every 5
# similarly: ws_compress_resize, ocr_seam_resize

# Selector evaluation (frozen rule)
python scripts/run_ocr_seam_resize_eval.py --n 1000 --write-gates
```
