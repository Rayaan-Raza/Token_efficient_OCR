# 00 — Hypothesis

## Why seam carving (OCR-protected) might help

The current resize reader shrinks the whole page uniformly. Small text (invoice
numbers, totals, dates, names) becomes blurry. OCR-aware compression removes
unimportant margins/whitespace first so text occupies more of the same VLM
pixel budget.

## Why classic seam carving is dangerous

Unprotected seams can warp tables, break words, distort numbers, remove thin
lines, and change form layout. Documents need:

- high energy / hard masks on OCR text boxes
- high energy on dark strokes and table/grid lines
- low energy on blank margins and inter-section whitespace
- seams forbidden through OCR boxes

## Cleanest experiment

Replace the resize reader; keep the same three-call budget and frozen selector.

Key question: does OCR-protected compression make the full-page reader stronger,
and does RAVEN-Select become significantly better when that reader improves?
