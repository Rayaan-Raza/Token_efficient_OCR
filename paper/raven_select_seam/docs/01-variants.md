# 01 — Full-page compression variants

All variants finish with the **same area-ratio resize** used by the baseline
(`area_ratio=0.25`) so pixel budgets stay comparable.

## 1. MarginCrop-Resize

1. Detect non-white content bounding box (with padding).
2. Crop margins.
3. Resize to area ratio 0.25.

Purpose: answer “is seam carving needed, or is cropping enough?”

## 2. WhitespaceCompress-Resize

1. Detect mostly blank horizontal/vertical bands.
2. Remove or compress blank bands while **never deleting OCR-box rows/cols**.
3. Resize to area ratio 0.25.

Safer than full seam carving for many documents.

## 3. OCR-SeamResize

1. Build energy map: high energy on OCR boxes, dark strokes, table-like lines;
   low energy on whitespace.
2. Remove vertical then horizontal seams from non-protected regions only.
3. Hard prohibition: seams cannot cross OCR boxes.
4. Save transformed image for audit under `outputs/transformed_images/`.
5. Resize to area ratio 0.25.

## Leakage

OCR boxes come from the existing EasyOCR cache (page text), never from gold
answers or answer-in-OCR flags derived from gold.
