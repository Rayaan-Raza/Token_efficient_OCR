# P24 — OCR Engine Robustness

## Scope

Keep VLM outputs fixed. Recompute OCR grounding with EasyOCR vs Tesseract page
OCR. Report page-only and mixed-provenance tests separately. Patch OCR remains
EasyOCR-derived unless rebuilt.

## Gate (ANLS drop = EasyOCR − Tesseract)

| Outcome | Condition |
|---------|-----------|
| **PASS** | Drop ≤ 0.015 |
| **PARTIAL** | 0.015 < drop ≤ 0.03 |
| **FAIL** | Drop > 0.03 |

## Commands

```text
python scripts/run_fullpage_ocr.py --manifest Data/manifests/docvqa_500.jsonl --engine tesseract --ocr-backend tesseract
# Then engine-aware OCR-presence + selector ablation
```

## Results

| OCR grounding | ANLS | EM | Drop vs EasyOCR | Gate |
|---------------|------|----|-----------------|------|
| EasyOCR (main) | — | — | 0 | — |
| Tesseract page | — | — | — | PENDING |

## Commit

- TBD
