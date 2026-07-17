# Restore Point — RAVEN-Select Journal Value-and-Limits Study

**Date:** 2026-07-18  
**Git tip at creation:** see commit that adds this file / subsequent seam branch tip.

## What this freeze preserves

Frozen production method **v1.0.0** (do not retune):

1. Readers: resize, BM25, LER-BOPS  
2. Conservative normalization  
3. OCR-ground on page or route-patch OCR  
4. Shortest grounded, else shortest nonempty  
5. Tie-break: resize → BM25 → LER-BOPS  

## Headline results (audited)

| Experiment | Outcome | Key numbers |
|------------|---------|-------------|
| DocVQA n=500 | PASS | 0.8053 / 0.706 vs resize 0.7840 / shortest 0.7965 |
| DocVQA n=1000 | **PARTIAL** | 0.8234 / 0.723; sig vs shortest only |
| Qwen2-VL-2B n=500 | PARTIAL | 0.6128 / 0.544; sig vs resize only |
| Tesseract grounding n=500 | PASS | identical decisions, drop 0 |
| InfographicVQA n=300 | **FAIL** | RAVEN 0.3167; BM25 alone 0.3713 |
| MP contact-sheet n=300 | **FAIL** | RAVEN 0.3360; not standard MP-DocVQA |

Full DocVQA validation is **blocked** by the PARTIAL n=1000 gate.

## Paper

- Workspace: `paper/raven_select_journal/`
- Framing: **value and limits** of OCR-grounded answer selection
- PDF: `main.pdf`

## How to come back

```text
git log --oneline -- paper/raven_select_journal
# Inspect this file + docs/P17…P25
# Do NOT retune method 1.0.0 from transfer/scale outcomes
```

Seam-carving / content-aware compression work lives in a **separate** folder:

`paper/raven_select_seam/`

and must not mutate the frozen selector.
