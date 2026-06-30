# BOPS — Budget-Aware OCR-Guided Patch Selection

Research pipeline for comparing preprocessing methods on text-rich images under equal visual budgets.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_preprocessing.py --config configs/smoke_test.yaml
python -m pytest tests/ -q
```

## Layout

- `src/` — core library (data, preprocessing, ocr, vlm, metrics)
- `configs/` — experiment YAML files
- `scripts/` — CLI entry points
- `data/` — manifests and raw data (large images gitignored)
- `outputs/` — experiment results (gitignored)

## Metrics

**Word recall (v1):** matched normalized GT tokens / total GT tokens (exact token match, count-aware).

**Budget fairness:** results with `invalid_budget=true` are excluded from aggregates.

See `Reference/Full Research Proposal and Implementation Plan.md` for full specification.
