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

## Reproduce experiments

```bash
# Phase gates
python scripts/run_preprocessing.py --config configs/smoke_test.yaml
python scripts/audit_datasets.py
python src/data/validate_manifest.py --manifest data/manifests/textocr_debug.jsonl

# Experiments (use --dry-run on OCR/VLM until PaddleOCR/GPU model installed)
python scripts/run_full_experiment.py --phase debug
python scripts/run_full_experiment.py --phase pilot
python scripts/run_full_experiment.py --phase ablation
python scripts/run_full_experiment.py --phase paper

# Paper assets
python scripts/make_paper_assets.py
python scripts/generate_plots.py
python -m pytest tests/ -q
```

Install OCR: `pip install easyocr` or `paddleocr` (if supported on your Python version).
Install VLM: requires GPU + `transformers`, `bitsandbytes` for Qwen2.5-VL-3B.
