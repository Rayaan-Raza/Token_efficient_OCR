# BOPS — Budget-Aware OCR-Guided Patch Selection

Research pipeline for comparing document-image preprocessing methods under **equal visual budgets**. The proposed method (BOPS) sends a low-resolution overview plus K OCR-guided high-resolution patches to downstream OCR or vision-language models, instead of naive resize or compression.

**Results & comparisons:** see [RESULTS.md](RESULTS.md) for dataset audits, metric tables, ablations, and phase status.

---

## What this repo does

1. **Preprocess** text-rich images with fair budgets (pixels, bytes, or patch count)
2. **Evaluate OCR** on TextOCR (CER, WER, word recall)
3. **Evaluate VLM QA** on DocVQA (Exact Match, ANLS)
4. **Compare baselines** (resize, JPEG, WebP, random/uniform patches) vs BOPS
5. **Generate paper assets** (tables, plots, failure analysis)

---

## Methods

| Method | Description |
|--------|-------------|
| `original` | Full-resolution baseline |
| `resize` | Area-ratio downscale (±3% pixel tolerance) |
| `jpeg` / `webp` | Byte-budget compression (±2% tolerance) |
| `bops` | Overview + K OCR-guided patches |
| `overview_only` | Global context only (VLM ablation) |
| `random` / `uniform` | Patch selection baselines (VLM ablation) |

Budget-fairness: rows with `invalid_budget=true` are excluded from aggregates. See [RESULTS.md §7](RESULTS.md#7-budget-fairness).

---

## Datasets

| Dataset | Role | Local path |
|---------|------|------------|
| **TextOCR** | OCR evaluation | `data/train_val_images/train_images/` + `data/raw/textocr/` |
| **DocVQA** | VLM QA evaluation | `data/raw/docvqa_hf/images/` (500 validation samples) |

Manifests live in `data/manifests/` (`textocr_debug`, `textocr_pilot`, `docvqa_debug`, `docvqa_pilot`, `docvqa_val_500`).

---

## Installation

```bash
pip install -r requirements.txt
```

**OCR backend** (pick one):

```bash
pip install easyocr          # works on Python 3.14
# or
pip install paddleocr paddlepaddle   # if supported on your platform
```

**VLM** (GPU required — RTX 3050 4 GB works with 4-bit Qwen):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers bitsandbytes accelerate
```

Model: `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit). See [RESULTS.md §10](RESULTS.md#10-qwen-local-inference). Close other GPU apps before eval.

---

## Quick start

```bash
# 1. Smoke test — resize + metadata
python scripts/run_preprocessing.py --config configs/smoke_test.yaml

# 2. Dataset audit (Phase 2A gate)
python scripts/audit_datasets.py

# 3. Validate a manifest
python src/data/validate_manifest.py --manifest data/manifests/textocr_debug.jsonl

# 4. Unit tests
python -m pytest tests/ -q
```

---

## Reproduce experiments

Use `--dry-run` on OCR/VLM scripts until backends are installed. Full phase orchestration:

```bash
python scripts/run_full_experiment.py --phase debug
python scripts/run_full_experiment.py --phase pilot
python scripts/run_full_experiment.py --phase ablation
python scripts/run_full_experiment.py --phase paper
```

Individual entry points:

```bash
# OCR on TextOCR
python scripts/run_ocr_eval.py \
  --manifest data/manifests/textocr_pilot.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_0.5 area_0.25 kb_200 \
  --limit 20

# VLM on DocVQA
python scripts/run_vlm_eval.py \
  --manifest data/manifests/docvqa_pilot.jsonl \
  --method bops --num-patches 4 --limit 10

# Paper tables + plots
python scripts/make_paper_assets.py
python scripts/generate_plots.py
python scripts/analyze_failures.py
```

After each batch, update [RESULTS.md](RESULTS.md) with new numbers.

---

## Repository layout

```
├── src/
│   ├── data/           # Manifest builders, validation, dataset loader
│   ├── preprocessing/  # Resize, compression, BOPS patch selection
│   ├── ocr/            # OCR backends + CER/WER/word recall
│   ├── vlm/            # Qwen2.5-VL harness + DocVQA metrics
│   ├── metrics/        # Aggregation + statistical tests
│   ├── visualization/  # Budget degradation plots
│   └── utils/          # Paths, config, I/O, budget checks
├── scripts/            # CLI entry points
├── configs/            # Experiment YAML files
├── data/               # Manifests + raw data (large files gitignored)
├── outputs/            # Metrics, plots, audit reports (gitignored)
├── paper/              # draft.tex, advisor deck, table CSVs
├── tests/              # Unit tests (14 tests)
└── RESULTS.md          # Living experiment log
```

---

## Metrics (canonical definitions)

| Metric | Formula / rule |
|--------|----------------|
| **Word recall (v1)** | matched normalized GT tokens ÷ total GT tokens |
| **CER / WER** | Standard character/word error rate |
| **Exact Match** | Normalized prediction equals any reference answer |
| **ANLS** | Average normalized Levenshtein similarity |

Implementation: `src/ocr/ocr_metrics.py`, `src/vlm/qa_metrics.py`.

---

## Data setup

**DocVQA** (streaming, 500 samples):

```bash
python scripts/download_docvqa_hf.py
```

**TextOCR** (convert ~280 MB annotation file):

```bash
python scripts/convert_textocr_annotations.py
python scripts/audit_datasets.py
```

See `Data/how_to_get_huggingface_dataset.txt` for Hugging Face details.

---

## Paper artifacts

| File | Purpose |
|------|---------|
| `paper/draft.tex` | Paper draft |
| `paper/advisor_deck.md` | 10-slide summary |
| `paper/tables/table_ocr_budget.csv` | OCR means by method × budget |
| `paper/tables/table_vlm_patches.csv` | VLM means by method × patch count |
| `RESULTS.md` | Full results log with comparisons |

---

## Current status

| Component | Status |
|-----------|--------|
| Pipeline code | Complete |
| Unit tests | 14/14 passing |
| Dataset audit | Passed (21,778 TextOCR, 500 DocVQA) |
| Real OCR/VLM runs | Pending — see [RESULTS.md](RESULTS.md) |

---

## Reference

Full research proposal: `Reference/Full Research Proposal and Implementation Plan.md`
