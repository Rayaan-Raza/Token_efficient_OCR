# BOPS — Budget-Aware OCR-Guided Patch Selection

Research pipeline for comparing document-image preprocessing methods under **equal visual budgets**. The proposed method (BOPS) sends a low-resolution overview plus K OCR-guided high-resolution patches to downstream OCR or vision-language models, instead of naive resize or compression.

**Results & comparisons:** see [RESULTS.md](RESULTS.md) for sanity/pilot numbers, bootstrap CIs, and gate verdicts.  
**Architecture:** see [LOGIC.md](LOGIC.md) for design, data flow, and module reference.

---

## What this repo does

1. **Preprocess** text-rich images with fair budgets (pixels, bytes, or patch count)
2. **Evaluate OCR** on TextOCR (CER, WER, word recall)
3. **Evaluate VLM QA** on DocVQA (Exact Match, ANLS)
4. **Compare baselines** (resize, JPEG, WebP, random/uniform patches) vs BOPS
5. **Generate paper assets** (tables, plots, bootstrap CIs, failure analysis)

---

## Methods

| Method | Description | Budget axis |
|--------|-------------|-------------|
| `original` | Full-resolution baseline | `area_1.0` / `reference` |
| `resize` | Area-ratio downscale (±3% pixel tolerance) | `area_*` |
| `jpeg` / `webp` | Byte-budget compression (`actual ≤ target`) | `kb_*` |
| `bops` | Overview + K OCR-guided patches (merge OCR for eval) | `patches_*` |
| `overview_only` | Global context only (VLM ablation) | `patches_0` |
| `random` / `uniform` | Patch selection baselines (VLM ablation) | `patches_K` |

**Fairness rules:**
- `not_applicable=true` — method does not use this budget axis (e.g. `jpeg` + `area_0.25`); skipped, not compared
- `invalid_budget=true` — budget was attempted but missed (pixel ±3%, byte over target, patch count ≠ K); excluded from aggregates
- `underutilized_budget=true` — byte compression used <70% of target; reported, **not** excluded

Paper tables filter: `dry_run=false`, `not_applicable=false`, `invalid_budget=false`, `experiment_stage ∈ {pilot, paper}`.

---

## Datasets

| Dataset | Role | Local path |
|---------|------|------------|
| **TextOCR** | OCR evaluation | `data/train_val_images/train_images/` + `data/raw/textocr/` |
| **DocVQA** | VLM QA evaluation | `data/raw/docvqa_hf/images/` (500 validation samples) |

Manifests in `data/manifests/` (or `Data/manifests/` on Windows):

| Manifest | Samples | Stage |
|----------|---------|-------|
| `textocr_debug.jsonl` | 50 | sanity / debug |
| `textocr_pilot.jsonl` | 200 | pilot OCR |
| `docvqa_debug.jsonl` | 20 | sanity / debug VLM |
| `docvqa_pilot.jsonl` | 100 | pilot VLM |
| `docvqa_val_500.jsonl` | 500 | full export |

---

## Installation

```bash
pip install -r requirements.txt
```

**OCR backend** (EasyOCR recommended on Python 3.14):

```bash
pip install easyocr torch   # GPU used when CUDA available
```

**VLM** (GPU required — RTX 3050 4 GB works with 4-bit Qwen):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers bitsandbytes accelerate qwen-vl-utils
```

Model: `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit). Close other GPU apps before long evals.

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

### Sanity (n=10)

```bash
python scripts/run_ocr_eval.py \
  --manifest data/manifests/textocr_debug.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_1.0 area_0.5 area_0.25 area_0.125 kb_500 kb_200 kb_100 kb_50 patches_2 patches_4 patches_8 \
  --limit 10 --experiment-stage sanity --overwrite

for method in resize overview_only random uniform bops; do
  python scripts/run_vlm_eval.py \
    --manifest data/manifests/docvqa_debug.jsonl \
    --method $method --num-patches 2 --limit 10 \
    --experiment-stage sanity --overwrite
done
python scripts/merge_vlm_metrics.py
```

### Pilot (real inference)

```bash
# OCR — resumable; checkpoints every 20 images (re-run same command to continue)
python scripts/run_ocr_eval.py \
  --manifest data/manifests/textocr_pilot.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_1.0 area_0.5 area_0.25 area_0.125 kb_500 kb_200 kb_100 kb_50 patches_2 patches_4 patches_8 \
  --limit 200 --experiment-stage pilot --checkpoint-every 20

# VLM — one CSV per method (no overwrite unless --overwrite)
for method in resize overview_only random uniform bops; do
  python scripts/run_vlm_eval.py \
    --manifest data/manifests/docvqa_pilot.jsonl \
    --method $method --num-patches 2 --limit 100 \
    --experiment-stage pilot --overwrite
done
python scripts/merge_vlm_metrics.py
```

### Paper assets

```bash
python scripts/make_paper_assets.py      # tables (pilot/paper rows only)
python scripts/generate_plots.py         # cer_vs_budget.png
python scripts/bootstrap_pilot_stats.py  # paired 95% CIs
python scripts/analyze_failures.py       # VLM failure cases
```

### Orchestration (explicit dry-run vs real)

```bash
python scripts/run_full_experiment.py --phase debug --dry-run
python scripts/run_full_experiment.py --phase pilot --real
```

---

## Repository layout

```
├── src/
│   ├── data/              # Manifest builders, validation, dataset loader
│   ├── preprocessing/     # Resize, compression, BOPS patch selection
│   ├── ocr/               # OCR backends, merge_patch_ocr, metrics
│   ├── vlm/               # Qwen harness, patch diagnostics, QA metrics
│   ├── metrics/           # Aggregation + bootstrap statistical tests
│   ├── visualization/     # Budget degradation plots
│   └── utils/             # paths, budget_check, budget_compat, experiment_io, ocr_cache
├── scripts/
│   ├── run_ocr_eval.py           # OCR eval (resumable checkpoints)
│   ├── run_vlm_eval.py           # VLM eval (per-method CSV)
│   ├── merge_vlm_metrics.py      # Combine VLM CSVs
│   ├── bootstrap_pilot_stats.py  # Paired bootstrap CIs
│   ├── make_paper_assets.py      # Paper table generation
│   └── run_full_experiment.py    # Phase orchestration (--dry-run / --real)
├── configs/               # Experiment YAML files
├── data/                  # Manifests + raw data (large files gitignored)
├── outputs/               # Metrics, plots, checkpoints, cache (gitignored)
├── paper/                 # draft.tex, advisor deck, table CSVs
├── tests/                 # Unit tests (31 tests)
├── RESULTS.md             # Living experiment log
└── LOGIC.md               # Architecture reference
```

---

## Metrics (canonical definitions)

| Metric | Formula / rule |
|--------|----------------|
| **Word recall (v1)** | matched normalized GT tokens ÷ total GT tokens (primary OCR sanity metric) |
| **CER / WER** | Standard character/word error rate (secondary) |
| **Exact Match** | Normalized prediction equals any reference answer |
| **ANLS** | Average normalized Levenshtein similarity |
| **byte_utilization** | `actual_bytes / target_bytes` for compression baselines |

Implementation: `src/ocr/ocr_metrics.py`, `src/vlm/qa_metrics.py`, `src/metrics/statistical_tests.py`.

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

---

## Paper artifacts

| File | Purpose |
|------|---------|
| `paper/draft.tex` | Paper draft |
| `paper/advisor_deck.md` | 10-slide summary |
| `paper/tables/table_ocr_budget.csv` | OCR means by method × budget (pilot) |
| `paper/tables/table_vlm_patches.csv` | VLM means by method × K (pilot) |
| `paper/tables/bootstrap_ci.csv` | Paired 95% confidence intervals |
| `RESULTS.md` | Full results log with gate verdicts |

---

## Current status (2026-07-02)

| Component | Status |
|-----------|--------|
| Pipeline code | Complete |
| Unit tests | **31/31** passing |
| Dataset audit | Passed (21,778 TextOCR, 500 DocVQA) |
| Real OCR (sanity n=10 + pilot n=200) | ✅ EasyOCR GPU |
| Real VLM (sanity n=10 + pilot n=100) | ✅ Qwen 4-bit on RTX 3050 |
| OCR pilot gate | **Passed** — BOPS p8 > resize a0.25 (bootstrap CI significant) |
| VLM pilot gate | **Inconclusive** — BOPS ≈ random; loses to uniform/resize |
| Paper-scale runs | Not started (`experiment_stage=paper`) |

**Pilot direction:** empirical study (clear OCR gains; VLM patch selection needs improvement). See [RESULTS.md](RESULTS.md) for numbers and claim boundaries.

**Not implemented:** seam carving (optional Phase 12 in proposal only).

---

## Reference

Full research proposal: `Reference/Full Research Proposal and Implementation Plan.md`
