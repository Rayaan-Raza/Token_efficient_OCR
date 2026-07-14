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
| `qe_bops` / `qe_bops_node_pair` | Question-conditioned evidence selection (G3 track) | `patches_K` |

**QE-BOPS / learned reranking (DocVQA evidence track):** See [LOGIC.md](LOGIC.md) §17. Heuristic G3 **failed/closed**. **G3-learned PASS** on docvqa_500 OOF. **G4 VLM PASS** on docvqa_100 (`lgbm_strict` OOF: ANLS +0.051 vs Q-BOPS). **G5 next:** docvqa_300 — watch **BM25-only** as headline comparator.

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
python scripts/generate_paper_figures.py  # paper/figures/*.pdf
python scripts/analyze_failures.py       # VLM failure cases
```

### Orchestration (explicit dry-run vs real)

```bash
python scripts/run_full_experiment.py --phase debug --dry-run
python scripts/run_full_experiment.py --phase pilot --real
```

### Learned evidence ranker (post-heuristic G3)

Heuristic QE-BOPS closed: Q-BOPS-fair is the strongest hand-built selector. Train a learned ranker next:

```bash
# Debug plumbing (n=100, OOF) — not for paper claims
python scripts/build_ranker_dataset.py --manifest Data/manifests/docvqa_100.jsonl --split-by image_id
python scripts/train_logreg_ranker.py --target strict_positive --from-dataset
python scripts/train_lgbm_ranker.py --objective lambdarank --target strict_positive --cv 5
python scripts/eval_learned_ranker_coverage.py \
  --manifest Data/manifests/docvqa_100.jsonl \
  --models lgbm_strict,lgbm_any,lgbm_combined,lgbm_qbops_hybrid \
  --baselines bops_qa_fair_pool,qe_bops_v2,bm25_only,bops_fair_pool \
  --k 1,2,4,8 --oof

# Paper path: repeat on docvqa_500 with --final-train / --held-out
```

Gate: learned strict@K > Q-BOPS strict@K and any@K ≥ Q-BOPS any@K (same eval images). **G4 VLM PASS** on docvqa_100. **G5:** `scripts/run_g5_vlm_pilot.ps1` on docvqa_300.

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
| `paper/tables/table_runtime.csv` | OCR/VLM runtime summary (pilot) |
| `paper/figures/*.pdf` | Publication figures (see below) |
| `RESULTS.md` | Full results log with gate verdicts |

**Paper figures** (regenerate with `python scripts/generate_paper_figures.py`):

| File | Content |
|------|---------|
| `figures/bops_pipeline.pdf` | Method schematic |
| `figures/ocr_word_recall_budget.pdf` | Primary OCR claim (word recall) |
| `figures/vlm_anls_methods.pdf` | VLM ANLS + EM bar chart |
| `figures/answer_coverage_diagnostics.pdf` | 76% vs 6% answer-in-OCR diagnostic |
| `figures/failure_panel.pdf` | Qualitative patch-selection failures |
| `figures/runtime_comparison.pdf` | OCR/VLM runtime cost |

---

## Current status (2026-07-15)

| Component | Status |
|-----------|--------|
| Pipeline code | Complete |
| Unit tests | Passing (`python -m pytest tests/ -q`) |
| QE-BOPS G1–G2 | **Passed** (candidate pool + oracle ceilings on docvqa_100) |
| QE-BOPS G3 heuristic | **Failed / closed** — no QE variant beat Q-BOPS at K=2 or K=4 |
| Learned evidence ranker | **G3-learned PASS** on docvqa_500 OOF — `lgbm_strict` +5.8pp / +6.6pp @ K=2 |
| G4 VLM (docvqa_100, K=2) | **PASS (strong)** — ANLS 0.829 vs Q-BOPS 0.778 (+0.051) |
| G5 VLM (docvqa_300, K=2) | **Next** — BM25-only (0.824 @ n=100) is key baseline |
| Dataset audit | Passed (21,778 TextOCR, 500 DocVQA) |
| Real OCR (sanity n=10 + pilot n=200) | ✅ EasyOCR GPU |
| Real VLM (G4 pilot n=100 + earlier sanity) | ✅ Qwen 4-bit on RTX 3050 |
| OCR pilot gate | **Passed** — BOPS p8 > resize a0.25 (bootstrap CI significant) |
| Paper-scale VLM | G5 n=300 then n=500 if G5 passes |

**Paper direction:** learned evidence reranking for budgeted Document VQA — coverage gains transfer to VLM. See [RESULTS.md](RESULTS.md).

**Not implemented:** seam carving (optional Phase 12 in proposal only).

---

## Reference

Full research proposal: `Reference/Full Research Proposal and Implementation Plan.md`  
QE-BOPS roadmap: `deep-research-report.md`
