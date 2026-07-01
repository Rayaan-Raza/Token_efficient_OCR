# BOPS Experiment Results

Living log of dataset audits, preprocessing checks, OCR metrics, VLM QA scores, and method comparisons.

**Last updated:** 2026-07-01  
**Pipeline status:** Real OCR (EasyOCR GPU) + real Qwen VLM verified — **sanity n=10 only, not final paper results**  
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU · PyTorch `2.12.1+cu126`  
**Unit tests:** 25/25 passing (`python -m pytest tests/ -q`)

> **Sanity checkpoint passed:** The experiment pipeline now produces valid research data (no dry-run rows in sanity CSVs). Paper tables remain empty until `experiment_stage=pilot` or `paper` runs. Do **not** claim BOPS improves OCR/VLM from these n=10 numbers alone.

---

## Real sanity run (2026-07-01) — n=10, not final

Commands executed:

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
python scripts/make_paper_assets.py
python scripts/generate_plots.py
python scripts/analyze_failures.py
```

| Track | Output | Rows | dry_run | Notes |
|-------|--------|------|---------|-------|
| OCR | `outputs/metrics/ocr_metrics_textocr_debug_sanity.csv` | 550 (160 applicable) | 0 | EasyOCR GPU; 390 `not_applicable` skipped |
| VLM | `outputs/metrics/vlm_metrics_merged.csv` | 50 (5 methods × 10) | 0 | Per-method CSVs under `outputs/metrics/vlm_metrics_docvqa_debug_*` |
| Paper tables | `paper/tables/*.csv` | 0 eligible | — | Filter requires `experiment_stage` ∈ {pilot, paper} |
| Plot | `outputs/plots/cer_vs_budget.png` | — | — | From sanity OCR CSV (includes sanity rows) |

### OCR sanity — word_recall (primary metric)

Mean word recall over 10 TextOCR samples (`not_applicable=false`, `dry_run=false`):

| method | budget | word_recall |
|--------|--------|-------------|
| original | area_1.0 | 0.317 |
| resize | area_1.0 | 0.317 |
| resize | area_0.5 | 0.269 |
| resize | area_0.25 | 0.177 |
| resize | area_0.125 | 0.102 |
| bops | patches_2 | 0.207 |
| bops | patches_4 | 0.235 |
| bops | patches_8 | 0.313 |

**Sanity pass criteria (primary):**

| Check | Result |
|-------|--------|
| `original` > `resize area_0.25` | ✅ 0.317 > 0.177 |
| `resize area_0.5` > `area_0.25` | ✅ 0.269 > 0.177 |
| OCR non-empty rate | ✅ 98.9% |
| BOPS `invalid_budget` | ✅ false for all patch rows |
| BOPS patches_2 → patches_8 trend | ✅ 0.207 → 0.313 (saturation OK) |

JPEG/WebP byte budgets hit `invalid_budget=true` on this small set (compression missed ±2% targets) but word_recall means are populated (kb_500 ≈ 0.328, kb_200 ≈ 0.325).

### VLM sanity — DocVQA n=10, K=2

Mean scores (`dry_run=false`, `experiment_stage=sanity`):

| method | EM | ANLS | mean runtime (s) |
|--------|-----|------|------------------|
| resize | 0.60 | 0.833 | 5.2 |
| overview_only | 0.20 | 0.329 | 3.9 |
| random | 0.20 | 0.200 | 4.8 |
| uniform | 0.20 | 0.374 | 8.2 |
| bops | 0.20 | 0.200 | 17.7 |

**Sanity pass criteria:**

| Check | Result |
|-------|--------|
| No dry_run rows | ✅ |
| `raw_prediction` / `parsed_prediction` populated | ✅ |
| `resize` not near-zero everywhere | ✅ EM=0.60 |
| `bops` not obviously worse than `random` | ✅ same EM/ANLS on n=10 |
| `overview_only` weaker on detail questions | ✅ observed on sample 3 (ITC vs ITC Limited) |
| Runtime tolerable (n=3 gate) | ✅ ~60–90 s first sample (model load), then 1–3 s/sample |

---

## Run log (2026-06-30)

Commands executed:

```bash
python scripts/run_full_experiment.py --phase pilot
python scripts/run_full_experiment.py --phase ablation
python scripts/run_full_experiment.py --phase paper
python scripts/audit_datasets.py
python scripts/make_paper_assets.py
python scripts/generate_plots.py
```

| Phase | OCR rows | VLM rows | Plot | Audit |
|-------|----------|----------|------|-------|
| pilot | 300 | — | ✅ | — |
| ablation | 15 (BOPS patches_4 × 3) | 50 (patch sweep 0/2/4/8/12) | ✅ | — |
| paper | 1,250 (50 samples × 5 methods × 5 budgets) | 10 (bops @ 4 patches, last write) | ✅ | ✅ |

**Outputs:**
- `outputs/metrics/ocr_metrics.csv` — 1,250 rows
- `outputs/metrics/vlm_metrics.csv` — 10 rows (last paper-phase method)
- `outputs/plots/cer_vs_budget.png`
- `paper/tables/table_ocr_budget.csv`
- `paper/tables/table_vlm_patches.csv`

---

## 1. Experiment design

### Methods compared

| Method | Type | Budget axis | Task |
|--------|------|-------------|------|
| `original` | Baseline | area / byte | OCR (TextOCR) |
| `resize` | Baseline | area ratio (±3% pixels) | OCR + VLM |
| `jpeg` | Baseline | byte target (±2%) | OCR |
| `webp` | Baseline | byte target (±2%) | OCR |
| `bops` | Proposed | patches + overview | OCR + VLM |
| `overview_only` | Ablation | 0 patches | VLM |
| `random` | Ablation | K patches | VLM |
| `uniform` | Ablation | K patches | VLM |

### Datasets & manifests

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `data/manifests/textocr_debug.jsonl` | 50 | Smoke / debug OCR |
| `data/manifests/textocr_pilot.jsonl` | 200 | Pilot OCR curves |
| `data/manifests/docvqa_debug.jsonl` | 20 | Smoke / debug VLM |
| `data/manifests/docvqa_pilot.jsonl` | 100 | Pilot VLM + ablations |
| `data/manifests/docvqa_val_500.jsonl` | 500 | Full DocVQA validation export |

### Metrics

| Metric | Definition | Used for |
|--------|------------|----------|
| **CER** | Character error rate | OCR |
| **WER** | Word error rate | OCR |
| **Word recall (v1)** | matched normalized GT tokens / total GT tokens | OCR |
| **Exact Match (EM)** | Normalized string match to any reference answer | DocVQA |
| **ANLS** | Average normalized Levenshtein similarity | DocVQA |
| **invalid_budget** | Row excluded if pixel ±3%, byte ±2%, or patch count ≠ target | Fairness filter |
| **not_applicable** | Method does not use this budget axis (e.g. `jpeg` + `area_0.25`) | Compatibility filter |
| **experiment_stage** | `debug`, `sanity`, `pilot`, `paper` — paper tables use pilot/paper only | Run tagging |

---

## 2. Dataset audit (Phase 2A)

**Gate:** passed  
**Report:** `outputs/audit/dataset_audit_report.json`

| Dataset | Check | Result |
|---------|-------|--------|
| TextOCR | Total images | 21,778 |
| TextOCR | Missing on disk | 0 |
| TextOCR | Sample checks (n=10) | All pass |
| DocVQA | Manifest rows | 500 |
| DocVQA | PNG files on disk | 500 |
| DocVQA | Duplicate IDs | 0 |

---

## 3. Preprocessing smoke test (Phase 1)

```bash
python scripts/run_preprocessing.py --config configs/smoke_test.yaml
```

| Method | Area ratio | Target pixels | Actual pixels | invalid_budget |
|--------|------------|---------------|---------------|----------------|
| resize | 0.5 | 1,990,674 | 1,988,965 | false |

---

## 4. OCR results (TextOCR)

**Latest real CSV:** `outputs/metrics/ocr_metrics_textocr_debug_sanity.csv` (550 rows, 160 applicable)  
**Legacy dry-run CSV:** `outputs/metrics/ocr_metrics.csv` (1,250 rows, CER/WER = 1.0)  
**Plot:** `outputs/plots/cer_vs_budget.png` (regenerated from sanity CSV)

Real OCR uses **EasyOCR** (GPU when CUDA available). BOPS runs merge overview + per-patch OCR via `merge_patch_ocr`.

### 4.1 Pilot-scale next step

```bash
python scripts/run_ocr_eval.py \
  --manifest data/manifests/textocr_pilot.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_1.0 area_0.5 area_0.25 area_0.125 kb_500 kb_200 kb_100 kb_50 patches_2 patches_4 patches_8 \
  --limit 50 --experiment-stage pilot
```

---

## 5. VLM results (DocVQA)

**Latest merged CSV:** `outputs/metrics/vlm_metrics_merged.csv` (50 sanity rows)  
**Per-method CSVs:** `outputs/metrics/vlm_metrics_docvqa_debug_{method}_{suffix}.csv`  
**Legacy dry-run CSV:** `outputs/metrics/vlm_metrics.csv`

### 5.1 Real Qwen inference (2026-06-30, RTX 3050)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install accelerate bitsandbytes transformers
python scripts/run_vlm_eval.py \
  --manifest data/manifests/docvqa_debug.jsonl \
  --method resize --num-patches 4 --limit 2
```

| image_id | question (short) | prediction | GT | EM* | ANLS* | runtime (s) |
|----------|------------------|------------|-----|-----|-------|-------------|
| docvqa_val_49153 | actual value per 1000, 1975? | 0.28 | 0.28 | 1.0 | 1.0 | 3055 |
| docvqa_val_24580 | name of university? | University of California | university of california | 1.0 | 0.71 | 88 |

\*Metrics after `parse_answer` strips the `assistant` prefix. Re-run eval to refresh CSV with cleaned predictions.

**4 GB VRAM tips:** Close GPU-heavy apps before eval; prefer `--method resize` or `--num-patches 2` for BOPS.

### 5.2 Prior dry-run aggregate (paper phase)

| method | num_patches | EM | ANLS | n |
|--------|-------------|-----|------|---|
| bops | 4 | 0.000 | 0.000 | 10 |

### 5.3 Pilot-scale next step

```bash
for method in resize overview_only random uniform bops; do
  python scripts/run_vlm_eval.py \
    --manifest data/manifests/docvqa_pilot.jsonl \
    --method $method --num-patches 2 --limit 20 \
    --experiment-stage pilot
done
python scripts/merge_vlm_metrics.py
```

---

## 6. Failure analysis

**Latest:** `outputs/failure_cases/vlm_failures.csv` — 36 VLM failures from sanity n=10 (EM=0 cases).  
Legacy dry-run failures were all `prediction = "dry-run"`.

---

## 7. Budget fairness

| Budget type | Tolerance | invalid rows (paper OCR) |
|-------------|-----------|--------------------------|
| Pixel (area) | ±3% | included in 187 total |
| Byte (JPEG/WebP) | ±2% | included in 187 total |
| Patch count | exact | 0 in BOPS ablation |

Invalid rows are excluded from `paper/tables/table_ocr_budget.csv`.

---

## 8. Phase checklist

| Phase | Status |
|-------|--------|
| 1 — Smoke test | ✅ |
| 2A — Dataset audit | ✅ |
| 2B — Manifests | ✅ |
| 3–8 — OCR pipeline (dry-run) | ✅ |
| 9–10 — VLM pipeline (dry-run) | ✅ |
| 11 — Ablations (dry-run) | ✅ |
| 13–16 — Paper tables + plot | ✅ |
| Real OCR inference (sanity n=10) | ✅ EasyOCR GPU |
| Real Qwen inference (sanity n=10) | ✅ Qwen 4-bit on RTX 3050 |
| Pilot-scale OCR/VLM | ⏳ next |
| Paper-scale tables | ⏳ needs `experiment_stage=paper` |

---

## 9. How to refresh this file

```bash
python scripts/run_full_experiment.py --phase paper   # add --dry-run until models ready
python scripts/make_paper_assets.py
python scripts/generate_plots.py
python scripts/analyze_failures.py
```

Update tables from `outputs/metrics/*.csv` and `paper/tables/*.csv`.

---

## 10. Qwen local inference

**Status:** Verified on RTX 3050 Laptop GPU (4 GB VRAM).

### One-time GPU setup (this machine)

PyTorch was CPU-only (`2.10.0+cpu`). Reinstalled with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install accelerate bitsandbytes transformers
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 3050 Laptop GPU
```

### How inference works (`src/vlm/run_vlm.py`)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit via `BitsAndBytesConfig` |
| Device | `device_map="auto"` (CUDA) |
| Image cap | Longest side ≤ 768 px (4 GB VRAM) |

**CLI (no `--dry-run`):**

```bash
python scripts/run_vlm_eval.py \
  --manifest data/manifests/docvqa_debug.jsonl \
  --method resize --limit 2
```

First run downloads ~7 GB of model weights from Hugging Face, then caches locally.

**What `--dry-run` does:** Skips `load_vlm()` and writes `prediction = "dry-run"`.
