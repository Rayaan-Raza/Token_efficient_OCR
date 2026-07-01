# BOPS Experiment Results

Living log of dataset audits, preprocessing checks, OCR metrics, VLM QA scores, and method comparisons.

**Last updated:** 2026-06-30  
**Pipeline status:** Real Qwen VLM verified on RTX 3050 (4 GB) — OCR still dry-run  
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU · PyTorch `2.12.1+cu126`  
**Unit tests:** 14/14 passing (`python -m pytest tests/ -q`)

> **OCR** still uses `--dry-run` until EasyOCR is installed. **VLM** runs locally via Qwen2.5-VL-3B (4-bit). Close other GPU apps (games, etc.) before long evals — first sample took ~51 min with Rainbow Six sharing the GPU; second took ~88 s.

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

## 4. OCR results (TextOCR) — dry-run

**Raw CSV:** `outputs/metrics/ocr_metrics.csv` (1,250 rows)  
**Paper table:** `paper/tables/table_ocr_budget.csv`  
**Plot:** `outputs/plots/cer_vs_budget.png`

### 4.1 Summary

| Stat | Value |
|------|-------|
| Samples | 50 (`textocr_pilot.jsonl`) |
| Methods | original, resize, jpeg, webp, bops |
| Budgets | area_1.0, area_0.5, area_0.25, kb_500, kb_200 |
| Mean runtime | 0.755 s/sample (preprocessing + dry OCR) |
| invalid_budget rows | 187 / 1,250 (excluded from paper table) |

### 4.2 Aggregated means (valid budget only)

From `paper/tables/table_ocr_budget.csv`. Empty cells = method/budget combo not applicable.

| method | budget | CER | WER | n |
|--------|--------|-----|-----|---|
| original | area_1.0 | 1.000 | 1.000 | 50 |
| original | area_0.5 | 1.000 | 1.000 | 50 |
| original | area_0.25 | 1.000 | 1.000 | 50 |
| original | kb_500 | 1.000 | 1.000 | 50 |
| original | kb_200 | 1.000 | 1.000 | 50 |
| resize | area_1.0 | 1.000 | 1.000 | 50 |
| resize | area_0.5 | 1.000 | 1.000 | 50 |
| resize | area_0.25 | 1.000 | 1.000 | 50 |
| jpeg | kb_500 | 1.000 | 1.000 | 1 |
| jpeg | kb_200 | 1.000 | 1.000 | 6 |
| webp | kb_500 | 1.000 | 1.000 | 1 |
| webp | kb_200 | 1.000 | 1.000 | 5 |
| bops | area_1.0 | 1.000 | 1.000 | 50 |
| bops | area_0.5 | 1.000 | 1.000 | 50 |
| bops | area_0.25 | 1.000 | 1.000 | 50 |
| bops | kb_500 | 1.000 | 1.000 | 50 |
| bops | kb_200 | 1.000 | 1.000 | 50 |

*CER/WER = 1.0 because `--dry-run` skips OCR and uses empty predictions.*

### 4.3 Re-run with real OCR

```bash
pip install easyocr
python scripts/run_ocr_eval.py \
  --manifest data/manifests/textocr_pilot.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_1.0 area_0.5 area_0.25 kb_500 kb_200 \
  --limit 50
```

---

## 5. VLM results (DocVQA)

**Raw CSV:** `outputs/metrics/vlm_metrics.csv`  
**Paper table:** `paper/tables/table_vlm_patches.csv`

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

### 5.3 Larger eval (recommended next)

```bash
python scripts/run_vlm_eval.py \
  --manifest data/manifests/docvqa_pilot.jsonl \
  --method resize --limit 10
```

---

## 6. Failure analysis

| failure_type | count |
|--------------|-------|
| other | 10 |

All failures are dry-run placeholders (`prediction = "dry-run"`).

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
| Real OCR inference | ⏳ needs EasyOCR |
| Real Qwen inference | ⏳ needs CUDA GPU |

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
